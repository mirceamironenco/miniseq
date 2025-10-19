import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from miniseq import configs as cfg
from miniseq import register_models
from miniseq.data import (
    PromptBatch,
    TrajectoryBatch,
    load_hf_pretrained_tokenzier,
    log_tokenizer,
)
from miniseq.datasets.math_verify import MathVerifier
from miniseq.logging import get_logger, log_config, setup_logging
from miniseq.machine import Machine, setup_default_machine
from miniseq.models import (
    LoRAConfig,
    ModelConfig,
    build_model,
    load_model_checkpoint,
    log_model,
)
from miniseq.recipes.algorithm import GeneratorEvalUnit
from miniseq.trainer import Trainer
from miniseq.training import set_seed
from miniseq.transformer import TransformerDecoderModel

# isort: split
from miniseq.recipes._common import (
    calculate_total_steps,
    create_generator,
    create_prompt_evals,
    setup_torch,
)
from miniseq.recipes._setup_model import setup_model, setup_reference_model
from miniseq.recipes.algorithm.online import GRPOConfig, GRPOUnit


def extract_gsm8k_answer(item: dict[str, str]) -> str:
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", item["answer"])
    assert solution is not None
    final_solution = solution.group(0)
    ground_truth = final_solution.split("#### ")[1].replace(",", "")

    if "\\boxed" not in ground_truth:
        boxed_ground_truth = f"\\boxed{{{ground_truth}}}"
    else:
        boxed_ground_truth = ground_truth

    return boxed_ground_truth


@dataclass(kw_only=True)
class OnlineRecipeConfig(cfg.TrainRecipeConfig):
    """Recipe config for online/rl fine-tuning. Example usage:
    ```
    import dataclasses
    from miniseq import cli, recipes
    from miniseq import configs as cfg

    @dataclasses.dataclass
    class Config(recipes.OnlineRecipeConfig):
        # --- Override fields from OnlineRecipeConfig below ---
        train_data: cfg.data.PromptDatasetConfig = ...
        valid_data: cfg.data.RegisteredDatasetConfig | None = ...
        model: cfg.ModelLoadConfig = ...
        optimizer: cfg.OptimizerConfig = ...
        lr_scheduler: cfg.LRSchedulerConfig = ...
        dp: cfg.DPStrategyConfig = ...
        compile: cfg.CompileConfig = ...
        lora: LoRAConfig | None = ...
        wandb: cfg.WandbConfig | None = ...
        profiler: cfg.TorchProfilerConfig | None = ...
        grpo: GRPOConfig = ...
        train: cfg.TrainConfig = ...
        generator: cfg.GeneratorConfig = ...
        sync_weights_every: int = ...
        packed: bool = ...
        prefix_share: bool = ...
        seed: int = ...

    config = cli.run_default_cli(Config)
    trainer = recipes.create_online_trainer(config)
    trainer.run()
    ```
    """

    train_data: cfg.data.PromptDatasetConfig = cfg.data.PromptDatasetConfig(
        path="openai/gsm8k",
        configuration="main",
        prompt_keymap="question",
        answer_keymap=extract_gsm8k_answer,
        split="train",
        test_split="test",
        apply_chat_template=True,
        verifier_factory=lambda: MathVerifier(),
    )

    valid_data: cfg.data.RegisteredDatasetConfig | None = None

    model: cfg.PretrainedModelConfig = cfg.PretrainedModelConfig(
        name="qwen2.5-math-1.5b-instruct", use_cce=True, dtype=torch.bfloat16
    )

    optimizer: cfg.OptimizerConfig = cfg.optim.AdamWConfig(lr=1e-6, weight_decay=0.01)

    lr_scheduler: cfg.LRSchedulerConfig = cfg.scheduler.LinearWarmupCosineConfig(
        warmup_steps=50
    )

    dp: cfg.DPStrategyConfig = cfg.dp.FSDP2Config(fsdp2_reshard_fwd=True)

    compile: cfg.CompileConfig = cfg.CompileConfig(model=True, dynamic=True)

    lora: LoRAConfig | None = None

    wandb: cfg.WandbConfig | None = None

    profiler: cfg.TorchProfilerConfig | None = None

    grpo: GRPOConfig = GRPOConfig(group_size=8)

    train: cfg.TrainConfig = cfg.TrainConfig(
        ac=True,
        max_epochs=None,
        max_steps=300,
        device_batch_size=64,
        rollout_batch_size=64,
        micro_batch_size=4,
        publish_metrics_every=2,
        no_sync=False,
        validate_at_start=False,
        validate_every=25,
    )

    generator: cfg.GeneratorConfig = field(
        default_factory=lambda: cfg.generation.VLLMConfig(
            engine=cfg.generation.VLLMEngineConfig(
                gpu_memory_utilization=0.3, enforce_eager=True
            ),
            sample=cfg.generation.VLLMSamplingConfig(
                temperature=1.0, top_p=1.0, max_tokens=1024
            ),
            eval_sample=cfg.generation.VLLMSamplingConfig(
                temperature=0.0, top_p=0.95, max_tokens=1024
            ),
        )
    )

    sync_weights_every: int = 1
    """How often to sync generator weights."""

    packed: bool = True

    prefix_share: bool = False
    """Enable prefix sharing packed batches."""

    seed: int = 2


def build_models(
    config: OnlineRecipeConfig, model_config: ModelConfig, machine: Machine
) -> tuple[TransformerDecoderModel, TransformerDecoderModel | None]:
    META = torch.device("meta")

    state_dict = load_model_checkpoint(
        model_config, cache_dir=config.cache_dir, machine=machine
    )

    ref_model = None

    if config.grpo.beta > 0.0:
        ref_model = build_model(model_config, device=META, dtype=config.model.dtype)

        # Parallelize, shard, compile, etc.
        ref_model = setup_reference_model(
            ref_model, machine, state_dict=state_dict, compile_config=config.compile
        )

        ref_model.requires_grad_(False).eval()

    model = build_model(model_config, device=META, dtype=config.model.dtype)

    model = setup_model(
        model,
        machine,
        mp_dtype=config.model.dtype,
        dp_config=config.dp,
        state_dict=state_dict,
        ac=config.train.ac,
        ac_freq=config.train.ac_freq,
        lora_config=config.lora,
        load_on_cpu=config.model.load_on_cpu,
        compile_config=config.compile,
    )

    return model, ref_model


def create_online_trainer(
    config: OnlineRecipeConfig, log: logging.Logger | None = None
) -> Trainer[TrajectoryBatch, PromptBatch]:
    register_models()

    if log is None:
        setup_logging(debug=False)

        log = get_logger()

    log_config(log, config)

    # expandable_segments=True not compatible with vLLM sleep mode.
    # https://github.com/vllm-project/vllm/pull/14189
    setup_torch(
        expandable_segments=False,
        capture_dynamic_shapes=True,
        capture_scalar_outputs=True,
    )

    machine: Machine = setup_default_machine(
        dp_replicate=config.dp.replicate,
        dp_shard=config.dp.shard,
        cpu_offloading=config.dp.cpu_offloading,
    )

    seed = config.seed

    set_seed(seed, torch.device("cpu"), machine.device)

    # If packed=True but no flex/fa2 specified, default to flex.
    use_fa2 = False
    if config.packed:
        if not (config.model.flash_attention2 or config.model.flex_attention):
            use_fa2 = True

    if config.prefix_share and (not config.model.flex_attention):
        raise ValueError("Prefix sharing only enabled for flex_attention.")

    model_config = ModelConfig.from_model_name(
        name=config.model.name,
        flex_attention=config.model.flex_attention,
        cut_cross_entropy=config.model.use_cce,
        finetune_repo_id=config.model.finetune_repo_id,
        reduce_flex_stages=config.model.reduce_flex,
        flash_attention2=config.model.flash_attention2 or use_fa2,
    )

    log.info(f"Loading tokenizer for model {model_config.base_model_repo_id}.")

    tokenizer = load_hf_pretrained_tokenzier(
        model_repo_id=model_config.base_model_repo_id,
        cache_dir=config.cache_dir,
        force_download=False,
        use_fast=True,
        set_none_pad_token_to_eos=True,
    )

    model, ref_model = build_models(config, model_config, machine)

    generator = create_generator(
        config.generator,
        mp_dtype=config.model.dtype,
        machine=machine,
        model_config=model_config,
        model=model,
        cache_dir=config.cache_dir,
        seed=seed,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    machine.barrier()

    optimizer = config.optimizer.build(params=model.parameters())

    dataset = config.train_data.build(cache_dir=config.cache_dir, seed=seed)

    batch_size = config.train.rollout_batch_size or config.train.device_batch_size

    assert config.sync_weights_every > 0
    assert batch_size > 0

    train_loader: Iterable[PromptBatch] = dataset.create_loader(
        tokenizer=tokenizer,
        machine=machine,
        batch_size=batch_size,
        seed=seed,
        for_evaluation=False,
    )

    total_train_steps = calculate_total_steps(
        loader=train_loader,
        max_steps=config.train.max_steps,
        max_epochs=config.train.max_epochs,
        grad_accum_steps=config.train.grad_accum_steps,
    )

    # Note: total_train_steps is an estimate in most cases; if max_epochs is
    # specified, training might terminate before total_train_steps
    # lr_schedulers that are sensitive to max_steps should specify their own estimate,
    # and have it take proirity over `max_num_steps`.
    lr_scheduler = config.lr_scheduler.build(
        optimizer=optimizer, max_num_steps=total_train_steps
    )

    reward_map = dataset.create_scorer(tokenizer=tokenizer)

    train_unit = GRPOUnit(
        model,
        machine,
        generator,
        trajectory_size=config.train.micro_batch_size,
        reference_model=ref_model,
        group_size=config.grpo.group_size,
        completion_scorer=reward_map,
        std_normalize_advantage=config.grpo.std_normalize_advantage,
        mu=config.grpo.mu,
        clip_eps_low=config.grpo.clip_eps_low,
        clip_eps_high=config.grpo.clip_eps_high,
        beta=config.grpo.beta,
        loss=config.grpo.loss,
        rollout_correction=config.grpo.rollout_correction,
        packed=config.packed,
        prefix_sharing=config.prefix_share,
        pad_index=tokenizer.pad_token_id or 0,
    )

    machine.barrier()

    valid_units, valid_loaders = [], []

    if config.valid_data is not None:
        valid_units, valid_loaders = create_prompt_evals(
            generator=generator,
            data=config.valid_data,
            tokenizer=tokenizer,
            machine=machine,
            cache_dir=config.cache_dir,
            seed=seed,
            pass_k=config.eval_pass_k,
            avg_n=config.eval_avg_n,
        )

    if "test" in dataset.splits():
        eval_loader = dataset.create_loader(
            machine=machine,
            batch_size=100,
            seed=seed,
            tokenizer=tokenizer,
            for_evaluation=True,
            split="test",
        )

        valid_loaders.append(eval_loader)

        valid_units.append(
            GeneratorEvalUnit(
                generator,
                completion_scorer=dataset.create_scorer(tokenizer=tokenizer),
                name="validation",
                pass_k=config.eval_pass_k,
                avg_n=config.eval_avg_n,
            )
        )

    log_tokenizer(log, tokenizer)
    log_model(log, model)

    trainer = Trainer[TrajectoryBatch, PromptBatch].from_configs(
        recipe_config=config,
        model_config=model_config,
        train_unit=train_unit,
        train_loader=train_loader,
        valid_units=valid_units,
        valid_loaders=valid_loaders,
        generator=generator,
        machine=machine,
        seed=seed,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        requires_rollout=True,
        total_steps=total_train_steps,
        log=log,
        compile_optimizer_step=config.compile.optimizer_step,
        rollout_sync_steps=config.sync_weights_every,
    )

    return trainer
