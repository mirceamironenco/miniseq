import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from miniseq import configs as cfg
from miniseq.data import SequenceBatch, load_hf_pretrained_tokenzier, log_tokenizer
from miniseq.generation import Generator
from miniseq.logging import get_logger, log_config, setup_logging
from miniseq.machine import Machine, setup_default_machine
from miniseq.models import (
    LoRAConfig,
    ModelConfig,
    build_model,
    load_model_checkpoint,
    log_model,
)
from miniseq.trainer import Trainer
from miniseq.training import set_seed

# isort: split
from miniseq.recipes._common import (
    calculate_total_steps,
    create_generator,
    create_prompt_evals,
    setup_torch,
)
from miniseq.recipes._setup_model import setup_model
from miniseq.recipes.algorithm import InstructionEvalUnit, InstructionUnit


@dataclass(kw_only=True)
class SFTRecipeConfig(cfg.TrainRecipeConfig):
    """Recipe config for supervised fine-tuning (SFT). Example usage:
    ```
    import dataclasses
    from miniseq import cli, recipes
    from miniseq import configs as cfg

    @dataclasses.dataclass
    class Config(recipes.SFTRecipeConfig):
        # --- Override fields from SFTRecipeConfig below ---
        optimizer: cfg.OptimizerConfig = ...
        lr_scheduler: cfg.LRSchedulerConfig = ...
        train: cfg.TrainConfig = ...
        train_data: cfg.data.SFTDatasetConfig = ...
        model: cfg.ModelLoadConfig = ...
        dp: cfg.DPStrategyConfig = ...
        compile: cfg.CompileConfig = ...
        validate: cfg.EvalTaskConfig | None = ...
        lora: LoRAConfig | None = ...
        wandb: cfg.WandbConfig | None = ...
        profiler: cfg.TorchProfilerConfig | None = ...
        packed: bool = ...
        seed: int = ...

    config = cli.run_default_cli(Config)
    trainer = recipes.create_finetune_trainer(config)
    trainer.run()
    ```
    """

    optimizer: cfg.OptimizerConfig = cfg.optim.AdamWConfig(lr=1e-6, weight_decay=0.01)

    lr_scheduler: cfg.LRSchedulerConfig = cfg.scheduler.LinearWarmupCosineConfig(
        warmup_steps=5
    )

    train: cfg.TrainConfig = cfg.TrainConfig(
        micro_batch_size=1, device_batch_size=4, no_sync=False, max_grad_norm=3.0
    )

    train_data: cfg.data.SFTDatasetConfig = cfg.data.SFTDatasetConfig(
        name="yahma/alpaca-cleaned",
        completions_only=True,
        packed_seqlen=4097,
        max_seqlen=2048,
        apply_chat_template=True,
        columns=("instruction", "output", "input", None),
        system_prompt=(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request."
        ),
    )

    model: cfg.PretrainedModelConfig = cfg.PretrainedModelConfig(
        name="qwen2.5-1.5b-instruct", use_cce=True, dtype=torch.bfloat16
    )

    dp: cfg.DPStrategyConfig = cfg.dp.FSDP2Config(fsdp2_reshard_fwd=True)

    compile: cfg.CompileConfig = cfg.CompileConfig(
        model=True, loss=False, dynamic=False
    )

    validate: cfg.EvalTaskConfig | None = None

    lora: LoRAConfig | None = None

    wandb: cfg.WandbConfig | None = None

    profiler: cfg.TorchProfilerConfig | None = None

    packed: bool = True

    seed: int = 2


def create_finetune_trainer(
    config: SFTRecipeConfig, log: logging.Logger | None = None
) -> Trainer[SequenceBatch, SequenceBatch]:
    if log is None:
        setup_logging(debug=False)

        log = get_logger()

    log_config(log, config)

    setup_torch(expandable_segments=True)

    machine: Machine = setup_default_machine(
        dp_replicate=config.dp.replicate,
        dp_shard=config.dp.shard,
        cpu_offloading=config.dp.cpu_offloading,
    )

    seed = config.seed

    set_seed(seed, torch.device("cpu"), machine.device)

    # If packed=True but no flex/fa2 specified, default to fa2.
    use_fa2 = False
    if config.packed:
        if not (config.model.flash_attention2 or config.model.flex_attention):
            use_fa2 = True

    model_config = ModelConfig.from_model_name(
        config.model.name,
        flex_attention=config.model.flex_attention,
        cut_cross_entropy=config.model.use_cce,
        finetune_repo_id=config.model.finetune_repo_id,
        reduce_flex_stages=config.model.reduce_flex,
        flash_attention2=config.model.flash_attention2 or use_fa2,
    )

    # Load the model on meta device on all ranks.
    model = build_model(
        model_config, device=torch.device("meta"), dtype=config.model.dtype
    )

    state_dict = load_model_checkpoint(
        model_config, cache_dir=config.cache_dir, machine=machine
    )

    log.info(f"Checkpoint for {model_config.base_model_repo_id} loaded.")

    # FSDP/DDP shard model, load state dict, activation checkpointing, model compile
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

    optimizer = config.optimizer.build(params=model.parameters())

    tokenizer = load_hf_pretrained_tokenzier(
        model_repo_id=model_config.base_model_repo_id,
        cache_dir=config.cache_dir,
        force_download=False,
        use_fast=True,
        set_none_pad_token_to_eos=True,  # only applied if pad token is None
    )

    dataset = config.train_data.build(cache_dir=config.cache_dir, seed=seed)

    machine.barrier()

    train_loader: Iterable[SequenceBatch] = dataset.create_loader(
        tokenizer=tokenizer,
        machine=machine,
        batch_size=config.train.micro_batch_size,
        seed=seed,
        packed=config.packed,
    )

    train_unit = InstructionUnit(model)

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

    generator: Generator | None = None

    valid_units, valid_loaders = [], []

    if config.validate is not None:
        generator = create_generator(
            config.validate.generator,
            mp_dtype=config.model.dtype,
            machine=machine,
            model_config=model_config,
            model=model,
            cache_dir=config.cache_dir,
            seed=seed,
            stop_token_ids=[tokenizer.eos_token_id],
        )

        valid_units, valid_loaders = create_prompt_evals(
            generator=generator,
            data=config.validate.data,
            tokenizer=tokenizer,
            machine=machine,
            cache_dir=config.cache_dir,
            seed=seed,
            pass_k=config.eval_pass_k,
            avg_n=config.eval_avg_n,
        )

        generator.after_generation()

        machine.barrier()

    if "test" in dataset.splits():
        valid_units.append(InstructionEvalUnit(model))

        valid_loader = dataset.create_loader(
            tokenizer=tokenizer,
            machine=machine,
            batch_size=100,
            seed=seed,
            split="test",
            for_evaluation=True,
            packed=config.packed,
        )

        valid_loaders.append(valid_loader)

    log_tokenizer(log, tokenizer)
    log_model(log, model)

    trainer = Trainer[SequenceBatch].from_configs(
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
        requires_rollout=False,
        total_steps=total_train_steps,
        log=log,
        compile_optimizer_step=config.compile.optimizer_step,
    )

    return trainer
