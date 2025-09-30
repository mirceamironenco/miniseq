import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from miniseq import configs as cfg
from miniseq.data import PreferenceBatch, load_hf_pretrained_tokenzier, log_tokenizer
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
from miniseq.recipes._common import calculate_total_steps, setup_torch
from miniseq.recipes._setup_model import setup_model, setup_reference_model
from miniseq.recipes.algorithm.preference import DPOFinetuneUnit


@dataclass(kw_only=True, frozen=True)
class DPOConfig:
    beta: float = 0.1

    nll_scale: float = 0.1

    length_normalize: bool = False


@dataclass(kw_only=True)
class PreferenceRecipeConfig(cfg.TrainRecipeConfig):
    """Recipe config for preference fine-tuning. Example usage:
    ```
    import dataclasses
    from miniseq import cli, recipes
    from miniseq import configs as cfg

    @dataclasses.dataclass
    class Config(recipes.PreferenceRecipeConfig):
        # --- Override fields from PreferenceRecipeConfig below ---
        optimizer: cfg.OptimizerConfig = ...
        lr_scheduler: cfg.LRSchedulerConfig = ...
        model: cfg.ModelLoadConfig = ...
        train_data: cfg.data.PreferenceDatasetConfig = ...
        train: cfg.TrainConfig = ...
        dp: cfg.DPStrategyConfig = ...
        compile: cfg.CompileConfig = ...
        lora: LoRAConfig | None = ...
        wandb: cfg.WandbConfig | None = ...
        profiler: cfg.TorchProfilerConfig | None = ...
        dpo: DPOConfig = ...
        packed: bool = ...
        seed: int = ...

    config = cli.run_default_cli(Config)
    trainer = recipes.create_preference_trainer(config)
    trainer.run()
    ```
    """

    optimizer: cfg.OptimizerConfig = cfg.optim.AdamWConfig(lr=1e-5, weight_decay=0.01)

    lr_scheduler: cfg.LRSchedulerConfig = cfg.scheduler.LinearWarmupCosineConfig(
        warmup_steps=5
    )

    model: cfg.PretrainedModelConfig = cfg.PretrainedModelConfig(
        name="qwen2.5-1.5b-instruct", use_cce=True, dtype=torch.bfloat16
    )

    train_data: cfg.data.PreferenceDatasetConfig = cfg.data.PreferenceDatasetConfig(
        name="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        preference_map=lambda item: {
            "prompt": item["prompt"],
            "chosen": item["chosen"][-1]["content"],
            "rejected": item["rejected"][-1]["content"],
        },
    )

    train: cfg.TrainConfig = cfg.TrainConfig(
        micro_batch_size=4, device_batch_size=4, ac=True, no_sync=False
    )

    dp: cfg.DPStrategyConfig = cfg.dp.FSDP2Config(fsdp2_reshard_fwd=False)

    compile: cfg.CompileConfig = cfg.CompileConfig(model=True, loss=False, dynamic=True)

    lora: LoRAConfig | None = None

    wandb: cfg.WandbConfig | None = None

    profiler: cfg.TorchProfilerConfig | None = None

    dpo: DPOConfig = DPOConfig(beta=0.1)

    packed: bool = True

    seed: int = 2


def create_preference_trainer(
    config: PreferenceRecipeConfig, log: logging.Logger | None = None
) -> Trainer[PreferenceBatch, PreferenceBatch]:
    if log is None:
        setup_logging(debug=False)

        log = get_logger()

    log_config(log, config)

    setup_torch(expandable_segments=False)

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

    model_config = ModelConfig.from_model_name(
        config.model.name,
        flex_attention=config.model.flex_attention,
        cut_cross_entropy=config.model.use_cce,
        finetune_repo_id=config.model.finetune_repo_id,
        reduce_flex_stages=config.model.reduce_flex,
        flash_attention2=config.model.flash_attention2 or use_fa2,
    )

    state_dict = load_model_checkpoint(
        model_config, cache_dir=config.cache_dir, machine=machine
    )

    model = build_model(
        model_config, device=torch.device("meta"), dtype=config.model.dtype
    )

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

    ref_model: nn.Module | None = None

    if config.dpo.beta != 0.0:
        ref_config = ModelConfig.from_model_name(
            config.model.name,
            flex_attention=config.model.flex_attention,
            cut_cross_entropy=config.model.use_cce,
            finetune_repo_id=config.model.finetune_repo_id,
        )

        ref_state_dict = load_model_checkpoint(
            ref_config, cache_dir=config.cache_dir, machine=machine
        )

        ref_model = build_model(
            ref_config, device=torch.device("meta"), dtype=config.model.dtype
        )

        ref_model = setup_reference_model(
            ref_model,
            machine,
            state_dict=ref_state_dict,
            compile_config=config.compile,
        )

        ref_model.requires_grad_(False).eval()

    machine.barrier()

    dpo_unit = DPOFinetuneUnit(
        model,
        ref_model,
        beta=config.dpo.beta,
        nll_scale=config.dpo.nll_scale,
        length_normalization=config.dpo.length_normalize,
    )

    optimizer = config.optimizer.build(params=model.parameters())

    tokenizer = load_hf_pretrained_tokenzier(
        model_repo_id=model_config.base_model_repo_id,
        cache_dir=config.cache_dir,
        force_download=False,
        use_fast=True,
        set_none_pad_token_to_eos=True,
    )

    machine.barrier()

    dataset = config.train_data.build(cache_dir=config.cache_dir, seed=seed)

    train_loader = dataset.create_loader(
        tokenizer=tokenizer,
        machine=machine,
        batch_size=config.train.micro_batch_size,
        seed=seed,
        packed=config.packed,
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

    log_tokenizer(log, tokenizer)
    log_model(log, model)

    trainer = Trainer[PreferenceBatch].from_configs(
        recipe_config=config,
        model_config=model_config,
        train_unit=dpo_unit,
        train_loader=train_loader,
        valid_units=None,
        valid_loaders=None,
        generator=None,
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
