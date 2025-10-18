import logging
from dataclasses import dataclass, field

import torch

from miniseq import configs as cfg
from miniseq import register_models
from miniseq.data import load_hf_pretrained_tokenzier, log_tokenizer
from miniseq.evaluator import Evaluator, build_evaluator
from miniseq.generation import VLLMEngineConfig, VLLMSamplingConfig
from miniseq.logging import create_rich_progress, get_logger, log_config, setup_logging
from miniseq.machine import Machine, setup_default_machine
from miniseq.models import ModelConfig, build_model, load_model_checkpoint, log_model
from miniseq.trainer import create_metric_writers
from miniseq.training import Profiler, create_memory_tracker, set_seed

# isort: split
from miniseq.recipes._common import create_generator, create_prompt_evals, setup_torch
from miniseq.recipes._setup_model import setup_reference_model


@dataclass(kw_only=True)
class EvalRecipeConfig(cfg.EvalRecipeConfig):
    data: cfg.data.RegisteredDatasetConfig = field(
        default_factory=lambda: cfg.data.RegisteredDatasetConfig(
            datasets=("aime24",), batch_size=100
        )
    )

    model: cfg.PretrainedModelConfig = cfg.PretrainedModelConfig(
        name="qwen2.5-math-1.5b-instruct"
    )

    generator: cfg.GeneratorConfig = field(
        default_factory=lambda: cfg.generation.VLLMConfig(
            engine=VLLMEngineConfig(gpu_memory_utilization=0.7),
            sample=VLLMSamplingConfig(
                temperature=0.0, top_p=1.0, top_k=-1, max_tokens=3600
            ),
        )
    )

    compile: cfg.CompileConfig = cfg.CompileConfig(model=True, dynamic=True)


def create_evaluator(
    config: EvalRecipeConfig, log: logging.Logger | None = None
) -> Evaluator:
    register_models()

    if log is None:
        setup_logging(debug=False)

        log = get_logger()

    log_config(log, config)

    setup_torch(expandable_segments=False)

    machine: Machine = setup_default_machine(
        cpu_offloading=False, dp_replicate=config.dp.replicate
    )

    memory_tracker = create_memory_tracker(device=machine.device)

    metric_writers = create_metric_writers(
        cache_dir=config.cache_dir,
        machine=machine,
        logger=log,
        tensorboard=config.tensorboard,
        wandb=config.wandb,
        max_per_row=-1,
    )

    profiler: Profiler | None = None

    if config.profiler is not None:
        profiler = config.profiler.build(cache_dir=config.cache_dir, machine=machine)

    seed = config.seed

    set_seed(seed, torch.device("cpu"), machine.device)

    model_config = ModelConfig.from_model_name(
        config.model.name,
        flex_attention=config.model.flex_attention,
        cut_cross_entropy=config.model.use_cce,
        finetune_repo_id=config.model.finetune_repo_id,
    )

    tokenizer = load_hf_pretrained_tokenzier(
        model_repo_id=model_config.base_model_repo_id,
        cache_dir=config.cache_dir,
        force_download=False,
        use_fast=True,
        set_none_pad_token_to_eos=True,
    )

    machine.barrier()

    state_dict = load_model_checkpoint(
        model_config, cache_dir=config.cache_dir, machine=machine
    )

    model = build_model(
        model_config, device=torch.device("meta"), dtype=config.model.dtype
    )

    model = setup_reference_model(
        model, machine, state_dict=state_dict, compile_config=config.compile
    )

    model.eval()

    machine.barrier()

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

    units, loaders = create_prompt_evals(
        generator=generator,
        data=config.data,
        tokenizer=tokenizer,
        machine=machine,
        cache_dir=config.cache_dir,
        seed=seed,
        pass_k=config.eval_pass_k,
        avg_n=config.eval_avg_n,
    )

    progress_reporter = create_rich_progress(disable=machine.rank != 0)

    log_tokenizer(log, tokenizer)
    log_model(log, model)

    evaluator = build_evaluator(
        units=units,
        loaders=loaders,
        generator=generator,
        machine=machine,
        memory_tracker=memory_tracker,
        progress_repoter=progress_reporter,
        metric_writers=metric_writers,
        profiler=profiler,
        seed=seed,
    )

    assert evaluator is not None

    return evaluator
