import os
import warnings
from collections.abc import Iterable, Sized
from pathlib import Path

import torch
import torch.nn as nn

import miniseq.configs as cfg
from miniseq.data import PretrainedHFTokenizer
from miniseq.evaluator import EvalUnit
from miniseq.generation import Generator
from miniseq.machine import Machine
from miniseq.models import ModelConfig
from miniseq.recipes.algorithm import GeneratorEvalUnit
from miniseq.utils import is_torchrun_cluster


def calculate_total_steps(
    *,
    loader: Iterable,
    max_steps: int | None,
    max_epochs: int | None,
    grad_accum_steps: int,
) -> int:
    len_dataloader = len(loader) if isinstance(loader, Sized) else None

    # Either our dataloader is not Sized (e.g. on-the-fly filtering) or max_epochs = None
    if len_dataloader is None or max_epochs is None:
        if max_steps is None:
            raise ValueError(
                "Either specify max_steps or max_epochs and a Sized data loader."
            )

        return max_steps

    # We have a Sized dataloader and max_epochs is not None.

    updates_per_epoch = max(
        1, (len_dataloader + grad_accum_steps - 1) // grad_accum_steps
    )

    return max_epochs * updates_per_epoch


def create_generator(
    config: cfg.GeneratorConfig,
    /,
    mp_dtype: torch.dtype,
    machine: Machine,
    model_config: ModelConfig,
    model: nn.Module,
    cache_dir: Path,
    seed: int,
    stop_token_ids: list[int],
) -> Generator:
    stop_token_ids = list(set(stop_token_ids))

    if isinstance(config, cfg.generation.VLLMConfig):
        distributed_backend: str | None = None

        if is_torchrun_cluster():
            # https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/torchrun_example.py
            distributed_backend = "external_launcher"

            if config.engine.tensor_parallel_size is None:
                config.engine.tensor_parallel_size = machine.size

        generator = config.build(
            machine=machine,
            model_config=model_config,
            model=model,
            dtype=mp_dtype,
            seed=seed,
            cache_dir=cache_dir,
            distributed_executor_backend=distributed_backend,
            stop_token_ids=stop_token_ids,
        )

        return generator

    raise ValueError(
        f"Currently supported generators are: vllm, torch; got config: {config}"
    )


def create_prompt_evals(
    *,
    generator: Generator,
    data: cfg.data.RegisteredDatasetConfig,
    tokenizer: PretrainedHFTokenizer,
    machine: Machine,
    cache_dir: Path,
    seed: int,
    pass_k: int = 1,
    avg_n: int = 1,
) -> tuple[list[EvalUnit], list[Iterable]]:
    units: list[EvalUnit] = []
    loaders: list[Iterable] = []

    datasets = data.build(cache_dir=cache_dir)

    for index, dataset in enumerate(datasets):
        loader = dataset.create_loader(
            machine=machine,
            batch_size=data.batch_size,
            seed=seed,
            tokenizer=tokenizer,
            for_evaluation=True,
        )

        completion_scorer = dataset.create_scorer(tokenizer=tokenizer)

        eval_unit = GeneratorEvalUnit(
            generator=generator,
            completion_scorer=completion_scorer,
            name=data.datasets[index].upper(),
            pass_k=pass_k,
            avg_n=avg_n,
        )

        units.append(eval_unit)

        loaders.append(loader)

    return units, loaders


def setup_torch(
    expandable_segments: bool = False,
    capture_dynamic_shapes: bool = False,
    capture_scalar_outputs: bool = False,
    bf16_reduced: bool = True,
) -> None:
    torch.set_float32_matmul_precision("high")

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bf16_reduced

    if expandable_segments and torch.cuda.is_available():
        # This may not have an effect and it may require setting before importing torch.
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # The following is independent of import order.
        ca_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", None)

        if ca_config is None or "expandable_segments:False" not in ca_config:
            try:
                # Avoid memory fragmentation and peak reserved memory increasing over time.
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            except RuntimeError:
                warnings.warn(
                    "Setting expandable_segments:True for CUDA allocator failed."
                )

    if capture_dynamic_shapes:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

    if capture_scalar_outputs:
        torch._dynamo.config.capture_scalar_outputs = True
