from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import torch
import torch.nn as nn
import tyro
from typing_extensions import override

from miniseq import cli
from miniseq.builder_config import BuilderConfig
from miniseq.generation import (
    Generator,
    VLLMEngineConfig,
    VLLMGenerator,
    VLLMSamplingConfig,
)
from miniseq.machine import Machine
from miniseq.models import ModelConfig


@cli.make_union_registry(name="generator")
@dataclass(kw_only=True, frozen=False)
class GeneratorConfig(BuilderConfig[Generator]):
    extra_stop_tokens: Annotated[list[int] | None, tyro.conf.Suppress] = None


@cli.union_struct_choice(registry="generator")
@tyro.conf.configure(tyro.conf.subcommand(name="vllm"), tyro.conf.OmitArgPrefixes)
@dataclass(kw_only=True, frozen=False)
class VLLMConfig(GeneratorConfig):
    _target: type[Generator] = field(
        init=False, repr=False, default_factory=lambda: VLLMGenerator
    )

    engine: VLLMEngineConfig = field(default_factory=lambda: VLLMEngineConfig())

    sample: VLLMSamplingConfig = field(default_factory=lambda: VLLMSamplingConfig())

    verbose: bool = True

    # Fixed, not available for CLI.
    eval_sample: Annotated[VLLMSamplingConfig | None, tyro.conf.Suppress] = None

    @override
    def build(
        self,
        *,
        machine: Machine,
        model_config: ModelConfig,
        model: nn.Module,
        dtype: torch.dtype,
        seed: int,
        cache_dir: Path,
        distributed_executor_backend: str | None = None,
        stop_token_ids: list[int] | None = None,
        **kwd_overrides: Any,
    ) -> Generator:
        assert not kwd_overrides

        self.engine.set_model_len(min(32768, model_config.max_seq_len))

        final_stop_tokens = []

        if stop_token_ids is not None:
            final_stop_tokens.extend(stop_token_ids)

        if self.extra_stop_tokens is not None:
            final_stop_tokens.extend(self.extra_stop_tokens)

        self.sample.stop_token_ids = list(set(final_stop_tokens))

        return VLLMGenerator(
            machine=machine,
            model_config=model_config,
            online_model=model,
            dtype=dtype,
            seed=seed,
            cache_dir=cache_dir,
            config=self.engine,
            sampling_config=self.sample,
            val_sampling_config=self.eval_sample,
            distributed_executor_backend=distributed_executor_backend,
            verbose=self.verbose,
        )
