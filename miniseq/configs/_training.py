from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from typing_extensions import override

from miniseq.builder_config import BuilderConfig, config_as_dict
from miniseq.machine import Machine
from miniseq.models import model_is_registered
from miniseq.training import TorchProfiler, WandbWriter


@tyro.conf.configure(tyro.conf.subcommand(name="on"))
@dataclass(kw_only=True, frozen=True)
class TorchProfilerConfig(BuilderConfig[TorchProfiler]):
    skip_n_steps: int = 4

    wait_n_steps: int = 1

    num_warmup_steps: int = 1

    num_active_steps: int = 4

    repeat: int = 1

    @override
    def build(
        self, *, cache_dir: Path, machine: Machine, **kwd_overrides: Any
    ) -> TorchProfiler:
        fields = config_as_dict(self, **kwd_overrides)

        tb_path = cache_dir.joinpath("tb")

        return TorchProfiler(log_dir=tb_path, machine=machine, **fields)


@tyro.conf.configure(tyro.conf.subcommand(name="on"))
@dataclass(kw_only=True, frozen=True)
class WandbConfig(BuilderConfig[WandbWriter]):
    project: str = "miniseq"

    run_name: str | None = None

    run_id: str | None = None

    group: str | None = None

    job_type: str | None = None

    @override
    def build(
        self,
        *,
        cache_dir: Path,
        config_to_save: dict[str, Any] | None = None,
        **kwd_overrides: Any,
    ) -> WandbWriter:
        fields = config_as_dict(self, **kwd_overrides)

        return WandbWriter(log_dir=cache_dir, config_to_save=config_to_save, **fields)


@dataclass(kw_only=True, frozen=True)
class CompileConfig:
    model: bool = False
    """Whether to compile the model."""

    loss: bool = False
    """Compile the loss function."""

    optimizer_step: bool = False
    """Compile optim+scheduler step."""

    fullgraph: bool = False
    """Compile using fullgraph=True."""

    dynamic: bool = False
    """Compile using dynamic=True."""


@dataclass(kw_only=True, frozen=True)
class TrainConfig:
    max_steps: int | None = 10000
    """Max number of steps to train."""

    max_epochs: int | None = 5
    """Max number of epochs to train."""

    micro_batch_size: int = 4
    """Per-GPU fwd+bwd pass batch size."""

    device_batch_size: int = 8
    """Per-GPU optimizer step batch size."""

    rollout_batch_size: int | None = None

    no_sync: bool = False

    anomaly: bool = False
    """Turns on anomaly detection."""

    max_grad_norm: float | None = None

    ac: bool = False
    """Enable activation checkpointing."""

    ac_freq: int = 1
    """Apply activ. ckpt. every 'ac_freq'-th layer."""

    checkpoint_every: int = 100

    checkpoint_last_n: int | None = 3
    """Number of ckpts. kept while training."""

    publish_metrics_every: int = 3

    validate_every: int = 25

    validate_at_start: bool = False

    save_model_only: bool = True

    resume_checkpoint: bool = False
    """Resume from local ckpt."""

    resume_model_only: bool = False

    @property
    def grad_accum_steps(self) -> int:
        return self.device_batch_size // self.micro_batch_size

    def __post_init__(self) -> None:
        if self.max_steps is None and self.max_epochs is None:
            raise ValueError("Either max_steps or max_epochs must be specified.")

        if self.resume_checkpoint and self.resume_model_only:
            raise ValueError(
                "Both resume_checkpoitn and resume_model_only are set to True."
            )

        if not self.device_batch_size % self.micro_batch_size == 0:
            raise ValueError("device_batch_size must be divisible by micro_batch_size")

        if self.rollout_batch_size is not None:
            if not self.device_batch_size % self.rollout_batch_size == 0:
                raise ValueError(
                    "device_batch_size must be divisible by rollout_batch_size"
                )


@dataclass(kw_only=True, frozen=True)
class PretrainedModelConfig:
    name: str = "llama-3.2-1b"

    dtype: torch.dtype = torch.bfloat16
    """Data type of the model."""

    flex_attention: bool = False
    """Force flex attention."""

    flash_attention2: bool = False

    reduce_flex: bool = False
    """Reduce flex triton kernel stages."""

    use_cce: bool = True
    """Whether to use cut cross entropy."""

    load_on_cpu: bool = True
    """Build the model on cpu on rank 0 if sharding (FSDP2)."""

    finetune_repo_id: str | None = None

    def __post_init__(self) -> None:
        if not model_is_registered(self.name):
            raise ValueError(f"Model with name {self.name} is not registered.")

        if self.flex_attention and self.flash_attention2:
            raise ValueError("Cannot have both flex and flash attention2 enabled.")
