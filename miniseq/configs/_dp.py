from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import override

from miniseq import cli


@cli.make_union_registry("dp")
@dataclass(frozen=True)
class DPStrategyConfig:
    """Data-parallel config."""

    @property
    def replicate(self) -> bool:
        return False

    @property
    def shard(self) -> bool:
        return False

    @property
    def cpu_offloading(self) -> bool:
        return False


@cli.union_struct_choice(registry="dp", command="ddp")
@dataclass(frozen=True, kw_only=True)
class DDPConfig(DPStrategyConfig):
    ddp_broadcast_buffers: bool = False

    ddp_find_unused_parameters: bool = False

    ddp_static_graph: bool = False

    @property
    @override
    def replicate(self) -> bool:
        return True


@cli.union_struct_choice(registry="dp", command="fsdp2")
@dataclass(frozen=True, kw_only=True)
class FSDP2Config(DPStrategyConfig):
    fsdp2_reshard_fwd: bool = True
    """Reshard params after forward."""

    fsdp2_cpu_offload: bool = False

    fsdp2_reshard_outer_fwd: bool = True
    """Reshard after fwd for outer-most module."""

    fsdp2_fp32_reduce: bool = False

    @property
    @override
    def shard(self) -> bool:
        return True

    @property
    @override
    def cpu_offloading(self) -> bool:
        return self.fsdp2_cpu_offload
