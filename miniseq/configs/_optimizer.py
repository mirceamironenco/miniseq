from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.optim.optimizer import Optimizer
from typing_extensions import override

from miniseq import cli
from miniseq._lazy import LazyModule
from miniseq.builder_config import BuilderConfig, config_as_dict


@cli.make_union_registry("optimizer")
@dataclass(kw_only=True, frozen=True)
class OptimizerConfig(BuilderConfig[Optimizer]):
    """Base optimizer config. Should not be instantiated."""

    lr: float = 1e-5
    weight_decay: float = 0.01

    @override
    def build(
        self, *, params: Iterable[torch.Tensor], **kwd_overrides: Any
    ) -> Optimizer:
        """Instantiate and return optimizer using config attributes and given params."""

        fields = config_as_dict(self, **kwd_overrides)

        # Make the lr a tensor by default, so that one can torch compile .step()
        if "lr" in fields:
            device = next(iter(params)).device

            # TODO: Should be documented we are wrapping in float32.
            fields["lr"] = torch.tensor(
                fields["lr"], device=device, dtype=torch.float32
            )

        return self._target(params=params, **fields)


@cli.union_struct_choice(registry="optimizer", command="adam")
@dataclass(kw_only=True, frozen=True)
class AdamConfig(OptimizerConfig):
    _target: type[Optimizer] = field(
        default_factory=lambda: torch.optim.Adam, init=False, repr=False
    )
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    fused: bool | None = None


@cli.union_struct_choice(registry="optimizer", command="adamw")
@dataclass(kw_only=True, frozen=True)
class AdamWConfig(OptimizerConfig):
    _target: type[Optimizer] = field(
        default_factory=lambda: torch.optim.AdamW, init=False, repr=False
    )
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    fused: bool | None = True


@cli.union_struct_choice(registry="optimizer", command="sgd")
@dataclass(kw_only=True, frozen=True)
class SGDConfig(OptimizerConfig):
    _target: type[Optimizer] = field(
        init=False, repr=False, default_factory=lambda: torch.optim.SGD
    )
    momentum: float = 0
    dampening: float = 0
    nesterov = False
    fused: bool | None = None


# TOOD: 8-bit Adam?

torchao = LazyModule("torchao", globals(), "torchao")


@cli.union_struct_choice(registry="optimizer", command="adam_fp8")
@dataclass(kw_only=True, frozen=True)
class AdamWFP8Config(OptimizerConfig):
    _target: type[Optimizer] = field(
        default_factory=lambda: torchao.optim.AdamWFp8,  # type: ignore
        init=False,
        repr=False,
    )
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    block_size: int = 256
    bf16_stochastic_round: bool = False
