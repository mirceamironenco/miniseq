import math
from dataclasses import dataclass, field
from typing import Any

from torch.optim.lr_scheduler import LinearLR, LRScheduler
from torch.optim.optimizer import Optimizer
from typing_extensions import override

from miniseq import cli
from miniseq.builder_config import BuilderConfig, config_as_dict
from miniseq.nn import CosineAnnealingWRLR, LinearWarmupCosineDecayLR, NoopLR


@cli.make_union_registry("scheduler")
@dataclass(frozen=True)
class LRSchedulerConfig(BuilderConfig[LRScheduler]):
    _target: type[LRScheduler]
    last_epoch: int = field(init=False, default_factory=lambda: -1)

    @override
    def build(
        self,
        *,
        optimizer: Optimizer,
        max_num_steps: int | None = None,
        **kwd_overrides: Any,
    ) -> LRScheduler:
        """Instantiate and return lr scheduler using config attributes and optimizer."""

        fields = config_as_dict(self, **kwd_overrides)

        return self._target(optimizer=optimizer, **fields)


@cli.union_struct_choice(registry="scheduler", command="constant")
@dataclass(frozen=True)
class NoopSchedulerConfig(LRSchedulerConfig):
    _target: type[LRScheduler] = field(
        init=False, repr=False, default_factory=lambda: NoopLR
    )


@cli.union_struct_choice(registry="scheduler", command="cosine_wr")
@dataclass(frozen=True)
class CosineAnnealingConfig(LRSchedulerConfig):
    _target: type[LRScheduler] = field(
        init=False, repr=False, default_factory=lambda: CosineAnnealingWRLR
    )
    cycle_len: int | None = None
    warmup_steps: int = 0
    lr_mul: float = 1.0
    start_lr: float = 0.0
    final_lr: float = 0.0

    @override
    def build(
        self,
        *,
        optimizer: Optimizer,
        max_num_steps: int | None = None,
        **kwd_overrides: Any,
    ) -> LRScheduler:
        fields = config_as_dict(self, **kwd_overrides)

        if self.cycle_len is None:
            if max_num_steps is None:
                raise ValueError(
                    "Either cycle_len or max_num_steps must be set to instantiate CosineAnnealingScheduler"
                )

            fields["cycle_len"] = max_num_steps - self.warmup_steps

        return self._target(optimizer=optimizer, **fields)


@cli.union_struct_choice(registry="scheduler", command="linear_decay")
@dataclass(frozen=True)
class LinearDecayConfig(LRSchedulerConfig):
    _target: type[LRScheduler] = field(
        init=False, repr=False, default_factory=lambda: LinearLR
    )
    start_factor: float = 1.0 / 3
    end_factor: float = 1.0
    total_iters: int | None = None

    @override
    def build(
        self,
        *,
        optimizer: Optimizer,
        max_num_steps: int | None = None,
        **kwd_overrides: Any,
    ) -> LRScheduler:
        fields = config_as_dict(self, **kwd_overrides)

        if self.total_iters is None:
            if max_num_steps is None:
                raise ValueError(
                    "Either total_iters or max_num_steps must be set to instantiate LinearLR."
                )

            fields["total_iters"] = max_num_steps

        return self._target(optimizer=optimizer, **fields)


@cli.union_struct_choice(registry="scheduler", command="cosine_decay")
@dataclass(frozen=True)
class LinearWarmupCosineConfig(LRSchedulerConfig):
    _target: type[LRScheduler] = field(
        init=False, repr=False, default_factory=lambda: LinearWarmupCosineDecayLR
    )

    warmup_steps: int = -1
    """Takes precedence over warmup_ratio."""

    warmup_ratio: float = 0.01

    def build(
        self,
        *,
        optimizer: Optimizer,
        max_num_steps: int | None = None,
        **kwd_overrides: Any,
    ) -> LRScheduler:
        assert not kwd_overrides

        if max_num_steps is None:
            raise ValueError(
                "LinearWarmupCosineDecay scheduler requires specifying number of training steps."
            )

        warmup_steps = self.warmup_steps

        if warmup_steps == -1:
            warmup_steps = math.ceil(max_num_steps * self.warmup_ratio)

        return LinearWarmupCosineDecayLR(
            optimizer,
            num_train_steps=max_num_steps,
            num_warmup_steps=self.warmup_steps,
        )
