import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_current_lr(scheduler: LRScheduler) -> float:
    return scheduler.get_last_lr()[0]


class NoopLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, *, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return self.base_lrs


class LinearWarmupCosineDecayLR(LambdaLR):
    _num_train_steps: int
    _num_warmup_steps: int
    _num_decay_steps: int

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        num_train_steps: int,
        num_warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if not num_train_steps > num_warmup_steps:
            raise ValueError(
                f"train_steps ({num_train_steps}) must be larger than warmup_step ({num_warmup_steps})."
            )

        self._num_train_steps = num_train_steps
        self._num_warmup_steps = num_warmup_steps
        self._num_decay_steps = max(1, num_train_steps - num_warmup_steps)

        def lr_lambda(step: int) -> float:
            if step < num_warmup_steps:
                return step / max(num_warmup_steps, 1)

            progress = (step - num_warmup_steps) / self._num_decay_steps

            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

            return multiplier

        super().__init__(optimizer, lr_lambda, last_epoch)


class CosineAnnealingWRLR(LRScheduler):
    """Cosine annealing with warm restarts."""

    _cycle_len: int
    _num_warmup_steps: int
    _lr_mul: float
    _start_lrs: list[float]
    _final_lrs: list[float]

    def __init__(
        self,
        optimizer: Optimizer,
        cycle_len: int,
        num_warmup_steps: int,
        *,
        lr_mul: float = 1.0,
        start_lr: float = 0.0,
        final_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self._cycle_len = cycle_len
        self._num_warmup_steps = num_warmup_steps
        self._lr_mul = lr_mul

        num_groups = len(optimizer.param_groups)
        self._start_lrs = [start_lr] * num_groups
        self._final_lrs = [final_lr] * num_groups

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        base_lrs = self.base_lrs

        # Linear increase to base value until warmup is done.
        if self.last_epoch < self._num_warmup_steps:
            c = self.last_epoch / self._num_warmup_steps
            return [s + (b - s) * c for b, s in zip(base_lrs, self._start_lrs)]

        curr_step = self.last_epoch - self._num_warmup_steps

        # Which cycle are we in.
        cycle_nr = curr_step // self._cycle_len

        cycle_len = self._cycle_len

        cycle_pos = curr_step - (cycle_nr * cycle_len)

        lr_mul = self._lr_mul**cycle_nr

        c = math.cos(math.pi * cycle_pos / cycle_len)

        min_lrs, max_lrs = self._final_lrs, base_lrs

        return [self._cycle_lr(mn, mx, lr_mul, c) for mn, mx in zip(min_lrs, max_lrs)]

    def _cycle_lr(self, min_lr: float, max_lr: float, lr_mul: float, c: float) -> float:
        min_lr *= lr_mul
        max_lr *= lr_mul

        return min_lr + 0.5 * (max_lr - min_lr) * (1 + c)
