from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TypeVar

import torch

from miniseq.metrics._base import Metric

TMean = TypeVar("TMean", bound="Mean")
TSum = TypeVar("TSum", bound="Sum")


def _mean_update(
    input: torch.Tensor, weight: float | int | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(weight, (float, int)):
        weighted_sum = float(weight) * torch.sum(input)
        weights = torch.tensor(float(weight) * input.numel(), device=input.device)
        return weighted_sum, weights

    if isinstance(weight, torch.Tensor) and input.size() == weight.size():
        return torch.sum(weight * input), torch.sum(weight)

    raise ValueError(
        "Weight must be a float/int or a tensor matching input size. "
        f"Got {type(weight)} with shape {getattr(weight, 'shape', None)}."
    )


def _sum_update(input: torch.Tensor, weight: float | int | torch.Tensor) -> torch.Tensor:
    if isinstance(weight, (float, int)):
        return torch.sum(input * float(weight))

    if isinstance(weight, torch.Tensor) and input.size() == weight.size():
        return torch.sum(input * weight)

    raise ValueError(
        "Weight must be a float/int or a tensor matching input size. "
        f"Got {type(weight)} with shape {getattr(weight, 'shape', None)}."
    )


class Mean(Metric[torch.Tensor]):
    def __init__(self: TMean, *, device: torch.device | None = None) -> None:
        super().__init__(device=device)
        self._add_state(
            "weighted_sum", torch.tensor(0.0, device=self.device, dtype=torch.float64)
        )
        self._add_state(
            "weights", torch.tensor(0.0, device=self.device, dtype=torch.float64)
        )

    @torch.inference_mode()
    def update(
        self: TMean,
        input: torch.Tensor | int | float,
        *,
        weight: float | int | torch.Tensor = 1.0,
    ) -> TMean:
        input = torch.as_tensor(input, device=self.device, dtype=torch.float64)
        if isinstance(weight, torch.Tensor):
            weight = weight.to(device=self.device, dtype=torch.float64)

        weighted_sum, weights = _mean_update(input, weight)

        self.weighted_sum += weighted_sum.to(self.device, dtype=torch.float64)
        self.weights += weights.to(self.device, dtype=torch.float64)

        return self

    @torch.inference_mode()
    def compute(self: TMean) -> torch.Tensor:
        if self.weights.item() == 0:
            logging.warning("No calls to update() have been made - returning 0.0")
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)

        return self.weighted_sum / self.weights

    @torch.inference_mode()
    def merge_state(self: TMean, metrics: Iterable[TMean]) -> TMean:
        for metric in metrics:
            self.weighted_sum += metric.weighted_sum.to(
                self.device, dtype=torch.float64
            )
            self.weights += metric.weights.to(self.device, dtype=torch.float64)

        return self


class Sum(Metric[torch.Tensor]):
    def __init__(self: TSum, *, device: torch.device | None = None) -> None:
        super().__init__(device=device)
        self._add_state(
            "weighted_sum", torch.tensor(0.0, device=self.device, dtype=torch.float64)
        )

    @torch.inference_mode()
    def update(
        self: TSum,
        input: torch.Tensor | int | float,
        *,
        weight: float | int | torch.Tensor = 1.0,
    ) -> TSum:
        input = torch.as_tensor(input, device=self.device, dtype=torch.float64)
        if isinstance(weight, torch.Tensor):
            weight = weight.to(device=self.device, dtype=torch.float64)

        self.weighted_sum += _sum_update(input, weight).to(
            self.device, dtype=torch.float64
        )

        return self

    @torch.inference_mode()
    def compute(self: TSum) -> torch.Tensor:
        return self.weighted_sum

    @torch.inference_mode()
    def merge_state(self: TSum, metrics: Iterable[TSum]) -> TSum:
        for metric in metrics:
            self.weighted_sum += metric.weighted_sum.to(self.device, dtype=torch.float64)

        return self
