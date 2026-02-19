# This file contains code adapted from the facebookresearch/fairseq2 project.
# The original copyright notice and license are provided below.
# ----
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/fairseq2.

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypeVar

import torch

from miniseq.machine import Machine
from miniseq.metrics import Metric as _Metric
from miniseq.metrics import _toolkit as toolkit
import miniseq.metrics as metrics

MetricT = TypeVar("MetricT", bound=_Metric[Any])


class MetricBag:
    _device: torch.device
    _metrics: dict[str, metrics.Metric[Any]]

    def __init__(self, device: torch.device) -> None:
        super().__setattr__("_metrics", {})

        self._device = device

    def get(self, kls: type[MetricT], name: str, *args: Any, **kwargs: Any) -> MetricT:
        metric = self._metrics.get(name)

        if metric is not None:
            if not isinstance(metric, kls):
                raise TypeError(
                    f"The '{name}' metric must be of type `{kls}`, but is of type `{type(metric)}` instead."
                )

            return metric

        metric = kls(*args, **kwargs, device=self._device)

        self._metrics[name] = metric

        return metric

    @property
    def metrics(self) -> Mapping[str, metrics.Metric[Any]]:
        """The metrics contained in this bag."""
        return self._metrics

    def state_dict(self) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}

        for name, metric in self._metrics.items():
            state_dict[name] = metric

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._metrics.clear()

        for name, metric in state_dict.items():
            if not isinstance(metric, metrics.Metric):
                raise ValueError(
                    f"`state_dict['{name}']` must be of type `{metrics.Metric}`, but is of type `{type(metric)}` instead."
                )

            metric.to(self._device)

            self._metrics[name] = metric

    @property
    def device(self) -> torch.device:
        return self._device


def sync_and_compute_metrics(
    bag: MetricBag, machine: Machine
) -> dict[str, object] | None:
    """Sync the metrics across all processes and and compute their values."""
    if bag.device != machine.device:
        raise ValueError("bag device and machine device do not match.")

    logging.disable(logging.WARNING)

    if machine.size == 1:
        values = {name: m.compute() for name, m in bag.metrics.items()}
    else:
        metrics = dict(bag.metrics)

        values = toolkit.sync_and_compute_collection(
            metrics=metrics, process_group=machine.process_group()
        )

    logging.disable(logging.NOTSET)

    return values


def extend_batch_metrics(
    metric_values: dict[str, object], num_batches: int, elapsed_time: float
) -> None:
    def get_value(name: str) -> int | float | torch.Tensor | None:
        try:
            value = metric_values[name]
        except KeyError:
            return None

        if not isinstance(value, (int, float, torch.Tensor)):
            return None

        return value

    num_examples = get_value("perf/num_examples")

    if num_examples is not None:
        if num_batches > 0:
            metric_values["batch/batch_size"] = num_examples // num_batches
        else:
            metric_values["batch/batch_size"] = 0

    num_elements = get_value("perf/num_elements")

    if num_elements is not None:
        if num_batches > 0:
            metric_values["batch/elements_per_batch"] = num_elements // num_batches
        else:
            metric_values["batch/elements_per_batch"] = 0

        if elapsed_time > 0.0:
            # TODO: Once tensor parallel is impl., need to also divide by tp_world_size
            metric_values["perf/(TPS) elements_per_second"] = (
                num_elements / elapsed_time
            )
        else:
            metric_values["perf/(TPS) elements_per_second"] = 0.0

        if num_elements > 0:
            padding = get_value("batch/padding")

            if padding is not None:
                metric_values["batch/padding_ratio"] = padding / (
                    num_elements + padding
                )

    rollout_batch_size = get_value("batch/rollout_batch_size")

    if rollout_batch_size is not None:
        metric_values["batch/rollout_batch_size"] = rollout_batch_size // num_batches
