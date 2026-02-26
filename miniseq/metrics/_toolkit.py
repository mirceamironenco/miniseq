from __future__ import annotations

from copy import deepcopy
from typing import Any, MutableMapping

import torch
import torch.distributed as dist

from miniseq.metrics._base import Metric
from miniseq.metrics._synclib import metrics_traversal_order, sync_states


def _get_world_size(process_group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1

    return dist.get_world_size(group=process_group)


def _get_rank(process_group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0

    return dist.get_rank(group=process_group)


def clone_metric(metric: Metric[Any]) -> Metric[Any]:
    return deepcopy(metric)


def _move_state_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_state_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move_state_to_device(v, device) for k, v in value.items()}

    return value


def _convert_to_pseudo_metric(metric_state_dict: dict[str, Any], device: torch.device) -> Any:
    device_state = {
        key: _move_state_to_device(value, device)
        for key, value in metric_state_dict.items()
    }

    return type("PseudoMetric", (), device_state)


def get_synced_metric_collection(
    metric_collection: MutableMapping[str, Metric[Any]],
    process_group: dist.ProcessGroup | None = None,
) -> MutableMapping[str, Metric[Any]]:
    world_size = _get_world_size(process_group)

    if world_size == 1:
        return metric_collection

    for metric in metric_collection.values():
        metric._prepare_for_merge_state()

    metric_state_data = {
        metric_name: metric.state_dict()
        for metric_name, metric in metric_collection.items()
    }
    traversal = metrics_traversal_order(metric_state_data)
    world_metric_data = sync_states(
        metric_state_data, traversal, process_group=process_group
    )

    local_rank = _get_rank(process_group)
    synced_metric_dict: dict[str, Metric[Any]] = {}

    for metric_name, metric in metric_collection.items():
        base_metric = clone_metric(metric).to(metric.device)
        other_rank_metrics = [
            _convert_to_pseudo_metric(world_metric_data[rank][metric_name], metric.device)
            for rank in range(world_size)
            if rank != local_rank
        ]
        synced_metric_dict[metric_name] = base_metric.merge_state(other_rank_metrics)

    return synced_metric_dict


def sync_and_compute_collection(
    metrics: MutableMapping[str, Metric[Any]],
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    synced_metrics = get_synced_metric_collection(metrics, process_group)
    return {key: metric.compute() for key, metric in synced_metrics.items()}
