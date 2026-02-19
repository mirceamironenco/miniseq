from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch.distributed as dist

from miniseq.metrics._base import TState


def metrics_traversal_order(
    state_dict: dict[str, dict[str, TState]],
) -> list[tuple[str, str]]:
    dict_items: list[tuple[str, str]] = []

    for outer_key in sorted(state_dict.keys()):
        inner_dict = state_dict[outer_key]
        for inner_key in sorted(inner_dict.keys()):
            dict_items.append((outer_key, inner_key))

    return dict_items


def _get_empty_metric_state_collection(
    traversal: list[tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    metric_state_collection: dict[str, dict[str, Any]] = {}

    for metric_name, state_name in traversal:
        metric_state_collection.setdefault(metric_name, {})
        metric_state_collection[metric_name][state_name] = {}

    return metric_state_collection


def sync_states(
    states: dict[str, dict[str, Any]],
    traversal: list[tuple[str, str]],
    process_group: dist.ProcessGroup | None = None,
) -> list[dict[str, dict[str, Any]]]:
    if not dist.is_available() or not dist.is_initialized():
        return [deepcopy(states)]

    world_size = dist.get_world_size(group=process_group)
    gathered_states = [
        _get_empty_metric_state_collection(traversal) for _ in range(world_size)
    ]

    for metric_name, state_name in traversal:
        my_state_data = states[metric_name][state_name]
        gathered_obj_data: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_obj_data, my_state_data, group=process_group)

        for rank, obj in enumerate(gathered_obj_data):
            gathered_states[rank][metric_name][state_name] = obj

    return gathered_states
