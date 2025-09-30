# This file contains code adapted from the facebookresearch/fairseq2 project.
# The original copyright notice and license are provided below.
# ----
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/fairseq2.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import Backend, ProcessGroup, ReduceOp
from typing_extensions import override

from miniseq.logging import get_logger
from miniseq.utils import (
    count_trainable_params,
    default_device,
    get_local_rank,
    get_local_world_size,
)

_log = get_logger()

RedOpType: TypeAlias = ReduceOp.RedOpType


def get_num_cpus(num_procs: int) -> int:
    num_cpus = os.cpu_count()

    try:
        affinity_mask = os.sched_getaffinity(0)  # type: ignore
    except AttributeError:
        # sched_getaffinity is not available on macOS.
        affinity_mask = None

    if num_cpus is None or affinity_mask is None:
        if num_cpus is not None:
            num_cpus = num_cpus // num_procs
        else:
            num_cpus = 1

        _log.warning(
            f"The number of CPUs cannot be precisely determined. Returning {num_cpus}."
        )

        return num_cpus

    # We should not exceed the number of cores available in the affinity mask.
    return min(max(num_cpus // num_procs, 1), len(affinity_mask))


class Machine(ABC):
    @abstractmethod
    def barrier(self) -> None: ...

    @abstractmethod
    def process_group(self) -> ProcessGroup: ...

    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor, op: RedOpType) -> None: ...

    @abstractmethod
    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor
    ) -> None: ...

    @abstractmethod
    def gather_object(self, object, object_gather_list, destination) -> None: ...

    @abstractmethod
    def all_gather_object(self, output_object_list: list[Any], object: Any) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def distributed(self) -> bool: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @property
    @abstractmethod
    def rank(self) -> int: ...

    @property
    @abstractmethod
    def dp_replicate_rank(self) -> int: ...

    @property
    @abstractmethod
    def dp_shard_rank(self) -> int: ...


class LocalMachine(Machine):
    _device: torch.device
    _rank: int
    _local_rank: int
    _world_size: int
    _dp_replicate_rank: int
    _dp_shard_rank: int

    def __init__(
        self, device: torch.device, rank: int = 0, world_size: int = 1
    ) -> None:
        self._device = device
        self._rank = rank
        self._local_rank = rank
        self._world_size = world_size
        self._dp_replicate_rank = rank
        self._dp_shard_rank = rank

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @override
    @override
    def barrier(self) -> None:
        pass

    @property
    @override
    def distributed(self) -> bool:
        return False

    @property
    @override
    def size(self) -> int:
        return self._world_size

    @property
    @override
    def rank(self) -> int:
        return self._rank

    @override
    def process_group(self) -> ProcessGroup:
        raise RuntimeError("ProcessGroup is not availalbe on LocalMachine.")

    @override
    def all_reduce(self, tensor: torch.Tensor, op: RedOpType) -> None:
        pass

    @override
    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor
    ) -> None:
        pass

    @override
    def gather_object(self, object, object_gather_list, destination) -> None:
        pass

    @override
    def all_gather_object(self, output_object_list: list[Any], object: Any) -> None:
        pass

    @override
    def close(self) -> None:
        pass

    @property
    @override
    def dp_replicate_rank(self) -> int:
        return self._dp_replicate_rank

    @property
    @override
    def dp_shard_rank(self) -> int:
        return self._dp_shard_rank


class DistributedMachine(Machine):
    _device: torch.device
    _rank: int
    _local_rank: int
    _world_size: int
    _pg: ProcessGroup
    _dp_replicate_rank: int
    _dp_shard_rank: int

    def __init__(
        self, device: torch.device, pg: ProcessGroup, *, replicate: bool, shard: bool
    ) -> None:
        if replicate and shard:
            raise ValueError("HSDP currently not supported.")

        if not (replicate or shard):
            raise ValueError("Distributed setting without data parallel not supported.")

        self._device = device
        self._rank = dist.get_rank(pg)
        self._world_size = dist.get_world_size(pg)
        self._local_rank = get_local_rank()
        self._pg = pg

        if replicate:
            self._dp_replicate_rank = self._local_rank
            self._dp_shard_rank = 0

        if shard:
            self._dp_replicate_rank = 0
            self._dp_shard_rank = self._local_rank

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @override
    def barrier(self) -> None:
        dist.barrier(group=self._pg, device_ids=[self._device.index])

    @property
    @override
    def distributed(self) -> bool:
        return self._world_size > 0

    @property
    @override
    def size(self) -> int:
        return self._world_size

    @property
    @override
    def rank(self) -> int:
        return self._rank

    @property
    @override
    def dp_replicate_rank(self) -> int:
        return self._dp_replicate_rank

    @property
    @override
    def dp_shard_rank(self) -> int:
        return self._dp_shard_rank

    @override
    def process_group(self) -> ProcessGroup:
        return self._pg

    @override
    def all_reduce(self, tensor: torch.Tensor, op: RedOpType) -> None:
        dist.all_reduce(tensor, op=op, group=self._pg)

    @override
    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor
    ) -> None:
        dist.all_gather_into_tensor(output_tensor, input_tensor, group=self._pg)

    @override
    def gather_object(self, object, object_gather_list, destination) -> None:
        dist.gather_object(object, object_gather_list, group=self._pg, dst=destination)

    @override
    def all_gather_object(self, output_object_list: list[Any], object: Any) -> None:
        torch.distributed.all_gather_object(output_object_list, object, group=self._pg)

    @override
    def close(self) -> None:
        dist.destroy_process_group(self._pg)

    @classmethod
    def init_default_process_group(
        cls,
        *,
        device: torch.device | None = None,
        dp_replicate: bool,
        dp_shard: bool,
        cpu_offloading: bool = False,
    ) -> DistributedMachine:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available.")

        num_procs = get_local_world_size()
        num_threads = None

        if num_procs > 1 and "OMP_NUM_THREADS" not in os.environ:
            num_threads = get_num_cpus(num_procs)

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        device = default_device()
        assert device.type in ("cpu", "cuda")

        backend = Backend.default_device_backend_map[device.type]

        if device.type == "cuda":
            # https://github.com/pytorch/pytorch/issues/46874.
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        if cpu_offloading:
            assert device.type == "cuda"

            # If using cpu_offloading with FSDP2, adding gloo backend is necessary.
            backend = f"{device.type}:{backend},cpu:gloo"

        dist.init_process_group(backend, timeout=timedelta(minutes=15))

        pg = dist.group.WORLD

        assert pg is not None

        return DistributedMachine(
            device=device, pg=pg, replicate=dp_replicate, shard=dp_shard
        )


def all_sum(machine: Machine, value: float | int) -> torch.Tensor:
    value_pt = torch.tensor(value, device=machine.device)

    machine.all_reduce(value_pt, ReduceOp.SUM)

    return value_pt


def setup_default_machine(
    *,
    device: torch.device | None = None,
    dp_replicate: bool = False,
    dp_shard: bool = False,
    cpu_offloading: bool = False,
) -> Machine:
    if dp_replicate and dp_shard:
        raise ValueError("HSDP not currently supported.")

    world_size = get_local_world_size()

    if world_size > 1:
        if not (dp_replicate or dp_shard):
            raise ValueError("world_size > 1 without data parallel not supported.")

        return DistributedMachine.init_default_process_group(
            device=device,
            cpu_offloading=cpu_offloading,
            dp_shard=dp_shard,
            dp_replicate=dp_replicate,
        )

    if device is None:
        device = default_device()

    return LocalMachine(device=device)


def all_ranks_same_trainable_params(model: nn.Module, machine: Machine) -> bool:
    """Checks that all ranks have the same number of trainable params."""

    if machine.size == 1:
        return True

    trainable_params = count_trainable_params(model)

    params_ = torch.zeros((machine.size,), device=machine.device, dtype=torch.int64)

    machine.all_gather(
        params_,
        torch.tensor(trainable_params, device=machine.device, dtype=torch.int64),
    )

    all_same = len(torch.unique(params_)) == 1

    return all_same
