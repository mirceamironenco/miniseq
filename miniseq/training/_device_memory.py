import logging
import typing
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeAlias, TypedDict

import torch
from typing_extensions import override


class DeviceMemoryTracker(ABC):
    @abstractmethod
    def get_stats(self, *, with_current: bool = False) -> dict[str, Any]: ...

    @abstractmethod
    def reset_stats(self) -> None: ...


class DummyMemoryTracker(DeviceMemoryTracker):
    @override
    def get_stats(self, *, with_current: bool = False) -> dict[str, Any]:
        return dict()

    @override
    def reset_stats(self) -> None:
        pass


class MemoryPoolMetrics(TypedDict):
    current: int
    peak: int
    allocated: int
    freed: int


MemoryPoolKey: TypeAlias = Literal["all", "small_pool", "large_pool"]
"""Memory pool types.

- ``all``: combined statistics across all memory pools.
- ``large_pool``: statistics for the large allocation pool
    (as of October 2019, for size >= 1MB allocations).
- ``small_pool``: statistics for the small allocation pool
    (as of October 2019, for size < 1MB allocations).
"""


class CudaMemoryStats(TypedDict):
    """Memory pools"""

    allocated: dict[MemoryPoolKey, MemoryPoolMetrics]
    """Number of allocation requests received by the memory allocator."""

    allocated_bytes: dict[MemoryPoolKey, MemoryPoolMetrics]
    """Amount of allocated memory."""

    segment: dict[MemoryPoolKey, MemoryPoolMetrics]
    """Number of reserved segments from ``cudaMalloc()``."""

    reserved_bytes: dict[MemoryPoolKey, MemoryPoolMetrics]
    """Amount of reserved memory."""

    active: dict[MemoryPoolKey, MemoryPoolMetrics]
    """Number of active memory blocks."""

    active_bytes: dict[MemoryPoolKey, MemoryPoolMetrics]
    """Amount of active memory."""

    inactive_split: dict[MemoryPoolKey, MemoryPoolMetrics]
    """Number of inactive, non-releasable memory blocks."""

    inactive_split_bytes: dict[MemoryPoolKey, MemoryPoolMetrics]
    """Amount of inactive, non-releasable memory."""

    """Event counters"""

    num_alloc_retries: int
    """Number of failed ``cudaMalloc`` calls that result in cache flush & retry."""

    num_ooms: int
    """Number of OOM errors thrown."""

    num_sync_all_streams: int
    """Number of ``synchronize_and_free_events`` calls."""

    num_device_alloc: int
    """Number of CUDA allocation calls. Includes both ``cuMemMap`` and ``cudaMalloc``."""

    num_device_free: int
    """Number of CUDA free calls. Includes both ``cuMemUnmap`` and ``cudaFree``."""

    """Caching allocator utilities"""

    max_split_size: int
    """Blocks above this size will not be split."""

    oversize_allocations: MemoryPoolMetrics
    """Number of over-size allocation allocation requests received."""

    oversize_segments: MemoryPoolMetrics
    """Number of over-size reserved segment from ``cudaMalloc()``."""

    requested_bytes: MemoryPoolMetrics
    """Memory requested by client code."""


class CudaMemoryTracker(DeviceMemoryTracker):
    _device: torch.device
    _total_memory: int

    def __init__(self, device: torch.device) -> None:
        self._device = device

        self._total_memory = torch.cuda.get_device_properties(device).total_memory

    @override
    def get_stats(self, *, with_current: bool = False) -> dict[str, Any]:
        stats = torch.cuda.memory_stats_as_nested_dict(device=self._device)
        stats = typing.cast(CudaMemoryStats, stats)

        peak_active_bytes = stats["active_bytes"]["all"]["peak"]
        peak_active_ratio = peak_active_bytes / self._total_memory

        peak_reserved = stats["reserved_bytes"]["all"]["peak"]
        peak_reserved_ratio = peak_reserved / self._total_memory

        memory_stats = {
            "memory/active_peak": peak_active_bytes / (1 << 30),
            "memory/active_pct": peak_active_ratio * 100,
            "memory/reserved_peak": peak_reserved / (1 << 30),
            "memory/reserved_pct": peak_reserved_ratio * 100,
            "memory/num_ooms": stats["num_ooms"],
            "memory/num_alloc_retries": stats["num_alloc_retries"],
        }

        if with_current:
            current_active_bytes = stats["active_bytes"]["all"]["current"]
            current_active_ratio = current_active_bytes / self._total_memory

            memory_stats["memory/current"] = current_active_bytes / (1 << 30)
            memory_stats["memory/current_pct"] = current_active_ratio * 100

        return memory_stats

    @override
    def reset_stats(self) -> None:
        # Reset the ``peak`` key in each memory pool stat dict.
        torch.cuda.reset_peak_memory_stats(device=self._device)


def create_memory_tracker(device: torch.device) -> DeviceMemoryTracker:
    if device.type == "cuda":
        return CudaMemoryTracker(device=device)

    return DummyMemoryTracker()


def log_memory(log: logging.Logger, tracker: DeviceMemoryTracker) -> None:
    if not isinstance(tracker, CudaMemoryTracker):
        log.info("Not on CUDA device, no memory stats to log.")

    memory_stats = tracker.get_stats(with_current=True)

    log.info(
        f"CUDA current memory usage:"
        f"{memory_stats['memory/current']:.2f}GiB"
        f"({memory_stats['memory/current_pct']:.2f}%)"
    )
