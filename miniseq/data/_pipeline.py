from __future__ import annotations

import functools
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    overload,
)

import torch
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from torchdata.nodes import (
    BaseNode,
    Batcher,
    IterableWrapper,
    Loader,
    Mapper,
    ParallelMapper,
    PinMemory,
    Prefetcher,
    SamplerWrapper,
)

from miniseq.data._utils import DistributedEvalSampler, MapDataset, SizedLoader
from miniseq.logging import get_console
from miniseq.utils import SupportsPinMemory

T = TypeVar("T", contravariant=True)
X = TypeVar("X", covariant=True)
listT = TypeVar("listT", bound=list)
PinMemoryT = TypeVar("PinMemoryT", bound=SupportsPinMemory, contravariant=True)


@dataclass
class _PipelineStep:
    operation: str
    params: dict[str, Any] | None

    def __repr__(self) -> str:
        step = f"{self.operation}"

        step += "("
        if self.params is not None:
            for index, (key, value) in enumerate(self.params.items()):
                if index > 0:
                    step += ", "
                step += f"{key}={str(value)}"
        step += ")"
        return step


def _get_callable_name(fn: Callable) -> str:
    if isinstance(fn, functools.partial):
        return _get_callable_name(fn.func)
    elif hasattr(fn, "__qualname__"):
        fn_name = fn.__qualname__
    elif hasattr(fn, "__name__"):
        fn_name = fn.__name__
    else:
        fn_name = f"{fn.__class__.__name__}"

    return fn_name


class PipelineBuilder(Generic[T]):
    _node: BaseNode[T]
    _pipeline_steps: list[_PipelineStep]
    _iterator_len: int | None

    def __init__(
        self,
        node: BaseNode[T],
        *,
        pipeline_steps: list[_PipelineStep] | None = None,
        iterator_len: int | None = None,
    ) -> None:
        self._node = node
        self._pipeline_steps = pipeline_steps or []
        self._iterator_len = iterator_len

    @classmethod
    def from_source_node(cls, source_node: BaseNode[X]) -> PipelineBuilder[X]:
        return PipelineBuilder(source_node)

    @classmethod
    def from_iterable(cls, iterable: Iterable[X]) -> PipelineBuilder[X]:
        source_node = IterableWrapper(iterable)

        iterator_len: int | None = None

        if isinstance(iterable, Sized):
            iterator_len = len(iterable)

        pipeline_step = _PipelineStep(operation="from_map_dataset", params=None)

        return PipelineBuilder(
            source_node, pipeline_steps=[pipeline_step], iterator_len=iterator_len
        )

    @overload
    @classmethod
    def from_map_dataset(
        cls,
        dataset: MapDataset[X],
        *,
        shuffle: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        allow_uneven: bool = False,
    ) -> PipelineBuilder[X]: ...

    @overload
    @classmethod
    def from_map_dataset(
        cls,
        dataset: MapDataset[X],
        *,
        sampler: Sampler[int],
    ) -> PipelineBuilder[X]: ...

    @classmethod
    def from_map_dataset(
        cls,
        dataset: MapDataset[X],
        *,
        shuffle: bool = False,
        sampler: Sampler[int] | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        allow_uneven: bool = False,
    ) -> PipelineBuilder[X]:
        if sampler is None:
            if world_size > 1:
                if allow_uneven:
                    sampler_cls = DistributedEvalSampler[int]
                else:
                    sampler_cls = DistributedSampler[int]

                sampler = sampler_cls(
                    dataset,  # type: ignore
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    seed=seed,
                    drop_last=False,
                )
            elif shuffle:
                generator = torch.Generator()

                generator.manual_seed(seed)

                sampler = RandomSampler(dataset, replacement=False, generator=generator)
            else:
                sampler = SequentialSampler(dataset)

        # torchdata MapStyleWrapper's map_dataset argument is typed as Mapping which is
        # not a protocol, so we reimplement it to avoid subclassing Mapping.
        sampler_node = SamplerWrapper[int](sampler=sampler)

        mapper_node = Mapper(source=sampler_node, map_fn=dataset.__getitem__)

        pipeline_step = _PipelineStep(
            operation="from_map_dataset",
            params={"sampler": str(sampler.__class__.__name__)},
        )

        iter_length = len(sampler) if isinstance(sampler, Sized) else len(dataset)

        return PipelineBuilder(
            node=mapper_node, pipeline_steps=[pipeline_step], iterator_len=iter_length
        )

    def as_node(self) -> BaseNode[T]:
        return self._node

    def to_node(self, node: BaseNode[X]) -> PipelineBuilder[X]:
        pipeline_steps = self._pipeline_steps[:]

        pipeline_steps.append(_PipelineStep(operation=str(node), params=None))

        return PipelineBuilder(node, pipeline_steps=pipeline_steps)

    def as_loader(self, *, restart_on_stop: bool = True) -> Loader[T]:
        if self._iterator_len is None:
            return Loader(self._node, restart_on_stop_iteration=restart_on_stop)

        return SizedLoader(
            self._node,
            length=self._iterator_len,
            restart_on_stop_iteration=restart_on_stop,
        )

    def map(
        self,
        fn: Callable[[T], X],
        *,
        num_parallel: int = 1,
        in_order: bool = True,
        method: Literal["thread", "process"] = "thread",
        mp_context: Literal["spawn", "fork", "forkserver"] | None = None,
        prebatch: int | None = None,
    ) -> PipelineBuilder[X]:
        if not num_parallel >= 1:
            raise ValueError(f"num_parallel must be positive, got {num_parallel}.")

        # NB: Mapper is just ParallelMapper(..., num_workers=0)
        num_workers = 0 if num_parallel == 1 else num_parallel

        node = ParallelMapper(
            self._node,
            map_fn=fn,
            num_workers=num_workers,
            in_order=in_order,
            method=method,
            multiprocessing_context=mp_context,
            prebatch=prebatch,
        )

        params = {
            "fn": _get_callable_name(fn),
            "num_parallel": num_parallel,
            "in_order": in_order,
            "method": f"'{method}'",
        }

        op_name = "map" if num_workers == 0 else "parallel_map"

        pipeline_steps = self._pipeline_steps[:]

        pipeline_steps.append(_PipelineStep(operation=op_name, params=params))

        return PipelineBuilder(
            node=node,
            pipeline_steps=pipeline_steps,
            iterator_len=self._iterator_len,
        )

    def batch(
        self, *, batch_size: int, drop_last: bool = True
    ) -> PipelineBuilder[list[T]]:
        node = Batcher(self._node, batch_size=batch_size, drop_last=drop_last)

        pipeline_steps = self._pipeline_steps[:]

        pipeline_steps.append(
            _PipelineStep(
                operation="batch",
                params={"batch_size": batch_size, "drop_last": drop_last},
            )
        )

        iter_len = self._iterator_len

        if iter_len is not None:
            if drop_last:
                iter_len //= batch_size
            else:
                iter_len = (iter_len + batch_size - 1) // batch_size

        return PipelineBuilder(
            node=node, pipeline_steps=pipeline_steps, iterator_len=iter_len
        )

    def collate(
        self: PipelineBuilder[listT],
        fn: Callable[[listT], X],
        *,
        num_parallel: int = 1,
        method: Literal["thread", "process"] = "thread",
    ) -> PipelineBuilder[X]:
        builder = self.map(fn, num_parallel=num_parallel, in_order=True, method=method)

        return builder

    def pin_memory(self: PipelineBuilder[PinMemoryT]) -> PipelineBuilder[PinMemoryT]:
        node = PinMemory(self._node)

        pipeline_steps = self._pipeline_steps[:]

        pipeline_steps.append(_PipelineStep(operation="pin_memory", params=None))

        return PipelineBuilder(
            node=node,
            pipeline_steps=pipeline_steps,
            iterator_len=self._iterator_len,
        )

    def prefetch(self, *, prefetch_factor: int) -> PipelineBuilder[T]:
        node = Prefetcher(self._node, prefetch_factor=prefetch_factor)

        pipeline_steps = self._pipeline_steps[:]

        pipeline_steps.append(
            _PipelineStep(
                operation="prefetch", params={"prefetch_factor": prefetch_factor}
            )
        )

        return PipelineBuilder(
            node=node,
            pipeline_steps=pipeline_steps,
            iterator_len=self._iterator_len,
        )

    @property
    def iterator_len(self) -> int:
        if self._iterator_len is None:
            return -1
        return self._iterator_len

    def pipeline_steps(self) -> list[_PipelineStep]:
        return self._pipeline_steps

    def print_pipeline(self, *, do_print: bool = True) -> None:
        # do_print - allows for rank-aware printing.

        if do_print:
            print_pipeline(self)


def print_pipeline(pipeline_builder: PipelineBuilder) -> None:
    CONSOLE = get_console()
    CONSOLE.rule("Data pipeline")
    for index, step in enumerate(pipeline_builder._pipeline_steps):
        CONSOLE.print(f"{index}: {str(step)}")
    CONSOLE.rule("")
