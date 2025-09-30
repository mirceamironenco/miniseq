from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence, Sized
from contextlib import nullcontext
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn
from rich.progress import Progress
from torch.profiler import record_function

from miniseq.generation import Generator
from miniseq.logging import get_logger
from miniseq.machine import Machine
from miniseq.metric_bag import (
    MetricBag,
    extend_batch_metrics,
    sync_and_compute_metrics,
)
from miniseq.training import (
    DeviceMemoryTracker,
    LogMetricWriter,
    MetricWriter,
    NoopProfiler,
    Profiler,
    StopWatch,
    log_memory,
    manual_seed,
)
from miniseq.utils import SupportsDeviceTransfer

_log = get_logger()

BatchT_contra = TypeVar(
    "BatchT_contra", bound=SupportsDeviceTransfer, contravariant=True
)


class EvalUnit(ABC, Generic[BatchT_contra]):
    @abstractmethod
    def __call__(self, batch: BatchT_contra, *, metric_bag: MetricBag) -> None: ...

    @property
    @abstractmethod
    def model(self) -> nn.Module: ...

    @property
    def name(self) -> str | None:
        return None


def build_evaluator(
    *,
    units: list[EvalUnit],
    loaders: list[Iterable],
    generator: Generator | None,
    machine: Machine,
    memory_tracker: DeviceMemoryTracker,
    progress_repoter: Progress,
    metric_writers: list[MetricWriter] | None = None,
    profiler: Profiler | None = None,
    seed: int | None = None,
) -> Evaluator | None:
    if not len(units) == len(loaders):
        raise ValueError(
            "Got mismatched number of eval units and corresponding loaders: "
            f"{len(units)} != {len(loaders)}"
        )

    if units:
        evaluator = Evaluator(
            units=units,
            loaders=loaders,
            machine=machine,
            memory_tracker=memory_tracker,
            progress_repoter=progress_repoter,
            metric_writers=metric_writers,
            profiler=profiler,
            generator=generator,
            seed=seed,
        )
    else:
        evaluator = None

    return evaluator


BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


class Evaluator(Generic[BatchT]):
    _units: Sequence[EvalUnit[BatchT]]
    _loaders: Sequence[Iterable[BatchT]]
    _machine: Machine
    _memory_tracker: DeviceMemoryTracker
    _progress: Progress
    _data_watch: StopWatch
    _compute_watch: StopWatch
    _metric_writers: list[MetricWriter]
    _profiler: Profiler
    _generator: Generator | None
    _seed: int | None

    def __init__(
        self,
        *,
        units: Sequence[EvalUnit[BatchT]],
        loaders: Sequence[Iterable[BatchT]],
        machine: Machine,
        memory_tracker: DeviceMemoryTracker,
        progress_repoter: Progress,
        metric_writers: list[MetricWriter] | None = None,
        profiler: Profiler | None = None,
        generator: Generator | None = None,
        seed: int | None = None,
    ) -> None:
        if len(units) != len(loaders):
            raise ValueError("Number of eval units != eval loaders.")

        self._units = units

        self._loaders = loaders

        self._machine = machine

        self._memory_tracker = memory_tracker

        self._compute_watch = StopWatch(device=machine.device)

        self._data_watch = StopWatch()

        self._progress = progress_repoter

        if metric_writers is None:
            metric_writers = []

            if self._machine.rank == 0:
                metric_writers.append(LogMetricWriter(_log, max_per_row=-1))

        self._metric_writers = metric_writers

        self._profiler = profiler or NoopProfiler()

        self._generator = generator

        self._seed = seed

        self._machine.barrier()

    @torch.inference_mode()
    def run(
        self, *, train_step_nr: int | None = None, train_epoch_nr: int | None = None
    ) -> float | None:
        scores: list[float] = []

        # If progress object already has tasks, let external caller manage state.
        progress_context = (
            nullcontext() if not self._progress.finished else self._progress
        )

        seed_context = nullcontext()

        if self._seed is not None:
            seed_context = manual_seed(
                self._seed + self._machine.rank,
                torch.device("cpu"),
                self._machine.device,
            )

        with progress_context, self._profiler, seed_context:
            if self._generator is not None:
                self._generator.prepare_for_generation()

                self._generator.update_model()

                self._generator.validation_mode(True)

            for unit, data_loader in zip(self._units, self._loaders):
                if unit.name is not None:
                    _log.info(f"Evaluating {unit.name}")

                score = self._run_unit(
                    unit=unit,
                    data_loader=data_loader,
                    train_step_nr=train_step_nr,
                    train_epoch_nr=train_epoch_nr,
                )

                if score is not None:
                    scores.append(score)

        if self._machine.rank == 0:
            if scores:
                _log.info(f"Validation score: {sum(scores)}")

        _log.info("Evaluation finished.")

        if self._generator is not None:
            self._generator.after_generation()

            self._generator.validation_mode(False)

        log_memory(_log, self._memory_tracker)

        total_score: float | None = None

        if scores:
            total_score = sum(scores)

        return total_score

    def close(self) -> None:
        if self._machine.rank == 0:
            for writer in self._metric_writers:
                writer.close()

        self._machine.close()

    def _run_unit(
        self,
        *,
        unit: EvalUnit[BatchT],
        data_loader: Iterable[BatchT],
        train_step_nr: int | None = None,
        train_epoch_nr: int | None = None,
    ) -> float | None:
        unit.model.eval()

        self._memory_tracker.reset_stats()

        metric_bag = MetricBag(device=self._machine.device)

        eval_task = self._progress.add_task(
            description="eval" if unit.name is None else unit.name,
            total=len(data_loader) if isinstance(data_loader, Sized) else None,
        )

        eod = False

        batches_read = 0

        data_iter = iter(data_loader)

        while not eod:
            with self._data_watch:
                batch = next(data_iter, None)

                if batch is None:
                    eod = True
                    continue

            with self._compute_watch:
                self._progress.update(eval_task, advance=1)

                batch = batch.to(self._machine.device, non_blocking=True)

                batches_read += 1

                with record_function(f"eval_step_{batches_read}"):
                    unit(batch, metric_bag=metric_bag)

                self._profiler.step()

        metric_values = self._publish_metrics(
            unit=unit,
            metric_bag=metric_bag,
            num_batches=batches_read,
            train_step_nr=train_step_nr,
            train_epoch_nr=train_epoch_nr,
        )

        score: float | None = None

        if "metric_name" in metric_values:
            assert isinstance(metric_values["metric_name"], (int, float, torch.Tensor))

            score = float(metric_values["metric_name"])

        self._progress.remove_task(eval_task)

        self._compute_watch.reset()

        self._data_watch.reset()

        self._memory_tracker.reset_stats()

        return score

    def _publish_metrics(
        self,
        *,
        unit: EvalUnit[BatchT],
        metric_bag: MetricBag,
        num_batches: int,
        train_step_nr: int | None = None,
        train_epoch_nr: int | None = None,
    ) -> dict[str, Any]:
        values = sync_and_compute_metrics(metric_bag, self._machine)

        if self._machine.rank == 0:
            if values is None:
                raise RuntimeError("Synchronizing metric values failed.")

            values = {k: v for k, v in values.items() if not k.startswith("total_")}

            compute_time = self._compute_watch.get_elapsed_time()

            data_time = self._data_watch.get_elapsed_time()

            extend_batch_metrics(
                values,
                num_batches=num_batches,
                elapsed_time=compute_time + data_time,
            )

            if train_epoch_nr is not None:
                values["epoch_step"] = train_epoch_nr

            values["compute_time"] = compute_time

            values["data_time"] = data_time

            values.update(self._memory_tracker.get_stats())

            section = "eval"

            if unit.name is not None:
                section = f"{unit.name}"

            step_nr = train_step_nr

            values = {f"{section}/{key}": value for key, value in values.items()}

            for writer in self._metric_writers:
                writer.record_metrics(
                    run=section, values=values, step_nr=step_nr, flush=True
                )

        self._machine.barrier()

        if values is None:
            return {}

        return values
