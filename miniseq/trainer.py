# This file contains code adapted from the facebookresearch/fairseq2 project.
# The original copyright notice and license are provided below.
# ----
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/fairseq2.

from __future__ import annotations

import collections
import dataclasses
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import (
    Any,
    Deque,
    Generic,
    Iterator,
    Mapping,
    Protocol,
    runtime_checkable,
)

import torch
import torch.nn as nn
from rich.progress import Progress, TaskID
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.profiler import record_function
from torch.utils.data import DataLoader
from torcheval.metrics import Mean
from typing_extensions import TypeVar, override

import miniseq.training.data_parallel as data_parallel
from miniseq.builder_config import DataclassInstance
from miniseq.configs import TrainRecipeConfig, WandbConfig
from miniseq.data import EpochSampler, TrajectoryBatch
from miniseq.evaluator import Evaluator, EvalUnit, build_evaluator
from miniseq.generation import Generator
from miniseq.logging import create_rich_progress, get_logger
from miniseq.machine import Machine
from miniseq.metric_bag import (
    MetricBag,
    extend_batch_metrics,
    sync_and_compute_metrics,
)
from miniseq.models import ModelConfig
from miniseq.nn import get_current_lr
from miniseq.training import (
    CheckpointManager,
    DeviceMemoryTracker,
    LogMetricWriter,
    MetricWriter,
    NoopProfiler,
    Profiler,
    StopWatch,
    TensorBoardWriter,
    clip_gradient_norm,
    create_checkpoint_manager,
    create_memory_tracker,
    log_memory,
    manual_seed,
    normalize_gradients,
)
from miniseq.utils import SupportsDeviceTransfer, clear_unused_memory

_log: logging.Logger = get_logger()


def create_metric_writers(
    *,
    cache_dir: Path,
    machine: Machine,
    logger: logging.Logger,
    tensorboard: bool = False,
    wandb: WandbConfig | None = None,
    max_per_row: int = 1,
    config: DataclassInstance | None = None,
) -> list[MetricWriter]:
    metric_writers = []

    if machine.rank == 0:
        metric_writers.append(LogMetricWriter(logger, max_per_row))

        if tensorboard:
            tb_path = cache_dir.joinpath("tb")

            metric_writers.append(TensorBoardWriter(tb_path))

        if wandb is not None:
            config_to_save: dict[str, Any] | None = None

            if config is not None:
                config_to_save = dataclasses.asdict(config)

            wandb_writer = wandb.build(
                cache_dir=cache_dir, config_to_save=config_to_save
            )

            metric_writers.append(wandb_writer)

    machine.barrier()

    return metric_writers


BatchT_contra = TypeVar(
    "BatchT_contra", bound=SupportsDeviceTransfer, contravariant=True
)
BatchT = TypeVar("BatchT", bound=SupportsDeviceTransfer)


class TrainUnit(ABC, Generic[BatchT_contra]):
    @abstractmethod
    def __call__(
        self, batch: BatchT_contra, *, metric_bag: MetricBag
    ) -> tuple[torch.Tensor, int | None]: ...

    @property
    @abstractmethod
    def model(self) -> nn.Module: ...


DataBatchT = TypeVar("DataBatchT", bound=SupportsDeviceTransfer, default=BatchT)
DataBatchT_contra = TypeVar(
    "DataBatchT_contra", bound=SupportsDeviceTransfer, contravariant=True
)


@runtime_checkable
class RolloutUnit(Protocol[DataBatchT_contra]):
    def update_rollout_model(self) -> None: ...

    def run_generation(
        self, batch: DataBatchT_contra, *, metric_bag: MetricBag
    ) -> list[TrajectoryBatch]: ...


class Trainer(Generic[BatchT, DataBatchT]):
    _model_config: ModelConfig | None
    _model: nn.Module
    _train_unit: TrainUnit[BatchT]
    _train_loader: Iterable[DataBatchT]
    _machine: Machine
    _seed: int
    _optimizer: Optimizer
    _lr_scheduler: LRScheduler
    _checkpoint_manager: CheckpointManager
    _memory_tracker: DeviceMemoryTracker
    _progress: Progress
    _metric_writers: list[MetricWriter]
    _profiler: Profiler
    _requires_rollout: bool
    _publish_metrics_every_n_steps: int
    _checkpoint_every_n_steps: int
    _validate_every_n_steps: int
    _validator: Evaluator | None
    _total_steps: int | None
    _total_epochs: int | None
    _grad_accum_steps: int
    _grad_accum_no_sync: bool
    _anomaly_detection: bool
    _rollout_sync_steps: int
    _metric_bag: MetricBag
    _max_gradient_norm: float | None
    _validate_at_start: bool
    _save_model_only: bool
    _resume_from_checkpoint: bool
    _resume_model_only: bool
    _global_step: int
    _epoch_step: int
    _compute_watch: StopWatch
    _data_watch: StopWatch
    _rollout_watch: StopWatch
    _num_effective_batches: int
    _train_iterator: Iterator[DataBatchT]
    _train_task_id: TaskID
    _rollout_unit: RolloutUnit[DataBatchT] | None
    _batch_queue: Deque[Any]

    def __init__(
        self,
        *,
        model_config: ModelConfig | None,
        train_unit: TrainUnit[BatchT],
        train_loader: Iterable[DataBatchT],
        machine: Machine,
        seed: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        requires_rollout: bool,
        checkpoint_manager: CheckpointManager,
        memory_tracker: DeviceMemoryTracker,
        progress_repoter: Progress,
        metric_writers: list[MetricWriter] | None = None,
        profiler: Profiler | None = None,
        publish_metrics_every_n_steps: int = 5,
        validate_every_n_steps: int = 50,
        checkpoint_every_n_steps: int = 100,
        validator: Evaluator | None = None,
        total_steps: int,
        total_epochs: int | None = None,
        grad_accum_steps: int = 1,
        grad_accum_no_sync: bool = False,
        max_gradient_norm: float | None = None,
        anomaly: bool = False,
        compile_optimizer_step: bool = False,
        validate_at_start: bool = False,
        save_model_only: bool = True,
        resume_from_checkpoint: bool = False,
        resume_model_only: bool = False,
        rollout_sync_steps: int = 1,
    ) -> None:
        self._model_config = model_config

        self._model = train_unit.model

        self._train_unit = train_unit

        self._train_loader = train_loader

        self._machine = machine

        self._seed = seed

        self._optimizer = optimizer

        self._lr_scheduler = lr_scheduler

        self._checkpoint_manager = checkpoint_manager

        self._memory_tracker = memory_tracker

        self._progress = progress_repoter

        if metric_writers is None:
            metric_writers = []

            if self._machine.rank == 0:
                metric_writers.append(LogMetricWriter(_log))

        self._metric_writers = metric_writers

        self._profiler = profiler or NoopProfiler()

        self._requires_rollout = requires_rollout

        # Currently, if doing rollout train_unit is also a rollout_unit.
        if requires_rollout:
            if not isinstance(self._train_unit, RolloutUnit):
                raise ValueError(
                    "Train unit does not have a `run_generation` metohd. "
                    "For online training, train units must be able to generate rollouts. "
                    f"Got unit type {type(self._train_unit)}."
                )

            self._rollout_unit = self._train_unit
        else:
            self._rollout_unit = None

        self._publish_metrics_every_n_steps = publish_metrics_every_n_steps

        self._checkpoint_every_n_steps = checkpoint_every_n_steps

        self._validate_every_n_steps = validate_every_n_steps

        self._validator = validator

        self._total_steps = total_steps

        self._total_epochs = total_epochs

        if not grad_accum_steps > 0:
            raise ValueError(
                f"`grad_accum_steps` must be positive integer, got {grad_accum_steps}."
            )

        self._grad_accum_steps = grad_accum_steps

        self._grad_accum_no_sync = grad_accum_no_sync

        self._compute_watch = StopWatch(device=machine.device)

        self._data_watch = StopWatch()

        self._rollout_watch = StopWatch(device=machine.device)

        self._max_gradient_norm = max_gradient_norm

        self._validate_at_start = validate_at_start

        self._save_model_only = save_model_only

        self._resume_from_checkpoint = resume_from_checkpoint

        self._resume_model_only = resume_model_only

        self._metric_bag = MetricBag(device=machine.device)

        self._anomaly_detection = anomaly

        self._rollout_sync_steps = rollout_sync_steps

        if compile_optimizer_step:
            self._run_optimizer_step = torch.compile(self._run_optimizer_step)

        self._num_effective_batches = 0

        self._epoch_step = 0

        self._global_step = 0

        self._batch_queue = collections.deque()

        self._machine.barrier()

    @classmethod
    def from_configs(
        cls,
        *,
        recipe_config: TrainRecipeConfig,
        model_config: ModelConfig | None = None,
        train_unit: TrainUnit[BatchT],
        train_loader: Iterable[DataBatchT],
        valid_units: list[EvalUnit] | None = None,
        valid_loaders: list[Iterable] | None = None,
        generator: Generator | None,
        machine: Machine,
        seed: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        requires_rollout: bool,
        total_steps: int,
        log: logging.Logger | None = None,
        compile_optimizer_step: bool = False,
        rollout_sync_steps: int = 1,
    ) -> Trainer[BatchT, DataBatchT]:
        device_memory_tracker = create_memory_tracker(device=machine.device)

        metric_writers = create_metric_writers(
            cache_dir=recipe_config.cache_dir,
            machine=machine,
            logger=log or _log,
            tensorboard=recipe_config.tensorboard,
            wandb=recipe_config.wandb,
            config=recipe_config,
        )

        checkpoint_manager = create_checkpoint_manager(
            recipe_config.cache_dir, machine, recipe_config.train.checkpoint_last_n
        )

        profiler: Profiler | None = None

        if recipe_config.profiler is not None:
            profiler = recipe_config.profiler.build(
                cache_dir=recipe_config.cache_dir, machine=machine
            )

        progress_reporter = create_rich_progress(disable=machine.rank != 0)

        validator = build_evaluator(
            units=valid_units or [],
            loaders=valid_loaders or [],
            generator=generator,
            machine=machine,
            memory_tracker=device_memory_tracker,
            progress_repoter=progress_reporter,
            metric_writers=metric_writers,
            profiler=profiler,
            seed=seed,
        )

        trainer = Trainer(
            model_config=model_config,
            train_unit=train_unit,
            train_loader=train_loader,
            machine=machine,
            seed=seed,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_manager=checkpoint_manager,
            memory_tracker=device_memory_tracker,
            progress_repoter=progress_reporter,
            requires_rollout=requires_rollout,
            metric_writers=metric_writers,
            profiler=profiler,
            publish_metrics_every_n_steps=recipe_config.train.publish_metrics_every,
            validate_every_n_steps=recipe_config.train.validate_every,
            checkpoint_every_n_steps=recipe_config.train.checkpoint_every,
            validator=validator,
            total_steps=total_steps,
            total_epochs=recipe_config.train.max_epochs,
            grad_accum_steps=recipe_config.train.grad_accum_steps,
            grad_accum_no_sync=recipe_config.train.no_sync,
            max_gradient_norm=recipe_config.train.max_grad_norm,
            anomaly=recipe_config.train.anomaly,
            compile_optimizer_step=compile_optimizer_step,
            validate_at_start=recipe_config.train.validate_at_start,
            save_model_only=recipe_config.train.save_model_only,
            resume_from_checkpoint=recipe_config.train.resume_checkpoint,
            resume_model_only=recipe_config.train.resume_model_only,
            rollout_sync_steps=rollout_sync_steps,
        )

        return trainer

    def run(self) -> None:
        try:
            with manual_seed(
                self._seed + self._machine.rank,
                self._machine.device,
                torch.device("cpu"),
            ):
                if self._resume_from_checkpoint or self._resume_model_only:
                    self._resume(model_only=self._resume_model_only)

                self._do_run()

        except KeyboardInterrupt:
            _log.info(f"Training terminated at step {self._global_step}.")

        if self._machine.rank == 0:
            for writer in self._metric_writers:
                writer.close()

        self._machine.close()

    def _resume(self, *, model_only: bool = False) -> None:
        last_step_nr = self._checkpoint_manager.last_saved_step(
            ignore_model_only=not model_only
        )

        if model_only:
            _log.info(f"Restoring model only checkpoint from step: {last_step_nr}.")

            self._checkpoint_manager.load_model_state(last_step_nr, self._model)
        else:
            _log.info(f"Restoring checkpoint from step: {last_step_nr}.")

            trainer_state = TrainerState(self)

            self._checkpoint_manager.load_checkpoint(
                last_step_nr,
                trainer_state,
                self._model,
                self._optimizer,
                self._train_loader,  # type: ignore
            )

        _log.info(f"Done. Training resuming from step: {self._global_step}.")

        self._machine.barrier()

    def _do_run(self) -> None:
        self._model.train()

        task_description = "train"

        if self._total_epochs is not None:
            task_description += f" epoch {self._epoch_step} / {self._total_epochs}"

        self._train_task_id = self._progress.add_task(
            task_description,
            start=True,
            total=self._total_steps,
            completed=self._global_step,
        )

        self._reset_iterator()

        with self._progress, self._profiler:
            if self._validate_at_start:
                self._validate()

            while not self._should_stop_training():
                self._run_step()

                if self._global_step % self._publish_metrics_every_n_steps == 0:
                    self._publish_metrics()

                if self._should_validate():
                    self._validate()

                if self._global_step % self._checkpoint_every_n_steps == 0:
                    self._checkpoint()

            self._stop()

    def _should_stop_training(self) -> bool:
        if self._total_epochs is not None:
            # If total_epochs is specified it takes priority over total_steps.
            return self._epoch_step >= self._total_epochs

        if self._total_steps is not None:
            return self._global_step >= self._total_steps

        return False

    def _stop(self) -> None:
        if self._validator is not None:
            self._validate()

        self._checkpoint()

    def _should_validate(self) -> bool:
        if self._validator is None:
            return False

        if self._global_step % self._validate_every_n_steps == 0:
            return True

        return False

    def _validate(self) -> None:
        if self._validator is None:
            raise RuntimeError("Attempting to run validation, but no validator set.")

        _log.info(f"Running validation at step: {self._global_step}")

        self._model.eval()

        with data_parallel.summon_full_parameters(self._model):
            score = self._validator.run(
                train_step_nr=self._global_step, train_epoch_nr=self._epoch_step
            )

        if score is not None:
            _log.info(f"Validation done, score is {score}.")

        self._model.train()

        self._machine.barrier()

        self._progress.refresh()

        clear_unused_memory()

    def _publish_metrics(self) -> None:
        values = sync_and_compute_metrics(self._metric_bag, self._machine)

        if self._machine.rank == 0:
            if values is None:
                raise RuntimeError("Synchronizing metric values failed.")

            compute_time = self._compute_watch.get_elapsed_time()

            data_time = self._data_watch.get_elapsed_time()

            rollout_time = self._rollout_watch.get_elapsed_time()

            extend_batch_metrics(
                values,
                num_batches=self._num_effective_batches,
                elapsed_time=compute_time + data_time,
            )

            values["learning_rate"] = get_current_lr(self._lr_scheduler)

            values["epoch_step"] = self._epoch_step

            values["perf/compute_time"] = compute_time

            values["perf/data_time"] = data_time

            values["perf/rollout_time"] = rollout_time

            values.update(self._memory_tracker.get_stats())

            # Avoid wandb double flush
            # train step is still logged but not shown.
            should_flush = not self._should_validate()

            for writer in self._metric_writers:
                writer.record_metrics(
                    run="train",
                    values=values,
                    step_nr=self._global_step,
                    flush=should_flush,
                )

        self._machine.barrier()

        # Reset metrics that are averaged over publish_every_n_steps.
        for name, metric in self._metric_bag.metrics.items():
            if not name.startswith("total_"):
                metric.reset()

        self._num_effective_batches = 0

        self._compute_watch.reset()

        self._data_watch.reset()

        self._rollout_watch.reset()

        self._memory_tracker.reset_stats()

    def _next_batch(self) -> DataBatchT:
        try:
            batch = next(self._train_iterator)

        except StopIteration:
            # Advance epoch step on iterator finish.
            self._epoch_step += 1

            if self._total_epochs is not None:
                task_description = (
                    f"train epoch {self._epoch_step} / {self._total_epochs}"
                )
                self._progress.update(self._train_task_id, description=task_description)

            # Reset iterator
            self._reset_iterator()

            # Continue with next batch
            return self._next_batch()

        return batch

    def _reset_iterator(self) -> None:
        if isinstance(self._train_loader, DataLoader):
            if isinstance(self._train_loader.sampler, EpochSampler):
                self._train_loader.sampler.set_epoch(self._epoch_step)

        # If we're not using a torch DataLoader, assume underlying sampler handles
        # set_epoch equivalent during iterator creation, as done e.g. torchdata Loader.
        self._train_iterator = iter(self._train_loader)

    def _populate_batch_queue(self) -> None:
        if not self._requires_rollout:
            with (
                self._data_watch,
                record_function(f"data_load_step_{self._global_step}"),
            ):
                self._batch_queue.extend(
                    [self._next_batch() for _ in range(self._grad_accum_steps)]
                )
        else:
            assert self._rollout_unit is not None

            batches_needed = self._grad_accum_steps * self._rollout_sync_steps

            model_synced = False

            while len(self._batch_queue) < batches_needed:
                if not model_synced:
                    self._rollout_unit.update_rollout_model()
                    model_synced = True

                with (
                    self._data_watch,
                    record_function(f"data_load_step_{self._global_step}"),
                ):
                    batch = self._next_batch()

                # Explicitly stop and start the profiler to not trace e.g. vllm internals.
                self._profiler.stop()

                # Hide progress bar to not mess with e.g. vllm tqdm bar when generating.
                self._progress.stop()

                # rollout_watch tracks: wake_up + sync weights + rollout + sleep + score
                with self._rollout_watch:
                    trajectories = self._rollout_unit.run_generation(
                        batch, metric_bag=self._metric_bag
                    )

                    random.shuffle(trajectories)

                    self._batch_queue.extend(trajectories)

                    clear_unused_memory()

                self._profiler.start()

                self._progress.start()

                self._machine.barrier()

    def _maybe_no_sync(
        self, curr_batch: int, batches_to_sync: int
    ) -> AbstractContextManager[None]:
        if (
            curr_batch < batches_to_sync - 1
            and self._machine.distributed
            and self._grad_accum_no_sync
        ):
            return data_parallel.no_sync(self._model)

        return nullcontext()

    def _run_step(self) -> None:
        detect_anomaly = torch.autograd.set_detect_anomaly(
            self._anomaly_detection, check_nan=True
        )

        self._global_step += 1

        # Fetch batch data if needed.
        self._populate_batch_queue()

        # Run the actual step
        with detect_anomaly:
            with record_function(f"train_step_{self._global_step}"):
                # Run the actual step.
                self._do_run_step()

        self._profiler.step()

        if self._global_step == 1:
            torch.cuda.empty_cache()

        if self._global_step % 15 == 0:
            clear_unused_memory()

        self._progress.update(self._train_task_id, advance=1)

        if self._requires_rollout:
            # Extra memory tracking for online training.
            log_memory(_log, self._memory_tracker)

    def _do_run_step(self) -> None:
        if self._global_step == 1:
            _log.info(
                "Starting training, expect slower start if model compilation enabled."
            )

        grad_steps = self._grad_accum_steps

        assert len(self._batch_queue) >= grad_steps

        with self._compute_watch:
            num_targets = 0

            # 1. Run a complete step, possibly with gradient accumulation.
            for batch_index in range(grad_steps):
                with self._maybe_no_sync(batch_index, batches_to_sync=grad_steps):
                    batch = self._batch_queue.popleft()

                    batch = batch.to(device=self._machine.device, non_blocking=True)

                    batch_loss, num_batch_targets = self._train_unit(
                        batch, metric_bag=self._metric_bag
                    )

                    if num_batch_targets is not None:
                        num_targets += num_batch_targets
                    else:
                        batch_loss = batch_loss / grad_steps

                    batch_loss.backward()

            # 2. Scale gradients by (world_size / num_target_tokens).
            if num_targets > 0:
                normalize_gradients(
                    self._model,
                    num_targets=num_targets,
                    machine=self._machine,
                    foreach=True,
                )

            # 3. Clip gradients
            grad_norm: torch.Tensor | None = None

            if self._max_gradient_norm is not None:
                grad_norm = clip_gradient_norm(
                    self._model, max_norm=self._max_gradient_norm
                )

            # 4. Run the optimizer + scheduler step.
            self._run_optimizer_step()

        if grad_norm is not None:
            self._metric_bag.get(Mean, "grad_norm").update(grad_norm)

        self._num_effective_batches += 1

    def _run_optimizer_step(self) -> None:
        """Run optimizer.step() -> scheduler.step() -> optimizer.zero_grad()."""

        # Note: To compile this function set compile_optimizer_step to True.

        self._optimizer.step()

        self._lr_scheduler.step()

        self._optimizer.zero_grad(set_to_none=True)

    def _checkpoint(self) -> None:
        step_nr = self._global_step

        _log.info(f"Saving checkpoint at step {step_nr}.")

        if self._save_model_only:
            self._checkpoint_manager.save_model_only(step_nr, self._model)
        else:
            self._checkpoint_manager.save_checkpoint(
                step_nr,
                TrainerState(self),
                self._model,
                self._optimizer,
                self._train_loader,  # type: ignore
            )

        _log.info(f"Saved checkpoint at step {step_nr}.")


class TrainerState(Stateful):
    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer

    @override
    def state_dict(self) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}

        state_dict["global_step"] = self._trainer._global_step

        state_dict["epoch_step"] = self._trainer._epoch_step

        state_dict["metric_bag"] = self._trainer._metric_bag.state_dict()

        state_dict["lr_scheduler"] = self._trainer._lr_scheduler.state_dict()

        return state_dict

    @override
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._trainer._global_step = state_dict["global_step"]

        self._trainer._epoch_step = state_dict["epoch_step"]

        self._trainer._metric_bag.load_state_dict(state_dict["metric_bag"])

        self._trainer._lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
