from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from typing_extensions import override

from miniseq.data import PromptBatch, TrajectoryBatch
from miniseq.generation import Generator
from miniseq.machine import Machine
from miniseq.metric_bag import MetricBag
from miniseq.trainer import TrainUnit
from miniseq.transformer import CausalTransformerModel


class OnlineTrainUnit(TrainUnit[TrajectoryBatch], ABC):
    _model: CausalTransformerModel
    _machine: Machine
    _generator: Generator
    _trajectory_size: int
    _trajectory_epochs: int

    def __init__(
        self,
        model: CausalTransformerModel,
        machine: Machine,
        generator: Generator,
        trajectory_size: int,
        trajectory_epochs: int = 1,
    ) -> None:
        self._model = model
        self._machine = machine
        self._generator = generator
        self._trajectory_size = trajectory_size

        # Equivalent to ppo_epochs
        self._trajectory_epochs = trajectory_epochs

    @abstractmethod
    def generate_and_score(
        self, batch: PromptBatch, metric_bag: MetricBag
    ) -> TrajectoryBatch: ...

    @abstractmethod
    def compute_loss(
        self, batch: TrajectoryBatch, metric_bag: MetricBag
    ) -> tuple[torch.Tensor, int | None]: ...

    def update_rollout_model(self) -> None:
        # Note: generator.model and _model are the same object.
        self._generator.update_model(only_trainable=True)

    def run_generation(
        self, batch: PromptBatch, *, metric_bag: MetricBag
    ) -> list[TrajectoryBatch]:
        assert batch.batch_size % self._trajectory_size == 0

        self._generator.prepare_for_generation()

        trajectory = self.generate_and_score(batch, metric_bag)

        self._generator.after_generation()

        trajectories = [
            traj
            for traj in trajectory.chunk(
                num_chunks=batch.batch_size // self._trajectory_size, and_prepare=True
            )
            for _ in range(self._trajectory_epochs)
        ]

        return trajectories

    @override
    def __call__(
        self, batch: TrajectoryBatch, *, metric_bag: MetricBag
    ) -> tuple[torch.Tensor, int | None]:
        return self.compute_loss(batch, metric_bag)

    @property
    @override
    def model(self) -> nn.Module:
        return self._model
