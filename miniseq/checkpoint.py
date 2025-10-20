# This file contains code adapted from the facebookresearch/fairseq2 project.
# The original copyright notice and license are provided below.
# ----
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/fairseq2.

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from pickle import PickleError
from shutil import rmtree
from typing import Any, Protocol, TypeAlias, runtime_checkable

import torch
import torch.nn as nn

from miniseq.logging import get_logger
from miniseq.machine import Machine
from miniseq.training import data_parallel

_log = get_logger()


@runtime_checkable
class Stateful(Protocol):
    """
    Stateful protocol for objects that can be checkpointed and restored.
    """

    def state_dict(self) -> dict[str, Any]:
        """
        Objects should return their state_dict representation as a dictionary.
        The output of this function will be checkpointed, and later restored in
        `load_state_dict()`.

        .. warning::
            Because of the inplace nature of restoring a checkpoint, this function
            is also called during `torch.distributed.checkpoint.load`.


        Returns:
            Dict: The objects state dict
        """

        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore the object's state from the provided state_dict.

        Args:
            state_dict: The state dict to restore from
        """

        ...


def torch_tensor_dump(data: Mapping[str, Any], path: Path) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", message=r".*Please use DTensor instead.*"
        )

        def dump_error() -> RuntimeError:
            return RuntimeError(
                path,
                f"The '{path}' tensor file cannot be dumped. See the nested exception for details.",  # fmt: skip
            )

        try:
            fp = path.open(mode="wb")
        except OSError as ex:
            raise dump_error() from ex

        try:
            torch.save(data, fp)
        except (RuntimeError, OSError, PickleError) as ex:
            raise dump_error() from ex
        finally:
            fp.close()


def create_checkpoint_manager(
    cache_dir: Path, machine: Machine, checkpoint_last_n: int | None = None
) -> CheckpointManager:
    checkpoint_dir = cache_dir.joinpath("checkpoints")

    return CheckpointManager(checkpoint_dir, machine, checkpoint_last_n)


CheckpointState: TypeAlias = dict[str, tuple[Path, dict[str, Any]]]


class CheckpointManager:
    _checkpoint_dir: Path
    _machine: Machine
    _checkpoint_last_n: int | None

    def __init__(
        self,
        checkpoint_dir: Path,
        machine: Machine,
        checkpoint_last_n: int | None = None,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir.expanduser().resolve()
        self._machine = machine
        self._checkpoint_last_n = checkpoint_last_n

    def save_model_only(self, step_nr: int, model: nn.Module) -> None:
        state: CheckpointState = {}

        self.delete_checkpoint(step_nr)

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if self._machine.rank == 0:
            tmp_step_dir.mkdir(parents=True, exist_ok=True)

        self._collect_model_state(step_nr, model, state)

        self._save_checkpoint(step_nr, state)

        self._maybe_delete_stale_checkpoints()

    def _maybe_delete_stale_checkpoints(self) -> None:
        if self._checkpoint_last_n is not None:
            steps = self._get_saved_steps(ignore_model_only=False)

            if len(steps) > self._checkpoint_last_n:
                _log.info(f"Deleting stale checkpoint from step {steps[0]}")

                self.delete_checkpoint(steps[0])

    def delete_checkpoint(self, step_nr: int) -> None:
        if self._machine.rank == 0:
            step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

            tmp_step_dir = step_dir.with_suffix(".tmp")

            try:
                rmtree(tmp_step_dir)
            except FileNotFoundError:
                pass

            if step_dir.exists():
                rmtree(step_dir)

        self._machine.barrier()

    def save_checkpoint(
        self,
        step_nr: int,
        trainer: Stateful,
        model: nn.Module,
        optimizer: Stateful,
        data_loader: Stateful,
    ) -> None:
        state: CheckpointState = {}

        self.delete_checkpoint(step_nr)

        tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

        if self._machine.rank == 0:
            tmp_step_dir.mkdir(parents=True, exist_ok=True)

        self._collect_model_state(step_nr, model, state)

        self._collect_trainer_state(step_nr, trainer, state)

        self._collect_optimizer_state(step_nr, optimizer, state)

        self._collect_data_loader_state(step_nr, data_loader, state)

        self._save_checkpoint(step_nr, state)

        self._maybe_delete_stale_checkpoints()

    def _save_state_files(self, step_nr: int, state: CheckpointState) -> None:
        for kind, (file, state_dict) in state.items():
            try:
                torch_tensor_dump(state_dict, file)
            except Exception as ex:
                raise RuntimeError(
                    step_nr,
                    f"{kind} state at step {step_nr} could not be saved to {str(file)}.",
                ) from ex

    def _collect_model_state(
        self, step_nr: int, model: nn.Module, state: CheckpointState
    ) -> None:
        if self._machine.dp_replicate_rank != 0:
            return

        pathname = f"step_{step_nr}.tmp/model/rank_{self._machine.dp_shard_rank}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if self._machine.dp_shard_rank == 0:
            file.parent.mkdir(parents=True, exist_ok=True)

        state_dict = data_parallel.state_dict(model)

        state["model"] = (file, state_dict)

    def _collect_trainer_state(
        self, step_nr: int, trainer: Stateful, state: CheckpointState
    ) -> None:
        pathname = f"step_{step_nr}.tmp/trainer/rank_{self._machine.rank}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if self._machine.rank == 0:
            file.parent.mkdir(parents=True, exist_ok=True)

        state_dict = trainer.state_dict()

        state["trainer"] = (file, state_dict)

    def _collect_optimizer_state(
        self, step_nr: int, optimizer: Stateful, state: CheckpointState
    ) -> None:
        if self._machine.dp_replicate_rank != 0:
            return

        pathname = f"step_{step_nr}.tmp/optimizer/rank_{self._machine.dp_shard_rank}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if self._machine.dp_shard_rank == 0:
            file.parent.mkdir(parents=True, exist_ok=True)

        state_dict = optimizer.state_dict()

        state["optimizer"] = (file, state_dict)

    def _collect_data_loader_state(
        self, step_nr, data_loader: Stateful, state: CheckpointState
    ) -> None:
        pathname = f"step_{step_nr}.tmp/loader/rank_{self._machine.rank}.pt"

        file = self._checkpoint_dir.joinpath(pathname)

        if self._machine.rank == 0:
            file.parent.mkdir(parents=True, exist_ok=True)

        state_dict = data_loader.state_dict()

        state["loader"] = (file, state_dict)

    def _save_checkpoint(self, step_nr: int, state: CheckpointState) -> None:
        self._machine.barrier()

        self._save_state_files(step_nr, state)

        del state

        self._machine.barrier()

        if self._machine.rank == 0:
            tmp_step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}.tmp")

            step_dir = tmp_step_dir.with_suffix("")

            tmp_step_dir.replace(step_dir)

        self._machine.barrier()

    def _get_saved_steps(self, *, ignore_model_only: bool = True) -> list[int]:
        steps = []

        for step_dir in filter(
            lambda file: file.is_dir(), self._checkpoint_dir.glob("step_*")
        ):
            try:
                step_nr = int(step_dir.name[5:])
            except ValueError:
                continue

            if not ignore_model_only or step_dir.joinpath("trainer").exists():
                steps.append(step_nr)

        if steps:
            steps.sort()

        return steps

    def last_saved_step(self, *, ignore_model_only: bool = True) -> int:
        steps = self._get_saved_steps(ignore_model_only=ignore_model_only)

        if not steps:
            raise FileNotFoundError(
                f"No checkpoints found at {str(self._checkpoint_dir)}."
            )

        return steps[-1]

    def load_model_state(self, step_nr: int, model: nn.Module) -> None:
        step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

        if not step_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint for step {step_nr} not found at {step_dir}."
            )

        model_path = step_dir.joinpath(f"model/rank_{self._machine.dp_shard_rank}.pt")
        model_state = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
        data_parallel.load_state_dict(model, model_state)

    def load_checkpoint(
        self,
        step_nr: int,
        trainer: Stateful,
        model: nn.Module,
        optimizer: Stateful,
        data_loader: Stateful,
    ) -> None:
        step_dir = self._checkpoint_dir.joinpath(f"step_{step_nr}")

        if not step_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint for step {step_nr} not found at {step_dir}."
            )

        # Load model state
        self.load_model_state(step_nr, model)

        CPU = torch.device("cpu")

        # Load trainer state
        trainer_path = step_dir.joinpath(f"trainer/rank_{self._machine.rank}.pt")
        trainer_state = torch.load(trainer_path, map_location=CPU, weights_only=False)
        trainer.load_state_dict(trainer_state)

        # Load optimizer state
        optimizer_path = step_dir.joinpath(
            f"optimizer/rank_{self._machine.dp_shard_rank}.pt"
        )
        optimizer_state = torch.load(
            optimizer_path, map_location=CPU, weights_only=False
        )
        optimizer.load_state_dict(optimizer_state)

        # Load data loader state
        loader_path = step_dir.joinpath(f"loader/rank_{self._machine.rank}.pt")
        loader_state = torch.load(loader_path, map_location=CPU, weights_only=False)
        data_loader.load_state_dict(loader_state)

        self._machine.barrier()
