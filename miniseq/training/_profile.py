# This file contains code adapted from the facebookresearch/fairseq2 project.
# The original copyright notice and license are provided below.
# ----
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/fairseq2.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from typing_extensions import override

from miniseq.machine import Machine


class Profiler(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    def __enter__(self) -> Self:
        self.start()

        return self

    def __exit__(self, *ex: Any) -> None:
        self.stop()


class TorchProfiler(Profiler):
    _profile: profile

    def __init__(
        self,
        *,
        skip_n_steps: int,
        wait_n_steps: int,
        num_warmup_steps: int,
        num_active_steps: int,
        repeat: int,
        log_dir: Path,
        machine: Machine,
    ) -> None:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        schedule_ = schedule(
            wait=wait_n_steps,
            warmup=num_warmup_steps,
            active=num_active_steps,
            repeat=repeat,
            skip_first=skip_n_steps,
        )

        trace_handler = tensorboard_trace_handler(
            str(log_dir), worker_name=f"rank_{machine.rank}", use_gzip=True
        )

        self._profile = profile(
            activities=activities,
            schedule=schedule_,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

    def start(self) -> None:
        self._profile.start()

    def stop(self) -> None:
        self._profile.stop()

    def step(self) -> None:
        self._profile.step()


class NoopProfiler(Profiler):
    @override
    def start(self) -> None:
        pass

    @override
    def stop(self) -> None:
        pass

    @override
    def step(self) -> None:
        pass
