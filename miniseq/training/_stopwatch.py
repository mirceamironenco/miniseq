# This file contains code adapted from the facebookresearch/fairseq2 project.
# The original copyright notice and license are provided below.
# ----
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of https://github.com/facebookresearch/fairseq2.

import time
from typing import Any, Self

import torch


class StopWatch:
    _is_running: bool
    _start_time: float
    _accumulated_duration: float
    _device: torch.device

    def __init__(self, *, device: torch.device | None = None) -> None:
        self._is_running = False

        self._start_time = 0.0

        self._accumulated_duration = 0.0

        if device is not None:
            if device.type not in ("cuda", "cpu"):
                raise ValueError(f"Got device type {device.type}, expected cuda/cpu.")

        self._device = device or torch.device("cpu")

    def start(self) -> None:
        if self._is_running:
            raise RuntimeError("StopWatch already running.")

        self._start_time = time.perf_counter()

        self._is_running = True

    def stop(self) -> None:
        if not self._is_running:
            return

        self._sync_device()

        self._accumulated_duration += time.perf_counter() - self._start_time

        self._is_running = False

    def reset(self) -> None:
        self._accumulated_duration = 0.0

        if self._is_running:
            self._sync_device()

            self._start_time = time.perf_counter()

    def get_elapsed_time(self) -> float:
        if not self._is_running:
            return self._accumulated_duration

        self._sync_device()

        return self._accumulated_duration + (time.perf_counter() - self._start_time)

    def _sync_device(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)

    def __enter__(self) -> Self:
        self.start()

        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._is_running
