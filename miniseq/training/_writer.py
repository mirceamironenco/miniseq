from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import override

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    has_tensorboard = False
else:
    has_tensorboard = True


class MetricWriter(ABC):
    @abstractmethod
    def record_metrics(
        self,
        run: str,
        values: dict[str, Any],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None: ...

    def record_config(self, run: str, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class LogMetricWriter(MetricWriter):
    def __init__(self, logger: logging.Logger, max_per_row: int = 1) -> None:
        self._log = logger
        self._max_per_row = max_per_row

    @override
    def record_metrics(
        self,
        run: str,
        values: dict[str, Any],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        if not self._log.isEnabledFor(logging.INFO):
            return

        formatted_values = []
        for name, value in values.items():
            if name.endswith(("loss", "metric")):
                formatted_values.append(f"{name}: {value:.7f}")
            elif name in ("lr", "learning_rate"):
                formatted_values.append(f"{name}: {value:.8f}")
            elif "num_" in name:
                formatted_values.append(f"{name}: {int(value)}")
            elif "pct" in name:
                formatted_values.append(f"{name}: {value:.4f}%")
            else:
                formatted_values.append(f"{name}: {value:.4f}")

        s = ""

        if run is not None:
            s = f"{run} - "

        if step_nr is not None:
            s += f"step: {step_nr}\n"

        max_per_row = self._max_per_row

        if max_per_row == -1:
            # -1 == single row
            max_per_row = len(formatted_values)

        groups = [
            formatted_values[index * max_per_row : (index + 1) * max_per_row]
            for index in range(math.ceil(len(formatted_values) / max_per_row))
        ]

        groups = [" | ".join(group) for group in groups]

        s += "\n".join(groups)

        self._log.info(s)

    @override
    def record_config(self, run: str, config: dict[str, Any]) -> None:
        pass

    @override
    def close(self) -> None:
        pass


class TensorBoardWriter(MetricWriter):
    _log_dir: Path
    _writers: dict[str, SummaryWriter]

    def __init__(self, log_dir: Path) -> None:
        if not has_tensorboard:
            raise ImportError(
                "tensorboard not found. Install with `pip install tensorboard`"
            )

        self._log_dir = log_dir
        self._writers = {}

    @override
    def record_metrics(
        self,
        run: str,
        values: dict[str, Any],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        writer = self._get_writer(run)

        for name, value in values.items():
            writer.add_scalar(name, value, step_nr)

        if flush:
            writer.flush()

    @override
    def record_config(self, run: str, config: dict[str, Any]) -> None:
        writer = self._get_writer(run)

        writer.add_text("config", str(config))

    def _get_writer(self, run: str) -> SummaryWriter:
        try:
            writer = self._writers[run]
        except KeyError:
            writer = SummaryWriter(self._log_dir)

            self._writers[run] = writer

        return writer

    @override
    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

        self._writers.clear()


if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class WandbWriter(MetricWriter):
    _run_name: str | None
    _log_dir: Path
    _project: str
    _run: Run

    def __init__(
        self,
        *,
        log_dir: Path,
        project: str,
        run_name: str | None = None,
        run_id: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        config_to_save: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb

        except ImportError:
            has_wandb = False
        else:
            has_wandb = True

        if not has_wandb:
            raise ImportError("wandb not installed. Install with `pip install wandb`")

        self._run_name = run_name

        self._log_dir = log_dir

        self._project = project

        self._config_to_save = config_to_save

        self._config_saved = False

        self._run = wandb.init(
            project=self._project,
            dir=self._log_dir,
            id=run_id,
            name=run_name,
            group=group,
            job_type=job_type,
        )

    @override
    def record_metrics(
        self,
        run: str,
        values: dict[str, Any],
        step_nr: int | None = None,
        *,
        flush: bool = True,
    ) -> None:
        run_to_use = self._run_name or self._run.name or run

        self._get_run(run_to_use).log(data=values, step=step_nr, commit=flush)

        if not self._config_saved and self._config_to_save is not None:
            self.record_config(run_to_use, self._config_to_save)

            self._config_saved = True

    @override
    def record_config(self, run: str, config: dict[str, Any]) -> None:
        self._get_run(run).config.update(config, allow_val_change=True)

    def _get_run(self, run: str):
        if self._run_name is not None and self._run_name != run:
            raise NotImplementedError(
                "Multi-run setting not yet implemented for wandb. "
                f"Original run name: {self._run_name}, got run name: {run}"
            )

        return self._run

    @override
    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
