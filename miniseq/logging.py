from __future__ import annotations

import logging
import math
import os
from logging import DEBUG, INFO, Formatter, Handler, NullHandler, getLogger
from typing import Final

import rich
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

_CONSOLE: Console | None = None
_ERROR_CONSOLE: Console | None = None


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "MINISEQ")


def setup_logging(*, debug: bool = False) -> None:
    rank = int(os.environ.get("LOCAL_RANK", 0))

    handlers: list[Handler] = []
    if rank == 0:
        console = get_error_console()

        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            keywords=[],
        )
        fmt = Formatter("%(name)s - %(message)s")
        handler.setFormatter(fmt)

        handlers.append(handler)

    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=DEBUG if debug else INFO, handlers=handlers, datefmt=datefmt, force=True
    )

    if rank != 0:
        getLogger().addHandler(NullHandler())


def get_console() -> Console:
    global _CONSOLE

    if _CONSOLE is None:
        _CONSOLE = rich.get_console()
    return _CONSOLE


def get_error_console() -> Console:
    global _ERROR_CONSOLE

    if _ERROR_CONSOLE is None:
        _ERROR_CONSOLE = Console(stderr=True, highlight=False)
    return _ERROR_CONSOLE


def create_rich_progress(
    disable: bool = False, *, speed_estimate: float = 180
) -> Progress:
    """Create a :class:`Progress` instance to report job progress.

    Note: If speed_estimate is too low, the progress bar will not show time remaining.
    See https://github.com/Textualize/rich/issues/3089.
    """

    console = get_error_console()

    columns = [
        TextColumn("{task.description}:"),
        TimeElapsedColumn(),
        BarColumn(),
        BasicMofNCompleteColumn(),
        TaskProgressColumn(show_speed=False),
        TimeRemainingColumn(),
    ]

    return Progress(
        *columns,
        transient=True,
        console=console,
        disable=disable,
        speed_estimate_period=speed_estimate,
    )


class BasicMofNCompleteColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.total is None:
            s = f"{task.completed:5d}"
        else:
            s = f"{task.completed:5d}/{task.total}"

        return Text(s, style="progress.download")


def log_table(rich_table: Table) -> Text:
    """Generate ascii formatted presentation of a Rich table, eliminates column style."""

    # Source:
    # https://github.com/Textualize/rich/discussions/1799#discussioncomment-1994605
    console = get_console()

    with console.capture() as capture:
        console.print(rich_table)

    return Text.from_ansi(capture.get())


def log_config(log: logging.Logger, config: object) -> None:
    log.info(
        f"Config:\n {pretty_repr(config, max_width=140, max_string=80, indent_size=1)}"
    )


_UNITS: Final = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]


def format_as_byte_size(value: object) -> str:
    """Format metric ``value`` in byte units."""

    if isinstance(value, float):
        size = value
    elif isinstance(value, (str, torch.Tensor, int)):
        try:
            size = float(value)
        except ValueError:
            return f"{value}"
    else:
        return f"{value}"

    unit_idx = 0

    if not math.isfinite(size) or size <= 0.0:
        return "0 B"

    while size >= 1024:
        size /= 1024

        unit_idx += 1

    try:
        return f"{size:.2f} {_UNITS[unit_idx]}"
    except IndexError:
        return "value is too big to be properly formatted."
