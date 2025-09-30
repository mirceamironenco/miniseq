import gc
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import (
    Callable,
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import torch
import torch.nn as nn
from torch.distributed import Backend

ModuleT = TypeVar("ModuleT", bound=nn.Module)


def count_trainable_params(model: nn.Module) -> int:
    return sum([param.numel() for param in model.parameters() if param.requires_grad])


@runtime_checkable
class SupportsDeviceTransfer(Protocol):
    def to(self, device: torch.device, *, non_blocking: bool = False) -> Self: ...


@runtime_checkable
class SupportsPinMemory(Protocol):
    def pin_memory(self) -> Self: ...


TorchCompileMode: TypeAlias = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]


def should_compile_flex() -> bool:
    return torch.cuda.is_available()


P = ParamSpec("P")
T = TypeVar("T")


def skip_compile(fn: Callable[P, T]) -> Callable[P, T]:
    """Equivalent to @torch.compiler.disable(recursive=False). Preserves type hints."""

    return torch.compiler.disable(fn=fn, recursive=False)  # type: ignore


def replace_method_signature_with(
    other_class_methods: Callable[P, T],
) -> Callable[[Callable], Callable[P, T]]: ...


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterator[None]:
    _dtype = torch.get_default_dtype()

    torch.set_default_dtype(dtype)

    try:
        yield
    finally:
        torch.set_default_dtype(_dtype)


def default_cuda_device() -> torch.device | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None

    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is None:
        assert get_local_world_size() == 1
        local_rank = 0

    device = torch.device("cuda", index=int(local_rank))
    return device


def default_device() -> torch.device:
    cuda_device = default_cuda_device()

    if cuda_device is None:
        return torch.device("cpu")

    torch.cuda.set_device(cuda_device)
    return cuda_device


def make_dtype(dtype: Literal["float32", "bfloat16", "float64"]) -> torch.dtype:
    torch_dtype = getattr(torch, dtype, None)

    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    raise ValueError(f"Expected valid torch.dtype, got {dtype} instead.")


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def on_local_rank_zero() -> bool:
    return get_local_rank() == 0


def get_distributed_backend(device: torch.device) -> str:
    assert device.type in ("cpu", "cuda")
    return Backend.default_device_backend_map[device.type]


def is_torchrun_cluster(env: Mapping[str, str] | None = None) -> bool:
    if env is None:
        env = os.environ

    return "TORCHELASTIC_RUN_ID" in env


def next_multiple(x: int, multiple: int) -> int:
    return multiple * ((x - 1) // multiple) + multiple


def clear_unused_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


CPU = torch.device("cpu")


def to_tensor(
    data: int | float | Sequence[int] | Sequence[float],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None or device.type != "cuda":
        return torch.tensor(data, dtype=dtype, device=device)

    t = torch.tensor(data, dtype=dtype, device=CPU, pin_memory=True)

    return t.to(device, non_blocking=True)
