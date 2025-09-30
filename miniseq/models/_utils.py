import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Callable, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn

from miniseq.logging import format_as_byte_size
from miniseq.machine import Machine
from miniseq.models._lora import LoRALayer
from miniseq.utils import get_local_rank


def merged_named_parameters(
    module: nn.Module,
    prefix: str = "",
    recruse: bool = True,
    remove_duplicate: bool = True,
) -> Iterator[tuple[str, nn.Parameter]]:
    """Similar to module.named_parameters() but if it encounters a LoRALayer, it will
    return the merged weights, and exclude the wrapped model weights.  If there are no
    LoRALayers this function is identical to calling module.named_parameters()."""

    def members_function(
        module: nn.Module,
    ) -> Iterable[tuple[str, nn.Parameter | None]]:
        if isinstance(module, LoRALayer):
            return module.merged_parameters()

        return module._parameters.items()

    for name, param in module._named_members(
        members_function,
        prefix=prefix,
        recurse=recruse,
        remove_duplicate=remove_duplicate,
    ):
        if "wrapped." not in name:
            yield name, param


def infer_device(module: nn.Module, *, recurse: bool = True) -> torch.device:
    """Checks whether all buffers and parameters of `module` are on the same device.
    If all tensors are on the same device, returns that device, otherwise errors out."""

    parameter_devices: set[torch.device] = set()

    buffer_devices: set[torch.device] = set()

    for parameter in module.parameters(recurse=recurse):
        parameter_devices.add(parameter.device)

    for parameter in module.buffers(recurse=recurse):
        buffer_devices.add(parameter.device)

    all_devices = parameter_devices | buffer_devices

    if len(all_devices) == 0:
        # Module has only non-persistent tensors. Default to cpu.
        return torch.device("cpu")

    if len(all_devices) > 1:
        device_types = ",".join([d.type for d in all_devices])

        param_types = ",".join([d.type for d in parameter_devices])

        buffer_types = ",".join([d.type for d in buffer_devices])

        raise RuntimeError(
            f"Found multiple device types ({device_types}) for module.\n"
            f"Rank: {get_local_rank()}, param types ({param_types}), buffer types ({buffer_types})"
        )

    return all_devices.pop()


def any_meta_device(module: nn.Module, *, recurse: bool = True) -> bool:
    """Returns whether any buffer or parameter of `module` is on meta device."""

    for param_or_buffer in chain(module.parameters(recurse), module.buffers(recurse)):
        if param_or_buffer.device == "meta":
            return True

    return False


@runtime_checkable
class ModuleWithNonPersistentBuffer(Protocol):
    def reset_non_persistent_buffers(self) -> None:
        """Reset the non-persistent buffers of the module."""


def reset_non_persistent_buffers(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, ModuleWithNonPersistentBuffer):
            m.reset_non_persistent_buffers()


@runtime_checkable
class ModuleWithParameter(Protocol):
    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""


def reset_parameters(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, ModuleWithParameter):
            m.reset_parameters()


def apply_to_parameters(
    module: nn.Module,
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    recurse: bool = True,
    memo: dict[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    if memo is None:
        memo = {}

    # Post-order apply.
    if recurse:
        for child in filter(None, module.children()):
            apply_to_parameters(child, fn, recurse=recurse, memo=memo)

    def call_fn(
        source: torch.Tensor, is_param: bool = False, requires_grad: bool = False
    ) -> torch.Tensor:
        if source in memo:
            return memo[source]

        target = fn(source)

        if is_param:
            target = nn.Parameter(target, requires_grad)
        elif requires_grad:
            target.requires_grad_(requires_grad)

        memo[source] = target

        return target

    for param_name, param in module.named_parameters(recurse=False):
        if param is None:
            continue

        with torch.no_grad():
            new_param = call_fn(param, is_param=True, requires_grad=param.requires_grad)

        setattr(module, param_name, new_param)

        if (grad := param.grad) is not None:
            with torch.no_grad():
                new_grad = call_fn(grad, requires_grad=grad.requires_grad)

            new_param.grad = new_grad

    for buffer_name, buffer in module.named_buffers(recurse=False):
        if buffer is None:
            continue

        setattr(module, buffer_name, call_fn(buffer))


def broadcast_model(
    model: nn.Module,
    machine: Machine,
    *,
    source_rank: int = 0,
    non_persistent_buffers: bool = False,
) -> None:
    if machine.size == 1:
        return

    if infer_device(model) != machine.device:
        raise ValueError(
            "broadcast_model expects model to be on correct cuda device on all ranks."
        )

    memo: set[torch.Tensor] = set()

    tensors = []

    def collect_tensors(m: nn.Module) -> None:
        for child in m.children():
            collect_tensors(child)

        for param in m.parameters(recurse=False):
            if param in memo:
                continue

            if param.grad is not None:
                raise RuntimeError(
                    "broadcast_module does not support syncing gradients. "
                )

            memo.add(param)

            tensors.append(param.detach())

        for buffer_name, buffer in m.named_buffers(recurse=False):
            if buffer in memo:
                continue

            memo.add(buffer)

            if (
                not non_persistent_buffers
                and buffer_name in m._non_persistent_buffers_set
            ):
                continue

            tensors.append(buffer.detach())

    collect_tensors(model)

    if not tensors:
        raise RuntimeError("broadcast_model could not collect params/buffers.")

    pg = machine.process_group()

    bucket_size = 250 * 1024 * 1024  # Same as DDP bucket size.

    from torch.distributed import _broadcast_coalesced  # type: ignore

    _broadcast_coalesced(pg, tensors, bucket_size, source_rank)


@dataclass(kw_only=True)
class ModuleSizeInfo:
    """Holds the size information of a module."""

    param_size: int = 0
    """The total size of all parameters."""

    param_size_bytes: int = 0
    """The total size of all parameters, in bytes."""

    trainable_param_size: int = 0
    """The total size of all trainable parameters."""

    trainable_param_size_bytes: int = 0
    """The total size of all trainable parameters, in bytes."""

    buffer_size: int = 0
    """The total size of all buffers."""

    buffer_size_bytes: int = 0
    """The total size of all buffers, in bytes."""

    total_size: int = 0
    """The total size of the module."""

    total_size_bytes: int = 0
    """The total size of the module, in bytes."""


def get_module_size_info(module: nn.Module) -> ModuleSizeInfo:
    """Return the size information of ``module`` and its descendant modules."""

    def get_numel(tensor: torch.Tensor) -> int:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            return cast(DTensor, tensor.detach()).to_local().numel()  # type: ignore[no-any-return]

        return tensor.numel()

    info = ModuleSizeInfo()

    param: torch.Tensor | None

    for param in module.parameters():
        if param is None:
            continue

        numel = get_numel(param)

        size_bytes = numel * param.element_size()

        info.param_size += numel
        info.param_size_bytes += size_bytes

        if param.requires_grad:
            info.trainable_param_size += numel
            info.trainable_param_size_bytes += size_bytes

        info.total_size += numel
        info.total_size_bytes += size_bytes

    for buffer in module.buffers():
        if buffer is None:
            continue

        numel = buffer.numel()

        size_bytes = numel * buffer.element_size()

        info.buffer_size += numel
        info.buffer_size_bytes += size_bytes

        info.total_size += numel
        info.total_size_bytes += size_bytes

    return info


def get_module_size(module: nn.Module) -> ModuleSizeInfo:
    """Return the size information of ``module`` and its descendant modules."""
    info = ModuleSizeInfo()

    for param in module.parameters():
        if param is not None:
            size = param.numel()
            size_bytes = size * param.element_size()

            info.param_size += size
            info.param_size_bytes += size_bytes

            if param.requires_grad:
                info.trainable_param_size += size
                info.trainable_param_size_bytes += size_bytes

            info.total_size += size
            info.total_size_bytes += size_bytes

    for buffer in module.buffers():
        size = buffer.numel()
        size_bytes = size * buffer.element_size()

        info.buffer_size += size
        info.buffer_size_bytes += size * size_bytes

        info.total_size += size
        info.total_size_bytes += size_bytes

    return info


def log_model(log: logging.Logger, model: nn.Module) -> None:
    """Log information about ``model``."""

    si = get_module_size_info(model)

    s = (
        f"Parameter Size: {si.param_size:,} | "
        f"Parameter Size (bytes): {format_as_byte_size(si.param_size_bytes)} | "
        f"Trainable Parameter Size: {si.trainable_param_size:,} | "
        f"Trainable Parameter Size (bytes): {format_as_byte_size(si.trainable_param_size_bytes)} | "
        f"Buffer Size: {si.buffer_size:,} | "
        f"Buffer Size (bytes): {format_as_byte_size(si.buffer_size_bytes)} | "
        f"Total Size: {si.total_size:,} | "
        f"Total Size (bytes): {format_as_byte_size(si.total_size_bytes)}"
    )

    log.info(f"Model - {s}\n{model}")
