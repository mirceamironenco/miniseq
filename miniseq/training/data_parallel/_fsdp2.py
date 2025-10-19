from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ReduceOp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    UnshardHandle,
    fully_shard,
)
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from miniseq.logging import get_logger
from miniseq.machine import Machine, all_ranks_same_trainable_params
from miniseq.models import (
    any_meta_device,
    apply_to_parameters,
    infer_device,
    reset_non_persistent_buffers,
)
from miniseq.transformer import TransformerDecoderModel
from miniseq.utils import ModuleT

_log = get_logger()

if TYPE_CHECKING:
    # Properly reflects class hierarchy.

    class FSDP2Module(nn.Module):
        def reshard(self) -> None: ...

        def unshard(self, async_op: bool = False) -> UnshardHandle | None: ...

        def set_is_last_backward(self, is_last_backward: bool) -> None: ...

        def set_requires_gradient_sync(
            self, requires_gradient_sync: bool, *, recurse: bool = True
        ) -> None: ...

        def set_reshard_after_backward(
            self, reshard_after_backward: bool, *, recurse: bool = True
        ) -> None: ...

else:
    FSDP2Module: TypeAlias = FSDPModule


def to_fsdp2(
    model: ModuleT,
    machine: Machine,
    *,
    reshard_after_forward: bool | int = True,
    mp_policy: MixedPrecisionPolicy,
    cpu_offload: bool = False,
    fsdp2_reshard_outer_fwd: bool = True,
) -> ModuleT:
    if not isinstance(model, TransformerDecoderModel):
        raise ValueError(
            f"to_fsdp2 is supported only for TransformedDecoder, got {type(model)}"
        )

    model_device = infer_device(model, recurse=True)

    _log.info(f"Starting FSDP2 sharding, current device={model_device=}")

    full_state_dict: dict[str, Any] | None = None

    if machine.rank == 0:
        if model_device.type == "meta":
            raise ValueError(
                "If init_sync=True, rank 0 must be on cpu/cuda and initialized, since we are broadcasting params/buffers from it."
            )

        # Take the state_dict from the first rank, and broadcast it
        full_state_dict = model.state_dict()

    if machine.rank != 0:
        if not model_device.type == "meta":
            raise ValueError("Model on rank != 0 must be on meta device.")

    if cpu_offload:
        try:
            machine.process_group()._get_backend(torch.device("cpu"))
        except Exception:
            # https://github.com/pytorch/torchtitan/blob/main/torchtitan/utils.py#L229
            raise ValueError(
                "FSDP2 cpu offloading requires adding `cpu:gloo` to process group backend."
            )

    device_mesh = DeviceMesh.from_group(
        machine.process_group(), device_type=machine.device.type
    )

    model = shard_transformer_decoder_(
        model=model,
        mesh=device_mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=mp_policy,
        cpu_offload=cpu_offload,
        reshard_outer_fwd=fsdp2_reshard_outer_fwd,
    )

    machine.barrier()

    # Sharded state_dict on every rank.
    meta_sharded_sd = model.state_dict()

    # To be used for load_state_dict.
    sharded_load_sd: dict[str, object] = {}

    if machine.rank == 0 and full_state_dict is None:
        raise ValueError("Failed to broadcast state_dict from rank 0.")

    for name, sharded_meta_param in meta_sharded_sd.items():
        assert sharded_meta_param is not None
        assert isinstance(sharded_meta_param, DTensor)
        assert sharded_meta_param.device_mesh == device_mesh

        if machine.rank == 0:
            assert full_state_dict is not None
            tensor_ = full_state_dict.get(name)

            assert tensor_ is not None
            tensor_ = tensor_.detach().to(machine.device)
        else:
            tensor_ = torch.empty(
                sharded_meta_param.size(),
                device=machine.device,
                dtype=sharded_meta_param.dtype,
            )

        dist.broadcast(tensor_, src=0, group=machine.process_group())

        sharded_tensor = distribute_tensor(
            tensor_, device_mesh=device_mesh, placements=sharded_meta_param.placements
        )

        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()

        sharded_load_sd[name] = sharded_tensor

    # NB: assign=True needed since rank != 0 is still on meta device.
    model.load_state_dict(sharded_load_sd, strict=True, assign=True)

    # Set the device for non-persistent buffers, which are not present in state_dict.
    for submodule in model.modules():
        for buffer_name, buffer in submodule.named_buffers(recurse=False):
            if buffer_name not in meta_sharded_sd:
                # Set the correct device.
                buffer = torch.empty_like(buffer, device=machine.device)

                setattr(submodule, buffer_name, buffer)

    # NB: Here we are reinitializing the non-persistent buffers, even on rank 0.
    reset_non_persistent_buffers(model)

    if any_meta_device(model):
        raise RuntimeError(
            f"FSDP2 sharding failed. Model on rank {machine.rank} still has params/buffers on meta device."
        )

    machine.barrier()

    if not all_ranks_same_trainable_params(model, machine):
        raise RuntimeError(
            "FSDP2 sharding failed. Different ranks have different number of trainable params."
        )

    _log.info(f"Finished FSDP2 model sharding and loading; world_size = {machine.size}")

    assert infer_device(model) == machine.device

    return model


def shard_transformer_decoder_(
    model: TransformerDecoderModel,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int = True,
    shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = None,
    mp_policy: MixedPrecisionPolicy,
    cpu_offload: bool = False,
    reshard_outer_fwd: bool = True,
    reduce_scatter_reduce_op: ReduceOp.RedOpType | None = None,  # experimental
) -> FSDP2Module:
    """Note: If reshard_after_fwd is False, reshard_outer_fwd will be set to False."""

    num_blocks_to_shard = len(model.decoder.layers)

    for block_index, (block_name, block_module) in enumerate(
        model.decoder.layers.named_children(), 1
    ):
        # From torchtitan: Do not reshard after fwd for last transformer block
        # Since FSDP would prefetch it immediatly.
        _reshard_after_fwd = reshard_after_forward and block_index < num_blocks_to_shard

        fully_shard_module_(
            block_module,
            mesh=mesh,
            reshard_after_forward=_reshard_after_fwd,
            shard_placement_fn=shard_placement_fn,
            mp_policy=mp_policy,
            cpu_offload=cpu_offload,
            reduce_scatter_reduce_op=reduce_scatter_reduce_op,
        )

    return fully_shard_module_(
        model,
        mesh=mesh,
        reshard_after_forward=reshard_outer_fwd and reshard_after_forward,
        shard_placement_fn=shard_placement_fn,
        mp_policy=mp_policy,
        cpu_offload=cpu_offload,
        reduce_scatter_reduce_op=reduce_scatter_reduce_op,
    )


def fully_shard_module_(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | int = True,
    shard_placement_fn: Callable[[nn.Parameter], Shard | None] | None = None,
    mp_policy: MixedPrecisionPolicy,
    cpu_offload: bool = False,
    reduce_scatter_reduce_op: ReduceOp.RedOpType | None = None,  # experimental
) -> FSDP2Module:
    # Note: Currently doesn't support module being a List[nn.Module].

    offload_policy = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()

    fsdp_module = fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        shard_placement_fn=shard_placement_fn,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )

    module = cast(FSDP2Module, fsdp_module)

    if reduce_scatter_reduce_op is not None:
        assert hasattr(fully_shard, "state")
        fsdp_param_group = fully_shard.state(module)._fsdp_param_group
        fsdp_param_group.reduce_scatter_reduce_op = reduce_scatter_reduce_op

    return module


@contextmanager
def fsdp2_summon_full_parameters(module: FSDP2Module) -> Iterator[None]:
    # Adapted from facebookresearch/fairseq2.

    state = fully_shard.state(module)

    mp: MixedPrecisionPolicy | None

    try:
        mp = state._mp_policy
    except AttributeError:
        mp = None

    def disable_hooks(module: nn.Module, hook_name: str) -> None:
        backup_key = f"__fs2_{hook_name}_backup__"

        original_hooks = getattr(module, hook_name)

        hooks = dict()

        # Remove any FSDP2 hook.
        for handle, hook in original_hooks.items():
            try:
                hook_module = hook.__module__
            except AttributeError:
                hook_module = ""

            if hook_module.startswith("torch.distributed.fsdp"):
                continue

            hooks[handle] = hook

        setattr(module, backup_key, original_hooks)

        setattr(module, hook_name, hooks)

    def enable_hooks(module: nn.Module, hook_name: str) -> None:
        backup_key = f"__fs2_{hook_name}_backup__"

        hooks = getattr(module, backup_key)

        setattr(module, hook_name, hooks)

        delattr(module, backup_key)

    def unshard(module: nn.Module) -> None:
        for child in module.children():
            unshard(child)

        if isinstance(module, FSDP2Module):
            module.unshard()

            disable_hooks(module, "_forward_hooks")
            disable_hooks(module, "_forward_pre_hooks")

    def reshard(module: nn.Module) -> None:
        for child in module.children():
            reshard(child)

        if isinstance(module, FSDP2Module):
            enable_hooks(module, "_forward_hooks")
            enable_hooks(module, "_forward_pre_hooks")

            module.reshard()

    def maybe_cast_dtype(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, nn.Parameter):
            return t

        if mp is None:
            return t

        if mp.param_dtype is None:
            return t

        return t.to(mp.param_dtype)

    unshard(module)

    apply_to_parameters(module, maybe_cast_dtype)

    try:
        yield
    finally:
        reshard(module)


def maybe_get_device_mesh(state_dict: dict[str, Any]) -> DeviceMesh | None:
    for item in state_dict.values():
        if isinstance(item, DTensor):
            return item.device_mesh

    return None


def fsdp2_local_state_dict(module: FSDP2Module) -> dict[str, Any]:
    # Note: Currently HSDP is not supported, however this implementation is compatible
    # with it, since replicated tensors are only saved on sharded rank 0.

    sharded_sd = module.state_dict()

    device_mesh = maybe_get_device_mesh(sharded_sd)

    if device_mesh is not None:
        sdp_rank = device_mesh.get_local_rank(0 if device_mesh.ndim == 1 else 1)
    else:
        sdp_rank = 0

    state_dict: dict[str, Any] = {}

    for key, value in sharded_sd.items():
        if isinstance(value, DTensor):
            state_dict[key] = cast(DTensor, value.detach()).to_local()
        elif sdp_rank == 0:
            if isinstance(value, torch.Tensor):
                value = value.detach()

            state_dict[key] = value

    return state_dict


def fsdp2_load_local_state_dict(
    module: FSDP2Module, state_dict: Mapping[str, Any]
) -> None:
    state_dict = dict(state_dict)

    for key, value in module.state_dict().items():
        if isinstance(value, DTensor):
            input_value = state_dict.get(key)
            if isinstance(input_value, torch.Tensor):
                cast(DTensor, value.detach()).to_local().copy_(input_value)

                state_dict[key] = value

    module.load_state_dict(state_dict)


@contextmanager
def fsdp2_no_sync(module: FSDP2Module) -> Iterator[None]:
    # Set if the module should sync gradients.
    module.set_requires_gradient_sync(False, recurse=True)

    # Set if the module should reshard parameters after backward.
    # Trades off memory for reduced comm. since unsharded params do not need to
    # be re-all-gathered before the next forward.
    module.set_reshard_after_backward(False, recurse=True)

    # Set if the next backward is the last one.
    module.set_is_last_backward(False)

    try:
        yield
    finally:
        module.set_requires_gradient_sync(True, recurse=True)

        module.set_reshard_after_backward(True, recurse=True)

        module.set_is_last_backward(True)
