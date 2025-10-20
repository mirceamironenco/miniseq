from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.nn.parallel import DistributedDataParallel as DDP

from miniseq.machine import Machine
from miniseq.models import infer_device
from miniseq.training.data_parallel._ddp import to_ddp
from miniseq.training.data_parallel._fsdp2 import (
    FSDP2Module,
    fsdp2_load_local_state_dict,
    fsdp2_local_state_dict,
    fsdp2_no_sync,
    fsdp2_summon_full_parameters,
    to_fsdp2,
)
from miniseq.utils import ModuleT


def to_data_parallel(
    model: ModuleT,
    machine: Machine,
    *,
    replicate: bool,
    shard: bool,
    mp_dtype: torch.dtype | None = None,
    ddp_broadcast_buffers: bool = False,
    ddp_find_unused_parameters: bool = False,
    ddp_static_graph: bool = False,
    fsdp2_reshard_fwd: bool | int = True,
    fsdp2_fp32_reduce: bool = False,
    fsdp2_cpu_offload: bool = False,
    fsdp2_reshard_outer_fwd: bool = True,
) -> ModuleT:
    if machine.size == 1:
        assert infer_device(model).type != "meta"
        assert infer_device(model) == machine.device
        return model

    if replicate:
        return to_ddp(
            model,
            machine,
            broadcast_buffers=ddp_broadcast_buffers,
            find_unused_parameters=ddp_find_unused_parameters,
            static_graph=ddp_static_graph,
        )
    elif shard:
        if mp_dtype is None or mp_dtype == torch.float32:
            mp_policy = MixedPrecisionPolicy()
        elif mp_dtype is not None:
            reduce_dtype = torch.float32 if fsdp2_fp32_reduce else mp_dtype

            mp_policy = MixedPrecisionPolicy(
                param_dtype=mp_dtype, reduce_dtype=reduce_dtype
            )

        return to_fsdp2(
            model,
            machine,
            reshard_after_forward=fsdp2_reshard_fwd,
            mp_policy=mp_policy,
            cpu_offload=fsdp2_cpu_offload,
            fsdp2_reshard_outer_fwd=fsdp2_reshard_outer_fwd,
        )
    else:
        raise ValueError(
            "If machine.size > 1, expected either replicate or shard to be True."
        )


def state_dict(model: nn.Module | DDP | FSDP2Module) -> dict[str, Any]:
    if isinstance(model, DDP):
        return model.module.state_dict()

    if isinstance(model, FSDP2Module):
        return fsdp2_local_state_dict(model)

    return model.state_dict()


def load_state_dict(
    model: nn.Module | DDP | FSDP2Module, state_dict: Mapping[str, Any]
) -> None:
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict)
    elif isinstance(model, FSDP2Module):
        fsdp2_load_local_state_dict(model, state_dict)
    else:
        model.load_state_dict(state_dict)


def summon_full_parameters(
    model: nn.Module | DDP | FSDP2Module,
) -> AbstractContextManager[None]:
    if isinstance(model, FSDP2Module):
        return fsdp2_summon_full_parameters(model)

    return nullcontext()


def no_sync(model: nn.Module | DDP | FSDP2Module) -> AbstractContextManager[None]:
    if isinstance(model, DDP):
        return model.no_sync()

    if isinstance(model, FSDP2Module):
        return fsdp2_no_sync(model)

    return nullcontext()


def base_module(model: nn.Module | DDP | FSDP2Module) -> nn.Module:
    if isinstance(model, DDP):
        return model.module

    if isinstance(model, FSDP2Module):
        return model

    return model
