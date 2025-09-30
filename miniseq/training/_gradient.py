import torch
import torch.nn as nn
from torch.distributed import ReduceOp

from miniseq.machine import Machine
from miniseq.utils import to_tensor


def clip_gradient_norm(
    module: nn.Module, max_norm: float | None, norm_type: float = 2.0
) -> torch.Tensor:
    if max_norm is None:
        max_norm = torch.inf

    return torch.nn.utils.clip_grad_norm_(
        module.parameters(), max_norm, norm_type, error_if_nonfinite=False
    )


def normalize_gradients(
    module: nn.Module, *, num_targets: int, machine: Machine, foreach: bool = False
) -> None:
    total_num_targets = to_tensor(num_targets, device=machine.device)

    machine.all_reduce(total_num_targets, op=ReduceOp.SUM)

    scale_gradients(module, machine.size / total_num_targets, foreach=foreach)


def scale_gradients(
    module: nn.Module, value: float | torch.Tensor, foreach: bool = False
) -> None:
    if foreach:
        grads = [param.grad for param in module.parameters() if param.grad is not None]
        torch._foreach_mul_(grads, value)
    else:
        for param in module.parameters():
            if param.grad is not None:
                param.grad *= value
