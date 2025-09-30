from typing import Literal

import torch
from torch.nn.functional import log_softmax
from torch.nn.functional import nll_loss as torch_nll_loss


@torch.compile(dynamic=True)
def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: Literal["none", "mean", "sum"] = "sum",
    loss_mask: torch.Tensor | None = None,
    pad_idx: int | None = None,
) -> torch.Tensor:
    batch_shape = targets.size()

    # (B, S, V) -> (B * S, V)
    logits = logits.contiguous().flatten(0, 1)

    # (B, S) -> (B * S,)
    targets = targets.contiguous().flatten(0, 1)

    lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

    if pad_idx is None:
        pad_idx = -100

    loss = torch_nll_loss(
        lprobs,
        targets,
        ignore_index=pad_idx,
        reduction=reduction if loss_mask is None else "none",
    )

    del lprobs

    if loss_mask is None:
        return loss

    # (B, S) -> (B * S,)
    loss_mask = loss_mask.contiguous().flatten(0, 1)

    loss = loss * loss_mask

    if reduction == "none":
        return loss.view(batch_shape)

    if reduction == "sum":
        return loss.sum()

    if reduction == "mean":
        return loss.mean()

    raise ValueError(
        f"`reduction` must be 'sum'/'mean'/'none', but is '{reduction}' instead."
    )


def repeat_interleave(x: torch.Tensor, *, dim: int, repeat: int) -> torch.Tensor:
    if repeat == 1:
        return x

    dim = dim + x.ndim if dim < 0 else dim

    shape = [-1] * (x.ndim + 1)

    shape[dim + 1] = repeat

    return x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)
