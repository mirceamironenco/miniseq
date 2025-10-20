from typing import Callable, Literal

import torch
from cut_cross_entropy.cce import CCEParams, linear_cross_entropy_apply
from cut_cross_entropy.utils import _handle_eps
from cut_cross_entropy.vocab_parallel import VocabParallelOptions
from typing_extensions import override

from miniseq.nn import Module
from miniseq.transformer._transformer import TransformerDecoderModel


# Note: Bellow (accum_* and filter_* correspond to 'cce_kahan_full_e'), see:
# https://github.com/apple/ml-cross-entropy/blob/main/cut_cross_entropy/cce_utils.py#L59
def cce_linear_cross_entropy_masked(
    e: torch.Tensor,  # pre_logit
    c: torch.Tensor,  # classifier_weight
    targets: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    *,
    num_valids: int | None = None,
    bias: torch.Tensor | None = None,
    softcap: float | None = None,
    reduction: str = "sum",  # Note: Changed default from 'mean'.
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = True,
    accum_c_fp32: bool = True,
    filter_e_grad: bool = False,
    filter_c_grad: bool = True,
    vocab_parallel_options: VocabParallelOptions | None = None,
) -> torch.Tensor:
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)

    batch_shape = targets.size()

    # (B, S, *) -> (B * S, *)
    e = e.contiguous().flatten(0, -2)

    # (B, S) -> (B * S,)
    targets = targets.contiguous().flatten()

    valids: torch.Tensor | None = None

    if target_mask is not None:
        # (B, S) -> (B * S,)
        target_mask = target_mask.contiguous().flatten()

        # nonzero_static avoids a cuda sync
        if num_valids is not None:
            valids = (
                target_mask.nonzero_static(size=num_valids).to(torch.int32).squeeze(1)
            )
        else:
            valids = target_mask.nonzero().to(torch.int32).squeeze(1)

    if (targets.data_ptr() % 16) != 0:
        targets = torch.nn.functional.pad(targets, (0, 1))[:-1]

    assert (targets.data_ptr() % 16) == 0

    loss, _ = linear_cross_entropy_apply(
        e=e,
        c=c,
        bias=bias,
        params=CCEParams(
            targets=targets,
            valids=valids,
            softcap=softcap,
            reduction=reduction,
            filter_eps=_handle_eps(filter_eps, e.dtype),
            shift=0,
            batch_shape=batch_shape,
            accum_e_fp32=accum_e_fp32,
            accum_c_fp32=accum_c_fp32,
            filter_e_grad=filter_e_grad and filter_eps is not None,
            filter_c_grad=filter_c_grad and filter_eps is not None,
            vocab_parallel_options=vocab_parallel_options,
            return_lse=False,
        ),
    )

    del e

    return loss


class LinearCrossEntropyMasked(Module[torch.Tensor]):
    def __init__(
        self,
        softcap: float | None = None,
        reduction: str = "sum",  # Note: Changed default from 'mean'.
        filter_eps: float | str | None = "auto",
        accum_e_fp32: bool = False,
        accum_c_fp32: bool = False,
        filter_e_grad: bool = True,
        filter_c_grad: bool = True,
    ) -> None:
        super().__init__()
        self.softcap = softcap
        self.reduction = reduction
        self.filter_eps = filter_eps
        self.accum_e_fp32 = accum_e_fp32
        self.accum_c_fp32 = accum_c_fp32
        self.filter_e_grad = filter_e_grad
        self.filter_c_grad = filter_c_grad

    def forward(
        self,
        e: torch.Tensor,
        c: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return cce_linear_cross_entropy_masked(
            e,
            c,
            targets,
            target_mask,
            bias=bias,
            reduction=self.reduction,
            # shift=0,
            filter_eps=self.filter_eps,
            accum_e_fp32=self.accum_e_fp32,
            accum_c_fp32=self.accum_c_fp32,
            filter_e_grad=self.filter_e_grad,
            filter_c_grad=self.filter_c_grad,
            vocab_parallel_options=None,
        )


def build_cce_nll_forward() -> Callable[..., torch.Tensor]:
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Cut Cross Entropy requires an ampere GPU or newer. "
            "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
        )

    @override
    def loss(
        self: TransformerDecoderModel,
        decoder_output: torch.Tensor,
        target_seqs: torch.Tensor,
        *,
        reduction: Literal["sum", "none", "mean"] = "sum",
        target_mask: torch.Tensor | None = None,
        ignore_prefix_size: int = 0,
        temperature: float | None = None,
        num_valids: int | None = None,
    ) -> torch.Tensor:
        if ignore_prefix_size > 0:
            decoder_output = decoder_output[..., ignore_prefix_size:, :]

            target_seqs = target_seqs[..., ignore_prefix_size:]

            if target_mask is not None:
                target_mask = target_mask[..., ignore_prefix_size:]

        if temperature is not None:
            decoder_output = decoder_output / temperature

        return cce_linear_cross_entropy_masked(
            e=decoder_output,
            c=self.final_proj.weight,
            targets=target_seqs,
            target_mask=target_mask,
            reduction=reduction,
            num_valids=num_valids,
        )

    return loss
