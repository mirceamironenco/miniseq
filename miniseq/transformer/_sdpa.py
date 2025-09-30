from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from typing_extensions import override

try:
    from flash_attn import (  # type: ignore
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    _has_flash_attn_2 = False
else:
    _has_flash_attn_2 = True

from miniseq.transformer._attention_mask import AttentionMask
from miniseq.utils import replace_method_signature_with


class SDPA(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: AttentionMask | None = None,
    ) -> torch.Tensor: ...

    if TYPE_CHECKING:

        @replace_method_signature_with(forward)
        def __call__(self, *args, **kwargs) -> None: ...


class TorchSDPA(SDPA):
    attn_scale: float | None
    _dropout_p: float
    _sdpa_backends: list[SDPBackend]  # type: ignore

    def __init__(
        self, attn_scale: float | None = None, dropout_p: float | None = None
    ) -> None:
        super().__init__()
        self._attn_scale = attn_scale
        self._dropout_p = dropout_p if dropout_p is not None else 0.0

        self._sdpa_backends = [SDPBackend.MATH]

        if torch.backends.cuda.mem_efficient_sdp_enabled():
            self._sdpa_backends.insert(0, SDPBackend.EFFICIENT_ATTENTION)

        if torch.backends.cuda.flash_sdp_enabled():
            self._sdpa_backends.insert(0, SDPBackend.FLASH_ATTENTION)

    @override
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: AttentionMask | None = None,
    ) -> torch.Tensor:
        dropout_p = 0.0 if not self.training else self._dropout_p

        is_causal, mask = True, None

        if attn_mask is not None:
            if (
                attn_mask.padded
                or attn_mask.packed
                or attn_mask.window_size is not None
            ):
                is_causal = False

                mask = attn_mask.materialize_dense()

        # (bsz, seqlen, num_*_heads, head_dim) -> (bsz, num_*_heads, seqlen, head_dim)
        q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        with sdpa_kernel(self._sdpa_backends, set_priority=True):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self._attn_scale,
            )

        # (bsz, num_heads, seqlen, head_dim) -> (bsz, seqlen, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)

        return attn_output

    @override
    def extra_repr(self) -> str:
        return f"scale={self._attn_scale:.2f}, dropout={self._dropout_p}, sdpa_kernel={self._sdpa_backends[0]}"


class Flash2SDPA(SDPA):
    _dropout_p: float

    def __init__(self, dropout_p: float | None = None) -> None:
        super().__init__()

        self._dropout_p = dropout_p if dropout_p is not None else 0.0

    @override
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: AttentionMask | None = None,
    ) -> torch.Tensor:
        dropout_p = 0.0 if not self.training else self._dropout_p

        window_size = (-1, -1)

        if attn_mask is not None and attn_mask.window_size is not None:
            window_size = (attn_mask.window_size, -1)

        if attn_mask is not None and attn_mask.packed:
            assert attn_mask.cu_seqlens is not None
            assert query.size(0) == 1

            seqlen = key.size(1)

            # Where padding starts.
            total_seq_len = attn_mask.total_seq_len

            # (1, packed_seqlen, num_*_heads, dim) -> (packed_seqlen, num_*_heads, dim)
            query.squeeze_(0)
            key.squeeze_(0)
            value.squeeze_(0)

            if total_seq_len < seqlen:
                query = query[:total_seq_len, ...]
                key = key[:total_seq_len, ...]
                value = value[:total_seq_len, ...]

            if _has_flash_attn_2:
                flash_output = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    attn_mask.cu_seqlens,
                    attn_mask.cu_seqlens,
                    attn_mask.max_seq_len,
                    attn_mask.max_seq_len,
                    dropout_p,
                    causal=True,
                    window_size=window_size,
                )
            else:
                flash_output, _, _, _, _ = torch.ops.aten._flash_attention_forward(
                    query,
                    key,
                    value,
                    attn_mask.cu_seqlens,
                    attn_mask.cu_seqlens,
                    attn_mask.max_seq_len,
                    attn_mask.max_seq_len,
                    dropout_p,
                    is_causal=True,
                    return_debug_mask=False,
                )

            flash_output = cast(torch.Tensor, flash_output)

            attn_output = torch.zeros(
                (1, seqlen, *flash_output.shape[-2:]),
                device=flash_output.device,
                dtype=flash_output.dtype,
            )

            if total_seq_len < seqlen:
                attn_output[0, :total_seq_len, ...] = flash_output
            else:
                attn_output[0, ...] = flash_output

        else:
            if _has_flash_attn_2:
                attn_output = flash_attn_func(
                    query, key, value, dropout_p, causal=True, window_size=window_size
                )
            else:
                attn_output, _, _, _, _ = torch.ops.aten._flash_attention_forward(
                    query,
                    key,
                    value,
                    None,
                    None,
                    attn_mask.max_seq_len if attn_mask is not None else query.size(1),
                    attn_mask.max_seq_len if attn_mask is not None else key.size(1),
                    dropout_p,
                    is_causal=True,
                    return_debug_mask=False,
                )

        return attn_output

    @override
    def extra_repr(self) -> str:
        return f"dropout={self._dropout_p}"


class NaiveSDPA(SDPA):
    _attn_scale: float | None
    _dropout_p: float
    _soft_cap: float | None

    def __init__(
        self,
        attn_scale: float | None = None,
        dropout_p: float | None = None,
        soft_cap: float | None = None,
    ) -> None:
        super().__init__()
        self._attn_scale = attn_scale
        self._dropout_p = dropout_p if dropout_p is not None else 0.0
        self._soft_cap = soft_cap

    @override
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: AttentionMask | None = None,
    ) -> torch.Tensor:
        # (bsz, seqlen, num_*_heads, head_dim) -> (bsz, num_*_heads, seqlen, head_dim)
        query, key, value = (
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )

        attn_scale = (
            (query.size(-1) ** -0.5) if self._attn_scale is None else self._attn_scale
        )

        scores = torch.matmul(query, key.transpose(-1, -2)) * attn_scale

        if self._soft_cap is not None:
            scores = self._soft_cap * torch.tanh(scores / self._soft_cap)

        if attn_mask is not None:
            mask = attn_mask.materialize_bias(seqs=query)
            scores = scores + mask

        # More stable in float32
        scores = F.softmax(scores, dim=-1, dtype=torch.float32)
        scores = scores.type_as(query)

        if self.training and self._dropout_p > 0.0:
            scores = F.dropout(scores, p=self._dropout_p)

        output = torch.matmul(scores, value)

        # (bsz, num_heads, seqlen, head_dim) -> (bsz, seqlen, num_heads, head_dim)
        output = output.transpose(1, 2)

        return output

    @override
    def extra_repr(self) -> str:
        return f"scale={self._attn_scale:.2f}, dropout={self._dropout_p}, softcap={self._soft_cap}"
