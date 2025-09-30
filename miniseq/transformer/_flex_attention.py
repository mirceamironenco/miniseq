from typing import Callable, NotRequired

import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import (
    flex_attention as _flex_attention,
)
from typing_extensions import TypedDict, override

from miniseq.transformer._attention_mask import AttentionMask, ScoreMod
from miniseq.transformer._sdpa import SDPA
from miniseq.utils import TorchCompileMode, should_compile_flex

_flex_attention_compiled = _flex_attention


class FlexCompileOptions(TypedDict):
    BLOCK_M: NotRequired[int]
    BLOCK_N: NotRequired[int]
    num_stages: NotRequired[int]
    num_warps: NotRequired[int]

    BLOCK_M1: NotRequired[int]
    BLOCK_N1: NotRequired[int]
    BLOCK_M2: NotRequired[int]
    BLOCK_N2: NotRequired[int]


def make_tanh_softcap_score(soft_cap: float) -> ScoreMod:
    def softcap_score_mod(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        return soft_cap * torch.tanh(score / soft_cap)

    return softcap_score_mod


class FlexSDPA(SDPA):
    _soft_cap: float | None
    _score_mod: Callable | None
    _flex_compiled: bool = False
    _kernel_options: FlexCompileOptions | None

    def __init__(
        self,
        *,
        attn_scale: float | None = None,
        soft_cap: float | None = None,
        reduce_stages: bool = False,
    ) -> None:
        super().__init__()
        assert _flex_attention is not None
        self._attn_scale = attn_scale
        self._soft_cap = soft_cap

        if soft_cap is not None:
            self._score_mod = make_tanh_softcap_score(soft_cap=soft_cap)
        else:
            self._score_mod = None

        if not FlexSDPA.is_flex_compiled():
            raise ValueError(
                "Flex attention is not compiled; use FlexSDPA.compile_flex_attention."
            )

        self._kernel_options = None

        if reduce_stages:
            self._kernel_options = {"num_stages": 2}

    @override
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: AttentionMask | None = None,
    ) -> torch.Tensor:
        if attn_mask is not None:
            mask = attn_mask.materialize_block()
        else:
            mask = None

        # (bsz, seqlen, num_*_heads, head_dim) -> (bsz, num_*_heads, seqlen, head_dim)
        q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        attn_output = self._apply_flex_attention(
            q,
            k,
            v,
            score_mod=self._score_mod,
            block_mask=mask,
            scale=self._attn_scale,
            kernel_options=self._kernel_options,
        )

        # (bsz, num_heads, seqlen, head_dim) -> (bsz, seqlen, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)

        return attn_output

    @staticmethod
    def _apply_flex_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: ScoreMod | None = None,
        block_mask: BlockMask | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        kernel_options: FlexCompileOptions | None = None,
    ) -> torch.Tensor:
        return _flex_attention_compiled(
            query,
            key,
            value,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=False,
            kernel_options=kernel_options,  # type: ignore
        )  # type: ignore

    @classmethod
    def compile_flex_attention(
        cls,
        fullgraph: bool = False,
        dynamic: bool | None = None,
        mode: TorchCompileMode | None = "default",
        options: dict[str, str | int | bool | Callable] | None = None,
    ) -> None:
        if not should_compile_flex():
            return

        if cls._flex_compiled:
            # Flex Attention already compiled, avoiding recompilation.
            return

        global _flex_attention_compiled
        _flex_attention_compiled = torch.compile(
            _flex_attention,
            fullgraph=fullgraph,
            dynamic=dynamic,
            mode=mode,
            options=options,
        )

        cls._flex_compiled = True

    @classmethod
    def is_flex_compiled(cls) -> bool:
        return cls._flex_compiled

    def extra_repr(self) -> str:
        return f"scale={self._attn_scale:.2f}, softcap={self._soft_cap}, kernel_options: {self._kernel_options}"
