from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from typing_extensions import override

from miniseq.builder_config import BuilderConfig
from miniseq.nn import Linear, Module, RMSNorm, repeat_interleave
from miniseq.transformer._attention_mask import AttentionMask
from miniseq.transformer._flex_attention import FlexSDPA
from miniseq.transformer._sdpa import SDPA, Flash2SDPA, NaiveSDPA, TorchSDPA
from miniseq.utils import replace_method_signature_with


class AttentionLayer(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: AttentionMask | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    if TYPE_CHECKING:

        @replace_method_signature_with(forward)
        def __call__(self, *args, **kwargs) -> None: ...


class MHAttention(AttentionLayer):
    config: AttentionConfig
    pos_encoder: Module | None
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    sdpa: SDPA
    head_dim: int
    _attn_scale: float
    q_norm: RMSNorm | None
    k_norm: RMSNorm | None

    def __init__(
        self,
        config: AttentionConfig,
        *,
        sdpa: SDPA | None = None,
        pos_encoder: Module | None = None,
    ) -> None:
        super().__init__()

        if not config.num_heads % config.num_kv_heads == 0:
            raise ValueError(
                f"num_heads ({config.num_heads}) % num_kv_heads ({config.num_kv_heads}) != 0"
            )

        if not config.num_kv_heads <= config.num_heads:
            raise ValueError(
                f"config.num_kv_heads ({config.num_kv_heads}) > config.num_heads ({config.num_heads})"
            )

        self.config = config

        head_dim = self.config.head_dim

        if head_dim is None:
            assert config.model_dim % config.num_heads == 0

            head_dim = config.model_dim // config.num_heads

        self.head_dim = head_dim
        self.model_dim = config.model_dim
        self.num_heads, self.num_kv_heads = config.num_heads, config.num_kv_heads
        self.query_per_kv = config.num_heads // config.num_kv_heads
        self.query_dim = config.num_heads * self.head_dim
        self.kv_dim = config.num_kv_heads * self.head_dim

        self.pos_encoder = pos_encoder

        self.q_proj = Linear(
            config.model_dim, self.query_dim, bias=config.bias, init_fn=init_qkv
        )

        self.k_proj = Linear(
            config.model_dim, self.kv_dim, bias=config.bias, init_fn=init_qkv
        )

        self.v_proj = Linear(
            config.model_dim, self.kv_dim, bias=config.bias, init_fn=init_qkv
        )

        self.o_proj = Linear(
            self.query_dim,
            config.model_dim,
            bias=config.output_bias,
            init_fn=init_outproj,
        )

        scale_factor = config.scale_factor or self.head_dim

        self._attn_scale = scale_factor**-0.5

        if config.qk_norm:
            self.q_norm = RMSNorm(dim=self.head_dim, eps=config.qk_norm_eps)
            self.k_norm = RMSNorm(dim=self.head_dim, eps=config.qk_norm_eps)
        else:
            self.register_module("q_norm", None)
            self.register_module("k_norm", None)

        if sdpa is not None:
            self.sdpa = sdpa
        else:
            self.sdpa = config.build_default_sdpa(
                attn_scale=self._attn_scale,
                dropout_p=config.dropout_p,
                soft_cap=config.soft_cap,
                force_flex=config.force_flex,
                force_flash2=config.force_flash2,
                reduce_flex_stages=config.reduce_flex_stages,
            )

    @override
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: AttentionMask | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # (bsz, seqlen, num_heads, head_dim)
        q = q.unflatten(-1, (self.num_heads, -1))

        # (bsz, seqlen, num_kv_heads, head_dim)
        k = k.unflatten(-1, (self.num_kv_heads, -1))
        v = v.unflatten(-1, (self.num_kv_heads, -1))

        if self.q_norm is not None:
            q = self.q_norm(q)

        if self.k_norm is not None:
            k = self.k_norm(k)

        if self.pos_encoder is not None:
            q = self.pos_encoder(q, input_pos=input_pos)
            k = self.pos_encoder(k, input_pos=input_pos)

        if self.query_per_kv > 1:
            k = repeat_interleave(k, dim=2, repeat=self.query_per_kv)
            v = repeat_interleave(v, dim=2, repeat=self.query_per_kv)

        # (bsz, seqlen, num_heads, head_dim)
        out = self.sdpa(query=q, key=k, value=v, attn_mask=attn_mask)

        del q, k, v

        out = out.flatten(2, 3)

        out = self.o_proj(out)

        return out

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}"


def init_qkv(projection: nn.Linear) -> None:
    nn.init.trunc_normal_(projection.weight, mean=0.0, std=0.02)


def init_outproj(projection: nn.Linear) -> None:
    nn.init.trunc_normal_(projection.weight, mean=0.0, std=0.02)


@dataclass(kw_only=True)
class AttentionConfig(BuilderConfig[MHAttention]):
    _target: type[MHAttention] = field(default_factory=lambda: MHAttention, init=False)

    model_dim: int = 4096

    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int | None = None
    bias: bool = False
    output_bias: bool = False
    qk_norm: bool = False
    qk_norm_eps: float = 1e-6
    scale_factor: int | None = None
    soft_cap: float | None = None
    dropout_p: float | None = None

    force_flex: bool = False
    force_flash2: bool = False
    reduce_flex_stages: bool = False

    @classmethod
    def build_default_sdpa(
        cls,
        *,
        attn_scale: float,
        dropout_p: float | None,
        soft_cap: float | None,
        force_flex: bool = False,
        force_flash2: bool = False,
        reduce_flex_stages: bool = False,
    ) -> SDPA:
        # FlexAttention doesn't support dropout.
        can_use_flex = not (dropout_p is not None and dropout_p > 0.0)

        if force_flex:
            if not can_use_flex:
                raise ValueError(
                    "Can't force flex_attention SDPA, dropout not supported."
                )

            return FlexSDPA(
                attn_scale=attn_scale,
                soft_cap=soft_cap,
                reduce_stages=reduce_flex_stages,
            )

        if soft_cap is not None:
            # F.sdpa does not support softcapping.
            if can_use_flex:
                return FlexSDPA(
                    attn_scale=attn_scale,
                    soft_cap=soft_cap,
                    reduce_stages=reduce_flex_stages,
                )
            else:
                return NaiveSDPA(
                    attn_scale=attn_scale, dropout_p=dropout_p, soft_cap=soft_cap
                )

        if force_flash2:
            return Flash2SDPA(dropout_p=dropout_p)

        return TorchSDPA(attn_scale=attn_scale, dropout_p=dropout_p)

    @override
    def build(
        self,
        *,
        sdpa: SDPA | None = None,
        pos_encoder: Module | None = None,
        **kwd_overrides: Any,
    ) -> MHAttention:
        assert not kwd_overrides

        return self._target(self, sdpa=sdpa, pos_encoder=pos_encoder)
