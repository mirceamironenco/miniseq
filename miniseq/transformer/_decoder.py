from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniseq.builder_config import BuilderConfig
from miniseq.nn import Linear, Module, RMSNorm
from miniseq.transformer._attention import AttentionLayer
from miniseq.transformer._attention_mask import AttentionMask


class TransformerNormOrder(Enum):
    PRE = 0
    POST = 1


class FeedForward(Module[torch.Tensor, [torch.Tensor]]):
    config: FFNConfig

    def __init__(self, config: FFNConfig) -> None:
        super().__init__()
        self.config = config
        inner_dim = config.inner_dim

        if inner_dim is None:
            inner_dim = 4 * config.model_dim

        if config.ffn_dim_multiplier != 1.0:
            inner_dim = int(inner_dim * config.ffn_dim_multiplier)

        multiple_of = config.multiple_of

        if multiple_of != 1:
            inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(config.model_dim, inner_dim, bias=config.bias)
        self.w3 = Linear(config.model_dim, inner_dim, bias=config.bias)
        self.w2 = Linear(inner_dim, config.model_dim, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def extra_repr(self) -> str:
        return f"multiple_of={self.config.multiple_of}, ffn_dim_multiplier={self.config.ffn_dim_multiplier:.2f}"


@dataclass(kw_only=True)
class FFNConfig(BuilderConfig[FeedForward]):
    _target: type = field(default_factory=lambda: FeedForward)

    model_dim: int = 4096
    inner_dim: int | None = None
    multiple_of: int = 4096
    ffn_dim_multiplier: float = 2 / 3
    bias: bool = False


class TransformerDecoderLayer(Module[torch.Tensor]):
    self_attn: AttentionLayer
    ffn: Module[torch.Tensor]
    attn_norm: RMSNorm
    ffn_norm: RMSNorm

    def __init__(
        self,
        attn: AttentionLayer,
        ffn: Module[torch.Tensor],
        model_dim: int,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.self_attn = attn
        self.ffn = ffn
        self.model_dim = model_dim
        self.norm_order = norm_order
        self.norm_eps = norm_eps

        self.attn_norm = RMSNorm(dim=self.model_dim, eps=norm_eps)

        self.ffn_norm = RMSNorm(dim=self.model_dim, eps=norm_eps)

    def _forward_attention(
        self,
        x: torch.Tensor,
        attn_mask: AttentionMask | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x

        if self.norm_order == TransformerNormOrder.PRE:
            x = self.attn_norm(x)

        x = self.self_attn(x, attn_mask=attn_mask, input_pos=input_pos)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = self.attn_norm(x)

        return x

    def _forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.norm_order == TransformerNormOrder.PRE:
            x = self.ffn_norm(x)

        x = self.ffn(x)

        x = x + residual

        if self.norm_order == TransformerNormOrder.POST:
            x = self.ffn_norm(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: AttentionMask | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._forward_attention(x, attn_mask=attn_mask, input_pos=input_pos)

        x = self._forward_ffn(x)

        return x


class TransformerDecoder(Module[torch.Tensor]):
    layers: nn.ModuleList
    norm_order: TransformerNormOrder
    norm: RMSNorm | None

    def __init__(
        self,
        layers: Sequence[TransformerDecoderLayer],
        model_dim: int,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert len(layers) > 0

        self.layers = nn.ModuleList(layers)
        self.norm_order = norm_order

        if norm_order != TransformerNormOrder.POST:
            self.norm = RMSNorm(model_dim, eps=norm_eps)
        else:
            self.register_module("norm", None)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: AttentionMask | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, input_pos=input_pos)

        if self.norm is not None:
            x = self.norm(x)

        return x
