from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.modules.feature_map import (
    DPFPFeatureMap,
    HadamardFeatureMap,
    HedgehogFeatureMap,
    T2RFeatureMap,
)
from fla.ops.gla import chunk_gla
from fla.ops.linear_attn import chunk_linear_attn
from typing_extensions import override

from miniseq import cli
from miniseq.builder_config import BuilderConfig, BuilderProtocol
from miniseq.nn import Linear, Module
from miniseq.transformer import AttentionConfig, AttentionLayer, AttentionMask


# Mixer classes for CLI entrypoint
@cli.make_union_registry("sequence_mixer")
class SequenceMixerConfig(BuilderProtocol[AttentionLayer], Protocol):
    model_dim: int


@cli.make_union_registry("state_mixer")
class StateMixerConfig(BuilderProtocol[Module[torch.Tensor]], Protocol):
    model_dim: int


# Register existing MHAttention implementation
cli.union_struct_choice("sequence_mixer", command="attention")(AttentionConfig)


FeatureMapName: TypeAlias = Literal[
    "hedgehog",
    "t2r",
    "elementwise_product",
    "dpfp",
    "elu",
    "relu",
    "identity",
]

FeatureMap: TypeAlias = (
    Module[torch.Tensor, [torch.Tensor]] | Callable[[torch.Tensor], torch.Tensor]
)


def make_feature_map(name: FeatureMapName, head_dim: int) -> FeatureMap | None:
    match name:
        case "hedgehog":
            return HedgehogFeatureMap(head_dim)
        case "t2r":
            return T2RFeatureMap(head_dim, bias=False)
        case "elementwise_product":
            return HadamardFeatureMap(head_dim)
        case "dpfp":
            return DPFPFeatureMap(head_dim)
        case "elu":

            def elu(x: torch.Tensor) -> torch.Tensor:
                return F.elu(x) + 1

            return elu

        case "relu":
            return nn.ReLU()

        case "identity":
            return None

        case _:
            raise ValueError(f"Unrecognized feature map: {name}.")


def init_linear(module: nn.Linear) -> None:
    nn.init.xavier_uniform_(module.weight, gain=2**-2.5)

    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_ffn_linear(module: nn.Linear) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=0.02)

    if module.bias is not None:
        nn.init.zeros_(module.bias)


@cli.union_struct_choice("sequence_mixer", command="linear_attention")
@dataclass(kw_only=True)
class LinearAttnConfig(BuilderConfig[AttentionLayer]):
    _target: type = field(default_factory=lambda: LinearAttention)

    model_dim: int = field(init=False, default=128)
    num_heads: int = 8
    feature_map: FeatureMapName = "elementwise_product"
    expand_k: float = 1.0
    expand_v: float = 1.0
    tie_feature_map_qk: bool = False
    output_norm: str = "rmsnorm"
    norm_q: bool = False
    norm_k: bool = False
    do_feature_map_norm: bool = False
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        assert int(self.model_dim * self.expand_k) % self.num_heads == 0
        assert int(self.model_dim * self.expand_v) % self.num_heads == 0
        assert self.output_norm in ("rmsnorm", "identity")
        assert self.feature_map in (
            "hedgehog",
            "t2r",
            "elementwise_product",
            "dpfp",
            "elu",
            "relu",
            "identity",
        )


class LinearAttention(AttentionLayer):
    config: LinearAttnConfig
    feature_map_q: FeatureMap | None
    feature_map_k: FeatureMap | None
    out_norm: nn.Module | None

    def __init__(self, config: LinearAttnConfig) -> None:
        super().__init__()

        self.config = config

        self.key_dim = int(config.model_dim * config.expand_k)
        self.value_dim = int(config.model_dim * config.expand_v)
        self.head_qk_dim = self.key_dim // config.num_heads
        self.head_v_dim = self.value_dim // config.num_heads

        self.feature_map_q = make_feature_map(config.feature_map, self.head_qk_dim)

        if config.tie_feature_map_qk:
            self.feature_map_k = self.feature_map_q
        else:
            self.feature_map_k = make_feature_map(config.feature_map, self.head_qk_dim)

        self.q_proj = Linear(
            config.model_dim, self.key_dim, bias=False, init_fn=init_linear
        )
        self.k_proj = Linear(
            config.model_dim, self.key_dim, bias=False, init_fn=init_linear
        )
        self.v_proj = Linear(
            config.model_dim, self.value_dim, bias=False, init_fn=init_linear
        )
        self.o_proj = Linear(
            self.value_dim, config.model_dim, bias=False, init_fn=init_linear
        )

        if config.output_norm == "rmsnorm":
            self.out_norm = RMSNorm(hidden_size=self.head_v_dim, eps=config.norm_eps)
        else:
            self.out_norm = None

    @override
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: AttentionMask | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = q.unflatten(-1, (self.num_heads, -1))
        k = k.unflatten(-1, (self.num_heads, -1))
        v = v.unflatten(-1, (self.num_heads, -1))

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.feature_map_q is not None:
            q = self.feature_map_q(q)

        if self.feature_map_k is not None:
            k = self.feature_map_k(k)

        if self.norm_q:
            q = q / (q.sum(-1, keepdim=True) + 1e-4)

        if self.norm_k:
            k = k / (k.sum(-1, keepdim=True) + 1e-4)

        output, _ = chunk_linear_attn(
            q, k, v, normalize=self.config.do_feature_map_norm
        )

        if self.out_norm is not None:
            output = self.out_norm(output)

        output = output.transpose(1, 2).flatten(2, 3)

        output = self.o_proj(output)

        return output


@cli.union_struct_choice("sequence_mixer", command="gated_linear_attn")
@dataclass(kw_only=True)
class GatedLinearAttnConfig(BuilderConfig[AttentionLayer], SequenceMixerConfig):
    _target: type = field(default_factory=lambda: GatedLinearAttention)

    model_dim: int = field(init=False, default=128)

    expand_v: float = 0.5
    expand_k: float = 1.0
    num_heads: int = 4
    gate_fn: str = "swish"
    norm_eps: float = 1e-5
    gate_logit_normalizer: int = 16
    gate_low_rank_dim: int = 16
    clamp_min: float | None = None
    fuse_norm: bool = True

    def __post_init__(self) -> None:
        assert int(self.model_dim * self.expand_k) % self.num_heads == 0
        assert int(self.model_dim * self.expand_v) % self.num_heads == 0


class GatedLinearAttention(AttentionLayer):
    config: GatedLinearAttnConfig

    def __init__(self, config: GatedLinearAttnConfig) -> None:
        super().__init__()

        # TODO: Allow for more options
        self.gate_fn = F.silu

        self.key_dim = int(config.model_dim * config.expand_k)
        self.value_dim = int(config.model_dim * config.expand_v)
        self.head_qk_dim = self.key_dim // config.num_heads
        self.head_v_dim = self.value_dim // config.num_heads

        self.q_proj = Linear(
            config.model_dim, self.key_dim, bias=False, init_fn=init_linear
        )
        self.k_proj = Linear(
            config.model_dim, self.key_dim, bias=False, init_fn=init_linear
        )
        self.v_proj = Linear(
            config.model_dim, self.value_dim, bias=False, init_fn=init_linear
        )
        self.gk_proj = nn.Sequential(
            Linear(
                config.model_dim,
                config.gate_low_rank_dim,
                bias=False,
                init_fn=init_linear,
            ),
            Linear(
                config.gate_low_rank_dim, self.key_dim, bias=True, init_fn=init_linear
            ),
        )
        self.g_proj = Linear(
            config.model_dim, self.value_dim, bias=False, init_fn=init_linear
        )
        self.o_proj = Linear(
            self.value_dim, config.model_dim, bias=False, init_fn=init_linear
        )

        self.fuse_norm_and_gate = config.gate_fn == "swish" and config.fuse_norm

        if self.fuse_norm_and_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(
                self.head_v_dim, eps=config.norm_eps
            )
        else:
            self.g_norm = RMSNorm(self.head_v_dim, eps=config.norm_eps)

    @override
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: AttentionMask | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, k, v, gk = self.q_proj(x), self.k_proj(x), self.v_proj(x), self.gk_proj(x)

        q = q.unflatten(-1, (self.num_heads, -1))
        k = k.unflatten(-1, (self.num_heads, -1))
        v = v.unflatten(-1, (self.num_heads, -1))
        gk = gk.unflatten(-1, (self.num_heads, -1))

        gk = F.logsigmoid(gk) / self.config.gate_logit_normalizer

        if self.config.clamp_min is not None:
            gk = torch.clamp_min(gk, self.config.clamp_min)

        # (bsz, seqlen, num_h, head_dim)
        output, _ = chunk_gla(q, k, v, gk)

        g = self.g_proj(x)

        if self.fuse_norm_and_gate:
            g = g.unflatten(-1, (self.num_heads, -1))

            output = self.g_norm_swish_gate(output, g)

            output = output.flatten(2, 3)
        else:
            output = self.g_norm(output)

            output = output.flatten(2, 3)

            output = output * self.gate_fn(g)

        output = self.o_proj(output)

        return output


@cli.union_struct_choice("state_mixer", command="mlp")
@dataclass(kw_only=True)
class MlpConfig(BuilderConfig[Module[torch.Tensor]]):
    _target: type = field(default_factory=lambda: Mlp)

    model_dim: int = field(init=False, default=128)
    dim_inner: int | None = None
    drop_rate: float = 0.0
    act: Callable[[torch.Tensor], torch.Tensor] = nn.GELU(approximate="tanh")
    bias: bool = True


class Mlp(nn.Module):
    config: MlpConfig
    drop1: nn.Dropout | None
    drop2: nn.Dropout | None

    def __init__(self, config: MlpConfig) -> None:
        super().__init__()
        self.config = config

        dim = config.model_dim

        dim_inner = dim * 4 if config.dim_inner is None else config.dim_inner
        self.fc = Linear(dim, dim_inner, bias=config.bias, init_fn=init_ffn_linear)
        self.proj = Linear(dim_inner, dim, bias=config.bias, init_fn=init_ffn_linear)
        self.act = config.act

        if config.drop_rate > 0:
            self.drop1 = nn.Dropout(config.drop_rate)
            self.drop2 = nn.Dropout(config.drop_rate)
        else:
            self.register_module("drop1", None)
            self.register_module("drop2", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.act(x)

        if self.drop1 is not None:
            x = self.drop1(x)

        x = self.proj(x)

        if self.drop2 is not None:
            x = self.drop2(x)
        return x


@cli.union_struct_choice("state_mixer", command="glu")
@dataclass(kw_only=True)
class GLUCfg(BuilderConfig[Module[torch.Tensor]]):
    _target: type = field(default_factory=lambda: GLU)

    model_dim: int = field(init=False, default=128)
    drop_rate: float = 0.0
    act: Callable[[torch.Tensor], torch.Tensor] = nn.Sigmoid()
    bias: bool = False
    multiple_of: int = 16


@cli.union_struct_choice("state_mixer", command="swiglu")
@dataclass(kw_only=True)
class SwiGLUCfg(GLUCfg):
    _target: type = field(default_factory=lambda: GLU)

    act: Callable[[torch.Tensor], torch.Tensor] = nn.SiLU()


class GLU(nn.Module):
    config: GLUCfg

    def __init__(self, config: GLUCfg) -> None:
        super().__init__()

        self.config = config

        dim = config.model_dim
        multiple_of = config.multiple_of

        self.act = config.act

        dim_inner = int(2 * dim * 4 / 3)
        dim_inner = multiple_of * ((dim_inner + multiple_of - 1) // multiple_of)

        self.w1 = Linear(dim, dim_inner, bias=config.bias, init_fn=init_ffn_linear)
        self.w2 = Linear(dim, dim_inner, bias=config.bias, init_fn=init_ffn_linear)
        self.w3 = Linear(dim_inner, dim, bias=config.bias, init_fn=init_ffn_linear)

        drop_rate = config.drop_rate

        self.drop1 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop2 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.drop3 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1(x)
        w2_out = self.w2(x)
        return self.drop3(self.w3(self.drop1(self.act(w1_out)) * self.drop2(w2_out)))
