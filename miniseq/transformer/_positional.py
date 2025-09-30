from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing_extensions import override

from miniseq.builder_config import DataclassBuilderConfig
from miniseq.nn import Module


@dataclass(kw_only=True)
class PositionalEncoderConfig(DataclassBuilderConfig[Module]):
    @override
    def build(self, *, dim: int, max_seq_len: int, **kwd_overrides: Any) -> Module:
        fields = self.get_fields(**kwd_overrides)

        return self._target(dim=dim, max_seq_len=max_seq_len, **fields)


@dataclass(kw_only=True)
class RotaryEncoderConfig(PositionalEncoderConfig):
    _target: type[Module] = field(init=False, default_factory=lambda: RotaryEncoderReal)

    rope_theta: float = 10_000.0
    freqs_init_fn: Callable[[RotaryEncoderReal], torch.Tensor] | None = None


@dataclass(kw_only=True)
class RotatedRotaryEncoderConfig(PositionalEncoderConfig):
    _target: type[Module] = field(
        init=False, default_factory=lambda: RotatedRotaryEncoderReal
    )

    rope_theta: float = 10_000.0
    freqs_init_fn: Callable[[RotatedRotaryEncoderReal], torch.Tensor] | None = None


class RotaryEncoderReal(Module[torch.Tensor]):
    """Reference: https://arxiv.org/abs/2104.09864"""

    dim: int
    max_seq_len: int
    rope_theta: float
    freqs_init_fn: Callable[[RotaryEncoderReal], torch.Tensor] | None
    freqs: torch.Tensor

    def __init__(
        self,
        *,
        dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        freqs_init_fn: Callable[[RotaryEncoderReal], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.freqs_init_fn = freqs_init_fn

        self.register_buffer(
            "freqs",
            torch.empty(max_seq_len, dim // 2, 2, dtype=torch.float32),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        device = self.freqs.device
        freqs_cis = torch.view_as_complex(self.freqs)

        if self.freqs_init_fn is None:
            idx = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
            freqs = 1.0 / (self.rope_theta ** (idx / self.dim))
        else:
            freqs = self.freqs_init_fn(self)

        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, freqs)

        torch.polar(torch.ones_like(freqs), freqs, out=freqs_cis)

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x shape (bsz, seqlen, nh, head_dim)
        # input_pos shape (*, seqlen)

        seqlen = x.size(-3)

        if input_pos is None:
            # (seqlen, dim / 2, 2)
            freqs_cis = self.freqs[0:seqlen]
        else:
            # (*input_pos.size(), dim / 2, 2)
            freqs_cis = self.freqs[input_pos]

            # Unsqueeze on num_heads dimension if input_pos has a batch_size.
            # (bsz, seqlen, dim / 2, 2) -> (bsz, seqlen, 1, dim / 2, 2)
            if input_pos.ndim == 2:
                freqs_cis.unsqueeze_(2)

        # (bsz, seqlen, nh, head_dim) -> (bsz, seqlen, nh, head_dim / 2, 2)
        x_ = x.unflatten(-1, (-1, 2)).float()

        x_ = torch.stack(
            [
                x_[..., 0] * freqs_cis[..., 0] - x_[..., 1] * freqs_cis[..., 1],
                x_[..., 1] * freqs_cis[..., 0] + x_[..., 0] * freqs_cis[..., 1],
            ],
            dim=-1,
        )

        # (bsz, seqlen, nh, head_dim / 2, 2) -> (bsz, seqlen, nh, head_dim)
        x_ = x_.flatten(3)

        return x_.type_as(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, rope_theta={self.rope_theta}"


class RotatedRotaryEncoderReal(Module[torch.Tensor]):
    dim: int
    max_seq_len: int
    rope_theta: float
    freqs_init_fn: Callable[[RotatedRotaryEncoderReal], torch.Tensor] | None
    freqs: torch.Tensor

    def __init__(
        self,
        *,
        dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        freqs_init_fn: Callable[[RotatedRotaryEncoderReal], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.freqs_init_fn = freqs_init_fn

        self.register_buffer(
            "freqs",
            torch.empty(max_seq_len, dim * 2, dtype=torch.float32),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.reset_non_persistent_buffers()

    def reset_non_persistent_buffers(self) -> None:
        device = self.freqs.device

        if self.freqs_init_fn is None:
            idx = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
            freqs = 1.0 / (self.rope_theta ** (idx / self.dim))
        else:
            freqs = self.freqs_init_fn(self)

        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(positions, freqs)

        freqs = torch.cat([freqs, freqs], dim=-1)

        out_freqs = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

        self.freqs.copy_(out_freqs)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.size(-1) // 2]
        x2 = x[..., x.size(-1) // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x shape (bsz, seqlen, nh, head_dim)
        # input_pos shape (*, seqlen)

        seqlen, head_dim = x.size(-3), x.size(-1)

        if input_pos is None:
            # (seqlen, dim * 2)
            freqs_cis = self.freqs[0:seqlen]
        else:
            # (*input_pos.size(), dim * 2)
            freqs_cis = self.freqs[input_pos]

        # (*, seqlen, dim * 2) -> (*, seqlen, 1, dim * 2)
        freqs_cis = freqs_cis.view(-1, seqlen, 1, head_dim * 2)

        # (*, seqlen, 1, dim)
        cos = freqs_cis[..., :head_dim]

        # (*, seqlen, 1, dim)
        sin = freqs_cis[..., head_dim:]

        x_ = x.float()

        x_ = (x_ * cos) + (self._rotate_half(x_) * sin)

        return x_.type_as(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, rope_theta={self.rope_theta}"


class LearnedPositionalEncoder(Module[torch.Tensor]):
    dim: int
    max_seq_len: int
    weight: Parameter

    def __init__(self, *, dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.weight = Parameter(torch.empty(max_seq_len, dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x shape (bsz, seqlen, dim)
        # input_pos shape (*, seqlen)

        seqlen = x.size(-2)

        if input_pos is None:
            steps = torch.arange(0, seqlen, device=x.device, dtype=torch.int64)

            # (dim,)
            embed = F.embedding(steps, self.weight)
        else:
            # (*input_pos.size(), dim)
            embed = F.embedding(input_pos, self.weight)

        return x + embed

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}"


@dataclass
class LLaMARoPEScaleConfig:
    """
    Holds the frequency scaling configuration for the Rotary position encoder
    in LLaMA models.
    """

    factor: float = 8.0
    """
    The ratio between the intended maximum context length and the original
    maximum context length of the model.
    """

    frequency_factors: tuple[float, float] = (1.0, 4.0)
    """The factor used to define low and high frequencies."""

    original_context_length: int = 8192
    """The original context length. Defaults to LLaMA 3's context length."""


def llama_scale_freqs(
    pos_encoder: RotaryEncoderReal, rope_scale: LLaMARoPEScaleConfig
) -> torch.Tensor:
    """Reference: https://github.com/facebookresearch/fairseq2"""

    device = pos_encoder.freqs.device

    idx = torch.arange(0, pos_encoder.dim, step=2, device=device, dtype=torch.float32)

    freqs = 1.0 / (pos_encoder.rope_theta ** (idx / pos_encoder.dim))

    if device.type == "meta":
        return freqs

    old_context_len = rope_scale.original_context_length
    scale_factor = rope_scale.factor
    l_freq_factor, h_freq_factor = rope_scale.frequency_factors

    l_freq_wavelen = old_context_len / l_freq_factor
    h_freq_wavelen = old_context_len / h_freq_factor

    new_freqs = []
    for freq in freqs.tolist():
        wavelen = 2 * math.pi / freq

        if wavelen < h_freq_wavelen:
            new_freqs.append(freq)
            continue

        if wavelen > l_freq_wavelen:
            new_freqs.append(freq / scale_factor)
            continue

        smooth = (old_context_len / wavelen - l_freq_factor) / (h_freq_factor - l_freq_factor)  # fmt: skip
        new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=device)
