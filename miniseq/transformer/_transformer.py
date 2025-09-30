from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Protocol, overload

import torch
import torch.nn as nn
from typing_extensions import override

from miniseq.nn import Embedding, Linear, TiedProjectionLayer, cross_entropy_loss
from miniseq.transformer._attention_mask import AttentionMask
from miniseq.transformer._decoder import TransformerDecoder


class CausalTransformerModel(nn.Module, ABC):
    @property
    @abstractmethod
    def max_seq_len(self) -> int: ...

    @overload
    def forward(
        self,
        seqs: torch.Tensor,
        *,
        attn_mask: AttentionMask,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        seqs: torch.Tensor,
        target_seqs: torch.Tensor,
        *,
        attn_mask: AttentionMask,
        input_pos: torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "sum",
        target_mask: torch.Tensor | None = None,
        ignore_prefix_size: int = 0,
        temperature: float | None = None,
        num_valids: int | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def forward(
        self,
        seqs: torch.Tensor,
        target_seqs: torch.Tensor | None = None,
        *,
        attn_mask: AttentionMask,
        input_pos: torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "sum",
        target_mask: torch.Tensor | None = None,
        ignore_prefix_size: int = 0,
        temperature: float | None = None,
        num_valids: int | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def loss(
        self,
        decoder_output: torch.Tensor,
        target_seqs: torch.Tensor,
        *,
        reduction: Literal["mean", "sum", "none"] = "sum",
        target_mask: torch.Tensor | None = None,
        ignore_prefix_size: int = 0,
        temperature: float | None = None,
        num_valids: int | None = None,
    ) -> torch.Tensor: ...

    if TYPE_CHECKING:

        @overload
        def __call__(
            self,
            seqs: torch.Tensor,
            *,
            attn_mask: AttentionMask,
            input_pos: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Performs a forward pass and returns the logits.

            Args:
                seqs: The input sequences.
                input_pos: The input positions for positional embeddings. Defaults to None.
                attn_mask: The attention mask.

            Returns:
                The logits from the model.
            """

        @overload
        def __call__(
            self,
            seqs: torch.Tensor,
            target_seqs: torch.Tensor,
            *,
            attn_mask: AttentionMask,
            input_pos: torch.Tensor | None = None,
            reduction: Literal["mean", "sum", "none"] = "sum",
            target_mask: torch.Tensor | None = None,
            ignore_prefix_size: int = 0,
            temperature: float | None = None,
            num_valids: int | None = None,
        ) -> torch.Tensor:
            """Performs a forward pass and computes the loss.

            Args:
                seqs: The input sequences.
                target_seqs: The target sequences for loss calculation.
                input_pos: The input positions for positional embeddings. Defaults to None.
                attn_mask: The attention mask.
                reduction: The loss reduction mode. Defaults to "sum".
                target_mask: The mask for target sequences. Defaults to None.
                ignore_prefix_size: Number of prefix tokens to ignore in loss calculation. Defaults to 0.
                temperature: The temperature for logits before loss calculation. Defaults to None.
                num_valids: The number of valid tokens for loss normalization. Defaults to None.

            Returns:
                The computed loss.
            """

        def __call__(
            self,
            seqs: torch.Tensor,
            target_seqs: torch.Tensor | None = None,
            *,
            attn_mask: AttentionMask,
            input_pos: torch.Tensor | None = None,
            reduction: Literal["mean", "sum", "none"] = "sum",
            target_mask: torch.Tensor | None = None,
            ignore_prefix_size: int = 0,
            temperature: float | None = None,
            num_valids: int | None = None,
        ) -> torch.Tensor: ...


class LossFunction(Protocol):
    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: Literal["none", "mean", "sum"] = "sum",
        loss_mask: torch.Tensor | None = None,
        pad_idx: int | None = None,
    ) -> torch.Tensor: ...


class TransformerDecoderModel(CausalTransformerModel):
    decoder_frontend: Embedding
    decoder: TransformerDecoder
    final_proj: Linear | TiedProjectionLayer
    _max_seq_len: int
    _pad_idx: int | None
    _loss_function: LossFunction

    def __init__(
        self,
        decoder_frontend: Embedding,
        decoder: TransformerDecoder,
        final_proj: Linear | TiedProjectionLayer,
        max_seq_len: int,
        pad_idx: int | None = None,
        loss_function: LossFunction = cross_entropy_loss,
    ) -> None:
        super().__init__()

        self.decoder_frontend = decoder_frontend

        self.decoder = decoder

        self.final_proj = final_proj

        self._max_seq_len = max_seq_len

        self._pad_idx = pad_idx

        self._loss_function = loss_function

    @property
    @override
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def decode(
        self,
        seqs: torch.Tensor,
        *,
        attn_mask: AttentionMask,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seqs = self.decoder_frontend(seqs, input_pos=input_pos)

        decoder_output = self.decoder(seqs, input_pos=input_pos, attn_mask=attn_mask)

        del seqs

        return decoder_output

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_proj(x)

    @override
    def forward(
        self,
        seqs: torch.Tensor,
        target_seqs: torch.Tensor | None = None,
        *,
        attn_mask: AttentionMask,
        input_pos: torch.Tensor | None = None,
        reduction: Literal["mean", "sum", "none"] = "sum",
        target_mask: torch.Tensor | None = None,
        ignore_prefix_size: int = 0,
        temperature: float | None = None,
        num_valids: int | None = None,
    ) -> torch.Tensor:
        if attn_mask.max_input_pos >= self.max_seq_len:
            raise ValueError(
                f"Input seq length {attn_mask.max_input_pos} >= model seq length {self.max_seq_len}"
            )

        decoder_output = self.decode(seqs, input_pos=input_pos, attn_mask=attn_mask)

        if target_seqs is None:
            return self.project(decoder_output)

        return self.loss(
            decoder_output,
            target_seqs,
            reduction=reduction,
            target_mask=target_mask,
            ignore_prefix_size=ignore_prefix_size,
            temperature=temperature,
            num_valids=num_valids,
        )

    @override
    def loss(
        self,
        decoder_output: torch.Tensor,
        target_seqs: torch.Tensor,
        *,
        reduction: Literal["mean", "sum", "none"] = "sum",
        target_mask: torch.Tensor | None = None,
        ignore_prefix_size: int = 0,
        temperature: float | None = None,
        num_valids: int | None = None,
    ) -> torch.Tensor:
        logits = self.project(decoder_output)

        del decoder_output

        if ignore_prefix_size > 0:
            logits = logits[..., ignore_prefix_size:, :]

            target_seqs = target_seqs[..., ignore_prefix_size:]

            if target_mask is not None:
                target_mask = target_mask[..., ignore_prefix_size:]

        if temperature is not None:
            logits = logits / temperature

        loss = self._loss_function(
            logits=logits,
            targets=target_seqs,
            reduction=reduction,
            loss_mask=target_mask,
        )

        del logits

        return loss
