import functools
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from layers import MlpConfig, SequenceMixerConfig, StateMixerConfig

from miniseq import cli
from miniseq.nn import Embedding, EmbeddingProjection, Linear, cross_entropy_loss
from miniseq.transformer import (
    AttentionConfig,
    LearnedPositionalEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderModel,
    TransformerNormOrder,
)
from miniseq.utils import on_local_rank_zero


@dataclass(kw_only=True)
class ModelConfig:
    seq_mixer: SequenceMixerConfig = field(
        default_factory=lambda: AttentionConfig(num_heads=8, num_kv_heads=8)
    )
    state_mixer: StateMixerConfig = field(default_factory=lambda: MlpConfig())
    model_dim: int = 128
    num_layers: int = 2
    max_seq_len: int = 1 << 15
    norm_order: TransformerNormOrder = TransformerNormOrder.PRE
    norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16
    embedding: Literal["index", "projection"] = "index"

    input_dim: int | None = None
    output_dim: int | None = None

    def __post_init__(self) -> None:
        self.seq_mixer.model_dim = self.model_dim
        self.state_mixer.model_dim = self.model_dim


def classifier_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: Literal["none", "mean", "sum"] = "sum",
    loss_mask: torch.Tensor | None = None,
    pad_idx: int | None = None,
    *,
    loss_type: Literal["mse", "bce"],
) -> torch.Tensor:
    if logits.size(-1) == 1:
        logits.squeeze_(-1)

    batch_shape = targets.size()

    # (B, S, V) -> (B * S, V)
    logits = logits.contiguous().flatten(0, 1)

    # (B, S) -> (B * S,)
    targets = targets.contiguous().flatten(0, 1)

    if loss_type == "mse":
        loss = F.mse_loss(
            logits,
            targets,
            reduction=reduction if loss_mask is None else "none",
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction=reduction if loss_mask is None else "none",
        )

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


class TransformerBuilder:
    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        loss: Literal["cross_entropy", "mse", "bce"] = "cross_entropy",
    ) -> None:
        self.config = config
        self.loss = loss

        assert self.config.seq_mixer.model_dim == self.config.state_mixer.model_dim

    def build_model(self) -> TransformerDecoderModel:
        frontend = self.build_frontend()

        decoder = self.build_decoder()

        output_dim = self.config.output_dim or 1

        final_proj = Linear(self.config.model_dim, output_dim, bias=False)

        match self.loss:
            case "cross_entropy":
                loss = cross_entropy_loss
            case "mse":
                loss = functools.partial(classifier_loss, loss_type="mse")
            case "bce":
                loss = functools.partial(classifier_loss, loss_type="bce")
            case _:
                raise ValueError(f"loss type {self.loss} not supported")

        model = TransformerDecoderModel(
            frontend,  # type: ignore
            decoder,
            final_proj,
            max_seq_len=self.config.max_seq_len,
            loss_function=loss,
        )

        return model

    def build_decoder(self) -> TransformerDecoder:
        layers = self.build_layers()

        return TransformerDecoder(
            layers,
            self.config.model_dim,
            norm_order=self.config.norm_order,
            norm_eps=self.config.norm_eps,
        )

    def build_frontend(self) -> Embedding | EmbeddingProjection:
        pos_encoder = LearnedPositionalEncoder(
            dim=self.config.model_dim, max_seq_len=self.config.max_seq_len
        )

        if self.config.embedding == "index":
            assert self.config.output_dim is not None

            return Embedding(
                num_embeddings=self.config.output_dim,
                embedding_dim=self.config.model_dim,
                pos_encoder=pos_encoder,
            )
        else:
            assert self.config.input_dim is not None

            return EmbeddingProjection(
                input_dim=self.config.input_dim,
                model_dim=self.config.model_dim,
                pos_encoder=pos_encoder,
            )

    def build_layers(self) -> list[TransformerDecoderLayer]:
        layers = []

        for _ in range(self.config.num_layers):
            attention = self.config.seq_mixer.build()

            ffn = self.config.state_mixer.build()

            layer = TransformerDecoderLayer(
                attn=attention,
                ffn=ffn,
                model_dim=self.config.model_dim,
                norm_order=self.config.norm_order,
                norm_eps=self.config.norm_eps,
            )

            layers.append(layer)

        return layers


def main(config: ModelConfig) -> None:
    print(config)


if __name__ == "__main__":
    config = cli.run_default_cli(ModelConfig, console_outputs=on_local_rank_zero())

    main(config)
