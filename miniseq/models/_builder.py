from types import MethodType

import torch

from miniseq.models._registry import ModelConfig
from miniseq.nn import (
    Embedding,
    Linear,
    Module,
    TiedProjectionLayer,
    cross_entropy_loss,
)
from miniseq.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderModel,
)
from miniseq.utils import default_dtype


class TransformerBuilder:
    config: ModelConfig

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

        assert self.config.attn_config.model_dim == self.config.ffn_config.model_dim

    def build_model(self) -> TransformerDecoderModel:
        frontend = self.build_frontend()

        decoder = self.build_decoder()

        if self.config.tie_weights:
            final_proj = TiedProjectionLayer(frontend)
        else:
            final_proj = Linear(
                self.config.model_dim, self.config.vocab_size, bias=False
            )

        loss_func = cross_entropy_loss

        if not self.config.cut_cross_entropy:
            loss_func = torch.compile(loss_func, dynamic=True)

        model = TransformerDecoderModel(
            frontend,
            decoder,
            final_proj,
            max_seq_len=self.config.max_seq_len,
            pad_idx=self.config.pad_idx,
            loss_function=loss_func,
        )

        if self.config.cut_cross_entropy:
            from miniseq.kernels._cce import build_cce_nll_forward

            # Patch the loss function if using custom loss.
            model.loss = MethodType(build_cce_nll_forward(), model)

        return model

    def build_decoder(self) -> TransformerDecoder:
        layers = self.build_layers()

        return TransformerDecoder(
            layers,
            self.config.model_dim,
            norm_order=self.config.norm_order,
            norm_eps=self.config.norm_eps,
        )

    def _build_rope(self) -> Module:
        head_dim = self.config.attn_config.head_dim

        if head_dim is None:
            assert not self.config.model_dim % self.config.attn_config.num_heads

            head_dim = self.config.model_dim // self.config.attn_config.num_heads

        return self.config.pos_encoder_config.build(
            dim=head_dim, max_seq_len=self.config.max_seq_len
        )

    def build_frontend(self) -> Embedding:
        return Embedding(
            num_embeddings=self.config.vocab_size, embedding_dim=self.config.model_dim
        )

    def build_layers(self) -> list[TransformerDecoderLayer]:
        pos_encoder = self._build_rope()

        layers = []

        for _ in range(self.config.num_layers):
            attention = self.config.attn_config.build(pos_encoder=pos_encoder)

            ffn = self.config.ffn_config.build()

            layer = TransformerDecoderLayer(
                attn=attention,
                ffn=ffn,
                model_dim=self.config.model_dim,
                norm_order=self.config.norm_order,
                norm_eps=self.config.norm_eps,
            )

            layers.append(layer)

        return layers


def build_model(
    config: ModelConfig, *, device: torch.device, dtype: torch.dtype
) -> TransformerDecoderModel:
    with default_dtype(dtype), device:
        model = TransformerBuilder(config).build_model()
    return model
