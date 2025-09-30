import torch

from miniseq.transformer import (
    AttentionConfig,
    FFNConfig,
    TransformerDecoder,
    TransformerDecoderLayer,
)


def test_decoder_layer_output_shape() -> None:
    # Test that the output shape of TransformerDecoderLayer is correct.
    attn_config = AttentionConfig(
        model_dim=128,
        num_heads=8,
        num_kv_heads=4,
    )
    ffn_config = FFNConfig(
        model_dim=128,
    )
    attn = attn_config.build()
    ffn = ffn_config.build()
    decoder_layer = TransformerDecoderLayer(attn, ffn, 128)
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, attn_config.model_dim)
    output = decoder_layer(x, attn_mask=None)
    assert output.shape == (batch_size, seq_len, attn_config.model_dim)


def test_transformer_decoder_output_shape() -> None:
    # Test that the output shape of TransformerDecoder is correct.
    attn_config = AttentionConfig(
        model_dim=128,
        num_heads=8,
        num_kv_heads=4,
    )
    ffn_config = FFNConfig(
        model_dim=128,
    )
    attn = attn_config.build()
    ffn = ffn_config.build()
    decoder_layer = TransformerDecoderLayer(attn, ffn, 128)
    decoder = TransformerDecoder([decoder_layer] * 2, 128)
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, attn_config.model_dim)
    output = decoder(x, attn_mask=None)
    assert output.shape == (batch_size, seq_len, attn_config.model_dim)
