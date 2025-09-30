import torch

from miniseq.transformer import AttentionConfig, AttentionMask


def test_mha_output_shape() -> None:
    # Test that the output shape of MHAttention is correct.
    config = AttentionConfig(
        model_dim=128,
        num_heads=8,
        num_kv_heads=4,
    )
    mha = config.build()
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.model_dim)
    output = mha(x, attn_mask=None)
    assert output.shape == (batch_size, seq_len, config.model_dim)


def test_mha_causal_mask() -> None:
    # Test that the causal mask is applied correctly.
    config = AttentionConfig(
        model_dim=128,
        num_heads=8,
        num_kv_heads=4,
    )
    mha = config.build()
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.model_dim)

    # Create a causal mask.
    attn_mask = AttentionMask.build_causal(
        q_len=seq_len,
        kv_len=seq_len,
        device="cpu",
        max_input_pos=seq_len,
        max_seq_len=seq_len,
        total_seq_len=seq_len,
        batch_size=batch_size,
    )

    # Forward pass with the causal mask.
    output1 = mha(x, attn_mask=attn_mask)

    # Change the last element of the sequence.
    x[:, -1, :] = torch.randn(batch_size, config.model_dim)

    # Forward pass again. The output for the first n-1 elements should be the same.
    output2 = mha(x, attn_mask=attn_mask)

    assert torch.allclose(output1[:, :-1, :], output2[:, :-1, :], atol=1e-6)
