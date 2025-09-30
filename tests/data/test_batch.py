import torch

from miniseq.data import SequenceBatch


def test_sequence_batch_properties():
    seqs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
    seq_lens = [3, 2]
    batch = SequenceBatch(seqs=seqs, seq_lens=seq_lens, is_padded=True, pad_idx=0)

    assert batch.batch_size == 2
    assert batch.num_elements == 5  # 3 + 2
    assert batch.padding == 3  # (2 * 4) - 5
    assert batch.min_seqlen == 2
    assert batch.max_seqlen == 3
    assert batch.num_target_elements == 5  # No target_mask, so same as num_elements
    assert torch.equal(batch.seqlens_pt, torch.tensor([3, 2], dtype=torch.int64))


def test_sequence_batch_properties_with_target_mask():
    seqs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
    seq_lens = [3, 2]
    target_mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False]], dtype=torch.bool
    )
    batch = SequenceBatch(
        seqs=seqs, seq_lens=seq_lens, target_mask=target_mask, is_padded=True, pad_idx=0
    )

    assert batch.num_target_elements == 3  # 2 from first seq, 1 from second


def test_sequence_batch_packed_properties():
    seqs = torch.tensor([[1, 2, 3, 4, 5, 6, 0]], dtype=torch.int64)
    seq_lens = [3, 3]
    input_pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], dtype=torch.int64)
    batch = SequenceBatch(
        seqs=seqs,
        seq_lens=seq_lens,
        input_pos=input_pos,
        is_packed=True,
        is_padded=True,
        pad_idx=0,
    )

    assert batch.batch_size == 1
    assert batch.num_elements == 6
    assert batch.padding == 1
    assert batch.min_seqlen == 3
    assert batch.max_seqlen == 3
    assert torch.equal(
        batch.document_ids, torch.tensor([[0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32)
    )


def test_sequence_batch_as_auto_regressive():
    seqs = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int64)
    seq_lens = [3, 2]
    batch = SequenceBatch(seqs=seqs, seq_lens=seq_lens, is_padded=True, pad_idx=0)

    input_batch, target_batch = batch.as_auto_regressive()

    # Check input batch
    assert torch.equal(input_batch.seqs, torch.tensor([[1, 2, 3], [4, 5, 0]]))
    assert input_batch.seq_lens == [3, 2]
    assert input_batch.is_padded

    # Check target batch
    assert torch.equal(target_batch.seqs, torch.tensor([[2, 3, 0], [5, 0, 0]]))
    assert target_batch.seq_lens == [2, 1]
    assert target_batch.is_padded


def test_sequence_batch_as_auto_regressive_packed():
    seqs = torch.tensor([[1, 2, 3, 4, 5, 6, 0]], dtype=torch.int64)
    seq_lens = [3, 3]
    input_pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], dtype=torch.int64)
    batch = SequenceBatch(
        seqs=seqs,
        seq_lens=seq_lens,
        input_pos=input_pos,
        is_packed=True,
        is_padded=True,
        pad_idx=0,
    )

    input_batch, target_batch = batch.as_auto_regressive()

    # Check input batch
    assert torch.equal(input_batch.seqs, torch.tensor([[1, 2, 3, 4, 5, 6]]))
    assert input_batch.seq_lens == [3, 3]
    assert input_batch.is_packed
    assert not input_batch.is_padded

    # Check target batch
    assert torch.equal(target_batch.seqs, torch.tensor([[2, 3, 4, 5, 6, 0]]))
    assert target_batch.seq_lens == [2, 3]  # First element of first seq is removed
    assert target_batch.is_packed
    assert target_batch.is_padded


def test_sequence_batch_as_auto_regressive_packed_no_padding():
    # Test that as_auto_regressive works correctly for a packed batch with no padding.
    seqs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    seq_lens = [4, 6]
    batch = SequenceBatch(seqs, seq_lens=seq_lens, is_packed=True)

    # First call to as_auto_regressive.
    input_batch, target_batch = batch.as_auto_regressive()

    assert input_batch.seq_lens == [4, 5]
    assert target_batch.seq_lens == [3, 6]
    assert not input_batch.is_padded
    assert not target_batch.is_padded
    assert torch.equal(input_batch.seqs, torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]))
    assert torch.equal(target_batch.seqs, torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10]]))

    # Check that original batch is not modified
    assert batch.seq_lens == [4, 6]


def test_as_auto_regressive_packed_with_padding():
    # Create a packed batch with padding.
    seqs = torch.tensor([[1, 2, 3, 4, 5, 0, 0]], dtype=torch.int64)
    seq_lens = [3, 2]
    input_pos = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], dtype=torch.int64)
    batch = SequenceBatch(
        seqs=seqs,
        seq_lens=seq_lens,
        input_pos=input_pos,
        is_packed=True,
        is_padded=True,
        pad_idx=0,
    )

    input_batch, _ = batch.as_auto_regressive()

    # The seq_lens of the input batch should not be modified when there is padding,
    # because the truncated token is a padding token.
    assert input_batch.seq_lens == [3, 2]
