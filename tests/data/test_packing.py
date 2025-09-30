import torch

from miniseq.data import SequenceExample, UnfinishedPack, UnfinishedPrefixPack


def test_unfinished_pack_target_mask():
    """Test case to verify the target_mask logic in UnfinishedPack."""
    unfinished_pack = UnfinishedPack(max_seq_len=10)
    sequence1 = SequenceExample.from_sequence([1, 2, 3])
    unfinished_pack.update(sequence1)
    sequence2 = SequenceExample.from_sequence([4, 5])
    unfinished_pack.update(sequence2)

    packed_sequence = unfinished_pack.finish(pad_idx=0)

    expected_target_mask = torch.tensor(
        [False, True, True, False, True, False, False, False, False, False]
    )
    assert torch.equal(packed_sequence.target_mask, expected_target_mask)


def test_unfinished_prefix_pack_basic():
    """Test basic packing with UnfinishedPrefixPack."""
    unfinished_pack = UnfinishedPrefixPack(max_seq_len=20, share_count=2)

    prompt = [1, 2, 3]
    completion1 = [4, 5, 6]
    completion2 = [7, 8]

    examples = [
        SequenceExample.from_instruction(
            prompt=torch.tensor(prompt),
            completion=torch.tensor(completion1),
            completion_only=True,
        ),
        SequenceExample.from_instruction(
            prompt=torch.tensor(prompt),
            completion=torch.tensor(completion2),
            completion_only=True,
        ),
    ]

    assert unfinished_pack.maybe_update(examples)

    packed_sequence = unfinished_pack.finish(pad_idx=0)

    assert packed_sequence.seq_lens == [6, 2]
    assert packed_sequence.padding == 20 - (3 + 3 + 2)
    assert torch.equal(
        packed_sequence.indices[:8], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    )
    assert packed_sequence.prefix_count == 2


def test_unfinished_prefix_pack_metadata():
    """Test metadata from UnfinishedPrefixPack."""
    unfinished_pack = UnfinishedPrefixPack(max_seq_len=20, share_count=2)

    prompt = [1, 2, 3]
    completion1 = [4, 5, 6]
    completion2 = [7, 8]

    examples = [
        SequenceExample.from_instruction(
            prompt=torch.tensor(prompt),
            completion=torch.tensor(completion1),
            completion_only=True,
        ),
        SequenceExample.from_instruction(
            prompt=torch.tensor(prompt),
            completion=torch.tensor(completion2),
            completion_only=True,
        ),
    ]

    assert unfinished_pack.maybe_update(examples)

    packed_sequence = unfinished_pack.finish(pad_idx=0)

    expected_starts = [0, 0, 0, 3, 3, 3, 6, 6]
    expected_inner_docs = [0, 0, 0, 1, 1, 1, 2, 2]

    expected_starts.extend([6] * 12)
    expected_inner_docs.extend([expected_inner_docs[-1]] * 12)

    assert torch.equal(
        torch.tensor(packed_sequence.inner_docs), torch.tensor(expected_inner_docs)
    )


def test_unfinished_prefix_pack_padding():
    """Test padding with UnfinishedPrefixPack."""
    unfinished_pack = UnfinishedPrefixPack(max_seq_len=10, share_count=1)

    prompt = [1, 2]
    completion1 = [3, 4, 5]

    examples = [
        SequenceExample.from_instruction(
            prompt=torch.tensor(prompt),
            completion=torch.tensor(completion1),
            completion_only=True,
        ),
    ]

    assert unfinished_pack.maybe_update(examples)
    packed_sequence = unfinished_pack.finish(pad_idx=-1)

    assert packed_sequence.padding == 5
    assert torch.equal(
        packed_sequence.indices, torch.tensor([1, 2, 3, 4, 5, -1, -1, -1, -1, -1])
    )
