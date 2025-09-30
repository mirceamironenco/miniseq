import torch

from miniseq.data import (
    SequenceBatch,
    SequenceExample,
    UnfinishedPack,
    UnfinishedPrefixPack,
)


def test_packed_attention_mask():
    """Verify attention mask for a simple packed sequence."""
    unfinished_pack = UnfinishedPack(max_seq_len=5)
    sequence1 = SequenceExample.from_sequence([1, 2, 3])
    unfinished_pack.update(sequence1)
    sequence2 = SequenceExample.from_sequence([4, 5])
    unfinished_pack.update(sequence2)

    packed_sequence = unfinished_pack.finish(pad_idx=0)
    batch = SequenceBatch(
        seqs=packed_sequence.indices.unsqueeze(0),
        seq_lens=packed_sequence.seq_lens,
        input_pos=packed_sequence.input_pos.unsqueeze(0),
        is_packed=True,
    )

    _, _, attention_mask = batch.as_input()
    dense_mask = attention_mask.materialize_dense().squeeze()

    # Manually construct the expected mask
    # Causal mask
    expected_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool), diagonal=0)
    # Document mask
    doc_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )

    expected_mask = expected_mask & doc_mask
    assert torch.equal(dense_mask, expected_mask)


def test_prefix_sharing_attention_mask():
    """Verify attention mask for a packed sequence with prefix sharing."""
    unfinished_pack = UnfinishedPrefixPack(max_seq_len=8, share_count=2)

    prompt = [1, 2, 3]
    completion1 = [4, 5]
    completion2 = [6, 7, 8]

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

    unfinished_pack.maybe_update(examples)
    packed_sequence = unfinished_pack.finish(pad_idx=0)

    SequenceBatch.using_flex_attention = True

    batch = SequenceBatch(
        seqs=packed_sequence.indices.unsqueeze(0),
        seq_lens=packed_sequence.seq_lens,
        input_pos=packed_sequence.input_pos.unsqueeze(0),
        is_packed=True,
        prefix_docs=packed_sequence.inner_docs,
        prefix_share_count=2,
    )

    _, _, attention_mask = batch.as_input()
    dense_mask = attention_mask.materialize_dense().squeeze()

    # Manually construct the expected mask
    # Causal mask
    expected_mask = torch.tril(torch.ones(8, 8, dtype=torch.bool), diagonal=0)

    # Document mask
    doc_ids = batch.document_ids.squeeze()
    doc_mask = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)
    expected_mask = expected_mask & doc_mask

    # Prefix sharing mask
    assert batch.prefix_docs is not None
    prefix_docs = torch.tensor(batch.prefix_docs)
    prefix_mask = (prefix_docs.unsqueeze(0) == 0) | (
        prefix_docs.unsqueeze(1) == prefix_docs.unsqueeze(0)
    )
    expected_mask = expected_mask & prefix_mask

    assert torch.equal(dense_mask, expected_mask)
