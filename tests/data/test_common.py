import torch

from miniseq.data import SequenceExample, right_pad_collate


def test_right_pad_collate_ragged():
    tensors = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    padded = right_pad_collate(tensors, pad_value=0)
    assert torch.equal(padded["seqs"], torch.tensor([[1, 2, 0], [3, 4, 5]]))
    assert padded["seq_lens"] == [2, 3]
    assert padded["is_ragged"]


def test_right_pad_collate_uniform():
    tensors = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    padded = right_pad_collate(tensors, pad_value=0)
    assert torch.equal(padded["seqs"], torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert padded["seq_lens"] == [3, 3]
    assert not padded["is_ragged"]


def test_sequence_example_collate():
    examples = [
        SequenceExample(
            indices=torch.tensor([1, 2]), target_mask=torch.tensor([True, True])
        ),
        SequenceExample(
            indices=torch.tensor([3, 4, 5]),
            target_mask=torch.tensor([True, True, False]),
        ),
    ]
    batch = SequenceExample.collate(examples, pad_idx=0)
    assert batch.target_mask is not None
    assert torch.equal(batch.seqs, torch.tensor([[1, 2, 0], [3, 4, 5]]))
    assert torch.equal(
        batch.target_mask, torch.tensor([[True, True, False], [True, True, False]])
    )
