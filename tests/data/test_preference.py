import pytest
import torch

from miniseq.data._preference import PreferenceBatch, collate
from miniseq.data._common import SequenceExample


def test_preference_batch_validation():
    # Test valid batch
    chosen = SequenceExample.from_sequence([1, 2, 3])
    rejected = SequenceExample.from_sequence([4, 5])
    batch_chosen = SequenceExample.collate([chosen], pad_idx=0)
    batch_rejected = SequenceExample.collate([rejected], pad_idx=0)

    PreferenceBatch(
        chosen_batch=batch_chosen,
        rejected_batch=batch_rejected,
        ref_chosen=None,
        ref_rejected=None,
    )

    # Test mismatched packed status
    batch_chosen.is_packed = True
    with pytest.raises(AssertionError):
        PreferenceBatch(
            chosen_batch=batch_chosen,
            rejected_batch=batch_rejected,
            ref_chosen=None,
            ref_rejected=None,
        )

    # Test mismatched ref scores
    with pytest.raises(AssertionError):
        PreferenceBatch(
            chosen_batch=batch_chosen,
            rejected_batch=batch_rejected,
            ref_chosen=torch.randn(1),
            ref_rejected=None,
        )


def test_collate_preference_examples():
    chosen1 = SequenceExample.from_sequence([1, 2, 3])
    rejected1 = SequenceExample.from_sequence([4, 5])
    chosen2 = SequenceExample.from_sequence([6, 7])
    rejected2 = SequenceExample.from_sequence([8, 9, 10])

    examples = [
        (chosen1, rejected1, 0.6, 0.4),
        (chosen2, rejected2, 0.7, 0.3),
    ]

    batch = collate(examples, pad_idx=0)

    assert batch.chosen_batch.batch_size == 2
    assert batch.rejected_batch.batch_size == 2
    assert not batch.is_packed
    assert torch.equal(batch.ref_chosen, torch.tensor([0.6, 0.7]))
    assert torch.equal(batch.ref_rejected, torch.tensor([0.4, 0.3]))
