import torch
from torchdata.nodes import IterableWrapper

from miniseq.data import Packer, SequenceExample, UnfinishedPack


def test_unfinished_pack_padding() -> None:
    """Test case to verify the input_pos & padding logic in UnfinishedPack."""

    unfinished_pack = UnfinishedPack(max_seq_len=10)
    sequence = SequenceExample.from_sequence([1, 2, 3])
    unfinished_pack.update(sequence)

    packed_sequence = unfinished_pack.finish(pad_idx=0)

    expected_input_pos = torch.tensor([0, 1, 2, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    assert torch.equal(packed_sequence.input_pos, expected_input_pos)


def test_unfinished_pack_can_update() -> None:
    unfinished_pack = UnfinishedPack(max_seq_len=10)
    sequence = SequenceExample.from_sequence([1, 2, 3])
    unfinished_pack.update(sequence)

    good_example = SequenceExample.from_sequence([1, 2, 3])
    assert unfinished_pack.can_update(good_example)

    good_example_full = SequenceExample.from_sequence(list(range(7)))
    assert unfinished_pack.can_update(good_example_full)

    bad_example = SequenceExample.from_sequence(list(range(8)))
    assert not unfinished_pack.can_update(bad_example)


def test_packer():
    # 1. Create a source of SequenceExamples
    examples = [
        SequenceExample.from_sequence([1, 2, 3]),
        SequenceExample.from_sequence([4, 5]),
        SequenceExample.from_sequence([6, 7, 8, 9]),
        SequenceExample.from_sequence([10]),
    ]
    source_node = IterableWrapper(examples)

    # 2. Create a Packer
    packer = Packer(source_node, max_seq_len=5, pad_idx=0)

    # 3. Collect the packed examples
    packed_examples = list(packer)

    # 4. Verify the output
    assert len(packed_examples) == 2

    # First packed example
    pack1 = packed_examples[0]
    assert pack1.seq_lens == [3, 2]
    assert pack1.padding == 0
    assert torch.equal(pack1.indices, torch.tensor([1, 2, 3, 4, 5]))
    assert torch.equal(pack1.input_pos, torch.tensor([0, 1, 2, 0, 1]))

    # Second packed example
    pack2 = packed_examples[1]
    assert pack2.seq_lens == [4, 1]
    assert pack2.padding == 0
    assert torch.equal(pack2.indices, torch.tensor([6, 7, 8, 9, 10]))
    assert torch.equal(pack2.input_pos, torch.tensor([0, 1, 2, 3, 0]))
