from unittest.mock import MagicMock

import torch

from miniseq.data import (
    ColumnMapperTransform,
    GenericInstructDataset,
    InstructionDict,
    PretrainedHFTokenizer,
)
from miniseq.machine import Machine


def test_generic_instruct_dataset():
    # 1. Create a simple in-memory dataset with items of different lengths
    dataset = [
        {"q": "question 1", "a": "answer 1"},
        {"q": "question 2", "a": "a much longer answer 2"},
    ]

    # 2. Create a ColumnMapperTransform
    transform = ColumnMapperTransform(
        columns=InstructionDict(instruction="q", completion="a")
    )

    # 3. Create a GenericInstructDataset
    instruct_dataset = GenericInstructDataset(
        {"train": dataset}, instruct_transform=transform
    )

    # 4. Mock a tokenizer and a machine
    tokenizer = MagicMock(spec=PretrainedHFTokenizer)
    # The default build_prompt_completion joins with no separator
    tokenizer.apply_chat_template.side_effect = lambda x, **kwargs: "".join(
        m["content"] for m in x
    )
    tokenizer.encode.side_effect = lambda s, **kwargs: [ord(c) for c in s]
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    machine = MagicMock(spec=Machine)
    machine.rank = 0
    machine.size = 1

    # 5. Create a data loader
    loader = instruct_dataset.create_loader(
        tokenizer=tokenizer, machine=machine, batch_size=2, seed=42, npc=1
    )

    # 6. Verify the output
    batch = next(iter(loader))

    assert batch.batch_size == 2
    # Check that lengths are different
    expected_lens = [
        len("question 1" + "answer 1"),
        len("question 2" + "a much longer answer 2"),
    ]
    assert batch.seq_lens == expected_lens
    # Since lengths are different, padding should be applied
    assert batch.is_padded
    assert not batch.is_packed

    expected_seqs_str = [
        "question 1" + "answer 1",
        "question 2" + "a much longer answer 2",
    ]
    expected_seqs = [[ord(c) for c in s] for s in expected_seqs_str]

    # Pad the shorter sequence
    max_len = max(len(s) for s in expected_seqs)
    for seq in expected_seqs:
        seq.extend([tokenizer.pad_token_id] * (max_len - len(seq)))

    assert torch.equal(batch.seqs, torch.tensor(expected_seqs, dtype=torch.int64))


def test_generic_instruct_dataset_with_chat_template():
    # 1. Create a simple in-memory dataset
    dataset = [
        {"q": "question 1", "a": "answer 1"},
        {"q": "question 2", "a": "answer 2"},
    ]

    # 2. Create a ColumnMapperTransform
    transform = ColumnMapperTransform(
        columns=InstructionDict(instruction="q", completion="a")
    )

    # 3. Create a GenericInstructDataset with apply_chat_template=True
    instruct_dataset = GenericInstructDataset(
        {"train": dataset}, instruct_transform=transform, apply_chat_template=True
    )

    # 4. Mock a tokenizer and a machine
    tokenizer = MagicMock(spec=PretrainedHFTokenizer)

    def mock_apply_chat_template(messages, **kwargs):
        # A more realistic mock where the prompt+completion starts with the prompt.
        text = ""
        for m in messages:
            text += f"{m['role']}:{m['content']};"
        return text

    tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    tokenizer.encode.side_effect = lambda s, **kwargs: [ord(c) for c in s]
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    machine = MagicMock(spec=Machine)
    machine.rank = 0
    machine.size = 1

    # 5. Create a data loader
    loader = instruct_dataset.create_loader(
        tokenizer=tokenizer, machine=machine, batch_size=2, seed=42, npc=1
    )

    # 6. Verify the output
    batch = next(iter(loader))

    assert batch.batch_size == 2
    # The mock template produces sequences of the same length, so no padding
    assert not batch.is_padded
    assert not batch.is_packed

    # Check that apply_chat_template was called twice per item
    assert tokenizer.apply_chat_template.call_count == 4

    # The create_loader pipeline calls template_prompt_completion, which returns
    # (prompt, completion), and then make_sequence_example concatenates them.
    expected_seq_strings = [
        "user:question 1;assistant:answer 1;",
        "user:question 2;assistant:answer 2;",
    ]

    expected_seqs = [[ord(c) for c in s] for s in expected_seq_strings]

    max_len = max(len(s) for s in expected_seqs)
    for seq in expected_seqs:
        seq.extend([tokenizer.pad_token_id] * (max_len - len(seq)))

    assert torch.equal(batch.seqs, torch.tensor(expected_seqs, dtype=torch.int64))
