from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence

from miniseq.data._batch import SequenceBatch


class SequenceData(TypedDict):
    seqs: torch.Tensor
    seq_lens: list[int]
    is_ragged: bool


def right_pad_collate(
    tensor_list: list[torch.Tensor], *, pad_value: int | float | bool
) -> SequenceData:
    if not tensor_list:
        raise ValueError("right_pad_collate got empty tensor_list.")

    is_ragged = False

    seq_lens = []

    shape_to_check = tensor_list[0].shape[1:]
    for index, tensor in enumerate(tensor_list):
        if tensor.shape[1:] != shape_to_check:
            raise ValueError(
                "All tensors in collate must have the same non-sequence-length shape. "
                f"Got shapes {shape_to_check} and {tensor.shape[1:]}"
            )

        seq_lens.append(tensor.size(0))

        if index > 0 and seq_lens[-1] != seq_lens[-2]:
            is_ragged = True

    if is_ragged:
        seqs = pad_sequence(
            tensor_list,
            batch_first=True,
            padding_side="right",
            padding_value=pad_value,
        )
    else:
        seqs = torch.stack(tensor_list, dim=0)

    return {
        "seqs": seqs,
        "seq_lens": seq_lens,
        "is_ragged": is_ragged,
    }


@dataclass(frozen=True)
class SequenceExample:
    indices: torch.Tensor
    """tokens ids. (seqlen,*)"""

    target_mask: torch.Tensor
    """boolean target mask. (seqlen,)"""

    completion_only: bool = False

    prompt: torch.Tensor | None = None

    completion: torch.Tensor | None = None

    def __post_init__(self) -> None:
        assert self.indices.ndim >= 1
        assert self.target_mask.ndim == 1
        assert self.indices.size(0) == self.target_mask.size(0)

    @classmethod
    def from_instruction(
        cls,
        prompt: torch.Tensor | list[int],
        completion: torch.Tensor | list[int],
        completion_only: bool,
    ) -> SequenceExample:
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.tensor(prompt, dtype=torch.int64)

        if not isinstance(completion, torch.Tensor):
            completion = torch.tensor(completion, dtype=torch.int64)

        seqs = torch.cat([prompt, completion])

        if completion_only:
            target_mask = torch.arange(len(seqs), device=prompt.device) >= len(prompt)
        else:
            target_mask = torch.full_like(seqs, fill_value=True)

        assert seqs.ndim == 1
        assert target_mask.ndim == 1

        return SequenceExample(seqs, target_mask, completion_only, prompt, completion)

    @classmethod
    def from_sequence(cls, sequence: torch.Tensor | list[int]) -> SequenceExample:
        """Construct example with no masked tokens."""

        if not isinstance(sequence, torch.Tensor):
            sequence = torch.tensor(sequence, dtype=torch.int64)

        target_mask = torch.full_like(sequence, fill_value=True)

        assert sequence.ndim == 1
        assert target_mask.ndim == 1

        return SequenceExample(sequence, target_mask, completion_only=False)

    @staticmethod
    def collate(
        examples: Sequence[SequenceExample], *, pad_idx: int = 0
    ) -> SequenceBatch:
        indices = right_pad_collate(
            [item.indices for item in examples], pad_value=pad_idx
        )

        masks = right_pad_collate(
            [item.target_mask for item in examples], pad_value=False
        )

        return SequenceBatch(
            indices["seqs"],
            seq_lens=indices["seq_lens"],
            target_mask=masks["seqs"],
            is_padded=indices["is_ragged"],
            pad_idx=pad_idx,
        )


PreferenceExample: TypeAlias = tuple[
    SequenceExample, SequenceExample, float | None, float | None
]


@dataclass(frozen=True)
class PackedSequenceExample:
    """Sequnce formed by packing smaller sequences + padding to a fixed length."""

    indices: torch.Tensor
    """tokens ids. (seqlen,)"""

    target_mask: torch.Tensor
    """boolean target mask. (seqlen,)"""

    input_pos: torch.Tensor
    """Input positions of tokens relative to each sequence. (seqlen,)"""

    seq_lens: list[int]
    """Lengths of sequences used to form the packed sequence."""

    padding: int
    """Amount of right-padding added."""

    inner_docs: list[int] | None = None

    prefix_count: int | None = None

    def __post_init__(self) -> None:
        if not self.indices.ndim == self.target_mask.ndim == self.input_pos.ndim == 1:
            raise ValueError("Packed sequence must be 1-dimensional.")

        length = len(self.indices)

        if not (length == len(self.target_mask) == len(self.input_pos)):
            raise ValueError(
                "Failed to construct packed sequence, got different lengths: "
                f"indices ({length}), target_mask ({len(self.target_mask)}) input_pos ({len(self.input_pos)})"
            )

        if not length == (sum(self.seq_lens) + self.padding):
            raise ValueError(
                f"Failed to construct packed sequence, seq_lens.sum() + padding != {length}."
            )

    @staticmethod
    def collate(
        examples: Sequence[PackedSequenceExample], *, pad_idx: int
    ) -> SequenceBatch:
        if not len(examples) == 1:
            raise ValueError(
                f"packed sequences are expected to have batch_size=1, got {len(examples)}"
            )

        # Assumes batch_size = 1.
        seq_lens = examples[0].seq_lens
        is_padded = examples[0].padding > 0
        inner_docs = examples[0].inner_docs

        # The following are implemented to work with bsz > 1 in case we later allow this.
        seqs = torch.stack([ex.indices for ex in examples], dim=0)

        input_pos = torch.stack([ex.input_pos for ex in examples], dim=0)

        target_mask = torch.stack([ex.target_mask for ex in examples], dim=0)

        examples_used = sum(len(ex.seq_lens) for ex in examples)

        return SequenceBatch(
            seqs=seqs,
            seq_lens=seq_lens,
            input_pos=input_pos,
            target_mask=target_mask,
            is_packed=True,
            is_padded=is_padded,
            pad_idx=pad_idx,
            examples_used=examples_used,
            prefix_docs=inner_docs,
            prefix_share_count=examples[0].prefix_count,
        )


PackedPreferenceExample: TypeAlias = tuple[
    PackedSequenceExample,
    PackedSequenceExample,
    Sequence[float | None],
    Sequence[float | None],
]
