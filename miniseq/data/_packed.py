from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torchdata.nodes import BaseNode

from miniseq.data._common import (
    PackedPreferenceExample,
    PackedSequenceExample,
    PreferenceExample,
    SequenceExample,
)
from miniseq.utils import next_multiple


@dataclass(kw_only=True)
class UnfinishedPrefixPack:
    max_seq_len: int

    share_count: int
    """With how many completions is a prefix (prompt) shared."""

    _current_len: int = field(init=False, default=0)

    _indices: list[torch.Tensor] = field(init=False, default_factory=lambda: [])
    _target_mask: list[torch.Tensor] = field(init=False, default_factory=lambda: [])
    _input_pos: list[torch.Tensor] = field(init=False, default_factory=lambda: [])
    _seq_lens: list[int] = field(init=False, default_factory=lambda: [])

    _inner_doc: list[int] = field(init=False, default_factory=lambda: [])
    """Prompts always have inner doc id 0.  
    Completions have doc id as (current_seq * share_count) + completion_index + 1."""

    _sequences_added: int = field(init=False, default=0)

    def maybe_update(self, examples: list[SequenceExample]) -> bool:
        if not (num_seqs := len(examples)) == self.share_count:
            raise ValueError(
                f"Expected {self.share_count} prefix-sharing sequences, got {num_seqs}."
            )

        assert examples[0].prompt is not None

        current_sequence = self._sequences_added

        shared_prompt: torch.Tensor = examples[0].prompt
        total_len = len(shared_prompt)
        completions: list[torch.Tensor] = []
        completions_masks: list[torch.Tensor] = []

        self._inner_doc.extend([0] * len(shared_prompt))

        self._seq_lens.append(len(shared_prompt))

        self._input_pos.append(
            torch.arange(
                end=len(shared_prompt), dtype=torch.int64, device=shared_prompt.device
            )
        )

        last_prompt_pos = int(self._input_pos[-1][-1])

        # Note: It is not checked at runtime that the prompt is actually identical
        # Beyond a length check.
        for index, ex in enumerate(examples):
            if not ex.completion_only:
                raise ValueError(
                    "Packing with shared prefixes only supported under completion_only setting."
                )

            prompt, completion = ex.prompt, ex.completion

            if prompt is None or completion is None:
                raise ValueError("Prefix shared packer expected prompt, completion.")

            if len(prompt) != len(shared_prompt):
                raise ValueError(
                    "Got prompts of different lengths while trying to construct shared prefix pack."
                )

            completions.append(completion)

            inner_document_id = (current_sequence * self.share_count) + index + 1

            self._inner_doc.extend([inner_document_id] * len(completion))

            total_len += len(completion)

            completions_masks.append(torch.full_like(completion, fill_value=True))

            if total_len > self.max_seq_len:
                raise ValueError(
                    "Splitting across pack not supported; increase max_seq_len, "
                    f"total examples length {total_len} > max_seq_len ({self.max_seq_len})"
                )

            if total_len + self._current_len > self.max_seq_len:
                return False

            if index == 0:
                # First completion and prompt are treated as 1 sequence.
                # This unfortunately no longer makes seq_lens accurate.
                # Operations that depend on 100% seqlen accuracy might be affected and
                # should use document ids and prefix document ids.
                self._seq_lens[-1] += len(completion)
            else:
                self._seq_lens.append(len(completion))

            self._input_pos.append(
                torch.arange(
                    start=last_prompt_pos + 1,
                    end=last_prompt_pos + 1 + len(completion),
                    dtype=torch.int64,
                    device=shared_prompt.device,
                )
            )

        completions_pt = torch.cat(completions, dim=0)

        self._indices.append(torch.cat([shared_prompt, completions_pt], dim=0))

        # Note: Setting self._target_mask[-1][0] = False is not necessary in this case
        # because we assume completion_only = True, so all prompts are masked out.
        self._target_mask.append(
            torch.cat(
                [
                    torch.full_like(shared_prompt, fill_value=False),
                    torch.cat(completions_masks, dim=0),
                ],
                dim=0,
            )
        )

        self._current_len += total_len

        self._sequences_added += 1

        return True

    def finish(
        self, *, pad_idx: int, max_seq_len: int | None = None
    ) -> PackedSequenceExample:
        if max_seq_len is not None:
            self.set_max_seq_len(max_seq_len)

        indices = torch.cat(self._indices, dim=0)
        target_mask = torch.cat(self._target_mask, dim=0)
        input_pos = torch.cat(self._input_pos, dim=0)

        if (to_pad := self.max_seq_len - self._current_len) > 0:
            indices = F.pad(indices, pad=(0, to_pad), mode="constant", value=pad_idx)

            target_mask = F.pad(
                target_mask, pad=(0, to_pad), mode="constant", value=False
            )

            # Last sequence is padded; instead of starting a new sequence from 0 for the
            # padding section, we start from 1, since padding pos doesn't matter.
            # In this way, the correct number of sequences in the pack can also be
            # determined by the number of 0's.
            padding_pos = torch.arange(
                start=1,
                end=1 + to_pad,
                dtype=torch.int64,
                device=input_pos.device,
            )

            input_pos = torch.cat((input_pos, padding_pos), dim=0)

            # Let padding be part of the same document as the last completion.
            # This is consistent with padding_pos, and can be masked out anyway.
            self._inner_doc.extend([self._inner_doc[-1]] * to_pad)

        return PackedSequenceExample(
            indices=indices,
            target_mask=target_mask,
            input_pos=input_pos,
            seq_lens=self._seq_lens,
            padding=to_pad,
            inner_docs=self._inner_doc,
            prefix_count=self.share_count,
        )

    def set_max_seq_len(self, new_max_length: int) -> None:
        if new_max_length < self.length:
            raise ValueError(
                f"Cannot update pack maxlen to {new_max_length}, pack is already {self.length} large."
            )

        self.max_seq_len = new_max_length

    @property
    def length(self) -> int:
        return self._current_len

    @property
    def num_sequences(self) -> int:
        return self._sequences_added


@dataclass(kw_only=True)
class UnfinishedPack:
    max_seq_len: int

    _current_len: int = field(init=False, default=0)
    _indices: list[torch.Tensor] = field(init=False, default_factory=lambda: [])
    _target_mask: list[torch.Tensor] = field(init=False, default_factory=lambda: [])
    _input_pos: list[torch.Tensor] = field(init=False, default_factory=lambda: [])
    _seq_lens: list[int] = field(init=False, default_factory=lambda: [])

    _sequences_added: int = field(init=False, default=0)

    def can_update(self, example: SequenceExample) -> bool:
        """Returns whether it's possible to add `example` to the pack."""

        seq_len = len(example.indices)

        if seq_len > self.max_seq_len:
            raise ValueError(
                "Splitting across pack not supported; increase max_seq_len, "
                f"example length {seq_len} > max_seq_len ({self.max_seq_len})"
            )

        if seq_len + self._current_len > self.max_seq_len:
            return False

        return True

    def update(self, example: SequenceExample) -> None:
        if not self.can_update(example):
            raise ValueError("Sequence is too large to be added to pack.")

        seq_len = len(example.indices)

        self._indices.append(example.indices)
        self._target_mask.append(example.target_mask)
        self._input_pos.append(
            torch.arange(end=seq_len, dtype=torch.int64, device=example.indices.device)
        )
        self._seq_lens.append(seq_len)

        # To prevent end of one sequence predicting beginning of next, set all starting
        # tokens as False in the target mask, indicating they can't be targets.
        # Example: packed seqs (a1,a2,a3,b1,b2,b3,b4), mask (F,T,T,F,T,T,T)
        # inputs = seqs[:-1] = (a1,a2,a3,b1,b2,b3), targets = seqs[1:] (a2,a3,b1,b2,b3,b4)
        # target_mask = mask[1:] = (T,T,F,T,T,T) - indicating that a3 -> b1 is invalid.
        # NOTE: This assumes that target_mask will be sliced i.e. as target_mask[1:],
        # before being used as auto regressive input.
        self._target_mask[-1][0] = False

        self._current_len += seq_len

        self._sequences_added += 1

    def batch_update(self, examples: list[SequenceExample]) -> None:
        for example in examples:
            self.update(example)

    def finish(
        self, *, pad_idx: int, max_seq_len: int | None = None
    ) -> PackedSequenceExample:
        if max_seq_len is not None:
            self.set_max_seq_len(max_seq_len)

        indices = torch.cat(self._indices, dim=0)
        target_mask = torch.cat(self._target_mask, dim=0)
        input_pos = torch.cat(self._input_pos, dim=0)

        if (to_pad := self.max_seq_len - self._current_len) > 0:
            indices = F.pad(indices, pad=(0, to_pad), mode="constant", value=pad_idx)

            target_mask = F.pad(
                target_mask, pad=(0, to_pad), mode="constant", value=False
            )

            # Last sequence is padded; instead of starting a new sequence from 0 for the
            # padding section, we start from 1, since padding pos doesn't matter.
            # In this way, the correct number of sequences in the pack can also be
            # determined by the number of 0's.
            padding_pos = torch.arange(
                start=1,
                end=1 + to_pad,
                dtype=torch.int64,
                device=input_pos.device,
            )

            input_pos = torch.cat((input_pos, padding_pos), dim=0)

        return PackedSequenceExample(
            indices=indices,
            target_mask=target_mask,
            input_pos=input_pos,
            seq_lens=self._seq_lens,
            padding=to_pad,
        )

    def set_max_seq_len(self, new_max_length: int) -> None:
        if new_max_length < self.length:
            raise ValueError(
                f"Cannot update pack maxlen to {new_max_length}, pack is already {self.length} large."
            )

        self.max_seq_len = new_max_length

    @property
    def length(self) -> int:
        """Total length of the all sequences in the pack."""
        return self._current_len

    @property
    def num_sequences(self) -> int:
        """Current numbers of sequences that have been added to the pack."""
        return self._sequences_added


class Packer(BaseNode[PackedSequenceExample]):
    def __init__(
        self,
        source: BaseNode[SequenceExample],
        *,
        max_seq_len: int,
        pad_idx: int,
        drop_on_reset: bool = False,
    ) -> None:
        super().__init__()
        self._source = source
        self._max_seq_len = max_seq_len
        self._pad_idx = pad_idx
        self._drop_on_reset = drop_on_reset
        self._current_pack = UnfinishedPack(max_seq_len=max_seq_len)

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)

        # If we're forcing a reset, and want a clean iterator.
        if self._drop_on_reset:
            self._current_pack = UnfinishedPack(max_seq_len=self._max_seq_len)

        if initial_state is not None:
            self._source.reset(initial_state["source"])
        else:
            self._source.reset()

    def next(self) -> PackedSequenceExample:
        while self._current_pack.length < self._max_seq_len:
            try:
                item = next(self._source)
            except StopIteration:
                break

            if self._current_pack.can_update(item):
                self._current_pack.update(item)
            else:
                pack = self._current_pack.finish(pad_idx=self._pad_idx)

                # Make a new pack
                self._current_pack = UnfinishedPack(max_seq_len=self._max_seq_len)

                self._current_pack.update(item)

                return pack

        # Either current_pack.length == max_seq_len or _source is exhausted.
        # If _source was exhausted but we still have some items in the current pack,
        # finish the pack and return it, so last batched are not dropped.
        if 0 < self._current_pack.length <= self._max_seq_len:
            pack = self._current_pack.finish(pad_idx=self._pad_idx)

            # Make a new pack
            self._current_pack = UnfinishedPack(max_seq_len=self._max_seq_len)

            return pack
        else:
            # current_pack.length is 0, meaning _source is exhausted.
            raise StopIteration()

    def get_state(self) -> dict[str, Any]:
        return {"source": self._source.state_dict()}

    def __repr__(self) -> str:
        return f"Packer(max_seq_len={self._max_seq_len}, pad_idx={self._pad_idx}, drop_on_reset={self._drop_on_reset})"


class BatchPacker(BaseNode[PackedSequenceExample]):
    def __init__(
        self,
        source: BaseNode[SequenceExample],
        *,
        batch_size: int,
        pad_idx: int,
        drop_last: bool = False,
        drop_on_reset: bool = False,
        multiple_of: int = 128,
    ) -> None:
        super().__init__()
        self._source = source
        self._batch_size = batch_size
        self._pad_idx = pad_idx
        self._drop_last = drop_last
        self._drop_on_reset = drop_on_reset
        self._multiple_of = multiple_of
        self._current_pack = UnfinishedPack(max_seq_len=1 << 30)

    def reset(self, initial_state: dict | None = None):
        super().reset(initial_state)

        # If we're forcing a reset, and want a clean iterator.
        if self._drop_on_reset:
            self._current_pack = UnfinishedPack(max_seq_len=1 << 30)

        if initial_state is not None:
            self._source.reset(initial_state["source"])
        else:
            self._source.reset()

    def _finish_pack(self) -> PackedSequenceExample:
        final_length = next_multiple(self._current_pack.length, self._multiple_of)

        pack = self._current_pack.finish(
            pad_idx=self._pad_idx, max_seq_len=final_length
        )

        return pack

    def next(self) -> PackedSequenceExample:
        while self._current_pack.num_sequences < self._batch_size:
            try:
                item = next(self._source)
            except StopIteration:
                break

            self._current_pack.update(item)

            if self._current_pack.num_sequences == self._batch_size:
                pack = self._finish_pack()

                # Make a new pack
                self._current_pack = UnfinishedPack(max_seq_len=1 << 30)

                return pack

        # Source is exhausted.
        if self._current_pack.num_sequences > 0 and not self._drop_last:
            pack = self._finish_pack()

            # Make a new pack
            self._current_pack = UnfinishedPack(max_seq_len=1 << 30)

            return pack
        else:
            # Make a new pack
            self._current_pack = UnfinishedPack(max_seq_len=1 << 30)

            raise StopIteration()

    def get_state(self) -> dict[str, Any]:
        return {"source": self._source.state_dict()}

    def __repr__(self) -> str:
        return f"BatchPacker(batch_size={self._batch_size}, pad_idx={self._pad_idx}, drop_on_reset={self._drop_on_reset}, drop_last={self._drop_last})"


class PreferencePakcer(BaseNode[PackedPreferenceExample]):
    def __init__(
        self,
        source: BaseNode[PreferenceExample],
        *,
        max_seq_len: int,
        pad_idx: int,
        drop_on_reset: bool = False,
    ) -> None:
        super().__init__()
        self._source = source
        self._max_seq_len = max_seq_len
        self._pad_idx = pad_idx
        self._drop_on_reset = drop_on_reset

        self._chosen_pack = UnfinishedPack(max_seq_len=max_seq_len)
        self._rejected_pack = UnfinishedPack(max_seq_len=max_seq_len)
        self._ref_chosen_scores: list[float | None] = []
        self._ref_rejected_scores: list[float | None] = []

    def reset(self, initial_state: dict | None = None) -> None:
        super().reset(initial_state)

        if self._drop_on_reset:
            self._clear_pack()

        if initial_state is not None:
            self._source.reset(initial_state["source"])
        else:
            self._source.reset()

    def _finish_pack(self) -> PackedPreferenceExample:
        chosen_pack = self._chosen_pack.finish(pad_idx=self._pad_idx)
        rejected_pack = self._rejected_pack.finish(pad_idx=self._pad_idx)
        ref_chosen_scores = self._ref_chosen_scores[:]
        ref_rejected_scores = self._ref_rejected_scores[:]

        return chosen_pack, rejected_pack, ref_chosen_scores, ref_rejected_scores

    def _clear_pack(self) -> None:
        self._chosen_pack = UnfinishedPack(max_seq_len=self._max_seq_len)
        self._rejected_pack = UnfinishedPack(max_seq_len=self._max_seq_len)
        self._ref_chosen_scores.clear()
        self._ref_rejected_scores.clear()

    def next(self) -> PackedPreferenceExample:
        max_seq_len = self._max_seq_len
        while max(self._chosen_pack.length, self._rejected_pack.length) < max_seq_len:
            try:
                chosen, rejected, ref_score_chosen, ref_score_rejected = next(self._source)  # fmt: skip
            except StopIteration:
                break

            if self._chosen_pack.can_update(chosen) and self._rejected_pack.can_update(rejected):  # fmt: skip
                self._chosen_pack.update(chosen)
                self._rejected_pack.update(rejected)

                self._ref_chosen_scores.append(ref_score_chosen)
                self._ref_rejected_scores.append(ref_score_rejected)
            else:
                pack = self._finish_pack()

                # Make new packs
                self._clear_pack()

                self._chosen_pack.update(chosen)
                self._rejected_pack.update(rejected)
                self._ref_chosen_scores.append(ref_score_chosen)
                self._ref_rejected_scores.append(ref_score_rejected)

                return pack

        if 0 < max(self._chosen_pack.length, self._rejected_pack.length) <= max_seq_len:
            pack = self._finish_pack()
            self._clear_pack()
            return pack
        else:
            raise StopIteration()

    def get_state(self) -> dict[str, Any]:
        return {"source": self._source.state_dict()}

    def __repr__(self) -> str:
        return f"PreferencePacker(max_seq_len={self._max_seq_len}, pad_idx={self._pad_idx}) drop_on_reset={self._drop_on_reset})"


class BatchPreferencePacker(BaseNode[PackedPreferenceExample]):
    def __init__(
        self,
        source: BaseNode[PreferenceExample],
        *,
        batch_size: int,
        pad_idx: int,
        drop_last: bool = False,
        drop_on_reset: bool = False,
        multiple_of: int = 128,
    ) -> None:
        super().__init__()
        self._source = source
        self._batch_size = batch_size
        self._pad_idx = pad_idx
        self._drop_last = drop_last
        self._drop_on_reset = drop_on_reset
        self._multiple_of = multiple_of

        # Use a large max_seq_len since we are batching by item count, not length.
        self._chosen_pack = UnfinishedPack(max_seq_len=1 << 30)
        self._rejected_pack = UnfinishedPack(max_seq_len=1 << 30)
        self._ref_chosen_scores: list[float | None] = []
        self._ref_rejected_scores: list[float | None] = []

    def reset(self, initial_state: dict | None = None) -> None:
        super().reset(initial_state)

        if self._drop_on_reset:
            self._clear_pack()

        if initial_state is not None:
            self._source.reset(initial_state["source"])
        else:
            self._source.reset()

    def _clear_pack(self) -> None:
        self._chosen_pack = UnfinishedPack(max_seq_len=1 << 30)
        self._rejected_pack = UnfinishedPack(max_seq_len=1 << 30)
        self._ref_chosen_scores.clear()
        self._ref_rejected_scores.clear()

    def _finish_pack(self) -> PackedPreferenceExample:
        # Determine the final length by the longer of the two packs
        max_len = max(self._chosen_pack.length, self._rejected_pack.length)
        final_length = next_multiple(max_len, self._multiple_of)

        # Finish both packs to the same final length
        chosen_pack = self._chosen_pack.finish(
            pad_idx=self._pad_idx, max_seq_len=final_length
        )
        rejected_pack = self._rejected_pack.finish(
            pad_idx=self._pad_idx, max_seq_len=final_length
        )

        ref_chosen_scores = self._ref_chosen_scores[:]
        ref_rejected_scores = self._ref_rejected_scores[:]

        return chosen_pack, rejected_pack, ref_chosen_scores, ref_rejected_scores

    def next(self) -> PackedPreferenceExample:
        while self._chosen_pack.num_sequences < self._batch_size:
            try:
                chosen, rejected, ref_score_chosen, ref_score_rejected = next(self._source)  # fmt: skip
            except StopIteration:
                break

            self._chosen_pack.update(chosen)
            self._rejected_pack.update(rejected)
            self._ref_chosen_scores.append(ref_score_chosen)
            self._ref_rejected_scores.append(ref_score_rejected)

            if self._chosen_pack.num_sequences == self._batch_size:
                pack = self._finish_pack()
                self._clear_pack()
                return pack

        # Source is exhausted.
        if self._chosen_pack.num_sequences > 0 and not self._drop_last:
            pack = self._finish_pack()
            self._clear_pack()
            return pack
        else:
            # Clear the pack to ensure a clean state and raise StopIteration.
            self._clear_pack()
            raise StopIteration()

    def get_state(self) -> dict[str, Any]:
        return {"source": self._source.state_dict()}

    def __repr__(self) -> str:
        return (
            f"BatchPreferencePacker(batch_size={self._batch_size}, pad_idx={self._pad_idx}, "
            f"drop_on_reset={self._drop_on_reset}, drop_last={self._drop_last})"
        )
