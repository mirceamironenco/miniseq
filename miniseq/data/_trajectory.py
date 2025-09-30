from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from miniseq.data._batch import SequenceBatch
from miniseq.data._common import (
    PackedSequenceExample,
    SequenceExample,
    right_pad_collate,
)
from miniseq.data._packed import UnfinishedPack, UnfinishedPrefixPack
from miniseq.utils import next_multiple, to_tensor


@dataclass
class TrajectoryBatch:
    prompt_ids: list[list[int]]
    """len(prompt_ids) = bsz * mc_samples"""

    completion_ids: list[list[int]]
    """len(completion_ids) = bsz * mc_samples"""

    advantages: torch.Tensor
    """(bsz, mc_samples)"""

    rewards: torch.Tensor
    """(bsz, mc_samples)"""

    pad_idx: int

    rollout_logps: list[list[float]] | None = None

    packed: bool = False
    """Whether to produce packed SequenceBatch batches."""

    prefix_sharing: bool = False

    and_prepare: bool = False
    """Pre-compute the input and target sequence batches."""

    _input_batch: SequenceBatch | None = field(init=False, repr=False, default=None)
    """input_batch.seqs shape (bsz * n, seqlen)"""

    _target_batch: SequenceBatch | None = field(init=False, repr=False, default=None)
    """target_batch.seqs shape (bsz * n, seqlen)"""

    _old_lprobs: torch.Tensor | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        assert len(self.prompt_ids) == len(self.completion_ids)

        if self.rollout_logps is not None:
            assert len(self.rollout_logps) == len(self.completion_ids)

        # Ensure that the number of unique prompts is correct.
        # assert len(set(map(tuple, self.prompt_ids))) == self.batch_size

        # For now assume GRPO style rewards
        assert self.rewards.ndim == 2

        assert self.advantages.ndim == 2

        # Validate trajectory shapes
        assert len(self.prompt_ids) % self.mc_samples == 0

        assert len(self.prompt_ids) // self.mc_samples == self.batch_size

        assert self.rewards.size() == self.advantages.size()

        if self.prefix_sharing:
            if not self.packed:
                raise ValueError("Prefix sharing enabled only for packed batches.")

        if self.and_prepare:
            self.prepare_batch()

    def prepare_batch(self) -> None:
        self._input_batch, self._target_batch = self._maybe_make_ar_batch()

    def to(
        self, device: torch.device, *, non_blocking: bool = False
    ) -> TrajectoryBatch:
        self.advantages = self.advantages.to(device, non_blocking=non_blocking)

        self.rewards = self.rewards.to(device, non_blocking=non_blocking)

        if self._input_batch is not None:
            self._input_batch = self._input_batch.to(device, non_blocking=non_blocking)

        if self._target_batch is not None:
            self._target_batch = self._target_batch.to(
                device, non_blocking=non_blocking
            )

        if self._old_lprobs is not None:
            self._old_lprobs = self._old_lprobs.to(device, non_blocking=non_blocking)

        return self

    @property
    def batch_size(self) -> int:
        return self.rewards.size(0)

    @property
    def mc_samples(self) -> int:
        return self.rewards.size(1)

    def _prepare_seqs_batch_prefix_packed(self) -> SequenceBatch:
        share_count = self.mc_samples

        # max_length is updated later.
        pack = UnfinishedPrefixPack(max_seq_len=int(1 << 20), share_count=share_count)

        device = self.advantages.device

        examples: list[SequenceExample] = []

        for index, (prompt_ids, completion_ids) in enumerate(
            zip(self.prompt_ids, self.completion_ids), start=1
        ):
            prompt_pt = to_tensor(prompt_ids, dtype=torch.int64, device=device)

            completion_pt = to_tensor(completion_ids, dtype=torch.int64, device=device)

            examples.append(
                SequenceExample.from_instruction(
                    prompt=prompt_pt, completion=completion_pt, completion_only=True
                )
            )

            if index % share_count == 0:
                pack.maybe_update(examples)

                examples.clear()

        packed_example = pack.finish(
            pad_idx=self.pad_idx, max_seq_len=next_multiple(pack.length, 128)
        )

        return PackedSequenceExample.collate([packed_example], pad_idx=self.pad_idx)

    def _prepare_seqs_batch_packed(self) -> SequenceBatch:
        # max_length is updated later.
        pack = UnfinishedPack(max_seq_len=int(1 << 20))

        device = self.advantages.device

        for prompt_ids, completion_ids in zip(self.prompt_ids, self.completion_ids):
            prompt_pt = to_tensor(prompt_ids, dtype=torch.int64, device=device)

            completion_pt = to_tensor(completion_ids, dtype=torch.int64, device=device)

            pack.update(
                SequenceExample.from_instruction(
                    prompt=prompt_pt, completion=completion_pt, completion_only=True
                )
            )

        packed_example = pack.finish(
            pad_idx=self.pad_idx, max_seq_len=next_multiple(pack.length, 128)
        )

        return PackedSequenceExample.collate([packed_example], pad_idx=self.pad_idx)

    def _prepare_seqs_batch(self) -> SequenceBatch:
        device = self.advantages.device

        full_seqs: list[torch.Tensor] = []

        prompt_seq_lens: list[int] | torch.Tensor = []

        for prompt_ids, completion_ids in zip(self.prompt_ids, self.completion_ids):
            full_seqs.append(
                to_tensor(prompt_ids + completion_ids, dtype=torch.int64, device=device)
            )

            prompt_seq_lens.append(len(prompt_ids))

        prompt_seq_lens = to_tensor(prompt_seq_lens, device=device, dtype=torch.int64)

        sequence = right_pad_collate(full_seqs, pad_value=self.pad_idx)

        # Build the target mask
        batch_size, batch_width = sequence["seqs"].size()

        seq_lens = to_tensor(sequence["seq_lens"], dtype=torch.int64, device=device)

        seq_lens = seq_lens.unsqueeze(1).expand(-1, batch_width)
        prompt_seq_lens = prompt_seq_lens.unsqueeze(1).expand(-1, batch_width)

        # After prompt, before padding.
        _indices = torch.arange(batch_width, device=device).expand(batch_size, -1)

        target_mask = (_indices > prompt_seq_lens) & (_indices <= seq_lens)

        return SequenceBatch(
            sequence["seqs"],
            seq_lens=sequence["seq_lens"],
            target_mask=target_mask,
            is_packed=False,
            is_padded=sequence["is_ragged"],
            pad_idx=self.pad_idx,
        )

    def chunk(
        self, *, num_chunks: int, and_prepare: bool = False
    ) -> list[TrajectoryBatch]:
        assert self._input_batch is None

        assert self._target_batch is None

        assert self._old_lprobs is None

        if num_chunks == 1:
            if and_prepare:
                self.prepare_batch()

            return [self]

        if num_chunks > self.batch_size:
            raise ValueError(
                f"Too many chunks, got {num_chunks}, available {self.batch_size}."
            )

        if self.batch_size % num_chunks != 0:
            raise ValueError("For now, batch_size must be divisible by num_chunks.")

        chunk_advantages = torch.chunk(self.advantages, chunks=num_chunks, dim=0)

        chunk_rewards = torch.chunk(self.rewards, chunks=num_chunks, dim=0)

        chunked_trajectories: list[TrajectoryBatch] = []

        for index in range(num_chunks):
            size = chunk_advantages[index].size(0) * self.mc_samples

            rollout_logps = None

            if self.rollout_logps is not None:
                rollout_logps = self.rollout_logps[size * index : size * (index + 1)]

            chunked_trajectories.append(
                TrajectoryBatch(
                    prompt_ids=self.prompt_ids[size * index : size * (index + 1)],
                    completion_ids=self.completion_ids[
                        size * index : size * (index + 1)
                    ],
                    advantages=chunk_advantages[index],
                    rewards=chunk_rewards[index],
                    rollout_logps=rollout_logps,
                    pad_idx=self.pad_idx,
                    packed=self.packed,
                    prefix_sharing=self.prefix_sharing,
                    and_prepare=and_prepare,
                )
            )

        return chunked_trajectories

    def set_log_probs(self, lprobs: torch.Tensor) -> None:
        if self._old_lprobs is None:
            self._old_lprobs = lprobs

    def old_lprobs(self) -> torch.Tensor | None:
        return self._old_lprobs

    def maybe_rollout_lprobs(self, input_batch: SequenceBatch) -> torch.Tensor | None:
        if self.rollout_logps is None:
            return None

        lprobs = []

        for index, completion_lprobs in enumerate(self.rollout_logps):
            prompt_probs = [0.0] * len(self.prompt_ids[index])

            full_probs = prompt_probs + completion_lprobs

            if input_batch.uses_prefix_sharing and self.prefix_sharing:
                if index % self.mc_samples != 0:
                    full_probs = completion_lprobs

            lprobs.append(
                to_tensor(
                    full_probs,
                    dtype=torch.float32,
                    device=input_batch.seqs.device,
                )
            )

        if input_batch.is_packed:
            lprobs_pt = torch.cat(lprobs, dim=0)

            to_pad = input_batch.seqs.size(1) - lprobs_pt.size(-1)

            # + 1 and slice to cover all cases.
            lprobs_pt = F.pad(lprobs_pt, pad=(0, to_pad + 1), value=0.0)

            return lprobs_pt[None, :-1]

        lprobs_pt = right_pad_collate(lprobs, pad_value=0.0)["seqs"]

        return lprobs_pt[..., :-1]

    def _maybe_make_ar_batch(self) -> tuple[SequenceBatch, SequenceBatch]:
        if self._input_batch is None or self._target_batch is None:
            if self.packed:
                if self.prefix_sharing:
                    batch = self._prepare_seqs_batch_prefix_packed()
                else:
                    batch = self._prepare_seqs_batch_packed()
            else:
                batch = self._prepare_seqs_batch()

            self._input_batch, self._target_batch = batch.as_auto_regressive()

        return self._input_batch, self._target_batch

    def auto_regressive_input(self) -> tuple[SequenceBatch, SequenceBatch]:
        return self._maybe_make_ar_batch()
