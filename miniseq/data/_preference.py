from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import NotRequired, Protocol, TypedDict, TypeVar

import torch
from typing_extensions import override

from miniseq.data._batch import SequenceBatch
from miniseq.data._common import (
    PackedPreferenceExample,
    PackedSequenceExample,
    PreferenceExample,
    SequenceExample,
)
from miniseq.data._packed import PreferencePakcer, BatchPreferencePacker
from miniseq.data._pipeline import PipelineBuilder
from miniseq.data._tokenizer import PretrainedHFTokenizer
from miniseq.data._utils import MapDataset
from miniseq.machine import Machine


@dataclass(kw_only=True)
class PreferenceBatch:
    chosen_batch: SequenceBatch
    rejected_batch: SequenceBatch
    ref_chosen: torch.Tensor | None
    ref_rejected: torch.Tensor | None

    def __post_init__(self) -> None:
        # Batches have to be either both pakced or both unpacked.
        assert not (self.chosen_batch.is_packed ^ self.rejected_batch.is_packed)

        # Reference scores must be provided for both or neither.
        assert not ((self.ref_chosen is None) ^ (self.ref_rejected is None))

        score_chosen, score_rejected = self.ref_chosen, self.ref_rejected

        if score_chosen is not None and score_rejected is not None:
            if not (score_chosen.ndim == score_rejected.ndim == 1):
                raise ValueError("Reference scores must be 1-dimensional.")

            if self.chosen_batch.is_packed:
                if score_chosen.size(0) != self.chosen_batch.seqlens_pt.size(0):
                    raise ValueError(
                        "Number of chosen reference scores must match the number of packed chosen sequences."
                    )

                if score_rejected.size(0) != self.rejected_batch.seqlens_pt.size(0):
                    raise ValueError(
                        "Number of rejected reference scores must match the number of packed rejected sequences."
                    )
            else:
                if score_chosen.size(0) != self.chosen_batch.seqs.size(0):
                    raise ValueError(
                        "Number of chosen reference scores must match the number of unpacked chosen sequences."
                    )

                if score_rejected.size(0) != self.rejected_batch.seqs.size(0):
                    raise ValueError(
                        "Number of rejected reference scores must match the number of unpacked rejected sequences."
                    )

    @property
    def is_packed(self) -> bool:
        return self.chosen_batch.is_packed and self.rejected_batch.is_packed

    @property
    def batch_size(self) -> int:
        return self.chosen_batch.batch_size

    def to(self, device: torch.device, *, non_blocking: bool = True) -> PreferenceBatch:
        self.chosen_batch.to(device, non_blocking=non_blocking)

        self.rejected_batch.to(device, non_blocking=non_blocking)

        if self.ref_chosen is not None:
            self.ref_chosen = self.ref_chosen.to(device, non_blocking=non_blocking)

        if self.ref_rejected is not None:
            self.ref_rejected = self.ref_rejected.to(device, non_blocking=non_blocking)

        return self

    def pin_memory(self) -> PreferenceBatch:
        self.chosen_batch.pin_memory()

        self.rejected_batch.pin_memory()

        if self.ref_chosen is not None:
            self.ref_chosen.pin_memory()

        if self.ref_rejected is not None:
            self.ref_rejected.pin_memory()

        return self


class PreferenceDataset(ABC):
    @abstractmethod
    def create_loader(
        self,
        *,
        tokenizer: PretrainedHFTokenizer,
        machine: Machine,
        batch_size: int,
        seed: int,
        split: str = "train",
        for_evaluation: bool = False,
        packed: bool = False,
        npc: int = 2,
    ) -> Iterable[PreferenceBatch]: ...


class PreferenceDict(TypedDict):
    prompt: str
    chosen: str
    rejected: str
    reference_score_chosen: NotRequired[float]
    reference_score_rejected: NotRequired[float]


MapItemT_contra = TypeVar("MapItemT_contra", bound=Mapping, contravariant=True)
MapItemT = TypeVar("MapItemT", bound=Mapping)


class PreferenceTransform(Protocol[MapItemT_contra]):
    def __call__(
        self,
        example: MapItemT_contra,
        *,
        tokenizer: PretrainedHFTokenizer | None = None,
    ) -> PreferenceDict: ...


def collate_packed(
    items: list[PackedPreferenceExample], *, pad_idx: int
) -> PreferenceBatch:
    if len(items) != 1:
        raise ValueError(
            f"packed sequences are expected to have batch_size=1, got {len(items)}"
        )

    chosen, reject, chosen_ref_scores, rejected_ref_scores = zip(*items, strict=True)

    batch_chosen = PackedSequenceExample.collate(chosen, pad_idx=pad_idx)
    batch_reject = PackedSequenceExample.collate(reject, pad_idx=pad_idx)

    ref_score_chosen: torch.Tensor | None = None
    ref_score_rejected: torch.Tensor | None = None

    chosen_ref_scores = list(filter(None, chosen_ref_scores[0]))
    if chosen_ref_scores:
        ref_score_chosen = torch.tensor(chosen_ref_scores).float()

    rejected_ref_scores = list(filter(None, rejected_ref_scores[0]))
    if rejected_ref_scores:
        ref_score_rejected = torch.tensor(rejected_ref_scores).float()

    return PreferenceBatch(
        chosen_batch=batch_chosen,
        rejected_batch=batch_reject,
        ref_chosen=ref_score_chosen,
        ref_rejected=ref_score_rejected,
    )


def collate(items: list[PreferenceExample], *, pad_idx: int) -> PreferenceBatch:
    chosen, rejected, chosen_ref_scores, rejected_ref_scores = zip(*items, strict=True)

    batch_chosen = SequenceExample.collate(chosen, pad_idx=pad_idx)
    batch_rejected = SequenceExample.collate(rejected, pad_idx=pad_idx)

    ref_score_chosen: torch.Tensor | None = None
    ref_score_rejected: torch.Tensor | None = None

    chosen_ref_scores = list(filter(None, chosen_ref_scores))
    if chosen_ref_scores:
        ref_score_chosen = torch.tensor(chosen_ref_scores).float()

    rejected_ref_scores = list(filter(None, rejected_ref_scores))
    if rejected_ref_scores:
        ref_score_rejected = torch.tensor(rejected_ref_scores).float()

    return PreferenceBatch(
        chosen_batch=batch_chosen,
        rejected_batch=batch_rejected,
        ref_chosen=ref_score_chosen,
        ref_rejected=ref_score_rejected,
    )


class GenericPreferenceDataset(PreferenceDataset):
    _data: Mapping[str, MapDataset]
    _preference_transform: PreferenceTransform
    _mask_source_tokens: bool
    _packed_seqlen: int | None
    _apply_chat_template: bool
    _max_seqlen: int | None

    def __init__(
        self,
        dataset: Mapping[str, MapDataset[MapItemT]],
        *,
        preference_transform: PreferenceTransform[MapItemT],
        mask_source_tokens: bool = True,
        packed_seqlen: int | None = None,
        apply_chat_template: bool = False,
        max_seqlen: int | None = None,
    ) -> None:
        self._data = dataset
        self._preference_transform = preference_transform
        self._mask_source_tokens = mask_source_tokens
        self._packed_seqlen = packed_seqlen
        self._apply_chat_template = apply_chat_template
        self._max_seqlen = max_seqlen

        if max_seqlen is not None and max_seqlen <= 0:
            raise ValueError(f"max_seqlen must be postiive, got {max_seqlen}.")

    def __getitem__(self, index: int, *, split: str = "train") -> Mapping:
        return self._data["train"][index]

    def __len__(self) -> int:
        return len(self._data)

    def get_split(self, split: str) -> MapDataset[Mapping]:
        return self._data[split]

    def get_preference(
        self,
        index: int,
        *,
        split: str = "train",
        tokenizer: PretrainedHFTokenizer | None = None,
    ) -> PreferenceDict:
        return self._preference_transform(self._data[split][index], tokenizer=tokenizer)

    @staticmethod
    def make_preference_example(
        example: PreferenceDict,
        *,
        tokenizer: PretrainedHFTokenizer,
        mask_source_tokens: bool = True,
        max_seqlen: int | None = None,
    ) -> PreferenceExample:
        prompt = tokenizer.encode(example["prompt"], add_special_tokens=False)

        chosen = tokenizer.encode(example["chosen"], add_special_tokens=False)

        rejected = tokenizer.encode(example["rejected"], add_special_tokens=False)

        if max_seqlen is not None:
            # Fit the prompt and as much as possible from the completion.
            # Allow for at least 1 token in the completion.
            # Ensure prompt is always the same for both.
            prompt_len = len(prompt)
            new_prompt_len = min(prompt_len, max_seqlen - 1)
            completion_len = max_seqlen - new_prompt_len

            if (prompt_len + len(chosen)) > max_seqlen:
                prompt = prompt[:new_prompt_len]
                chosen = chosen[:completion_len]

            if (prompt_len + len(rejected)) > max_seqlen:
                prompt = prompt[:new_prompt_len]
                rejected = rejected[:completion_len]

        chosen_ex = SequenceExample.from_instruction(
            prompt=prompt, completion=chosen, completion_only=mask_source_tokens
        )

        rejected_ex = SequenceExample.from_instruction(
            prompt=prompt, completion=rejected, completion_only=mask_source_tokens
        )

        return (
            chosen_ex,
            rejected_ex,
            example.get("reference_score_chosen"),
            example.get("reference_score_rejected"),
        )

    @override
    def create_loader(
        self,
        *,
        tokenizer: PretrainedHFTokenizer,
        machine: Machine,
        batch_size: int,
        seed: int,
        split: str = "train",
        for_evaluation: bool = False,
        packed: bool = False,
        npc: int = 2,
    ) -> Iterable[PreferenceBatch]:
        # Note: seed should be identical on all ranks here.
        builder = PipelineBuilder.from_map_dataset(
            self.get_split(split),
            shuffle=not for_evaluation,
            rank=machine.rank,
            world_size=machine.size,
            seed=seed,
            allow_uneven=for_evaluation,
        )

        preference_transform = functools.partial(
            self._preference_transform, tokenizer=tokenizer
        )

        # Dataset item -> PreferenceDict
        builder = builder.map(preference_transform, num_parallel=npc)

        make_preference_example = functools.partial(
            self.make_preference_example,
            tokenizer=tokenizer,
            mask_source_tokens=self._mask_source_tokens,
            max_seqlen=self._max_seqlen,
        )

        # PreferenceDict -> PreferenceExample
        builder = builder.map(
            make_preference_example, num_parallel=npc, method="thread"
        )

        assert tokenizer.pad_token_id is not None

        pad_index = tokenizer.pad_token_id

        if packed:
            if self._packed_seqlen is not None:
                builder = builder.to_node(
                    PreferencePakcer(
                        builder.as_node(),
                        max_seq_len=self._packed_seqlen,
                        pad_idx=pad_index,
                    )
                )
            else:
                builder = builder.to_node(
                    BatchPreferencePacker(
                        builder.as_node(),
                        batch_size=batch_size,
                        drop_last=False,
                        pad_idx=pad_index,
                        multiple_of=128,
                    )
                )

            # packed batches have only 1 packed sequence, shape is (1, packed_seqlen)
            builder = builder.batch(batch_size=1, drop_last=False)

            builder = builder.collate(
                functools.partial(collate_packed, pad_idx=pad_index),
                num_parallel=npc,
                method="thread",
            )
        else:
            builder = builder.batch(batch_size=batch_size, drop_last=False)

            builder = builder.collate(
                functools.partial(collate, pad_idx=pad_index),
                num_parallel=npc,
                method="thread",
            )

        if torch.cuda.is_available():
            builder = builder.pin_memory()

        builder = builder.prefetch(prefetch_factor=8)

        loader = builder.as_loader()

        builder.print_pipeline(do_print=machine.rank == 0)

        return loader
