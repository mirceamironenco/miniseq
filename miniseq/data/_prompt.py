from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import torch

from miniseq.data._tokenizer import PretrainedHFTokenizer
from miniseq.machine import Machine
from miniseq.metric_bag import MetricBag

T = TypeVar("T")


@dataclass(kw_only=True)
class PromptBatch(Generic[T]):
    """Batch holding a list of prompts."""

    prompt_ids: list[list[int]]
    """Tokenized input prompts, bsz = len(prompt_ids)"""

    prompt_ids_pt: list[torch.Tensor] | None = None
    """Tokenized input prompts as 1-dimensional tensors."""

    batch_extras: Sequence[T]
    """Typically the list of dataset items used to construct the batch. 
    Should be of equal length to `prompt_ids`."""

    prompt_strs: Sequence[str] | None = None
    """The processed prompts which were tokenized to obtain `prompt_ids`."""

    def __post_init__(self) -> None:
        if self.prompt_ids_pt is not None:
            if not (pt_lengths := len(self.prompt_ids_pt)) == self.batch_size:
                raise ValueError(
                    f"prompt_input_ids_pt has length ({pt_lengths}) != input_ids ({self.batch_size})."
                )

        if not (extras_length := len(self.batch_extras)) == len(self.prompt_ids):
            raise ValueError(
                f"Batch extras has length ({extras_length}) != input_ids ({self.batch_size})."
            )

        if self.prompt_strs is not None:
            if not (strs_length := len(self.prompt_strs)) == len(self.prompt_ids):
                raise ValueError(
                    f"`prompt_strs` length must be the same as prompt_ids, got {strs_length}"
                )

    @property
    def batch_size(self) -> int:
        return len(self.prompt_ids)

    def pin_memory(self) -> PromptBatch:
        if self.prompt_ids_pt is not None:
            self.prompt_ids_pt = list(map(lambda x: x.pin_memory(), self.prompt_ids_pt))

        return self

    def to(self, device: torch.device, *, non_blocking: bool = False) -> PromptBatch:
        if self.prompt_ids_pt is not None:
            self.prompt_ids_pt = list(
                map(
                    lambda x: x.to(device, non_blocking=non_blocking),
                    self.prompt_ids_pt,
                )
            )

        return self


class CompletionScorer(Protocol[T]):
    def __call__(
        self,
        *,
        completions: list[list[int]],
        batch: PromptBatch[T],
        repetitions: int = 1,
        metric_bag: MetricBag | None = None,
    ) -> list[float]: ...


InfoT = TypeVar("InfoT", covariant=True)


class PromptDataset(ABC, Generic[InfoT]):
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
        npc: int = 2,
    ) -> Iterable[PromptBatch[InfoT]]: ...

    @abstractmethod
    def create_scorer(
        self, *, tokenizer: PretrainedHFTokenizer, **kwargs: Any
    ) -> CompletionScorer[InfoT]: ...

    @abstractmethod
    def splits(self) -> list[str]: ...
