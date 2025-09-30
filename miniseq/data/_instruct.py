from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, NotRequired, Protocol, TypedDict, TypeVar

import torch
from typing_extensions import override

from miniseq.data._batch import SequenceBatch
from miniseq.data._common import PackedSequenceExample, SequenceExample
from miniseq.data._packed import BatchPacker, Packer
from miniseq.data._pipeline import PipelineBuilder
from miniseq.data._tokenizer import PretrainedHFTokenizer, make_chat_prefix
from miniseq.data._utils import MapDataset
from miniseq.machine import Machine


class InstructDataset(ABC):
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
    ) -> Iterable[SequenceBatch]: ...

    @abstractmethod
    def splits(self) -> list[str]: ...


class InstructionDict(TypedDict):
    system: NotRequired[str]
    instruction: str
    input: NotRequired[str]
    completion: str


MapItemT_contra = TypeVar("MapItemT_contra", bound=Mapping, contravariant=True)


class InstructionTransform(Protocol[MapItemT_contra]):
    def __call__(
        self,
        example: MapItemT_contra,
        *,
        tokenizer: PretrainedHFTokenizer | None = None,
    ) -> InstructionDict: ...


class ColumnMapperTransform(InstructionTransform[dict[str, Any]]):
    """A generic column mapper which can be specified using an InstructDict."""

    def __init__(
        self, columns: InstructionDict, *, system_prompt: str | None = None
    ) -> None:
        self._columns = columns
        self._system_prompt = system_prompt

    @override
    def __call__(
        self, example: dict[str, Any], *, tokenizer: PretrainedHFTokenizer | None = None
    ) -> InstructionDict:
        instruction: InstructionDict = {
            "instruction": example[self._columns["instruction"]],
            "completion": example[self._columns["completion"]],
        }

        # If the system prompt is specified as a column, we prioritize it
        # over the passed in system prompt.
        if (system_column := self._columns.get("system")) is not None:
            if (system_msg := example.get(system_column)) is not None:
                instruction["system"] = system_msg
        elif self._system_prompt is not None:
            instruction["system"] = self._system_prompt

        if (input_column := self._columns.get("input")) is not None:
            if (input_msg := example.get(input_column)) is not None:
                instruction["input"] = input_msg

        return instruction


def truncate_prompt_completion(
    prompt_completion: tuple[torch.Tensor, torch.Tensor], *, max_seqlen: int
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_pt, completion_pt = prompt_completion

    if max_seqlen is not None and (len(prompt_pt) + len(completion_pt)) > max_seqlen:
        # By default, try to truncate the prompt and fit the entire completion.
        # Allow for at least 1 token in the prompt.
        if len(completion_pt) < max_seqlen:
            prompt_pt = prompt_pt[: max_seqlen - len(completion_pt)]
        else:
            # Otherwise, fit the prompt and as much as possible from the completion.
            # Allow for at least 1 token in the completion.
            prompt_len = min(len(prompt_pt), max_seqlen - 1)
            completion_len = max_seqlen - prompt_len

            prompt_pt = prompt_pt[:prompt_len]
            completion_pt = completion_pt[:completion_len]

    return prompt_pt, completion_pt


def template_prompt_completion(
    instruction: InstructionDict, *, tokenizer: PretrainedHFTokenizer
) -> tuple[str, str]:
    user_prompt = instruction["instruction"]

    if (input_str := instruction.get("input")) is not None and len(input_str) > 0:
        user_prompt = "\n".join([user_prompt, input_str])

    prompt_msg = make_chat_prefix(
        user_message=user_prompt,
        system_message=instruction.get("system"),
    )

    prompt = tokenizer.apply_chat_template(
        prompt_msg,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    )

    prompt_completion_msg = make_chat_prefix(
        user_message=user_prompt,
        assistant_message=instruction["completion"],
        system_message=instruction.get("system"),
    )

    prompt_completion = tokenizer.apply_chat_template(
        prompt_completion_msg,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    )

    if not prompt_completion.startswith(prompt):
        raise ValueError(
            "tokenizer apply_chat_template not compatible with instruction dataset."
        )

    completion = prompt_completion[len(prompt) :]

    return prompt, completion


def build_prompt_completion(instruction: InstructionDict) -> tuple[str, str]:
    """Builds prompt and completion strings from an InstructionDict."""

    prompt_parts = []

    if (system_msg := instruction.get("system")) is not None:
        prompt_parts.append(system_msg)

    prompt_parts.append(instruction["instruction"])

    if (input_prompt := instruction.get("input")) is not None:
        prompt_parts.append(input_prompt)

    prompt = "".join(prompt_parts)

    completion = instruction["completion"]

    return prompt, completion


MapItemT = TypeVar("MapItemT", bound=Mapping)


class GenericInstructDataset(InstructDataset):
    _data: Mapping[str, MapDataset]
    _instruct_transform: InstructionTransform
    _completions_only: bool
    _packed_seqlen: int | None
    _apply_chat_template: bool
    _max_seqlen: int | None

    def __init__(
        self,
        dataset: Mapping[str, MapDataset[MapItemT]],
        *,
        instruct_transform: InstructionTransform[MapItemT],
        completions_only: bool = True,
        packed_seqlen: int | None = None,
        apply_chat_template: bool = False,
        max_seqlen: int | None = None,
    ) -> None:
        self._data = dataset
        self._instruct_transform = instruct_transform
        self._completions_only = completions_only
        self._packed_seqlen = packed_seqlen
        self._apply_chat_template = apply_chat_template
        self._max_seqlen = max_seqlen

        if max_seqlen is not None and max_seqlen <= 0:
            raise ValueError(f"max_seqlen must be postiive, got {max_seqlen}.")

    def __getitem__(self, index: int, *, split: str = "train") -> Mapping:
        return self._data[split][index]

    def __len__(self) -> int:
        return len(self._data)

    def get_split(self, split: str) -> MapDataset[Mapping]:
        return self._data[split]

    def get_instruction(
        self,
        index: int,
        *,
        split: str = "train",
        tokenizer: PretrainedHFTokenizer | None = None,
    ) -> InstructionDict:
        return self._instruct_transform(self._data[split][index], tokenizer=tokenizer)

    @staticmethod
    def make_sequence_example(
        prompt_completion: tuple[str, str],
        *,
        tokenizer: PretrainedHFTokenizer,
        max_seqlen: int | None,
        completions_only: bool = True,
    ) -> SequenceExample:
        prompt, completion = prompt_completion

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        completion_ids = tokenizer.encode(completion, add_special_tokens=False)

        prompt_pt = torch.tensor(prompt_ids, dtype=torch.int64)

        completion_pt = torch.tensor(completion_ids, dtype=torch.int64)

        if max_seqlen is not None:
            prompt_pt, completion_pt = truncate_prompt_completion(
                (prompt_pt, completion_pt), max_seqlen=max_seqlen
            )

        return SequenceExample.from_instruction(
            prompt=prompt_pt,
            completion=completion_pt,
            completion_only=completions_only,
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
    ) -> Iterable[SequenceBatch]:
        assert tokenizer.pad_token_id is not None

        # Note: seed should be identical on all ranks here.
        builder = PipelineBuilder.from_map_dataset(
            self.get_split(split),
            shuffle=not for_evaluation,
            rank=machine.rank,
            world_size=machine.size,
            seed=seed,
            allow_uneven=for_evaluation,
        )

        # Dataset item -> InstructionDict
        builder = builder.map(
            functools.partial(self._instruct_transform, tokenizer=tokenizer),
            num_parallel=npc,
        )

        # InstructionDict -> (prompt: str, completion: str) tuple to be tokenized.
        if self._apply_chat_template:
            prompt_completion_map = functools.partial(
                template_prompt_completion, tokenizer=tokenizer
            )
        else:
            prompt_completion_map = build_prompt_completion

        builder = builder.map(prompt_completion_map, num_parallel=npc)

        # (prompt: str, completion: str) -> SequenceExample
        make_sequence_example = functools.partial(
            self.make_sequence_example,
            tokenizer=tokenizer,
            max_seqlen=self._max_seqlen,
            completions_only=self._completions_only,
        )

        builder = builder.map(make_sequence_example, num_parallel=npc)

        pad_token_id = tokenizer.pad_token_id

        if packed:
            # packed_seqlen takes priority over batch_size
            if self._packed_seqlen is not None:
                builder = builder.to_node(
                    Packer(
                        builder.as_node(),
                        max_seq_len=self._packed_seqlen,
                        pad_idx=pad_token_id,
                    )
                )
            else:
                builder = builder.to_node(
                    BatchPacker(
                        builder.as_node(),
                        batch_size=batch_size,
                        drop_last=False,
                        pad_idx=pad_token_id,
                        multiple_of=128,
                    )
                )

            # packed batches have only 1 packed sequence, shape is (1, packed_seqlen)
            builder = builder.batch(batch_size=1, drop_last=False)

            builder = builder.collate(
                fn=functools.partial(
                    PackedSequenceExample.collate, pad_idx=pad_token_id
                ),
                num_parallel=npc,
            )
        else:
            builder = builder.batch(batch_size=batch_size, drop_last=False)
            builder = builder.collate(
                fn=functools.partial(SequenceExample.collate, pad_idx=pad_token_id),
                num_parallel=npc,
            )

        if torch.cuda.is_available():
            builder = builder.pin_memory()

        builder = builder.prefetch(prefetch_factor=8)

        loader = builder.as_loader()

        builder.print_pipeline(do_print=machine.rank == 0)

        return loader

    @override
    def splits(self) -> list[str]:
        return list(self._data.keys())
