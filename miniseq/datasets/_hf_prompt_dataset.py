from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from typing_extensions import is_typeddict, override

from miniseq.data import (
    CompletionScorer,
    HFDataset,
    PipelineBuilder,
    PretrainedHFTokenizer,
    PromptBatch,
    PromptDataset,
    load_hf_map_dataset,
    log_dataset,
)
from miniseq.datasets._prompt_builder import (
    ChatTemplatePromptBuilder,
    ConcatenatePromptBuilder,
    PromptBuilder,
    maybe_map_to_callable,
)
from miniseq.datasets._scorer import VerificationScorer
from miniseq.datasets._verifiers import Verifier
from miniseq.logging import get_logger
from miniseq.machine import Machine

_log = get_logger()

HFItemT = TypeVar("HFItemT", bound=Mapping)


@dataclass(kw_only=True)
class HFPromptDatasetSpec(Generic[HFItemT]):
    path: str
    configuration: str | None
    prompt_keymap: str | Callable[[HFItemT], str]
    answer_keymap: str | Callable[[HFItemT], str]
    system_message: str | None = None
    assistant_message: str | None = None
    prompt_transform: Callable[[str], str] | None = None
    schema: type[HFItemT]
    apply_chat_template: bool = True
    verifier: Verifier
    extras: dict[str, Any] = field(default_factory=dict)


HF_PROMPT_DATASET_REGISTRY: dict[str, HFPromptDatasetSpec] = {}


def all_registered_datasets(newest_first: bool = True) -> list[str]:
    datasets = list(HF_PROMPT_DATASET_REGISTRY.keys())

    if newest_first:
        return datasets[::-1]

    return datasets


def register_prompt_dataset(
    dataset_name: str,
    *,
    path: str,
    prompt_keymap: str | Callable[[HFItemT], str],
    answer_keymap: str | Callable[[HFItemT], str],
    configuration: str | None = None,
    system_message: str | None = None,
    assistant_message: str | None = None,
    prompt_transform: Callable[[str], str] | None = None,
    apply_chat_template: bool = True,
    verifier: Verifier,
    schema: type[HFItemT] = dict[str, Any],
    **kwargs,
) -> None:
    """Registers a HuggingFace prompt dataset."""

    if dataset_name in HF_PROMPT_DATASET_REGISTRY:
        raise ValueError(
            f"HuggingFace dataset with name {dataset_name} already registered."
        )

    keys_to_check: list[str] = [
        key for key in (prompt_keymap, answer_keymap) if isinstance(key, str)
    ]

    if keys_to_check and is_typeddict(schema):
        _required_keys = schema.__required_keys__  # type: ignore
        if _required_keys:
            for item_key in keys_to_check:
                if item_key not in _required_keys:
                    raise ValueError(
                        f"Could not register dataset '{dataset_name}'; '{item_key}' not a required key. "
                        f"Specified schema has required keys: {_required_keys}."
                    )

    HF_PROMPT_DATASET_REGISTRY[dataset_name] = HFPromptDatasetSpec(
        path=path,
        configuration=configuration,
        prompt_keymap=prompt_keymap,
        answer_keymap=answer_keymap,
        system_message=system_message,
        assistant_message=assistant_message,
        prompt_transform=prompt_transform,
        apply_chat_template=apply_chat_template,
        schema=schema,
        verifier=verifier,
        extras=kwargs,
    )


def build_from_registry(
    name: str,
    cache_dir: Path,
    *,
    apply_chat_template: bool | None = None,
    system_message: str | None = None,
    assistant_message: str | None = None,
    prompt_transform: Callable[[str], str] | None = None,
) -> HFPromptDataset:
    spec: HFPromptDatasetSpec = HF_PROMPT_DATASET_REGISTRY[name]

    # Allowed overrides which can be model-dependent.
    overrides = dict(
        apply_chat_template=apply_chat_template,
        system_message=system_message,
        assistant_message=assistant_message,
        prompt_transform=prompt_transform,
    )

    overrides = {key: value for key, value in overrides.items() if value is not None}

    if overrides:
        spec = replace(spec, **overrides)

    if spec.apply_chat_template:
        prompt_builder = ChatTemplatePromptBuilder(
            prompt_keymap=spec.prompt_keymap,
            system_message=spec.system_message,
            assistant_message=spec.assistant_message,
            prompt_transform=spec.prompt_transform,
        )
    else:
        prompt_builder = ConcatenatePromptBuilder(
            prompt_keymap=spec.prompt_keymap,
            system_message=system_message,
            assistant_message=spec.assistant_message,
            prompt_transform=spec.prompt_transform,
        )

    # https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
    add_special_tokens = not spec.apply_chat_template

    return HFPromptDataset.from_hugging_face(
        path=spec.path,
        cache_dir=cache_dir,
        prompt_builder=prompt_builder,
        answer_keymap=spec.answer_keymap,
        name=spec.configuration,
        kls=spec.schema,
        add_special_tokens=add_special_tokens,
        verifier=spec.verifier,
        **spec.extras,
    )


class HFPromptDataset(PromptDataset[HFItemT]):
    _dataset: dict[str, HFDataset[HFItemT]]
    _prompt_builder: PromptBuilder[HFItemT]
    _answer_keymap: Callable[[HFItemT], str]
    _verifier: Verifier
    _add_special_tokens: bool

    def __init__(
        self,
        data: dict[str, HFDataset[HFItemT]],
        *,
        prompt_builder: PromptBuilder[HFItemT],
        answer_keymap: str | Callable[[HFItemT], str],
        verifier: Verifier,
        add_special_tokens: bool = True,
    ) -> None:
        self._dataset = data
        self._prompt_builder = prompt_builder
        self._answer_keymap = maybe_map_to_callable(answer_keymap)
        self._verifier = verifier
        self._add_special_tokens = add_special_tokens

    @classmethod
    def from_hugging_face(
        cls,
        *,
        path: str,
        cache_dir: Path,
        prompt_builder: PromptBuilder[HFItemT],
        answer_keymap: str | Callable[[HFItemT], str],
        verifier: Verifier,
        seed: int = 0,
        split: str | None = None,
        test_split: str | None = None,
        test_split_ratio: float = 0.0,
        name: str | None = None,
        kls: type[HFItemT] = dict[str, Any],
        add_special_tokens: bool = True,
        log_ds: bool = False,
        **kwargs,
    ) -> HFPromptDataset[HFItemT]:
        dataset = load_hf_map_dataset(
            path,
            cache_dir=cache_dir,
            name=name,
            split=split,
            test_split=test_split,
            test_split_ratio=test_split_ratio,
            seed=seed,
            kls=kls,
            **kwargs,
        )

        if log_ds:
            log_dataset(_log, dataset["train"])

        return HFPromptDataset(
            dataset,
            prompt_builder=prompt_builder,
            answer_keymap=answer_keymap,
            verifier=verifier,
            add_special_tokens=add_special_tokens,
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
        npc: int = 2,
    ) -> Iterable[PromptBatch[HFItemT]]:
        # Note: seed should be identical on all ranks here.
        builder = PipelineBuilder.from_map_dataset(
            self._dataset[split],
            shuffle=not for_evaluation,
            rank=machine.rank,
            world_size=machine.size,
            seed=seed,
            allow_uneven=for_evaluation,
        )

        prompt_map = functools.partial(self._prompt_builder, tokenizer=tokenizer)

        builder = builder.map(fn=prompt_map, num_parallel=npc, method="thread")

        builder = builder.batch(batch_size=batch_size, drop_last=False)

        add_special_tokens = self._add_special_tokens

        def _collate_map(items: list[tuple[HFItemT, str]]) -> PromptBatch[HFItemT]:
            examples, prompts = zip(*items, strict=True)

            encoding = tokenizer(
                text=prompts, padding=False, add_special_tokens=add_special_tokens
            )

            input_ids: list[list[int]] = encoding["input_ids"]

            return PromptBatch(
                prompt_ids=input_ids, batch_extras=examples, prompt_strs=prompts
            )

        builder = builder.collate(fn=_collate_map, num_parallel=npc, method="thread")

        builder = builder.prefetch(prefetch_factor=8)

        builder.print_pipeline(do_print=machine.rank == 0)

        return builder.as_loader()

    @override
    def create_scorer(
        self, *, tokenizer: PretrainedHFTokenizer, **kwargs: Any
    ) -> CompletionScorer[HFItemT]:
        return VerificationScorer(
            tokenizer, self._answer_keymap, verifier=self._verifier
        )

    @override
    def splits(self) -> list[str]:
        return list(self._dataset.keys())
