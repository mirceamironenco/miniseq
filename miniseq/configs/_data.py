from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Generic

import tyro
from typing_extensions import TypeVar, override

from miniseq.builder_config import BuilderConfig
from miniseq.data import (
    ColumnMapperTransform,
    GenericInstructDataset,
    GenericPreferenceDataset,
    InstructionDict,
    InstructionTransform,
    PreferenceDict,
    PreferenceTransform,
    PretrainedHFTokenizer,
    PromptDataset,
    load_hf_map_dataset,
)
from miniseq.datasets import (
    ChainedVerifier,
    ChatTemplatePromptBuilder,
    ConcatenatePromptBuilder,
    HFPromptDataset,
    Verifier,
    all_registered_datasets,
    build_from_registry,
)

MapItemT = TypeVar("MapItemT", bound=Mapping, default=Mapping, covariant=True)


@dataclass(kw_only=True, frozen=True)
class SFTDatasetConfig(BuilderConfig[GenericInstructDataset], Generic[MapItemT]):
    name: str
    configuration: str | None = None
    data_files: str | None = None
    split: str | None = None
    test_split: str | None = None
    test_split_ratio: float = 0.0

    completions_only: bool = True
    packed_seqlen: int | None = None
    apply_chat_template: bool = False
    max_seqlen: int | None = None
    system_prompt: Annotated[str | None, tyro.conf.Suppress] = None

    instruct_transform: InstructionTransform | None = None
    instruct_map: Callable[[MapItemT], InstructionDict] | None = None

    columns: tuple[str, str, str | None, str | None] | None = None
    """instruction, completion, input, system"""

    @override
    def build(
        self,
        *,
        cache_dir: Path,
        seed: int,
        **kwd_overrides: Any,
    ) -> GenericInstructDataset:
        split, data_files = self.split, self.data_files

        if kwd_overrides:
            assert "name" not in kwd_overrides

            if "split" in kwd_overrides:
                split = kwd_overrides["split"]

            if "data_files" in kwd_overrides:
                data_files = kwd_overrides["data_files"]

        hf_data = load_hf_map_dataset(
            path=self.name,
            name=self.configuration,
            cache_dir=cache_dir,
            data_files=data_files,
            split=split,
            test_split=self.test_split,
            test_split_ratio=self.test_split_ratio,
            seed=seed,
        )

        if self.instruct_transform is not None:
            instruct_transform = self.instruct_transform

            if self.instruct_map is not None or self.columns is not None:
                raise ValueError(
                    "Must specify only one of instruct_transform, instruct_map or columns"
                )
        elif self.instruct_map is not None:
            instruct_map = self.instruct_map

            def _transform(
                example: MapItemT, *, tokenizer: PretrainedHFTokenizer | None = None
            ) -> InstructionDict:
                return instruct_map(example)

            instruct_transform = _transform

        else:
            if self.columns is None:
                raise ValueError(
                    "Either instruct_transform, instruct_map or column_map must be set."
                )

            instruction, completion, input, system = self.columns

            columns: InstructionDict = {
                "instruction": instruction,
                "completion": completion,
            }

            if input is not None:
                columns["input"] = input

            if system is not None:
                columns["system"] = system

            instruct_transform = ColumnMapperTransform(
                columns, system_prompt=self.system_prompt
            )

        dataset_kwargs: dict[str, Any] = dict(
            completions_only=self.completions_only,
            packed_seqlen=self.packed_seqlen,
            apply_chat_template=self.apply_chat_template,
            max_seqlen=self.max_seqlen,
        )

        for key, value in kwd_overrides.items():
            if key in dataset_kwargs:
                dataset_kwargs[key] = value

        return GenericInstructDataset(
            hf_data, instruct_transform=instruct_transform, **dataset_kwargs
        )


@dataclass(kw_only=True, frozen=True)
class PreferenceDatasetConfig(
    BuilderConfig[GenericPreferenceDataset], Generic[MapItemT]
):
    name: str
    split: str | None = None
    test_split: str | None = None
    test_split_ratio: float = 0.0
    data_files: str | None = None
    mask_source_tokens: bool = True
    packed_seqlen: int | None = None
    apply_chat_template: bool = False
    max_seqlen: int | None = None

    preference_transform: PreferenceTransform | None = None
    preference_map: Callable[[MapItemT], PreferenceDict] | None = None

    @override
    def build(
        self, *, cache_dir: Path, seed: int, **kwd_overrides: Any
    ) -> GenericPreferenceDataset:
        assert not kwd_overrides

        preference_transform = self.preference_transform
        if preference_transform is None:
            if self.preference_map is None:
                raise ValueError(
                    "Either preference_transform or preference_map must be specified."
                )

            preference_map = self.preference_map

            def _transform(
                example: MapItemT, *, tokenizer: PretrainedHFTokenizer | None = None
            ) -> PreferenceDict:
                return preference_map(example)

            preference_transform = _transform
        else:
            if self.preference_map is not None:
                raise ValueError(
                    "Either preference_transform or preference_map must be specified but both."
                )

        hf_data = load_hf_map_dataset(
            path=self.name,
            cache_dir=cache_dir,
            data_files=self.data_files,
            split=self.split,
            test_split=self.test_split,
            test_split_ratio=self.test_split_ratio,
            seed=seed,
        )

        return GenericPreferenceDataset(
            hf_data,
            preference_transform=preference_transform,
            mask_source_tokens=self.mask_source_tokens,
            packed_seqlen=self.packed_seqlen,
            apply_chat_template=self.apply_chat_template,
            max_seqlen=self.max_seqlen,
        )


@dataclass(kw_only=True, frozen=True)
class PromptDatasetConfig(BuilderConfig[PromptDataset], Generic[MapItemT]):
    path: str
    prompt_keymap: str | Callable[[MapItemT], str]
    answer_keymap: str | Callable[[MapItemT], str]
    configuration: str | None = None
    split: str | None = None
    test_split: str | None = None
    test_split_ratio: float = 0.0
    data_files: str | None = None
    data_dir: str | None = None
    filter_map: Callable[[MapItemT], bool] | None = None
    system_message: Annotated[str | None, tyro.conf.Suppress] = None
    assistant_message: Annotated[str | None, tyro.conf.Suppress] = None
    prompt_transform: Callable[[str], str] | None = None
    apply_chat_template: bool = True

    verifiers: Verifier | list[Verifier] | None = None
    verifier_factory: Callable[..., Verifier | list[Verifier]] | None = None

    @override
    def build(
        self, *, cache_dir: Path, seed: int, **kwd_overrides: Any
    ) -> PromptDataset:
        assert not kwd_overrides

        if self.apply_chat_template:
            prompt_builder = ChatTemplatePromptBuilder(
                prompt_keymap=self.prompt_keymap,
                system_message=self.system_message,
                assistant_message=self.assistant_message,
                prompt_transform=self.prompt_transform,
            )
        else:
            prompt_builder = ConcatenatePromptBuilder(
                prompt_keymap=self.prompt_keymap,
                system_message=self.system_message,
                assistant_message=self.assistant_message,
                prompt_transform=self.prompt_transform,
            )

        verifier, verifier_factory = self.verifiers, self.verifier_factory

        if verifier is None:
            if verifier_factory is None:
                raise ValueError("Either verifiers or verifier_factory must be set.")

            verifier = verifier_factory()

        if isinstance(verifier, list):
            verifier = ChainedVerifier(*verifier)

        return HFPromptDataset.from_hugging_face(
            path=self.path,
            cache_dir=cache_dir,
            prompt_builder=prompt_builder,
            answer_keymap=self.answer_keymap,
            verifier=verifier,
            name=self.configuration,
            split=self.split,
            test_split=self.test_split,
            test_split_ratio=self.test_split_ratio,
            seed=seed,
            data_files=self.data_files,
            data_dir=self.data_dir,
            filter_map=self.filter_map,
            log_ds=True,
        )


RegisteredDatasetNames = Annotated[
    tuple[str, ...],
    tyro.conf.arg(
        constructor_factory=lambda: tuple[
            tyro.extras.literal_type_from_choices(all_registered_datasets()), ...
        ]
    ),
]


@dataclass(frozen=False, kw_only=True)
class RegisteredDatasetConfig(BuilderConfig[list[PromptDataset]]):
    datasets: RegisteredDatasetNames
    batch_size: int = 64

    # Model-dependent options which can be used to override registered dataset spec.
    apply_chat_template: Annotated[bool | None, tyro.conf.Suppress] = None
    system_message: Annotated[str | None, tyro.conf.Suppress] = None
    assistant_message: Annotated[str | None, tyro.conf.Suppress] = None
    prompt_transform: Annotated[Callable[[str], str] | None, tyro.conf.Suppress] = None

    @override
    def build(
        self,
        *,
        cache_dir: Path,
        **kwd_overrides: Any,
    ) -> list[PromptDataset]:
        assert not kwd_overrides

        return [
            build_from_registry(
                dataset,
                cache_dir,
                apply_chat_template=self.apply_chat_template,
                system_message=self.system_message,
                assistant_message=self.assistant_message,
                prompt_transform=self.prompt_transform,
            )
            for dataset in self.datasets
        ]
