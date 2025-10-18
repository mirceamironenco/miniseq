from __future__ import annotations

import logging
import typing
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    TypeVar,
    is_typeddict,
    overload,
)

from rich.pretty import pretty_repr

HFItemT = TypeVar("HFItemT", bound=Mapping)


if TYPE_CHECKING:
    from datasets import Dataset

    class HFDataset(Dataset, Generic[HFItemT]):
        """At runtime, this class is a no-op, but helps with type info preservation."""

        # fmt: off
        @overload
        def __getitem__(self, key: int | slice | Iterable[int]) -> HFItemT: ...

        @overload
        def __getitem__(self, key: str) -> list: ...

        def __getitem__(self, key: int | slice | Iterable[int] | str) -> HFItemT | list: ... # type: ignore

        def __len__(self) -> int: ...
        # fmt: on

        def select(
            self,
            indices: Iterable,
            keep_in_memory: bool = False,
            indices_cache_file_name: str | None = None,
            writer_batch_size: int | None = 1000,
            new_fingerprint: str | None = None,
        ) -> HFDataset[HFItemT]: ...
else:
    # HFDataset is not used at runtime, yet we still need a subscriptable type
    # for typing.cast(HFDataset[...], dataset_obj)
    HFDataset = list


OutT = TypeVar("OutT", bound=Mapping)


@overload
def process_dataset(
    dataset: HFDataset[HFItemT],
    *,
    function: Callable[[HFItemT], dict[str, Any]],
    num_proc: int | None = None,
    disable_nullable: bool = False,
    description: str | None = None,
) -> HFDataset[HFItemT]: ...


@overload
def process_dataset(
    dataset: HFDataset[HFItemT],
    *,
    function: Callable[[HFItemT], OutT],
    num_proc: int | None = None,
    disable_nullable: bool = False,
    description: str | None = None,
) -> HFDataset[OutT]: ...


def process_dataset(
    dataset: HFDataset[HFItemT] | HFDataset[Mapping[str, Any]],
    *,
    function: Callable[[HFItemT], OutT],
    num_proc: int | None = None,
    disable_nullable: bool = False,
    description: str | None = None,
) -> HFDataset[OutT] | HFDataset[HFItemT]:
    """A simplified (and less capable) version of dataset.map which preserves type info."""

    out = dataset.map(
        function,
        with_indices=False,
        with_rank=False,
        batched=False,
        remove_columns=None,
        keep_in_memory=False,
        disable_nullable=disable_nullable,
        num_proc=num_proc,
        desc=description,
    )

    return out  # type: ignore


def filter_dataset(
    dataset: HFDataset[HFItemT],
    *,
    function: Callable[[HFItemT], bool],
    num_proc: int | None = None,
) -> HFDataset[HFItemT]:
    out = dataset.filter(
        function,
        batched=False,
        with_indices=False,
        with_rank=False,
        keep_in_memory=False,
        num_proc=num_proc,
    )

    return out  # type: ignore


def load_hf_map_dataset(
    path: str,
    *,
    cache_dir: Path,
    name: str | None = None,
    data_dir: str | None = None,
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
    split: str | None = None,
    test_split: str | None = None,
    test_split_ratio: float = 0.0,
    seed: int = 0,
    filter_map: Callable[[HFItemT], bool] | None = None,
    kls: type[HFItemT] = dict[str, Any],
) -> dict[str, HFDataset[HFItemT]]:
    """Equivalent to datasets.load_dataset(...).  Additionally checks that:

    - We are working with a non-iterable Dataset object.
    - if `kls` is provided and is a TypedDict, validates the item schema.
    """

    if split is None:
        if test_split is not None:
            raise ValueError("Must specify either both split and test_split or None.")

    if test_split is not None and test_split_ratio > 0.0:
        raise ValueError(
            "Either `test_split` is provided or test_split_ratio > 0.0, but not both."
        )

    # Lazy-load since this has significant import time
    from datasets import Dataset, DatasetDict, load_dataset

    if Path(path).is_dir():
        dataset = Dataset.load_from_disk(path)
    else:
        dataset = load_dataset(
            path,
            name=name,
            data_dir=data_dir,
            data_files=data_files,
            cache_dir=str(cache_dir),
            split=split if test_split is None else None,
            streaming=False,
        )

        assert isinstance(dataset, (Dataset, DatasetDict))

    train_dataset: Dataset
    test_dataset: Dataset | None = None

    if isinstance(dataset, DatasetDict):
        # If the dataset has more than 1 split on the hub, user is expected
        # to specify it explicitly.
        if len(dataset.keys()) == 1:
            split = next(iter(dataset.keys()))

            train_dataset = dataset[split]
        elif split is None:
            splits = ", ".join(dataset.keys())

            raise ValueError(
                "No dataset split provided. "
                " Provide `split` (and optionally `test_split`) to load HF dataset."
                f"Available splits for {path} are: {splits}."
            )
        else:
            assert split is not None and test_split is not None

            train_dataset = dataset[split]

            test_dataset = dataset[test_split]
    else:
        train_dataset = dataset

    if test_split_ratio > 0.0:
        assert test_dataset is None

        test_size = int(len(train_dataset) * test_split_ratio)

        split_dataset = train_dataset.train_test_split(
            test_size=test_size, seed=seed, shuffle=True
        )

        train_dataset, test_dataset = split_dataset["train"], split_dataset["test"]

    if not isinstance(train_dataset, Dataset):
        raise ValueError(
            f"Dataset {path} is not a map-style dataset, it is of type {type(train_dataset)}."
        )

    train_item = train_dataset[0]

    if not isinstance(train_item, dict):
        raise ValueError(f"Map dataset items must be dicts, got {type(train_item)}.")

    # Soft runtime check to see if dict keys match, in case kls is TypedDict.
    if is_typeddict(kls):
        _req_keys = kls.__required_keys__  # type: ignore

        # Required keys must be present; dataset item can contain additional keys.
        if not _req_keys <= (item_keys := frozenset(train_item.keys())):
            raise ValueError(
                f"HF dataset {path} item schema does not match provided dict type. "
                f"Mismatched keys: {_req_keys - item_keys}"
            )

    train_dataset = typing.cast(HFDataset[HFItemT], train_dataset)

    if test_dataset is not None:
        test_dataset = typing.cast(HFDataset[HFItemT], test_dataset)

    if filter_map is not None:
        train_dataset = filter_dataset(train_dataset, function=filter_map)

        if test_dataset is not None:
            test_dataset = filter_dataset(test_dataset, function=filter_map)

    if test_dataset is not None:
        return {"train": train_dataset, "test": test_dataset}

    return {"train": train_dataset}


def log_dataset(log: logging.Logger, dataset: HFDataset) -> None:
    log.info(
        f"Dataset(length = {len(dataset)}):\n {pretty_repr(dataset.info, max_width=88)}"
    )
