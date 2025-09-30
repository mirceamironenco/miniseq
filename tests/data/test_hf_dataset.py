from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest
from datasets import Dataset

from miniseq.data._hf_dataset import load_hf_map_dataset


class MyTypedDict(TypedDict):
    text: str


def test_load_hf_map_dataset_schema_validation(tmp_path: Path):
    # 1. Create a dummy dataset with a schema that matches MyTypedDict.
    matching_data = [{"text": "hello"}, {"text": "world"}]
    matching_dataset = Dataset.from_list(matching_data)
    matching_dataset.save_to_disk(str(tmp_path / "matching_dataset"))

    # 2. This should pass without raising an exception.
    load_hf_map_dataset(
        str(tmp_path / "matching_dataset"),
        cache_dir=tmp_path,
        kls=MyTypedDict,
    )

    # 3. Create a dummy dataset with a schema that does NOT match MyTypedDict.
    mismatching_data = [{"wrong_key": "hello"}, {"wrong_key": "world"}]
    mismatching_dataset = Dataset.from_list(mismatching_data)
    mismatching_dataset.save_to_disk(str(tmp_path / "mismatching_dataset"))

    # 4. This should raise a ValueError because the keys do not match.
    with pytest.raises(ValueError, match=r"Mismatched keys: frozenset\({'text'}\)"):
        load_hf_map_dataset(
            str(tmp_path / "mismatching_dataset"),
            cache_dir=tmp_path,
            kls=MyTypedDict,
        )
