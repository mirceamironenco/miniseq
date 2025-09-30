from pathlib import Path
from typing import TypeVar

import torch
import yaml

from miniseq.builder_config import DataclassInstance
from miniseq.utils import make_dtype


def dtype_representer(dumper: yaml.Dumper, data: torch.dtype) -> yaml.ScalarNode:
    dtype_string = str(data).split(".")[1]

    try:
        make_dtype(dtype_string)  # type: ignore
    except ValueError as ex:
        raise ValueError(
            f"Invalid torch dtype for yaml serializtion: {dtype_string}."
        ) from ex

    return dumper.represent_scalar("!torch.dtype", dtype_string)


def dtype_constructor(loader: yaml.Loader, node: yaml.ScalarNode) -> torch.dtype:
    return make_dtype(node.value)


yaml.add_representer(torch.dtype, dtype_representer)
yaml.add_constructor("!torch.dtype", dtype_constructor, Loader=yaml.Loader)  # type: ignore


def save_config(path: Path, config: DataclassInstance, name: str = "config") -> Path:
    path.mkdir(exist_ok=True, parents=True)

    config_path = path / f"{name}.yaml"

    config_path.write_text(yaml.dump(config), "utf8")

    return config_path


DataclassT = TypeVar("DataclassT", bound=DataclassInstance)


def load_config(path: Path, *, kls: type[DataclassT]) -> DataclassT:
    config = yaml.load(path.read_text(), Loader=yaml.Loader)

    if not isinstance(config, kls):
        raise ValueError(f"YAML config loading failed, loaded config is of type {kls}.")

    return config
