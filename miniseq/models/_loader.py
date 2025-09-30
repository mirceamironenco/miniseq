import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from miniseq.logging import get_logger
from miniseq.machine import Machine
from miniseq.models._builder import ModelConfig, build_model
from miniseq.models._download import download_checkpoint
from miniseq.models._registry import get_from_hf_ckpt_converter, get_model_family
from miniseq.transformer import TransformerDecoderModel

_log = get_logger()


def convert_model_state_dict(
    state_dict: dict[str, Any], key_map: Mapping[str, str]
) -> dict[str, Any]:
    new_state_dict = {}

    for old_key in state_dict.keys():
        replacement_key = old_key

        for old_pattern, replacement in key_map.items():
            if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                replacement_key = new_key
                break

        new_state_dict[replacement_key] = state_dict[old_key]

    return new_state_dict


def load_model_hf_checkpoint(
    *,
    repo_id: str,
    cache_dir: Path,
    machine: Machine,
) -> dict[str, Any]:
    model_name = repo_id.split("/")[-1]

    output_dir = cache_dir / f"{model_name}"

    if machine.rank == 0:
        download_checkpoint(
            repo_id,
            output_dir=output_dir,
            force=False,
            ignore_patterns="original/consolidated*",
        )

    machine.barrier()

    assert output_dir.exists(), f"Failed to find model state dict at {output_dir}."

    _log.info(f"Loading checkpoint from {str(output_dir)}.")

    try:
        from safetensors import safe_open
    except ImportError:
        raise RuntimeError("Safetensors not found. Use `pip install safetensors`.")

    files = list(output_dir.glob("*.safetensors"))

    if not files:
        raise RuntimeWarning(f"No safetensors files found at {output_dir}.")

    state_dict = {}

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    return state_dict


def convert_hf_sd_to_mini(
    state_dict: dict[str, Any], config: ModelConfig
) -> dict[str, Any]:
    family = get_model_family(config.model)

    ckpt_converter = get_from_hf_ckpt_converter(family)

    state_dict = ckpt_converter(state_dict, config)

    return state_dict


def load_model_checkpoint(
    config: ModelConfig, *, cache_dir: Path, machine: Machine
) -> dict[str, Any]:
    state_dict = load_model_hf_checkpoint(
        repo_id=config.repo_id,
        cache_dir=cache_dir,
        machine=machine,
    )

    state_dict = convert_hf_sd_to_mini(state_dict, config=config)

    return state_dict


def load_model_and_sd(
    config: ModelConfig,
    *,
    device: torch.device,
    machine: Machine,
    cache_dir: Path,
    dtype: torch.dtype,
) -> tuple[TransformerDecoderModel, dict[str, Any]]:
    model = build_model(config, device=device, dtype=dtype)

    state_dict = load_model_hf_checkpoint(
        repo_id=config.repo_id,
        cache_dir=cache_dir,
        machine=machine,
    )

    state_dict = convert_hf_sd_to_mini(state_dict, config=config)

    return model, state_dict
