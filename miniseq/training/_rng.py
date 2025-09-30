import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import torch


def device_to_generator(device: torch.device) -> torch.Generator:
    if device.type not in ("cuda", "cpu"):
        raise ValueError(
            f"device type {device.type} not compatible with torch.generator"
        )

    if device.type == "cpu":
        return torch.default_generator

    index = device.index

    if index is None:
        index = torch.cuda.current_device()

    return torch.cuda.default_generators[index]


def unique_generators(*devices: torch.device) -> list[torch.Generator]:
    seen = set()
    generators = []
    for generator in map(device_to_generator, devices):
        if generator not in seen:
            seen.add(generator)
            generators.append(generator)
    return generators


def set_seed(seed: int, *devices: torch.device) -> None:
    assert seed >= 0 and seed < 1 << 32

    generators = unique_generators(*devices)

    for g in generators:
        g.manual_seed(seed)


@contextmanager
def manual_seed(seed: int, *devices: torch.device) -> Iterator[None]:
    assert seed >= 0 and seed < 1 << 32

    generators = unique_generators(*devices)

    states = []

    for g in generators:
        states.append(g.get_state())

        g.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        yield
    finally:
        for generator, original_state in zip(generators, states):
            generator.set_state(original_state)


@contextmanager
def manual_state(
    new_states: Sequence[torch.Tensor], devices: Sequence[torch.device]
) -> Iterator[None]:
    assert len(new_states) > 0
    assert len(new_states) == len(devices)

    generators = unique_generators(*devices)

    states = []

    for generator, new_state in zip(generators, new_states):
        states.append(generator.get_state())

        generator.set_state(new_state)

    try:
        yield
    finally:
        for generator, original_state in zip(generators, states):
            generator.set_state(original_state)
