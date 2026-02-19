from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Generic, TypeVar

import torch

TSelf = TypeVar("TSelf", bound="Metric[Any]")
TComputeReturn = TypeVar("TComputeReturn")
TState = torch.Tensor | list[torch.Tensor] | dict[Any, torch.Tensor] | int | float


class Metric(Generic[TComputeReturn], ABC):
    _state_name_to_default: dict[str, TState]
    _device: torch.device

    def __init__(self: TSelf, *, device: torch.device | None = None) -> None:
        self._state_name_to_default = {}
        self._device = torch.device("cpu") if device is None else device

    def _add_state(self: TSelf, name: str, default: TState) -> None:
        _check_state_variable_type(name, default)
        setattr(self, name, deepcopy(default))
        self._state_name_to_default[name] = deepcopy(default)

    @abstractmethod
    @torch.inference_mode()
    def update(self: TSelf, *_: Any, **__: Any) -> TSelf: ...

    @abstractmethod
    @torch.inference_mode()
    def compute(self: TSelf) -> TComputeReturn: ...

    @abstractmethod
    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: list[TSelf]) -> TSelf: ...

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TSelf) -> None:
        pass

    def reset(self: TSelf) -> TSelf:
        for state_name, default in self._state_name_to_default.items():
            setattr(self, state_name, _clone_to_device(default, self.device))
        return self

    def state_dict(self: TSelf) -> dict[str, TState]:
        state_dict: dict[str, TState] = {}

        for state_name in self._state_name_to_default:
            value = getattr(self, state_name)
            _check_state_variable_type(state_name, value)

            if isinstance(value, torch.Tensor):
                state_dict[state_name] = value.detach().clone()
            elif isinstance(value, list):
                state_dict[state_name] = [tensor.detach().clone() for tensor in value]
            elif isinstance(value, dict):
                state_dict[state_name] = {
                    key: tensor.detach().clone() for key, tensor in value.items()
                }
            elif isinstance(value, (int, float)):
                state_dict[state_name] = value
            else:
                raise TypeError(f"Unsupported state type for {state_name}: {type(value)}")

        return state_dict

    def load_state_dict(
        self: TSelf, state_dict: dict[str, Any], strict: bool = True
    ) -> None:
        metric_state_names = set(self._state_name_to_default.keys())
        incoming = deepcopy(state_dict)

        for state_name in metric_state_names:
            if state_name in incoming:
                value = incoming[state_name]
                _check_state_variable_type(state_name, value)
                setattr(self, state_name, _clone_to_device(value, self.device))

        if strict:
            incoming_names = set(incoming.keys())
            unexpected_keys = incoming_names.difference(metric_state_names)
            missing_keys = metric_state_names.difference(incoming_names)
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}. "
                    f"Encountered missing keys: {missing_keys} and unexpected "
                    f"keys: {unexpected_keys}."
                )

    def to(self: TSelf, device: str | torch.device, *_: Any, **__: Any) -> TSelf:
        device = torch.device(device) if isinstance(device, str) else device

        for state_name in self._state_name_to_default:
            value = getattr(self, state_name)
            _check_state_variable_type(state_name, value)
            setattr(self, state_name, _move_to_device(value, device))

        self._device = device
        return self

    @property
    def device(self: TSelf) -> torch.device:
        return self._device


def _clone_to_device(value: TState, device: torch.device) -> TState:
    if isinstance(value, torch.Tensor):
        return value.clone().to(device)
    if isinstance(value, list):
        return [tensor.clone().to(device) for tensor in value]
    if isinstance(value, dict):
        return {key: tensor.clone().to(device) for key, tensor in value.items()}
    return deepcopy(value)


def _move_to_device(value: TState, device: torch.device) -> TState:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [tensor.to(device) for tensor in value]
    if isinstance(value, dict):
        return {key: tensor.to(device) for key, tensor in value.items()}
    return value


def _check_state_variable_type(name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, list) and all(isinstance(x, torch.Tensor) for x in value):
        return
    if isinstance(value, dict) and all(
        isinstance(x, torch.Tensor) for x in value.values()
    ):
        return
    if isinstance(value, (int, float)):
        return

    raise TypeError(
        "The value of state variable must be a torch.Tensor, list[torch.Tensor], "
        f"dict[Any, torch.Tensor], int, or float. Got {name}={value}."
    )
