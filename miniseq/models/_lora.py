from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from rich.table import Table
from typing_extensions import override

from miniseq.logging import get_console
from miniseq.machine import Machine, all_ranks_same_trainable_params
from miniseq.nn import Module
from miniseq.utils import count_trainable_params


@tyro.conf.configure(tyro.conf.subcommand("on"))
@dataclass(frozen=True, kw_only=True)
class LoRAConfig:
    r: int = 64
    alpha: float = 128.0
    dropout_p: float = 0.0
    keys: list[str] = field(
        default_factory=lambda: [
            ".*decoder.layers.*.self_attn.*(q_proj|v_proj|o_proj)$"
        ]
    )


class LoRALayer(ABC):
    def __init__(self, config: LoRAConfig) -> None:
        self.r = config.r
        self.alpha = config.alpha
        self.scaling = self.alpha / self.r
        self.dropout_p = config.dropout_p

    @property
    @abstractmethod
    def wrapped_module(self) -> nn.Module: ...

    @abstractmethod
    def merge(self) -> None: ...

    @abstractmethod
    def unmerge(self) -> None: ...

    @abstractmethod
    def merged_parameters(self) -> Iterator[tuple[str, nn.Parameter]]: ...


class LoRAEmbedding(Module[torch.Tensor, [torch.Tensor]], LoRALayer):
    wrapped: nn.Embedding
    weight: nn.Parameter
    lora_A: nn.Parameter
    lora_B: nn.Parameter
    merged: bool

    def __init__(
        self,
        wrapped: nn.Embedding,
        config: LoRAConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        Module.__init__(self)
        LoRALayer.__init__(self, config)

        self.wrapped = wrapped
        self.wrapped.weight.requires_grad_(False)
        self.num_embeddings = wrapped.num_embeddings
        self.embedding_dim = wrapped.embedding_dim

        assert isinstance(self.wrapped.weight, nn.Parameter)

        self.weight = self.wrapped.weight

        self.lora_A = nn.Parameter(
            torch.empty((self.r, self.num_embeddings), device=device, dtype=dtype)
        )

        self.lora_B = nn.Parameter(
            torch.empty((self.embedding_dim, self.r), device=device, dtype=dtype)
        )

        self.merged = False

        self.reset_lora_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.embedding(x, self.weight)

        return F.embedding(
            x, self.weight + (self.lora_B @ self.lora_A).T * self.scaling
        )

    def reset_lora_parameters(self) -> None:
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)

    @property
    @override
    def wrapped_module(self) -> nn.Embedding:
        return self.wrapped

    @override
    def merge(self) -> None:
        if self.merged:
            return

        with torch.no_grad():
            self.weight += (self.lora_B @ self.lora_A).T * self.scaling  # type: ignore

        self.merged = True

    @override
    def unmerge(self) -> None:
        if not self.merged:
            return

        with torch.no_grad():
            self.weight -= (self.lora_B @ self.lora_A).T * self.scaling  # type: ignore

        self.merged = False

    @override
    def merged_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        with torch.no_grad():
            merged_weight = self.weight + (self.lora_B @ self.lora_A).T * self.scaling

        yield "weight", nn.Parameter(merged_weight)


class LoRALinear(Module[torch.Tensor, [torch.Tensor]], LoRALayer):
    wrapped: nn.Linear
    weight: nn.Parameter
    bias: nn.Parameter | None
    lora_A: nn.Parameter
    lora_B: nn.Parameter
    dropout: nn.Dropout | None
    skip_init: bool
    merged: bool

    def __init__(
        self,
        wrapped: nn.Linear,
        config: LoRAConfig,
        skip_init: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        Module.__init__(self)
        LoRALayer.__init__(self, config)

        self.wrapped = wrapped
        self.in_features, self.out_features = wrapped.in_features, wrapped.out_features
        self.wrapped.weight.requires_grad_(False)

        assert isinstance(self.wrapped.weight, nn.Parameter)

        self.weight = self.wrapped.weight

        if self.wrapped.bias is not None:
            self.wrapped.bias.requires_grad_(False)
            self.bias = self.wrapped.bias
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(
            torch.empty((self.r, self.in_features), device=device, dtype=dtype)
        )

        self.lora_B = nn.Parameter(
            torch.empty((self.out_features, self.r), device=device, dtype=dtype)
        )

        if self.dropout_p > 0.0:
            self.dropout = nn.Dropout(self.dropout_p)
        else:
            self.register_module("dropout", None)

        self.merged = False

        self.skip_init = skip_init

        self.reset_lora_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        h1 = F.linear(x, self.weight, self.bias)

        if self.dropout is not None:
            x = self.dropout(x)

        h2 = F.linear(x, self.lora_B @ self.lora_A * self.scaling)

        return h1 + h2

    def reset_lora_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if not self.skip_init:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @property
    @override
    def wrapped_module(self) -> nn.Linear:
        return self.wrapped

    @override
    def merge(self) -> None:
        if self.merged:
            return

        with torch.no_grad():
            self.weight += self.lora_B @ self.lora_A * self.scaling  # type: ignore

        self.merged = True

    @override
    def unmerge(self) -> None:
        if not self.merged:
            return

        with torch.no_grad():
            self.weight -= self.lora_B @ self.lora_A * self.scaling  # type: ignore

        self.merged = False

    @override
    def merged_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        with torch.no_grad():
            merged_weight = self.weight + self.lora_B @ self.lora_A * self.scaling

        yield "weight", nn.Parameter(merged_weight)

        if self.bias is not None:
            yield "bias", self.bias


def wrap_lora(
    module: nn.Module, config: LoRAConfig, skip_init: bool = False
) -> nn.Module:
    found_module = False
    for name, submodule in module.named_modules():
        if not _is_target_module(name, config.keys):
            continue

        submodule_path = name.split(".")
        parent = module.get_submodule(".".join(submodule_path[:-1]))
        submodule_name = submodule_path[-1]

        lora_layer: LoRALayer | None = None

        if isinstance(submodule, nn.Linear):
            lora_layer = LoRALinear(
                wrapped=submodule,
                config=config,
                skip_init=skip_init,
                device=submodule.weight.device,
                dtype=submodule.weight.dtype,
            )
        elif isinstance(submodule, nn.Embedding):
            lora_layer = LoRAEmbedding(
                wrapped=submodule,
                config=config,
                device=submodule.weight.device,
                dtype=submodule.weight.dtype,
            )
        else:
            raise ValueError(
                f"Cannot wrap module {name} with lora as module type {type(submodule)} is not supported."
            )

        lora_layer.train(mode=submodule.training)
        setattr(parent, submodule_name, lora_layer)
        found_module = True

    if not found_module:
        raise ValueError(f"Did not wrap any modules with LoRA, for keys: {config.keys}")

    return module


def unwrap_lora(module: nn.Module, merge: bool = True) -> nn.Module:
    if merge:
        merge_lora(module)

    for name, submodule in module.named_modules():
        if not isinstance(submodule, LoRALayer):
            continue

        submodule_path = name.split(".")
        parent = module.get_submodule(".".join(submodule_path[:-1]))
        submodule_name = submodule_path[-1]

        unwrapped_layer: nn.Module | None = None
        if isinstance(submodule, LoRALayer):
            unwrapped_layer = submodule.wrapped_module
        else:
            raise ValueError(
                f"Cannot unwrap the module '{name}' as the module type '{type(submodule).__name__}' is not supported."
            )

        setattr(parent, submodule_name, unwrapped_layer)

    return module


def merge_lora(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, LoRALayer):
            submodule.merge()


def unmerge_lora(module: nn.Module) -> None:
    for submodule in module.modules():
        if isinstance(submodule, LoRALayer):
            submodule.unmerge()


def lora_state_dict(module: nn.Module) -> dict[str, object]:
    lora_names = []
    for name, submodule in module.named_modules():
        if isinstance(submodule, LoRALayer):
            lora_names.append(name)

    state_dict = module.state_dict()
    lora_states = {name: state_dict[name] for name in lora_names}
    return lora_states


def freeze_non_lora_params(
    module: nn.Module, unfreeze_bias: Literal["none", "all", "lora_only"] = "none"
) -> None:
    for submodule in module.modules():
        if isinstance(submodule, LoRALayer):
            for param_name, param in submodule.named_parameters(recurse=False):
                if param_name in ["lora_A", "lora_B"]:
                    param.requires_grad = True
                elif param_name == "bias" and unfreeze_bias in ["all", "lora_only"]:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param_name, param in submodule.named_parameters(recurse=False):
                if param_name == "bias" and unfreeze_bias == "all":
                    param.requires_grad = True
                else:
                    param.requires_grad = False


def lora_wrap_model(
    model: nn.Module,
    lora_cfg: LoRAConfig,
    machine: Machine,
    freeze_non_lora: bool = True,
    log_model: bool = True,
) -> nn.Module:
    # Save model repr and # of trainable params in case we want to log them.
    pre_wrap_model_str = str(model)
    pre_freeze_params = count_trainable_params(model)

    # wrap LoRA
    model = wrap_lora(model, lora_cfg, skip_init=not machine.rank == 0)

    if freeze_non_lora:
        freeze_non_lora_params(model)

    if not all_ranks_same_trainable_params(model, machine):
        raise ValueError(
            "Found inconsistent # of trainable params between ranks after LoRA wrap."
        )

    if machine.rank == 0 and log_model:
        post_freeze_params = count_trainable_params(model)

        CONSOLE = get_console()

        post_wrap_model_str = str(model)

        table = Table("Original model", "LoRA model")
        table.add_row(
            CONSOLE.render_str(pre_wrap_model_str),
            CONSOLE.render_str(post_wrap_model_str),
        )
        table.add_section()
        table.add_row(
            CONSOLE.render_str(f"# Trainable params: {pre_freeze_params:,}"),
            CONSOLE.render_str(f"# Trainable params: {post_freeze_params:,}"),
        )
        CONSOLE.print(table)
    return model


def _is_target_module(name: str, target_keys: list[str]) -> bool:
    return any(name == key or re.match(key, name) for key in target_keys)
