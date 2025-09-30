from unittest.mock import MagicMock

import torch
import torch.nn as nn

from miniseq.models._lora import (
    LoRAConfig,
    lora_wrap_model,
)
from miniseq.models._utils import merged_named_parameters


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_lora_merged_named_parameters_keys() -> None:
    lora_config = LoRAConfig(
        r=4,
        alpha=8.0,
        keys=[".*linear"],
    )
    model = SimpleModel()
    machine = MagicMock()
    machine.size = 1
    machine.rank = 0
    machine.device = torch.device("cpu")

    lora_model = lora_wrap_model(
        model=SimpleModel(),
        lora_cfg=lora_config,
        machine=machine,
        log_model=False,
    )

    model_named_params = list(model.named_parameters())

    lora_model_named_params = merged_named_parameters(lora_model)

    assert sorted([key for key, _ in model_named_params]) == sorted(
        [key for key, _ in lora_model_named_params]
    )
