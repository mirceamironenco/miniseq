import typing
from typing import Any

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
)

from miniseq.transformer import TransformerDecoderLayer, TransformerDecoderModel


def apply_ac(
    model: nn.Module, *, every_nth_layer: int = 1, preserve_rng_state: bool = True
) -> None:
    if not isinstance(model, TransformerDecoderModel):
        raise ValueError(
            "Activaton checkpointing only available for Transformer models."
        )

    model = typing.cast(TransformerDecoderModel, model)

    for layer_index, (layer_name, layer) in enumerate(
        model.decoder.layers.named_children()
    ):
        assert isinstance(layer, TransformerDecoderLayer)

        if layer_index % every_nth_layer == 0:
            wrapper = CheckpointWrapper(
                layer,
                CheckpointImpl.NO_REENTRANT,
                preserve_rng_state=preserve_rng_state,
            )

            model.decoder.layers.register_module(layer_name, wrapper)


def maybe_unwrap_ac_checkpoint(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        k.replace("._checkpoint_wrapped_module", ""): v for (k, v) in state_dict.items()
    }
