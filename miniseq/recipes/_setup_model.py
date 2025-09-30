from typing import Any, Callable

import torch

import miniseq.configs as cfg
from miniseq.builder_config import config_as_dict
from miniseq.logging import get_logger
from miniseq.machine import Machine
from miniseq.models import (
    LoRAConfig,
    broadcast_model,
    infer_device,
    lora_wrap_model,
    reset_non_persistent_buffers,
    reset_parameters,
)
from miniseq.training import apply_ac
from miniseq.training.data_parallel import to_data_parallel
from miniseq.transformer import TransformerDecoderModel
from miniseq.utils import ModuleT, TorchCompileMode

_log = get_logger()


def compile_transformer(
    model: TransformerDecoderModel,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    mode: TorchCompileMode | None = "default",
    options: dict[str, str | int | bool | Callable] | None = None,
    compile_loss: bool = False,
) -> TransformerDecoderModel:
    if mode is not None and options is not None:
        raise RuntimeError("Compile error: either mode OR options can be specified.")

    model.decoder_frontend.compile(
        fullgraph=fullgraph, dynamic=dynamic, mode=mode, options=options
    )

    for layer in model.decoder.layers:
        layer.compile(fullgraph=fullgraph, dynamic=dynamic, mode=mode, options=options)

    if model.decoder.norm is not None:
        model.decoder.norm.compile(
            fullgraph=fullgraph, dynamic=dynamic, mode=mode, options=options
        )

    if compile_loss:
        model.loss = torch.compile(
            model.loss, dynamic=dynamic, fullgraph=fullgraph, mode=mode, options=options
        )

    return model


def setup_reference_model(
    model: ModuleT,
    machine: Machine,
    *,
    state_dict: dict[str, Any] | None = None,
    compile_config: cfg.CompileConfig = cfg.CompileConfig(),
) -> ModuleT:
    if not (model_device := infer_device(model)) == torch.device("meta"):
        raise ValueError(f"Expected reference model on meta device, got {model_device}")

    model = model.to_empty(device=machine.device)

    if machine.rank == 0:
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)

    if machine.size > 1:
        _log.info("Broadcasting reference model on all processes from rank 0.")

        broadcast_model(model, machine, source_rank=0, non_persistent_buffers=False)

        _log.info("Broadcasting done.")

    machine.barrier()

    reset_non_persistent_buffers(model)

    if compile_config.model:
        if not isinstance(model, TransformerDecoderModel):
            raise ValueError(
                f"Compile only supported for TransformerDecoderModel, got {type(model)}."
            )

        model = compile_transformer(
            model,
            fullgraph=compile_config.fullgraph,
            dynamic=compile_config.dynamic,
            compile_loss=compile_config.loss,
        )

    return model


def setup_model(
    model: ModuleT,
    machine: Machine,
    *,
    mp_dtype: torch.dtype,
    dp_config: cfg.DPStrategyConfig = cfg.dp.DDPConfig(),
    state_dict: dict[str, Any] | None = None,
    ac: bool = False,
    ac_freq: int = 1,
    lora_config: LoRAConfig | None = None,
    load_on_cpu: bool = True,
    compile_config: cfg.CompileConfig = cfg.CompileConfig(),
) -> ModuleT:
    """Parallelize, checkpoint, compile, lora wrap, load the state dict for the model."""

    if not (model_device := infer_device(model)) == torch.device("meta"):
        raise ValueError(f"Expected model on meta device, got {model_device}")

    init_device = machine.device

    if machine.size > 1 and dp_config.shard:
        # If we're sharding, allow loading on CPU to avoid OOM.
        # Apply this option only for FSDP2 parallelism, since DDP assumes model fits.
        init_device = torch.device("cpu") if load_on_cpu else machine.device

    if machine.rank == 0 or dp_config.replicate:
        model = model.to_empty(device=init_device)

    # Rank 0 used for broadcast for DDP and FSDP2, so we load state dict on it.
    if state_dict is not None and machine.rank == 0:
        # Note: After this point RoPE/non-persistent buffers are not initialized.
        model.load_state_dict(state_dict, strict=True)

    if state_dict is None and machine.rank == 0:
        reset_parameters(model)

    if lora_config is not None:
        lora_wrap_model(model, lora_config, machine)

    # For world_size == 1, we reset non-persistent buffers now.
    # For world_size > 1 (DDP/FSDP2) this is done after wrapping/sharding the model.
    if machine.size == 1:
        reset_non_persistent_buffers(model)

    machine.barrier()

    if ac:
        apply_ac(model=model, every_nth_layer=ac_freq, preserve_rng_state=True)

    if compile_config.model:
        if not isinstance(model, TransformerDecoderModel):
            raise ValueError(
                f"Compile only supported for TransformerDecoderModel, got {type(model)}."
            )

        model = compile_transformer(
            model,
            fullgraph=compile_config.fullgraph,
            dynamic=compile_config.dynamic,
            compile_loss=compile_config.loss,
        )

    machine.barrier()

    model = to_data_parallel(
        model,
        machine=machine,
        replicate=dp_config.replicate,
        shard=dp_config.shard,
        mp_dtype=mp_dtype,
        **config_as_dict(dp_config),
    )

    machine.barrier()

    return model
