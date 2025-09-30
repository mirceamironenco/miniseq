from torch.nn.parallel import DistributedDataParallel as DDP

from miniseq.machine import Machine, all_ranks_same_trainable_params
from miniseq.models import any_meta_device, infer_device, reset_non_persistent_buffers
from miniseq.utils import ModuleT


def to_ddp(
    model: ModuleT,
    machine: Machine,
    *,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> ModuleT:
    model_device = infer_device(model, recurse=True)

    if machine.rank == 0:
        if not model_device == machine.device:
            raise ValueError(
                "Rank 0 model must be on correct cuda device and initialized"
                f"(e.g. randomly or via load_state_dict())!. Got device {model_device}"
            )
    else:
        if model_device.type == "meta":
            model = model.to_empty(device=machine.device)
        else:
            if model_device != machine.device:
                model = model.to(machine.device)

    machine.barrier()

    # NB:init_sync must be True to broadcast params/buffers from rank 0 to others.
    # For DDP assume the model fits on 1 GPU.
    ddp_model = DDP(
        model,
        broadcast_buffers=broadcast_buffers,
        init_sync=True,
        process_group=machine.process_group(),
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
    )

    if any_meta_device(ddp_model) == "meta":
        raise ValueError(
            f"Model on rank {machine.rank} still has params on meta device."
        )

    # Reset non-persistent buffers, e.g. RoPE.
    reset_non_persistent_buffers(ddp_model)

    machine.barrier()

    if not all_ranks_same_trainable_params(model, machine):
        raise RuntimeError(
            "DDP wrap failed. Different ranks have different number of trainable params."
        )

    assert infer_device(model) == machine.device

    return ddp_model  # type: ignore
