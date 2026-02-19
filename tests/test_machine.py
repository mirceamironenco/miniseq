import os
from unittest.mock import patch

import pytest
import torch

from miniseq.machine import LocalMachine, all_sum, setup_default_machine


def test_local_machine_properties():
    device = torch.device("cpu")
    machine = LocalMachine(device=device)

    assert machine.device == device
    assert not machine.distributed
    assert machine.rank == 0
    assert machine.size == 1
    assert machine.dp_replicate_rank == 0
    assert machine.dp_shard_rank == 0

    with pytest.raises(RuntimeError):
        machine.process_group()


def test_all_sum_local_machine():
    machine = LocalMachine(device=torch.device("cpu"))
    result = all_sum(machine, 10.5)
    assert torch.isclose(result, torch.tensor(10.5))


@patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"})
def test_setup_default_machine_local():
    machine = setup_default_machine(device=torch.device("cpu"))
    assert isinstance(machine, LocalMachine)
    assert machine.device.type == "cpu"  # Default device without CUDA
    assert machine.rank == 0
    assert machine.size == 1


@patch.dict(os.environ, {}, clear=True)
def test_setup_default_machine_no_env_vars():
    # When no env vars are set, it should default to a LocalMachine.
    machine = setup_default_machine()
    assert isinstance(machine, LocalMachine)
    assert machine.rank == 0
    assert machine.size == 1
