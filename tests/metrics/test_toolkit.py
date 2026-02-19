import torch

from miniseq.metrics import Mean, Sum
from miniseq.metrics._synclib import metrics_traversal_order, sync_states
from miniseq.metrics._toolkit import sync_and_compute_collection


def test_metrics_traversal_order_is_deterministic() -> None:
    state_dict = {
        "metric_b": {"z": torch.tensor(1.0), "a": torch.tensor(2.0)},
        "metric_a": {"b": torch.tensor(3.0)},
    }

    assert metrics_traversal_order(state_dict) == [
        ("metric_a", "b"),
        ("metric_b", "a"),
        ("metric_b", "z"),
    ]


def test_sync_states_without_distributed_returns_local_copy() -> None:
    states = {
        "mean": {
            "weighted_sum": torch.tensor(6.0, dtype=torch.float64),
            "weights": torch.tensor(3.0, dtype=torch.float64),
        }
    }
    traversal = metrics_traversal_order(states)

    gathered = sync_states(states, traversal)

    assert len(gathered) == 1
    torch.testing.assert_close(
        gathered[0]["mean"]["weighted_sum"], torch.tensor(6.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        gathered[0]["mean"]["weights"], torch.tensor(3.0, dtype=torch.float64)
    )

    states["mean"]["weighted_sum"] += 1.0
    torch.testing.assert_close(
        gathered[0]["mean"]["weighted_sum"], torch.tensor(6.0, dtype=torch.float64)
    )


def test_sync_and_compute_collection_single_process() -> None:
    mean = Mean()
    total = Sum()

    mean.update(torch.tensor([2.0, 3.0]), weight=torch.tensor([0.25, 0.75]))
    total.update(torch.tensor([1.0, 2.0, 3.0]), weight=0.5)

    values = sync_and_compute_collection({"mean": mean, "sum": total})

    torch.testing.assert_close(values["mean"], torch.tensor(2.75, dtype=torch.float64))
    torch.testing.assert_close(values["sum"], torch.tensor(3.0, dtype=torch.float64))
