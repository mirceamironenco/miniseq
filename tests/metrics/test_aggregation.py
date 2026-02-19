import pytest
import torch

from miniseq.metrics import Mean, Sum


def _assert_scalar_close(actual: torch.Tensor, expected: float) -> None:
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))


def test_mean_unweighted_updates() -> None:
    metric = Mean()

    metric.update(1)
    metric.update(torch.tensor([2, 3]))

    _assert_scalar_close(metric.compute(), 2.0)


def test_mean_weighted_updates_with_scalar_and_tensor_weights() -> None:
    metric = Mean()

    metric.update(torch.tensor([2, 3]), weight=torch.tensor([0.2, 0.8]))
    _assert_scalar_close(metric.compute(), 2.8)

    metric.update(torch.tensor([4, 5]), weight=0.5)
    _assert_scalar_close(metric.compute(), 3.65)

    metric.update(torch.tensor([6]), weight=2)
    _assert_scalar_close(metric.compute(), 4.825)


def test_mean_invalid_weight_shape_raises_value_error() -> None:
    metric = Mean()

    with pytest.raises(ValueError):
        metric.update(torch.tensor([1.0, 2.0]), weight=torch.tensor([1.0]))


def test_mean_compute_without_updates_returns_zero() -> None:
    metric = Mean()

    _assert_scalar_close(metric.compute(), 0.0)


def test_mean_merge_state() -> None:
    m1 = Mean()
    m2 = Mean()
    m3 = Mean()

    m1.update(torch.tensor([1.0, 2.0]))
    m2.update(torch.tensor([3.0, 5.0]), weight=0.5)
    m3.update(torch.tensor([7.0]), weight=2)

    m1.merge_state([m2, m3])

    # weighted_sum = 3 + 4 + 14 = 21
    # weights = 2 + 1 + 2 = 5
    _assert_scalar_close(m1.compute(), 4.2)


def test_mean_reset_and_state_dict_roundtrip() -> None:
    source = Mean()
    source.update(torch.tensor([1.0, 2.0, 3.0]), weight=2)
    source.update(torch.tensor([4.0]), weight=1)
    expected = source.compute()

    target = Mean()
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(target.compute(), expected)

    source.reset()
    _assert_scalar_close(source.compute(), 0.0)


def test_sum_unweighted_updates() -> None:
    metric = Sum()

    metric.update(1)
    metric.update(torch.tensor([2, 3]))

    _assert_scalar_close(metric.compute(), 6.0)


def test_sum_weighted_updates_with_scalar_and_tensor_weights() -> None:
    metric = Sum()

    metric.update(torch.tensor([2, 3]), weight=torch.tensor([0.1, 0.6]))
    _assert_scalar_close(metric.compute(), 2.0)

    metric.update(torch.tensor([2, 3]), weight=0.5)
    _assert_scalar_close(metric.compute(), 4.5)

    metric.update(torch.tensor([4, 6]), weight=1)
    _assert_scalar_close(metric.compute(), 14.5)


def test_sum_invalid_weight_shape_raises_value_error() -> None:
    metric = Sum()

    with pytest.raises(ValueError):
        metric.update(torch.tensor([1.0, 2.0]), weight=torch.tensor([1.0]))


def test_sum_merge_state_reset_and_state_dict_roundtrip() -> None:
    s1 = Sum()
    s2 = Sum()

    s1.update(torch.tensor([1.0, 2.0]), weight=2)
    s2.update(torch.tensor([4.0, 6.0]), weight=torch.tensor([0.5, 1.5]))

    s1.merge_state([s2])

    # 2 * (1 + 2) + (0.5 * 4 + 1.5 * 6) = 6 + 11 = 17
    _assert_scalar_close(s1.compute(), 17.0)

    saved = s1.state_dict()
    loaded = Sum()
    loaded.load_state_dict(saved)
    torch.testing.assert_close(loaded.compute(), s1.compute())

    s1.reset()
    _assert_scalar_close(s1.compute(), 0.0)
