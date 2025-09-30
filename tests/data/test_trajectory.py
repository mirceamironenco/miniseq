import pytest
import torch

from miniseq.data import TrajectoryBatch


def test_trajectory_batch_properties():
    rewards = torch.randn(2, 3)
    advantages = torch.randn(2, 3)
    prompts = [[1, 2], [1, 2], [1, 2], [7, 8], [7, 8], [7, 8]]
    completions = [[13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24]]

    batch = TrajectoryBatch(
        prompt_ids=prompts,
        completion_ids=completions,
        rewards=rewards,
        advantages=advantages,
        pad_idx=0,
    )

    assert batch.batch_size == 2
    assert batch.mc_samples == 3


def test_trajectory_batch_to_device():
    rewards = torch.randn(2, 3)
    advantages = torch.randn(2, 3)
    prompts = [[1, 2], [1, 2], [1, 2], [7, 8], [7, 8], [7, 8]]
    completions = [[13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24]]

    batch = TrajectoryBatch(
        prompt_ids=prompts,
        completion_ids=completions,
        rewards=rewards,
        advantages=advantages,
        pad_idx=0,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        batch.to(device)
        assert batch.rewards.device.type == "cuda"
        assert batch.advantages.device.type == "cuda"


def test_trajectory_batch_chunking():
    rewards = torch.randn(4, 2)
    advantages = torch.randn(4, 2)
    prompts = [[i // 2] for i in range(8)]
    completions = [[i + 8] for i in range(8)]

    batch = TrajectoryBatch(
        prompt_ids=prompts,
        completion_ids=completions,
        rewards=rewards,
        advantages=advantages,
        pad_idx=0,
    )

    chunks = batch.chunk(num_chunks=2)
    assert len(chunks) == 2
    assert chunks[0].batch_size == 2
    assert chunks[1].batch_size == 2
    assert len(chunks[0].prompt_ids) == 4

    with pytest.raises(ValueError):
        batch.chunk(num_chunks=3)  # Not divisible

    with pytest.raises(ValueError):
        batch.chunk(num_chunks=5)  # Too many chunks


def test_trajectory_batch_auto_regressive_input():
    rewards = torch.randn(1, 2)
    advantages = torch.randn(1, 2)
    prompts = [[1, 2], [1, 2]]
    completions = [[5, 6, 7], [8, 9]]

    # Test with padding
    batch = TrajectoryBatch(
        prompt_ids=prompts,
        completion_ids=completions,
        rewards=rewards,
        advantages=advantages,
        pad_idx=0,
        packed=False,
    )

    input_batch, target_batch = batch.auto_regressive_input()

    assert not input_batch.is_padded
    assert target_batch.is_padded
    assert input_batch.seqs.shape == (2, 4)
    assert target_batch.seqs.shape == (2, 4)

    # Test with packing
    batch = TrajectoryBatch(
        prompt_ids=prompts,
        completion_ids=completions,
        rewards=rewards,
        advantages=advantages,
        pad_idx=0,
        packed=True,
    )

    input_batch, target_batch = batch.auto_regressive_input()

    assert input_batch.is_packed
    assert target_batch.is_packed
