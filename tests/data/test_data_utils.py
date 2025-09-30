import torch
from torch.utils.data import TensorDataset

from miniseq.data._utils import DistributedEvalSampler, SizedLoader


def test_sized_loader():
    loader = SizedLoader(root=None, length=123)
    assert len(loader) == 123


def test_distributed_eval_sampler():
    dataset = TensorDataset(torch.arange(10))

    # Test with num_replicas that does not divide the dataset size
    sampler = DistributedEvalSampler(dataset, num_replicas=3, rank=0, shuffle=False)
    indices = list(sampler)
    assert len(indices) == 4  # Rank 0 gets 4 samples
    assert indices == [0, 3, 6, 9]

    sampler = DistributedEvalSampler(dataset, num_replicas=3, rank=1, shuffle=False)
    indices = list(sampler)
    assert len(indices) == 3  # Rank 1 gets 3 samples
    assert indices == [1, 4, 7]

    sampler = DistributedEvalSampler(dataset, num_replicas=3, rank=2, shuffle=False)
    indices = list(sampler)
    assert len(indices) == 3  # Rank 2 gets 3 samples
    assert indices == [2, 5, 8]

    # Test with shuffle
    sampler = DistributedEvalSampler(
        dataset, num_replicas=2, rank=0, shuffle=True, seed=42
    )
    indices = list(sampler)
    assert len(indices) == 5
    assert set(indices) == {0, 1, 2, 3, 4}  # Seeded shuffle
