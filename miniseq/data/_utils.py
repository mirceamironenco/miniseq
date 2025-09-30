from collections.abc import Sized
from typing import Protocol, TypeVar, runtime_checkable

from torch.utils.data import Dataset, DistributedSampler
from torchdata.nodes import Loader
from torchdata.nodes.base_node import BaseNode

X = TypeVar("X", covariant=True)


@runtime_checkable
class MapDataset(Protocol[X]):
    def __getitem__(self, index: int, /) -> X: ...
    def __len__(self) -> int: ...


@runtime_checkable
class EpochSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


class SizedLoader(Loader[X]):
    _length: int

    def __init__(
        self, root: BaseNode[X], *, length: int, restart_on_stop_iteration: bool = True
    ) -> None:
        super().__init__(root, restart_on_stop_iteration)
        self._length = length

    def __len__(self) -> int:
        return self._length


class DistributedEvalSampler(DistributedSampler[X]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        if self.drop_last:
            raise ValueError(
                "drop_last = True not compatible with DistribuetdEvalSampler"
            )

        if not isinstance(self.dataset, Sized):
            raise ValueError("dataset is not Sized, could not determine its length.")

        if len(self.dataset) % self.num_replicas != 0:
            if self.rank >= (len(self.dataset) % self.num_replicas):
                self.num_samples -= 1

            self.total_size = len(self.dataset)
