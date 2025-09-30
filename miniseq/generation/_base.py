from abc import ABC, abstractmethod

import torch.nn as nn


class Generator(ABC):
    @abstractmethod
    def prepare_for_generation(self, **kwargs) -> None: ...

    @abstractmethod
    def after_generation(self, **kwargs) -> None: ...

    @abstractmethod
    def generate(
        self, input_ids: list[list[int]]
    ) -> tuple[list[list[int]], list[list[float]] | None]: ...

    @property
    @abstractmethod
    def model(self) -> nn.Module: ...

    @abstractmethod
    def update_model(self, *, only_trainable: bool = False) -> None: ...

    @abstractmethod
    def validation_mode(self, validation: bool) -> None: ...
