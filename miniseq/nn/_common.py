from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing_extensions import ParamSpec

# Return type of the Module
ReturnT = TypeVar("ReturnT")

# ParamSpec for input parameters; default allows for no params to be specified.
P = ParamSpec("P", default=...)


class Module(nn.Module, Generic[ReturnT, P]):
    """nn.Module wrapper allowing type-hinting __call__.
    As with nn.Module, the user only needs to implement `forward`, ensuring the return
    type and optionally the input arguments match the specified generic arguments.
    Note: No runtime checks occur.

    Example usage:

        ```python
        # type-hint only the return type.
        class Decoder(Module[Tensor]):
            def forward(self, *args, **kwargs) -> Tensor:
                ...

        # type-hint return as well as input arguments.
        class Decoder(Module[Tensor, [Tensor]]):
            def forward(self, x: Tensor) -> Tensor:
                ...
        ```
    """

    __call__: Callable[P, ReturnT]


class Linear(nn.Linear):
    """Typed nn.Linear which preserves parent while allowing for init_fn."""

    in_features: int
    out_features: int
    weight: Parameter
    bias: Parameter | None
    init_fn: Callable[[nn.Linear], None] | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        init_fn: Callable[[nn.Linear], None] | None = None,
    ) -> None:
        self.init_fn = init_fn

        super().__init__(in_features, out_features, bias, device, dtype)

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.init_fn(self)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            if self.bias is not None:
                bound = 1 / math.sqrt(self.weight.size(1))
                nn.init.uniform(self.bias, -bound, bound)

    if TYPE_CHECKING:

        def __call__(self, input: torch.Tensor) -> torch.Tensor: ...


class Embedding(nn.Embedding):
    """Typed nn.Embedding which preserves parent while allowing for init_fn.

    Also allows one to specify a learned positional encoder."""

    num_embeddings: int
    embedding_dim: int
    init_fn: Callable[[nn.Embedding], None] | None
    weight: Parameter
    _pos_encoder: Module[torch.Tensor] | None

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: torch.Tensor | None = None,
        _freeze: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        init_fn: Callable[[nn.Embedding], None] | None = None,
        pos_encoder: Module[torch.Tensor] | None = None,
    ) -> None:
        self.init_fn = init_fn

        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )

        self._pos_encoder = pos_encoder

    def reset_parameters(self) -> None:
        if self.init_fn is not None:
            self.init_fn(self)
        else:
            nn.init.normal_(self.weight)

    def forward(
        self, input: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        if self._pos_encoder is not None:
            x = self._pos_encoder(x, input_pos=input_pos)

        return x

    if TYPE_CHECKING:

        def __call__(
            self, input: torch.Tensor, input_pos: torch.Tensor | None = None
        ) -> torch.Tensor: ...


class TiedProjectionLayer(Module[torch.Tensor, [torch.Tensor]]):
    """Needed for FSDP2 compatibility."""

    def __init__(self, embed: Embedding) -> None:
        super().__init__()
        self.embed = embed
        self._shape = self.embed.weight.size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embed.weight)

    def extra_repr(self) -> str:
        return f"in_dim={self._shape[1]}, out_dim={self._shape[0]}"

    @property
    def weight(self) -> torch.Tensor:
        return self.embed.weight


class EmbeddingProjection(Module[torch.Tensor]):
    _embedding: Linear
    _pos_encoder: Module[torch.Tensor] | None

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        bias: bool = False,
        pos_encoder: Module[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()

        self._embedding = Linear(input_dim, model_dim, bias=bias)

        self._pos_encoder = pos_encoder

    def forward(
        self, input: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self._embedding(input)

        if self._pos_encoder is not None:
            x = self._pos_encoder(x)

        return x

    if TYPE_CHECKING:

        def __call__(
            self, input: torch.Tensor, input_pos: torch.Tensor | None = None
        ) -> torch.Tensor: ...


class RMSNorm(Module[torch.Tensor, [torch.Tensor]]):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
