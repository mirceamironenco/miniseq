from __future__ import annotations

import functools
from typing import Callable, TypeAlias

import torch
import torch.nn.functional as F
from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask,
    _convert_mask_to_block_mask,
    _create_sparse_block_from_block_mask,
    _mask_mod_signature,
    _vmap_for_bhqkv,
    and_masks,
    create_block_mask,
)

from miniseq.utils import should_compile_flex

MaskMod: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]

ScoreMod: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def build_block_mask_from_dense(
    dense_mask: torch.Tensor,
    mask_mod: MaskMod | None,
    Q_LEN: int,
    KV_LEN: int,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> BlockMask:
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
        dense_mask,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        separate_full_blocks=True,
    )
    return _create_sparse_block_from_block_mask(
        (partial_block_mask, full_block_mask),
        mask_mod,
        (Q_LEN, KV_LEN),
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )


def _custom_create_mask(
    mod_fn: _mask_mod_signature,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    device: str | torch.device,
) -> torch.Tensor:
    r"""This function creates a mask tensor from a mod_fn function.

    Args:
        mod_fn (_mask_mod_signature): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of query heads.
        Q_LEN (int): Sequence length of query.
        KV_LEN (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    batch_size = B or 1
    num_heads = H or 1

    b = torch.arange(0, batch_size, device=device)
    h = torch.arange(0, num_heads, device=device)
    m = torch.arange(0, Q_LEN, device=device)
    n = torch.arange(0, KV_LEN, device=device)

    with TransformGetItemToIndex():
        mask_mod = mod_fn
        mask_mod = _vmap_for_bhqkv(mask_mod, prefix=())
        mask = mask_mod(b, h, m, n)
        return mask


build_dense_mask_from_mod = _custom_create_mask


def causal_mask_mod(
    batch: torch.Tensor,
    head: torch.Tensor,
    query_index: torch.Tensor,
    kv_index: torch.Tensor,
) -> torch.Tensor:
    return query_index >= kv_index


@functools.lru_cache
def make_sliding_window_mod(window_size: int) -> MaskMod:
    def window_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        return query_index - kv_index <= window_size

    return window_mask_mod


@functools.lru_cache
def make_sliding_window_causal(window_size: int) -> MaskMod:
    return and_masks(causal_mask_mod, make_sliding_window_mod(window_size))


class AttentionMask:
    mask_mod: MaskMod
    _bsz: int
    _num_heads: int
    _q_len: int
    _kv_len: int
    seq_lens_pt: torch.Tensor | None
    _max_input_pos: torch.Tensor
    _max_seq_len: torch.Tensor
    _total_seq_len: torch.Tensor
    cu_seqlens: torch.Tensor | None
    window_size: int | None
    block_sizes: tuple[int, int]
    _dense_mask: torch.Tensor | None
    _dense_bias: torch.Tensor | None
    _block_mask: BlockMask | None
    padded: bool
    packed: bool

    _flex_compiled: bool = False

    def __init__(
        self,
        *,
        mask_mod: MaskMod,
        batch_size: int | None,
        num_heads: int | None,
        q_len: int,
        kv_len: int,
        device: str | torch.device,
        max_input_pos: int,
        max_seq_len: int,
        total_seq_len: int,
        seq_lens_pt: torch.Tensor | None = None,
        BLOCK_SIZE: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
        uses_block_mask: bool = False,
        padded: bool = False,
        packed: bool = False,
        window_size: int | None = None,
    ) -> None:
        self.compile_mask_builders()

        self.mask_mod = mask_mod
        self._bsz = batch_size or 1
        self._num_heads = num_heads or 1
        self._q_len = q_len
        self._kv_len = kv_len
        self.device = device
        self.seq_lens_pt = seq_lens_pt

        # https://github.com/pytorch/pytorch/issues/129623
        self._max_input_pos = torch.empty((0, max_input_pos))
        torch._dynamo.mark_dynamic(self._max_input_pos, 1)

        self._max_seq_len = torch.empty((0, max_seq_len))
        torch._dynamo.mark_dynamic(self._max_seq_len, 1)

        self._total_seq_len = torch.empty((0, total_seq_len))
        torch._dynamo.mark_dynamic(self._total_seq_len, 1)

        if seq_lens_pt is not None:
            self.cu_seqlens = F.pad(seq_lens_pt.cumsum(0, dtype=torch.int32), (1, 0))

            torch._dynamo.maybe_mark_dynamic(self.cu_seqlens, 0)
        else:
            self.cu_seqlens = None

        self.window_size = window_size

        if isinstance(BLOCK_SIZE, int):
            self.block_sizes = (BLOCK_SIZE, BLOCK_SIZE)
        else:
            self.block_sizes = BLOCK_SIZE

        self._dense_mask = None
        self._dense_bias = None
        self._block_mask = None
        self.padded = padded
        self.packed = packed

        # If block_mask=True, don't construct lazily - build the block mask immediatly.
        # Otherwise, always build a dense mask used to make the block mask if needed.
        # Note: Pass uses_block_mask=True if running into `get_device` dynamo errors.
        if not uses_block_mask:
            self._dense_mask = self._build_dense()
        else:
            self._block_mask = create_block_mask(
                self.mask_mod,
                *self.shape,
                self.device,  # type: ignore
                BLOCK_SIZE=self.block_sizes,
            )

    @classmethod
    def compile_mask_builders(cls) -> None:
        if not should_compile_flex():
            return

        if cls._flex_compiled:
            return

        global build_dense_mask_from_mod
        global build_block_mask_from_dense
        global create_block_mask

        build_dense_mask_from_mod = torch.compile(_custom_create_mask, dynamic=True)
        build_block_mask_from_dense = torch.compile(
            build_block_mask_from_dense, dynamic=True
        )

        create_block_mask = torch.compile(create_block_mask, fullgraph=True)

        cls._flex_compiled = True

    @classmethod
    def build_causal(
        cls,
        *,
        q_len: int,
        kv_len: int,
        device: str | torch.device,
        max_input_pos: int,
        max_seq_len: int,
        total_seq_len: int,
        seq_lens_pt: torch.Tensor | None = None,
        batch_size: int | None = None,
        BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
        window_size: int | None = None,
        other_mod: MaskMod | None = None,
        uses_block_mask: bool = False,
        padded: bool = False,
        packed: bool = False,
    ) -> AttentionMask:
        if window_size is not None:
            if not window_size < kv_len:
                raise ValueError(
                    f"window_size must be smaller then key length ({kv_len}), got {window_size}."
                )
            mask_mod = make_sliding_window_causal(window_size=window_size)
        else:
            mask_mod = causal_mask_mod

        if other_mod is not None:
            mask_mod = and_masks(mask_mod, other_mod)

        return AttentionMask(
            mask_mod=mask_mod,
            batch_size=batch_size,
            num_heads=None,
            q_len=q_len,
            kv_len=kv_len,
            device=device,
            max_input_pos=max_input_pos,
            max_seq_len=max_seq_len,
            total_seq_len=total_seq_len,
            seq_lens_pt=seq_lens_pt,
            BLOCK_SIZE=BLOCK_SIZE,
            uses_block_mask=uses_block_mask,
            padded=padded,
            packed=packed,
            window_size=window_size,
        )

    def _build_dense(self) -> torch.Tensor:
        return build_dense_mask_from_mod(self.mask_mod, *self.shape, self.device)  # type: ignore

    def _build_block_from_dense(self) -> BlockMask:
        return build_block_mask_from_dense(
            self.materialize_dense(),
            mask_mod=self.mask_mod,
            Q_LEN=self._q_len,
            KV_LEN=self._kv_len,
            Q_BLOCK_SIZE=self.block_sizes[0],
            KV_BLOCK_SIZE=self.block_sizes[1],
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self._bsz, self._num_heads, self._q_len, self._kv_len)

    def materialize_block(self) -> BlockMask:
        if self._block_mask is None:
            self._block_mask = self._build_block_from_dense()

        return self._block_mask

    def materialize_dense(self) -> torch.Tensor:
        """Boolean mask where True indicates the element should take part in attn."""

        if self._dense_mask is None:
            self._dense_mask = self._build_dense()

        return self._dense_mask

    def materialize_bias(self, seqs: torch.Tensor) -> torch.Tensor:
        """Float mask that can be added to the attention scores."""
        if self._dense_bias is None:
            mask = self.materialize_dense()
            float_mask = torch.zeros_like(mask, dtype=seqs.dtype)
            self._dense_bias = torch.where(mask, float_mask, -torch.inf)
        return self._dense_bias

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len.size(1)

    @property
    def total_seq_len(self) -> int:
        return self._total_seq_len.size(1)

    @property
    def max_input_pos(self) -> int:
        return self._max_input_pos.size(1)
