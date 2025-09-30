from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import and_masks

from miniseq.transformer import AttentionMask, MaskMod
from miniseq.utils import to_tensor


@dataclass
class SequenceBatch:
    """A container for sequence batches, handling packing, padding, and metadata. Assumes right-padding."""

    seqs: torch.Tensor
    """Sequence of token ids; shape (batch_size, sequence_length, *) or (1, packed_length)."""

    seq_lens: list[int] | None
    """List of true unpadded lengths for each sequence or packed segment."""

    input_pos: torch.Tensor | None = None
    """Positions of seqs. Used for document masking if packed, shape: seqs.size()[:2]."""

    target_mask: torch.Tensor | None = None
    """Bool mask specifying elements of seqs used for loss computation, shape: seqs.size()[:2]."""

    is_packed: bool = False
    """If True, `seqs` is a single packed sequence (bsz=1) of concatenated segments."""

    is_padded: bool = False
    """If True, sequences in `seqs` are right-padded; `seq_lens` must be provided."""

    pad_idx: int | None = None
    """Token id used for padding. Used for log/debug, will not be checked at runtime."""

    examples_used: int | None = None
    """Number of dataset examples used to construct the batch. Used for logging."""

    prefix_docs: list[int] | None = None
    """Document id of completion or prompt, if the batch is packed + prefix sharing."""

    prefix_share_count: int | None = None
    """The number of completions per shared prompt. Used when packed + prefix sharing."""

    max_input_pos: int = -1
    """Maximum value to be used for positional encoding. If -1, will be calculated."""

    num_valids: int = -1
    """Number of valid targets if used for auto-regression. If -1 will be calculated."""

    _document_ids: torch.Tensor | None = field(init=False, repr=False, default=None)
    """Document ids used for attention mask. Used when packed."""

    _min_seqlen: int = field(init=False, repr=False)
    """Smallest sequence length in the batch."""

    _max_seqlen: int = field(init=False, repr=False)
    """Largest sequence length in the batch."""

    _total_seqlen: int = field(init=False, repr=False)
    """Sum of sequence lengths (no padding) in the batch."""

    _full_seqlens: list[int] | None = field(init=False, repr=False, default=None)
    """Used when packed. List of sequence lengths, final seq_len has padding length."""

    _seqlens_pt: torch.Tensor = field(init=False, repr=False)

    _total_seqlen_pt: torch.Tensor = field(init=False, repr=False)

    _is_split: bool = field(init=False, default=False)
    """Batch was produced by as_auto_regressive method."""

    using_flex_attention: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.is_packed:
            if not (self.seqs.ndim == 2 and self.seqs.size(0) == 1):
                raise ValueError(
                    f"Packed batches have shape (1, packed_seqlen), got {self.seqs.size()}"
                )

        if self.is_padded or self.is_packed:
            if self.seq_lens is None:
                raise ValueError("A padded/packed SequenceBatch must provide seq_lens.")

        if self.seq_lens is not None:
            if not isinstance(self.seq_lens, list) or not isinstance(
                self.seq_lens[0], int
            ):
                raise TypeError(
                    f"seq_lens must be list[int], got {type(self.seq_lens)}."
                )

            if not self.is_packed:
                if not (num_seqs := len(self.seq_lens)) == self.seqs.size(0):
                    raise ValueError(
                        f"len(self.seq_lens) ({num_seqs}) != seqs.size(0) ({self.seqs.size(0)})"
                    )

            total_seqlen = 0
            self._min_seqlen, self._max_seqlen = self.seqs.size(1), -1
            for seqlen in self.seq_lens:
                assert seqlen > 0

                if seqlen < self._min_seqlen:
                    self._min_seqlen = seqlen

                if seqlen > self._max_seqlen:
                    self._max_seqlen = seqlen

                total_seqlen += seqlen

            self._total_seqlen = total_seqlen

            self._seqlens_pt = to_tensor(
                self.seq_lens, dtype=torch.int64, device=self.seqs.device
            )

            self._full_seqlens = None

            if self.is_packed:
                self._full_seqlens = self.seq_lens[:]

                self._full_seqlens[-1] += self.padding

        else:
            self._min_seqlen = self._max_seqlen = self.seqs.size(1)

            self._total_seqlen = self.seqs.numel()

            self._seqlens_pt = torch.full(
                (self.seqs.size(0),),
                fill_value=self.seqs.size(1),
                device=self.seqs.device,
            )

        if not self._min_seqlen > 0:
            raise ValueError("seq_lens contains at lesat one 0-length sequence.")

        self._total_seqlen_pt = to_tensor(
            self._total_seqlen, dtype=torch.int32, device=self.seqs.device
        )

        seqs_shape = self.seqs.size()

        if self.input_pos is not None:
            if not (pos_size := self.input_pos.size()) == seqs_shape[:2]:
                raise ValueError(
                    f"input_pos.size() {pos_size} != seqs.size()[:2] ({seqs_shape[:2]})"
                )

        if self.target_mask is not None:
            if not (target_size := self.target_mask.size()) == seqs_shape[:2]:
                raise ValueError(
                    f"target_mask.size() {target_size} != seqs.size()[:2] ({seqs_shape[:2]})"
                )

        # max_input_pos and num_valids are calculated here to avoid a cuda sync.
        # If num_valids == -1 or max_input_pos == -1, we are in the original batch
        # produced by the data loader When as_auto_regressive is called, num_valids is
        # passed to the target batch and represents the number of valid prediction
        # targets. max_input_pos is passed to the input batch.

        if self.max_input_pos == -1:
            if self.input_pos is not None:
                self.max_input_pos = int(self.input_pos.max())
            else:
                self.max_input_pos = self.seqs.size(1) - 1

        if self.num_valids == -1:
            if self.target_mask is not None:
                num_valids = int(self.target_mask.sum())
            else:
                num_valids = self._total_seqlen

            # Chop off targets corresponding to the slice [1:]
            if self.is_packed:
                num_valids -= (
                    1 if self.target_mask is None else int(self.target_mask[0][0])
                )
            else:
                num_valids -= (
                    self.seqs.size(0)
                    if self.target_mask is None
                    else int(self.target_mask[:, 0].sum())
                )

            self.num_valids = num_valids

        torch._dynamo.maybe_mark_dynamic(self._seqlens_pt, 0)
        torch._dynamo.maybe_mark_dynamic(self._total_seqlen_pt, 0)

    @property
    def batch_size(self) -> int:
        # Note: Not equivalent to the number of dataset examples if batch is packed.
        return self.seqs.size(0)

    @property
    def num_examples(self) -> int:
        """Total number of dataset items used to construct the batch."""

        # Note: `examples_used` should preferably be set if working with a packed batch.
        # However this is not enforced since `num_examples` is used for logging only.
        if self.examples_used is not None:
            return self.examples_used

        # If seq_lens is provided, fall back to that.
        if self.seq_lens is not None:
            return len(self.seq_lens)

        # Otherwise, fallback to standard batch_size counting.
        return self.batch_size

    @property
    def num_target_elements(self) -> int:
        """Total number of tokens which can produce gradients (i.e. not masked/padded)."""
        if self._is_split:
            return self.num_valids

        if self.target_mask is not None:
            return int(self.target_mask.sum())

        return self._total_seqlen

    @property
    def num_elements(self) -> int:
        """Total number of (masked and non-masked) tokens minus padding."""
        return self._total_seqlen

    @property
    def padding(self) -> int:
        # TODO: Adapt for 3d seqs case.
        return self.seqs.numel() - self.num_elements

    @property
    def min_seqlen(self) -> int:
        return self._min_seqlen

    @property
    def max_seqlen(self) -> int:
        return self._max_seqlen

    @property
    def seqlens_pt(self) -> torch.Tensor:
        return self._seqlens_pt

    def uses_prefix_sharing(self) -> bool:
        return (
            self.is_packed
            and self.prefix_docs is not None
            and self.prefix_share_count is not None
        )

    @property
    def document_ids(self) -> torch.Tensor:
        if not self.is_packed:
            raise ValueError("document_ids available only for packed batches.")

        if self._document_ids is None:
            if self.input_pos is None:
                raise ValueError(
                    "SequenceBatch is packed but no input_pos was provided."
                )

            # Note: Cast to int32 to have the triton kernel consume less shmem.
            self._document_ids = input_pos_to_document_ids(
                self.input_pos, dtype=torch.int32
            )

        return self._document_ids

    def _maybe_packing_mask(self) -> MaskMod | None:
        if self.is_packed:
            document_ids = self.document_ids

            mask_doc_ids = document_ids.flatten()

            prefix_docs = self.prefix_docs

            valid_len = self._total_seqlen_pt

            if prefix_docs is not None:
                if not SequenceBatch.using_flex_attention:
                    raise ValueError(
                        "Prefix sharing compatible only with Flex Attention SDPA."
                    )

                prefix_docs_pt = to_tensor(
                    prefix_docs, dtype=torch.int32, device=self.seqs.device
                )

                if self.padding > 0:
                    return build_doc_maskmod_prefix_pad(
                        mask_doc_ids, prefix_docs_pt, valid_len
                    )
                else:
                    return build_doc_maskmod_prefix(mask_doc_ids, prefix_docs_pt)

            if self.padding > 0:
                return build_doc_maskmod_pad(mask_doc_ids, valid_len)

            return build_doc_maskmod(mask_doc_ids)

    def _maybe_padding_mask(self) -> MaskMod | None:
        if self.padding == 0 or self.is_packed:
            return None

        if self.seq_lens is None:
            raise ValueError("Could not create padding mask, batch seq_lens is None.")

        # Note: Cast to int32 to have the triton kernel consume less shmem.
        mask_seq_lens = self.seqlens_pt.to(dtype=torch.int32)

        return right_side_pad_mask_mod(mask_seq_lens)

    def as_auto_regressive(self) -> tuple[SequenceBatch, SequenceBatch]:
        if self._total_seqlen < 2:
            raise ValueError(f"Expected sequence length > 1, got {self._total_seqlen}.")

        assert not self._is_split

        batch_width = self.seqs.size(1)

        seqs, targets = self.seqs[:, :-1, ...], self.seqs[:, 1:, ...]

        seq_lens: list[int] | None = None
        target_seq_lens: list[int] | None = None

        still_padded = self.is_padded

        if self.seq_lens is not None:
            if self._min_seqlen <= 1 and not self.uses_prefix_sharing():
                raise ValueError(
                    f"To produce (input,target) batches a (packed) sequence requires length >= 2, got {self._min_seqlen}, seq_lens: {self.seq_lens}"
                )

            if self.is_packed:
                # For the input batch (`seqs`), the last token of the entire packed
                # sequence is truncated. If there is no padding, this truncated token
                # belongs to the last sequence in the pack.
                seq_lens = self.seq_lens[:]
                if self.padding == 0:
                    seq_lens[-1] -= 1

                # For the target batch (`targets`), the first token of the entire
                # packed sequence is truncated. This token always belongs to the
                # first sequence in the pack.
                target_seq_lens = self.seq_lens[:]
                target_seq_lens[0] -= 1

                # Whether input batch still padded.
                still_padded = still_padded and self._total_seqlen + 1 < batch_width
            else:
                # clamp_max since it is assumed batch might have right-side padding.
                seq_lens, target_seq_lens = [], []
                for seqlen in self.seq_lens:
                    seq_lens.append(min(batch_width - 1, seqlen))
                    target_seq_lens.append(seqlen - 1)

                # Corner case if we trimmed off all padding for input batch.
                still_padded = still_padded and self.min_seqlen + 1 < batch_width

        input_pos: torch.Tensor | None = None

        if self.input_pos is not None:
            input_pos = self.input_pos[:, :-1]

        if self.target_mask is None:
            target_mask = None
        else:
            target_mask = self.target_mask[:, 1:]

        batch = SequenceBatch(
            seqs,
            seq_lens=seq_lens,
            input_pos=input_pos,
            target_mask=None,
            is_packed=self.is_packed,
            is_padded=still_padded,
            pad_idx=self.pad_idx,
            examples_used=self.num_examples,
            prefix_docs=self.prefix_docs[:-1] if self.prefix_docs else None,
            prefix_share_count=self.prefix_share_count,
            max_input_pos=self.max_input_pos,
        )

        target_batch = SequenceBatch(
            targets,
            seq_lens=target_seq_lens,
            input_pos=None,
            target_mask=target_mask,
            is_packed=self.is_packed,
            is_padded=self.is_padded,
            pad_idx=self.pad_idx,
            examples_used=self.num_examples,
            num_valids=self.num_valids,
        )

        batch._is_split = target_batch._is_split = True

        return batch, target_batch

    def as_input(self) -> tuple[torch.Tensor, torch.Tensor | None, AttentionMask]:
        if self.is_packed and self.input_pos is None:
            raise ValueError(
                "Packed SequenceBatch must provide input_pos to be used as input."
            )

        batch_mask = None

        if (padding_mask := self._maybe_padding_mask()) is not None:
            batch_mask = padding_mask

        if (document_mask := self._maybe_packing_mask()) is not None:
            if batch_mask is not None:
                batch_mask = and_masks(batch_mask, document_mask)
            else:
                batch_mask = document_mask

        batch_size, batch_width = self.seqs.size()[:2]

        seq_lens = None

        if self.is_packed and not SequenceBatch.using_flex_attention:
            # Used for cu_seqlens in flash_attn_varlen_func
            seq_lens = self.seqlens_pt

        attention_mask = AttentionMask.build_causal(
            q_len=batch_width,
            kv_len=batch_width,
            device=self.seqs.device,
            max_input_pos=self.max_input_pos,
            max_seq_len=self.max_seqlen,
            total_seq_len=self._total_seqlen,
            seq_lens_pt=seq_lens,
            batch_size=batch_size,
            other_mod=batch_mask,
            uses_block_mask=SequenceBatch.using_flex_attention,
            padded=self.is_padded,
            packed=self.is_packed,
        )

        return self.seqs, self.input_pos, attention_mask

    def pin_memory(self) -> SequenceBatch:
        # Note: Will pin to torch.cuda.current_device()

        self.seqs = self.seqs.pin_memory()

        if self.input_pos is not None:
            self.input_pos = self.input_pos.pin_memory()

        if self.target_mask is not None:
            self.target_mask = self.target_mask.pin_memory()

        self._seqlens_pt = self._seqlens_pt.pin_memory()

        return self

    def to(self, device: torch.device, *, non_blocking: bool = False) -> SequenceBatch:
        self.seqs = self.seqs.to(device, non_blocking=non_blocking)

        if self.input_pos is not None:
            self.input_pos = self.input_pos.to(device, non_blocking=non_blocking)

        if self.target_mask is not None:
            self.target_mask = self.target_mask.to(device, non_blocking=non_blocking)

        self._seqlens_pt = self._seqlens_pt.to(device, non_blocking=non_blocking)

        return self

    def full_lengths(self) -> torch.Tensor:
        assert self._full_seqlens is not None

        full_seqleqns_pt = to_tensor(
            self._full_seqlens, dtype=torch.int32, device=self.seqs.device
        )

        return full_seqleqns_pt

    def nested_layout_offsets(self) -> tuple[torch.Tensor, int, int]:
        """Can be used to make a nested tensor with same shape as the seqs tensor."""

        if not self.is_packed:
            raise ValueError("Nested layout is available only for packed batch.")

        if self.prefix_docs is not None:
            raise ValueError(
                "Nested layout is not available only for prefix+packed batch."
            )

        lengths = self.full_lengths()

        offsets_pt = F.pad(lengths.cumsum(0), (1, 0))

        return offsets_pt, self.min_seqlen, self.max_seqlen


def build_doc_mask_mod(mask_doc_ids: torch.Tensor) -> MaskMod:
    def doc_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        return mask_doc_ids[query_index] == mask_doc_ids[kv_index]

    return doc_mask_mod


def build_doc_maskmod_pad(
    mask_doc_ids: torch.Tensor, valid_len: torch.Tensor
) -> MaskMod:
    def doc_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (mask_doc_ids[query_index] == mask_doc_ids[kv_index])
            & (query_index < valid_len)
            & (kv_index < valid_len)
        )

    return doc_mask_mod


def build_doc_maskmod(mask_doc_ids: torch.Tensor) -> MaskMod:
    def doc_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        return mask_doc_ids[query_index] == mask_doc_ids[kv_index]

    return doc_mask_mod


def build_doc_maskmod_prefix_pad(
    mask_doc_ids: torch.Tensor, prefix_docs: torch.Tensor, valid_len: torch.Tensor
) -> MaskMod:
    def doc_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        same_doc = mask_doc_ids[query_index] == mask_doc_ids[kv_index]

        # we are in the same completion (or the prefix prompt).
        # this relies on external document mask, as all prompts have id 0
        same_completion = prefix_docs[query_index] == prefix_docs[kv_index]

        # kv is at a prompt, assuming document mask is also applied and true
        # this is the shared prefix prompt.
        kv_in_prefix_prompt = prefix_docs[kv_index] == 0

        padding_mask = (query_index < valid_len) & (kv_index < valid_len)

        return padding_mask & same_doc & (kv_in_prefix_prompt | same_completion)

    return doc_mask_mod


def build_doc_maskmod_prefix(
    mask_doc_ids: torch.Tensor, prefix_docs: torch.Tensor
) -> MaskMod:
    def doc_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        """Mask for packed batch of the form: [A, a1, a2, a3, B, b1, b2, b3, ...]"""

        # We are in the same prompt group, e.g. [A, a1, a2, a3]
        same_doc = mask_doc_ids[query_index] == mask_doc_ids[kv_index]

        # We are in the same completion or the prefix prompt.
        # this relies on the document mask, as all prompts have prefix doc id = 0
        same_completion = prefix_docs[query_index] == prefix_docs[kv_index]

        # kv is at a prompt, assuming document mask is also applied and true
        # this will be the shared prefix prompt.
        kv_in_prefix_prompt = prefix_docs[kv_index] == 0

        return same_doc & (kv_in_prefix_prompt | same_completion)

    return doc_mask_mod


def input_pos_to_document_ids(
    input_pos: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    assert input_pos.ndim == 2
    # input_pos: (bsz, seqlen). Assumes that input_pos[index] is of the form
    # [0,1,2,0,1,...], where a 0 will mark the beginning of a new document.
    # we subtract by 1 to have the documents be 0-indexed.
    return (input_pos == 0).cumsum(dim=1, dtype=dtype) - 1


def right_side_pad_mask_mod(seq_lens: torch.Tensor) -> MaskMod:
    def pad_mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
    ) -> torch.Tensor:
        valid_len = seq_lens[batch]

        return (query_index < valid_len) & (kv_index < valid_len)

    return pad_mask_mod
