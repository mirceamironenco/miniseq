import torch

from miniseq.data import SequenceBatch
from miniseq.transformer import CausalTransformerModel


def model_logps(
    input_batch: SequenceBatch,
    model: CausalTransformerModel,
    *,
    target_batch: SequenceBatch,
    temperature: float | None = None,
) -> torch.Tensor:
    seqs, input_pos, attn_mask = input_batch.as_input()

    # (B, S)
    nll = model(
        seqs,
        target_batch.seqs,
        attn_mask=attn_mask,
        input_pos=input_pos,
        target_mask=target_batch.target_mask,
        reduction="none",
        temperature=temperature,
        num_valids=target_batch.num_target_elements,
    )

    # (B, S)
    logprobs = -nll

    return logprobs


def model_sequence_logps(
    input_batch: SequenceBatch,
    model: CausalTransformerModel,
    *,
    target_batch: SequenceBatch,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    seqs, input_pos, attn_mask = input_batch.as_input()

    target_mask = target_batch.target_mask

    assert target_mask is not None

    # (B, S)
    nll = model(
        seqs,
        target_batch.seqs,
        attn_mask=attn_mask,
        input_pos=input_pos,
        reduction="none",
        target_mask=target_mask,
        temperature=temperature,
        num_valids=target_batch.num_target_elements,
    )

    # (B, )
    if input_batch.is_packed:
        logprobs = -packed_scatter_sum_reduce(
            nll,
            document_ids=input_batch.document_ids,
            num_sequences=input_batch.num_examples,
        )

        sequence_target_mask = packed_scatter_sum_reduce(
            target_mask,
            document_ids=input_batch.document_ids,
            num_sequences=input_batch.num_examples,
        )

        avg_logprobs = logprobs / sequence_target_mask
    else:
        logprobs = -nll.sum(dim=-1)

        avg_logprobs = logprobs / target_mask.sum(-1)

    return logprobs, avg_logprobs


@torch.compile(dynamic=True)
def packed_scatter_sum_reduce(
    input: torch.Tensor, *, document_ids: torch.Tensor, num_sequences: int
) -> torch.Tensor:
    assert input.ndim == 2 and input.size(0) == 1

    if not document_ids.size() == input.size():
        raise ValueError(
            f"Incompatible document ids in reduction, expected shape {input.size()} got {document_ids.size()}"
        )

    # Upcast to float32 for precise accumulation, and convert back afterwards.
    result = torch.zeros(num_sequences, device=input.device, dtype=torch.float32)

    result.scatter_add_(
        0, index=document_ids.flatten().long(), src=input.flatten().float()
    )

    return result.type_as(input)


def prefix_packed_scatter_sum_reduce(
    input: torch.Tensor, *, prefix_docs: list[int], share_count: int
) -> torch.Tensor:
    assert input.ndim == 2 and input.size(0) == 1

    assert len(prefix_docs) == input.size(1)

    # Upcast to float32 for precise accumulation, and convert back afterwards.
    result = torch.zeros(
        int(prefix_docs[-1] + 1), device=input.device, dtype=torch.float32
    )

    document_ids = torch.tensor(prefix_docs, device=input.device, dtype=torch.int64)

    result.scatter_add_(0, index=document_ids, src=input.flatten().float())

    result = result.type_as(input)[1:].view(-1, share_count)

    return result


@torch.compile(dynamic=True)
def packed_segment_sum_reduce(
    input: torch.Tensor, *, seq_lens: torch.Tensor
) -> torch.Tensor:
    # Note: It is recommended to use a reduce based on scatter_add_ since it is
    # less dependent on seq_lens. seq_lens for packed + prefix sharing will
    # merge the prompt and first completion, making the seq_len for the remaining
    # completions inaccurate. Since prefix sharing is only supported for completion_only
    # training, the result is however identical after proper masking.
    assert input.ndim == 2 and input.size(0) == 1

    # Note: unsafe=True because seq_lens.sum() != input.flatten().numel()
    # since input might have padding.
    result = torch.segment_reduce(
        input.flatten().float(), reduce="sum", lengths=seq_lens, unsafe=True
    )

    return result.type_as(input)
