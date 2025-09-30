import torch

from miniseq.data import SequenceBatch, SequenceExample, UnfinishedPrefixPack
from miniseq.recipes.algorithm._common import (
    packed_scatter_sum_reduce,
    prefix_packed_scatter_sum_reduce,
)


def test_prefix_packed_reduction():
    """
    Tests that prefix-packed reduction gives the same result as packed reduction
    when there is only one completion per prompt.
    """
    # 1. Create a prefix-packed batch with share_count=1
    share_count = 1
    pack = UnfinishedPrefixPack(max_seq_len=100, share_count=share_count)

    prompts = [
        torch.tensor([1, 2]),
        torch.tensor([3, 4, 5]),
    ]
    completions = [
        torch.tensor([6, 7, 8]),
        torch.tensor([9, 10]),
    ]

    for prompt, completion in zip(prompts, completions):
        pack.maybe_update(
            [
                SequenceExample.from_instruction(
                    **{
                        "prompt": prompt,
                        "completion": completion,
                        "completion_only": True,
                    }
                )
            ]
        )

    packed_example = pack.finish(pad_idx=0)
    batch = SequenceBatch(
        seqs=packed_example.indices.unsqueeze(0),
        seq_lens=packed_example.seq_lens,
        input_pos=packed_example.input_pos.unsqueeze(0),
        target_mask=packed_example.target_mask.unsqueeze(0),
        is_packed=True,
        is_padded=packed_example.padding > 0,
        pad_idx=0,
        examples_used=len(prompts),
        prefix_docs=packed_example.inner_docs,
        prefix_share_count=share_count,
    )

    # 2. Create some dummy logprobs
    assert batch.target_mask is not None
    logprobs = torch.randn_like(batch.seqs, dtype=torch.float32)
    logprobs = logprobs * batch.target_mask

    # 3. Reduce using both methods
    assert batch.prefix_docs is not None
    prefix_reduced = prefix_packed_scatter_sum_reduce(
        logprobs,
        prefix_docs=batch.prefix_docs,
        share_count=share_count,
    )
    packed_reduced = packed_scatter_sum_reduce(
        logprobs,
        document_ids=batch.document_ids,
        num_sequences=batch.num_examples,
    )

    # 4. Compare the results
    # The shapes will be different, (N, 1) vs (N,), so we flatten.
    assert torch.allclose(prefix_reduced.flatten(), packed_reduced.flatten())
