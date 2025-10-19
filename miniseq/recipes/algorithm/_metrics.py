import torch

from miniseq.data import PreferenceBatch, SequenceBatch
from miniseq.metric_bag import MetricBag, metrics
from miniseq.utils import to_tensor


@torch.inference_mode()
def update_sum_loss(
    metric_bag: MetricBag,
    loss: torch.Tensor,
    num_targets: int | None = None,
    name: str = "nll_loss",
) -> None:
    loss = loss.detach().float()

    n = num_targets or 1

    metric_bag.get(metrics.Mean, name).update(loss / n, weight=n)


@torch.inference_mode()
def update_seq_batch_metrics(metric_bag: MetricBag, batch: SequenceBatch) -> None:
    num_examples, num_elements, num_target_elements, padding = map(
        lambda t: to_tensor(t, device=metric_bag.device),
        (
            batch.num_examples,
            batch.num_elements,
            batch.num_target_elements,
            batch.padding,
        ),
    )

    metric_bag.get(metrics.Sum, "perf/num_examples").update(num_examples)

    metric_bag.get(metrics.Sum, "perf/num_elements").update(num_elements)

    metric_bag.get(metrics.Sum, "perf/num_target_elements").update(num_target_elements)

    metric_bag.get(metrics.Sum, "perf/total_num_examples").update(num_examples)

    metric_bag.get(metrics.Sum, "perf/total_num_elements").update(num_elements)

    metric_bag.get(metrics.Sum, "perf/total_num_target_elements").update(
        num_target_elements
    )

    metric_bag.get(metrics.Sum, "batch/padding").update(padding)


@torch.inference_mode()
def update_preference_seqlens(
    metric_bag: MetricBag, chosen_batch: SequenceBatch, rejected_batch: SequenceBatch
) -> None:
    if chosen_batch.is_packed:
        assert chosen_batch.seq_lens is not None
        assert rejected_batch.seq_lens is not None
        weight_chosen = len(chosen_batch.seq_lens)
        weight_rejected = len(rejected_batch.seq_lens)
    else:
        weight_chosen = chosen_batch.batch_size
        weight_rejected = rejected_batch.batch_size

    metric_bag.get(metrics.Mean, "chosen_lengths").update(
        to_tensor(
            chosen_batch.num_target_elements / weight_chosen,
            device=metric_bag.device,
        ),
        weight=weight_chosen,
    )

    metric_bag.get(metrics.Mean, "rejected_lengths").update(
        to_tensor(
            rejected_batch.num_target_elements / weight_rejected,
            device=metric_bag.device,
        ),
        weight=weight_rejected,
    )


@torch.inference_mode()
def update_lengths(
    metric_bag: MetricBag,
    total_length: torch.Tensor,
    num_sequences: int,
    name: str = "completion_len",
) -> None:
    assert total_length.numel() == 1

    metric_bag.get(metrics.Mean, name).update(
        total_length / num_sequences, weight=num_sequences
    )


@torch.inference_mode()
def update_logps(
    metric_bag: MetricBag,
    batch: PreferenceBatch,
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
) -> None:
    metric_bag.get(metrics.Mean, "chosen_logps").update(
        chosen_logps.detach().sum() / batch.chosen_batch.num_examples,
        weight=batch.chosen_batch.num_examples,
    )

    metric_bag.get(metrics.Mean, "rejected_logps").update(
        rejected_logps.detach().sum() / batch.rejected_batch.num_examples,
        weight=batch.rejected_batch.num_examples,
    )
