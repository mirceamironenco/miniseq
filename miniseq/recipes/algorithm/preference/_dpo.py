import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from miniseq.data import PreferenceBatch
from miniseq.metric_bag import MetricBag, metrics
from miniseq.recipes.algorithm import (
    model_sequence_logps,
    update_logps,
    update_preference_seqlens,
    update_seq_batch_metrics,
    update_sum_loss,
)
from miniseq.trainer import TrainUnit
from miniseq.transformer import CausalTransformerModel


@torch.inference_mode()
def update_dpo_loss(
    metric_bag: MetricBag, loss: torch.Tensor, batch: PreferenceBatch
) -> None:
    metric_bag.get(metrics.Mean, "dpo_loss").update(
        loss.detach() / batch.chosen_batch.num_examples,
        weight=batch.chosen_batch.num_examples,
    )


def compute_dpo_loss(
    chosen_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logp_ratio_chosen = beta * (chosen_logps - ref_chosen_logps)

    logp_ratio_rejected = beta * (rejected_logps - ref_rejected_logps)

    dpo_loss = -F.logsigmoid(logp_ratio_chosen - logp_ratio_rejected).sum()

    return dpo_loss, logp_ratio_chosen, logp_ratio_rejected


class DPOFinetuneUnit(TrainUnit[PreferenceBatch]):
    _model: CausalTransformerModel
    _reference_model: CausalTransformerModel | None
    _beta: float
    _nll_scale: float
    _length_normalization: bool

    def __init__(
        self,
        model: CausalTransformerModel,
        reference_model: CausalTransformerModel | None,
        *,
        beta: float = 0.1,
        nll_scale: float = 1.0,
        length_normalization: bool = False,
    ) -> None:
        self._model = model
        self._reference_model = reference_model
        self._beta = beta
        self._nll_scale = nll_scale
        self._length_normalization = length_normalization

    @override
    def __call__(
        self, batch: PreferenceBatch, *, metric_bag: MetricBag
    ) -> tuple[torch.Tensor, int | None]:
        chosen_input, chosen_target = batch.chosen_batch.as_auto_regressive()
        rejected_input, rejected_target = batch.rejected_batch.as_auto_regressive()

        if chosen_target.target_mask is None or rejected_target.target_mask is None:
            raise RuntimeError("target_mask must be specified for DPO loss.")

        # (B,)
        chosen_logps, avg_chosen_logps = model_sequence_logps(
            chosen_input, self._model, target_batch=chosen_target
        )

        # (B,)
        rejected_logps, avg_rejected_logps = model_sequence_logps(
            rejected_input, self._model, target_batch=rejected_target
        )

        if self._reference_model is not None:
            with torch.no_grad():
                # (B,)
                ref_chosen_logps, ref_avg_chosen_logps = model_sequence_logps(
                    chosen_input, self._reference_model, target_batch=chosen_target
                )

                # (B,)
                ref_rejected_logps, ref_avg_rejected_logps = model_sequence_logps(
                    rejected_input, self._reference_model, target_batch=rejected_target
                )
        elif batch.ref_chosen is not None and batch.ref_rejected is not None:
            ref_chosen_logps = batch.ref_chosen
            ref_avg_chosen_logps = ref_chosen_logps / chosen_target.target_mask.sum(-1)
            ref_rejected_logps = batch.ref_rejected
            ref_avg_rejected_logps = (
                ref_rejected_logps / rejected_target.target_mask.sum(-1)
            )
        else:
            raise RuntimeError(
                "Reference model is None & batch does not have reference score."
            )

        if self._length_normalization:
            dpo_loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
                avg_chosen_logps,
                ref_avg_chosen_logps,
                avg_rejected_logps,
                ref_avg_rejected_logps,
                self._beta,
            )
        else:
            dpo_loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
                chosen_logps,
                ref_chosen_logps,
                rejected_logps,
                ref_rejected_logps,
                self._beta,
            )

        if self._nll_scale > 0.0:
            nll_loss = -chosen_logps.sum()
        else:
            nll_loss = torch.full((), 0.0, device=batch.chosen_batch.seqs.device)

        loss = (
            dpo_loss
            + self._nll_scale
            * nll_loss
            * chosen_target.num_examples
            / chosen_target.num_target_elements
        )

        # Record metrics
        update_sum_loss(
            metric_bag,
            chosen_rewards.sum(),
            num_targets=chosen_target.num_examples,
            name="chosen_rewards",
        )

        update_sum_loss(
            metric_bag,
            rejected_rewards.sum(),
            num_targets=rejected_target.num_examples,
            name="rejected_rewards",
        )

        reward_margin = chosen_rewards - rejected_rewards

        update_sum_loss(
            metric_bag,
            reward_margin.sum(),
            num_targets=chosen_target.num_examples,
            name="reward_margin",
        )

        update_sum_loss(
            metric_bag,
            (reward_margin > 0).sum(),
            num_targets=chosen_target.num_examples,
            name="accuracy",
        )

        if self._length_normalization:
            update_logps(metric_bag, batch, avg_chosen_logps, avg_rejected_logps)
        else:
            update_logps(metric_bag, batch, chosen_logps, rejected_logps)

        update_dpo_loss(metric_bag, dpo_loss, batch)

        update_preference_seqlens(metric_bag, chosen_target, rejected_target)

        update_sum_loss(metric_bag, nll_loss, chosen_target.num_target_elements)

        update_seq_batch_metrics(metric_bag, chosen_target)

        update_seq_batch_metrics(metric_bag, rejected_target)

        return loss, chosen_target.num_examples

    @property
    @override
    def model(self) -> nn.Module:
        return self._model
