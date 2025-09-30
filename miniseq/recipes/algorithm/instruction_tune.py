from typing import Protocol

import torch
import torch.nn as nn
from typing_extensions import override

from miniseq.data import SequenceBatch
from miniseq.evaluator import EvalUnit
from miniseq.metric_bag import MetricBag
from miniseq.recipes.algorithm._metrics import (
    update_seq_batch_metrics,
    update_sum_loss,
)
from miniseq.trainer import TrainUnit
from miniseq.transformer import CausalTransformerModel


class InstructionSumLoss(Protocol):
    def __call__(
        self,
        model: CausalTransformerModel,
        input_batch: SequenceBatch,
        target_batch: SequenceBatch,
    ) -> torch.Tensor: ...


def nll_loss(
    model: CausalTransformerModel,
    input_batch: SequenceBatch,
    target_batch: SequenceBatch,
) -> torch.Tensor:
    seqs, input_pos, attention_mask = input_batch.as_input()

    loss = model(
        seqs,
        target_batch.seqs,
        attn_mask=attention_mask,
        input_pos=input_pos,
        reduction="sum",
        target_mask=target_batch.target_mask,
        num_valids=target_batch.num_target_elements,
    )

    return loss


class InstructionUnit(TrainUnit[SequenceBatch]):
    _model: CausalTransformerModel
    _sum_loss: InstructionSumLoss
    _name: str

    def __init__(
        self,
        model: CausalTransformerModel,
        sum_loss: InstructionSumLoss = nll_loss,
        name: str = "nll_loss",
    ) -> None:
        self._model = model
        self._sum_loss = sum_loss
        self._name = name

    @override
    def __call__(
        self, batch: SequenceBatch, *, metric_bag: MetricBag
    ) -> tuple[torch.Tensor, int | None]:
        input_batch, target_batch = batch.as_auto_regressive()

        loss = self._sum_loss(self._model, input_batch, target_batch)

        update_sum_loss(
            metric_bag,
            loss,
            num_targets=target_batch.num_target_elements,
            name=self._name,
        )

        update_seq_batch_metrics(metric_bag, target_batch)

        return loss, target_batch.num_target_elements

    @property
    @override
    def model(self) -> nn.Module:
        return self._model


class InstructionEvalUnit(EvalUnit[SequenceBatch]):
    _model: CausalTransformerModel
    _sum_loss: InstructionSumLoss
    _name: str

    def __init__(
        self,
        model: CausalTransformerModel,
        sum_loss: InstructionSumLoss = nll_loss,
        name: str = "eval_nll",
    ) -> None:
        self._model = model
        self._sum_loss = sum_loss
        self._name = name

    @override
    def __call__(self, batch: SequenceBatch, *, metric_bag: MetricBag) -> None:
        input_batch, target_batch = batch.as_auto_regressive()

        loss = self._sum_loss(self._model, input_batch, target_batch)

        update_sum_loss(
            metric_bag,
            loss,
            num_targets=target_batch.num_target_elements,
            name=self._name,
        )

        update_seq_batch_metrics(metric_bag, target_batch)

    @property
    @override
    def model(self) -> nn.Module:
        return self._model

    @property
    @override
    def name(self) -> str | None:
        return self._name


class AcuracyEvalUnit(EvalUnit[SequenceBatch]):
    _model: CausalTransformerModel

    def __init__(self, model: CausalTransformerModel) -> None:
        self._model = model

    @override
    def __call__(self, batch: SequenceBatch, *, metric_bag: MetricBag) -> None:
        input_batch, target_batch = batch.as_auto_regressive()

        seqs, input_pos, attention_mask = input_batch.as_input()

        logits = self._model(seqs, attn_mask=attention_mask, input_pos=input_pos)

        # (B, S, V) -> (B, S)
        preds = logits.argmax(dim=-1)

        assert target_batch.target_mask is not None

        accuracy = (preds == target_batch.seqs).float() * target_batch.target_mask

        update_sum_loss(
            metric_bag,
            accuracy.sum(),
            num_targets=target_batch.num_target_elements,
            name="accuracy",
        )

        update_seq_batch_metrics(metric_bag, target_batch)

    @property
    @override
    def model(self) -> nn.Module:
        return self._model

    @property
    @override
    def name(self) -> str | None:
        return "accuracy_unit"
