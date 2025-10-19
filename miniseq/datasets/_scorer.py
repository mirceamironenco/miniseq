from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import torch
from typing_extensions import override

from miniseq.data import CompletionScorer, PretrainedHFTokenizer, PromptBatch
from miniseq.datasets._verifiers import Verifier
from miniseq.metric_bag import MetricBag, metrics
from miniseq.utils import to_tensor

T = TypeVar("T")


class AbstractScorer(CompletionScorer[T], ABC):
    _tokenizer: PretrainedHFTokenizer
    _answer_keymap: Callable[..., str]

    def __init__(
        self, *, tokenizer: PretrainedHFTokenizer, answer_keymap: Callable[..., str]
    ) -> None:
        self._tokenizer = tokenizer
        self._answer_keymap = answer_keymap

    @abstractmethod
    def __call__(
        self,
        *,
        completions: list[list[int]],
        batch: PromptBatch[T],
        repetitions: int = 1,
        metric_bag: MetricBag | None = None,
    ) -> list[float]: ...


@torch.inference_mode()
def update_verifier_score(
    metric_bag: MetricBag, total_score: torch.Tensor, num_answers: int, name: str
) -> None:
    assert total_score.numel() == 1

    metric_bag.get(metrics.Mean, name).update(
        total_score / num_answers, weight=num_answers
    )


class VerificationScorer(AbstractScorer):
    _verifier: Verifier

    def __init__(
        self,
        tokenizer: PretrainedHFTokenizer,
        answer_keymap: Callable[..., str],
        *,
        verifier: Verifier,
    ) -> None:
        super().__init__(tokenizer=tokenizer, answer_keymap=answer_keymap)
        self._verifier = verifier

    @override
    def __call__(
        self,
        *,
        completions: list[list[int]],
        batch: PromptBatch,
        repetitions: int = 1,
        metric_bag: MetricBag | None = None,
    ) -> list[float]:
        num_completions = len(completions)

        if num_completions != repetitions * (n_extr := len(batch.batch_extras)):
            raise ValueError(
                f"num_completions ({num_completions}) != repetitions ({repetitions}) * len(promp_extras) ({n_extr})"
            )

        total_scores = [0.0] * num_completions

        verifier_total_scores: dict[str, float] = collections.defaultdict(float)

        for index in range(num_completions):
            prompt = batch.prompt_ids[index // repetitions]

            prompt_and_response = self._tokenizer.decode(
                prompt + completions[index], skip_special_tokens=False
            )

            document = batch.batch_extras[index // repetitions]

            answer = self._answer_keymap(document)

            verifier_scores = self._verifier(
                guess=prompt_and_response, gold=answer, document=document
            )

            for score_name, score_value in verifier_scores.items():
                total_scores[index] += score_value
                verifier_total_scores[score_name] += score_value

        if metric_bag is not None:
            for score_name, total_score in verifier_total_scores.items():
                score_pt = to_tensor(
                    total_score,
                    device=metric_bag.device,
                    dtype=torch.float32,
                )

                update_verifier_score(metric_bag, score_pt, num_completions, score_name)

        return total_scores
