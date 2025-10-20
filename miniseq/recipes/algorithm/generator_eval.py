import itertools

import torch
import torch.nn as nn
from typing_extensions import override

from miniseq.data import CompletionScorer, PromptBatch
from miniseq.evaluator import EvalUnit
from miniseq.generation import Generator
from miniseq.metric_bag import MetricBag, metrics
from miniseq.recipes.algorithm import update_lengths
from miniseq.utils import to_tensor


@torch.inference_mode()
def update_scores(
    metric_bag: MetricBag, total_score: torch.Tensor, num_answers: int
) -> None:
    assert total_score.numel() == 1

    metric_bag.get(metrics.Mean, "total_score").update(
        total_score.detach() / num_answers, weight=num_answers
    )


class GeneratorEvalUnit(EvalUnit[PromptBatch]):
    _generator: Generator
    _completion_scorer: CompletionScorer
    _name: str | None
    _pass_k: int
    _avg_n: int

    def __init__(
        self,
        generator: Generator,
        completion_scorer: CompletionScorer,
        name: str | None = None,
        pass_k: int = 1,
        avg_n: int = 1,
    ) -> None:
        self._generator = generator
        self._completion_scorer = completion_scorer
        self._name = name
        self._pass_k = pass_k
        self._avg_n = avg_n

        assert min(avg_n, pass_k) >= 1

        if pass_k > 1 and avg_n > 1:
            raise ValueError(
                f"Either avg@n must be 1 (got {avg_n}) or pass@k must be 1 (got {pass_k})."
            )

    @override
    def __call__(self, batch: PromptBatch, *, metric_bag: MetricBag) -> None:
        input_prompt_ids = batch.prompt_ids

        repeats = 1

        if self._pass_k > 1 or self._avg_n > 1:
            repeats = max(self._pass_k, self._avg_n)

            input_prompt_ids = [
                input_ids[:]
                for prompt_ids in batch.prompt_ids
                for input_ids in itertools.repeat(prompt_ids, repeats)
            ]

        # Generate completions
        completions, _ = self._generator.generate(input_prompt_ids)

        # Score completions
        scores = self._completion_scorer(
            completions=completions,
            batch=batch,
            repetitions=repeats,
            metric_bag=metric_bag,
        )

        device = metric_bag.device

        prompt_lens = to_tensor(
            list(map(len, input_prompt_ids)),
            dtype=torch.int64,
            device=metric_bag.device,
        )

        completion_lens = to_tensor(
            list(map(len, completions)), dtype=torch.float, device=device
        )

        scores_pt = to_tensor(scores, dtype=torch.float, device=device)

        if self._pass_k > 1:
            scores_pt = (scores_pt.reshape(-1, self._pass_k).sum(-1) > 0).float()

        update_scores(metric_bag, scores_pt.sum(), num_answers=scores_pt.numel())

        update_lengths(
            metric_bag,
            prompt_lens.sum(),
            num_sequences=prompt_lens.numel(),
            name="prompt_len",
        )

        update_lengths(
            metric_bag,
            completion_lens.sum(),
            num_sequences=completion_lens.numel(),
            name="completion_len",
        )

    @property
    @override
    def model(self) -> nn.Module:
        return self._generator.model

    @property
    @override
    def name(self) -> str | None:
        return self._name
