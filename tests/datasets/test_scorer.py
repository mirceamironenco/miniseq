from unittest.mock import MagicMock

import torch

from miniseq.data import PromptBatch
from miniseq.datasets import VerificationScorer
from miniseq.metric_bag import MetricBag


def test_verification_scorer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x, **kwargs: "".join(map(str, x))

    mock_verifier = MagicMock()
    mock_verifier.return_value = {"accuracy": 1.0}

    scorer = VerificationScorer(
        tokenizer=mock_tokenizer,
        answer_keymap=lambda x: x["answer"],
        verifier=mock_verifier,
    )

    batch = PromptBatch(
        prompt_ids=[[1, 2], [3, 4]],
        batch_extras=[{"answer": "125"}, {"answer": "348"}],
    )

    completions = [[5], [8]]

    scores = scorer(completions=completions, batch=batch)

    assert scores == [1.0, 1.0]
    mock_verifier.assert_any_call(guess="125", gold="125", document={"answer": "125"})
    mock_verifier.assert_any_call(guess="348", gold="348", document={"answer": "348"})


def test_verification_scorer_with_metric_bag():
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x, **kwargs: "".join(map(str, x))

    mock_verifier = MagicMock()
    mock_verifier.return_value = {"accuracy": 1.0, "f1": 0.5}

    scorer = VerificationScorer(
        tokenizer=mock_tokenizer,
        answer_keymap=lambda x: x["answer"],
        verifier=mock_verifier,
    )

    batch = PromptBatch(
        prompt_ids=[[1, 2]],
        batch_extras=[{"answer": "125"}],
    )

    completions = [[5]]
    metric_bag = MetricBag(device=torch.device("cpu"))

    scorer(completions=completions, batch=batch, metric_bag=metric_bag)

    values = {name: m.compute().float() for name, m in metric_bag.metrics.items()}

    assert "accuracy" in metric_bag.metrics
    assert "f1" in metric_bag.metrics
    assert torch.isclose(values["accuracy"], torch.tensor(1.0))
    assert torch.isclose(values["f1"], torch.tensor(0.5))
