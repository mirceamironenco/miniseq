from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypeVar

import math_verify
from math_verify.parser import (
    ExprExtractionConfig,
    ExtractionTarget,
    LatexExtractionConfig,
)
from typing_extensions import override

from miniseq.logging import get_logger

MapItemT = TypeVar("MapItemT", bound=Mapping[str, Any], contravariant=True)


class VerifierMap(Protocol[MapItemT]):
    """A callable that verifies a guess against a gold standard.

    Example:
        A simple implementation that checks for an exact match:

        >>> def exact_match_verifier(
        ...     *, guess: str, gold: str, document: dict | None = None
        ... ) -> dict[str, float]:
        ...     \"\"\"Returns 1.0 if guess and gold are identical, else 0.0.\"\"\"
        ...     score = 1.0 if guess.strip() == gold.strip() else 0.0
        ...     return {"score": score}
        ...
        >>> # This VerifierMap is used to create a Verifier.
        >>> verifier = Verifier.from_verifier_map(exact_match_verifier)
        >>> # The Verifier can then be called.
        >>> result = verifier(guess="hello world", gold="hello world")
        >>> print(result) # {'score': 1.0}
    """

    def __call__(
        self, *, guess: str, gold: str, document: MapItemT | None = None
    ) -> dict[str, float]: ...


class OutcomeMap(Protocol[MapItemT]):
    """A callable that evaluates a guess and returns a direct outcome.

    Example:
        A simple boolean implementation that checks for an exact match:

        >>> def is_exact_match(
        ...     *, guess: str, gold: str, document: dict | None = None
        ... ) -> bool:
        ...     \"\"\"Returns True if guess and gold are identical, else False.\"\"\"
        ...     return guess.strip() == gold.strip()
        >>> # This OutcomeMap is used to create a Verifier.
        >>> verifier = Verifier.from_outcome_map(is_exact_match, name="exact_match")
        >>> # The Verifier returns a standardized dictionary.
        >>> result = verifier(guess="hello world", gold="hello world")
        >>> print(result) # {'exact_match': 1.0}
        >>> result = verifier(guess="hello", gold="world")
        >>> print(result) # {'exact_match': 0.0}
    """

    def __call__(
        self, *, guess: str, gold: str, document: MapItemT | None = None
    ) -> float | int | bool: ...


class Verifier(ABC):
    @abstractmethod
    def __call__(
        self, *, guess: str, gold: str, document: Mapping[str, Any] | None = None
    ) -> dict[str, float]: ...

    def __add__(self, other: Verifier) -> ChainedVerifier:
        return ChainedVerifier(self, other)

    @classmethod
    def from_verifier_map(cls, verifier_map: VerifierMap) -> MapVerifier:
        return MapVerifier(verifier_map)

    @classmethod
    def from_outcome_map(
        cls, outcome_map: OutcomeMap, name: str | None = None
    ) -> MapVerifier:
        _score_name = name or "score"

        def _verifier_map(*, guess, gold, document=None) -> dict[str, float]:
            score = outcome_map(guess=guess, gold=gold, document=document)
            return {_score_name: float(score)}

        return MapVerifier(_verifier_map)

    @classmethod
    def from_outcome_table(
        cls, table: list[OutcomeMap], names: list[str] | None = None
    ) -> ChainedVerifier:
        verifiers: list[Verifier] = []

        for index in range(len(table)):
            verifiers.append(
                Verifier.from_outcome_map(
                    table[index], names[index] if names is not None else None
                )
            )

        return ChainedVerifier(*verifiers)


class MapVerifier(Verifier):
    def __init__(self, verifier_map: VerifierMap) -> None:
        self._verifier_map = verifier_map

    @override
    def __call__(
        self, *, guess: str, gold: str, document: Mapping[str, Any] | None = None
    ) -> dict[str, float]:
        return self._verifier_map(guess=guess, gold=gold, document=document)


class ChainedVerifier(Verifier):
    def __init__(self, *verifiers: Verifier) -> None:
        self._verifiers = verifiers

    @override
    def __call__(
        self, *, guess: str, gold: str, document: Mapping[str, Any] | None = None
    ) -> dict[str, float]:
        all_scores: dict[str, float] = {}

        for index, verifier in enumerate(self._verifiers):
            scores = verifier(guess=guess, gold=gold, document=document)

            for score_name, score_value in scores.items():
                final_key = score_name

                # Allow for verifiers with the same score names.
                # Only change the key if the overlaps occur
                if final_key in all_scores:
                    final_key = f"{score_name}_{index}"

                    assert final_key not in all_scores

                all_scores.update({final_key: score_value})

        return all_scores


class EqualityVerifier(Verifier):
    @override
    def __call__(
        self, *, guess: str, gold: str, document: Mapping[str, Any] | None = None
    ) -> dict[str, float]:
        score = 1.0 if guess == gold else 0.0

        return {"equality": score}


_log = get_logger()


class MathVerifier(Verifier):
    _guess_extraction_mode: Literal["first_match", "any_match"]
    _gold_extraction_mode: Literal["first_match", "any_match"]

    def __init__(
        self,
        float_rounding: int = 6,
        numeric_precision: int = 15,
        strict: bool = True,
        timeout_seconds: int = 5,
        guess_extraction_mod: Literal["first_match", "any_match"] = "any_match",
        gold_extraction_mod: Literal["first_match", "any_match"] = "any_match",
        gold_extraction_target: Sequence[ExtractionTarget] = [
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ],
        guess_extraction_target: Sequence[ExtractionTarget] = [
            LatexExtractionConfig(),
            ExprExtractionConfig(),
        ],
        verbose: bool = False,
    ) -> None:
        self._float_rounding = float_rounding
        self._numeric_precision = numeric_precision
        self._strict = strict
        self._timeout_seconds = timeout_seconds
        self._guess_extraction_mode = guess_extraction_mod
        self._gold_extraction_mode = gold_extraction_mod
        self._guess_extraction_target = guess_extraction_target
        self._gold_extraxtion_target = gold_extraction_target
        self._verbose = verbose

    @override
    def __call__(
        self, *, guess: str, gold: str, document: Mapping[str, Any] | None = None
    ) -> dict[str, float]:
        parsed_guess = math_verify.parse(
            guess,
            extraction_mode=self._guess_extraction_mode,
            extraction_config=self._guess_extraction_target,
        )

        parsed_answer = math_verify.parse(
            gold,
            extraction_mode=self._gold_extraction_mode,
            extraction_config=self._gold_extraxtion_target,
        )

        is_correct = math_verify.verify(
            gold=parsed_answer,
            target=parsed_guess,
            float_rounding=self._float_rounding,
            numeric_precision=self._numeric_precision,
            strict=self._strict,
            timeout_seconds=self._timeout_seconds,
        )

        if not is_correct and self._verbose:
            _log.info(
                "===\n"
                + f"Incorrect math_verify \n completion: {guess[:500]} \n...\n {guess[-150:]}"
                + f"\n guess: {parsed_guess} \n answer: {parsed_answer}\n"
                + "\n ==="
            )

        # Always print that we couldn't parse the answer
        if len(parsed_answer) == 0:
            _log.info(f"Math verify could not parse answer: {gold}")

        return {"math_verify": float(is_correct)}
