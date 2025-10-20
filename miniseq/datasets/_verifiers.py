from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

from typing_extensions import override

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
