from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import math_verify
from math_verify.parser import (
    ExprExtractionConfig,
    ExtractionTarget,
    LatexExtractionConfig,
)
from typing_extensions import override

from miniseq.datasets._verifiers import Verifier
from miniseq.logging import get_logger

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
