import math_verify
from typing_extensions import TypedDict

from miniseq.datasets import register_prompt_dataset
from miniseq.datasets.math_verify import MathVerifier


def register_eval_datasets() -> None:
    register_prompt_dataset(
        "math500",
        path="HuggingFaceH4/MATH-500",
        prompt_keymap="problem",
        answer_keymap="solution",
        split="test",
        assistant_message=None,
        prompt_transform=None,
        apply_chat_template=True,
        schema=TypedDict[{"problem": str, "solution": str}],
        verifier=MathVerifier(
            verbose=False,
            gold_extraction_target=[
                math_verify.LatexExtractionConfig(),
                math_verify.ExprExtractionConfig(),
            ],
            guess_extraction_target=[
                math_verify.ExprExtractionConfig(),
                math_verify.LatexExtractionConfig(boxed_match_priority=0),
            ],
        ),
    )

    register_prompt_dataset(
        "olympiad",
        path="knoveleng/OlympiadBench",
        prompt_keymap="question",
        answer_keymap="answer",
        split="train",
        assistant_message=None,
        prompt_transform=None,
        apply_chat_template=True,
        verifier=MathVerifier(
            verbose=False,
            gold_extraction_target=[math_verify.LatexExtractionConfig()],
            guess_extraction_target=[
                math_verify.ExprExtractionConfig(),
                math_verify.LatexExtractionConfig(boxed_match_priority=0),
            ],
        ),
        schema=TypedDict[{"question": str, "answer": str}],
    )

    register_prompt_dataset(
        "minerva",
        path="knoveleng/Minerva-Math",
        prompt_keymap="problem",
        answer_keymap="solution",
        split="train",
        assistant_message=None,
        prompt_transform=None,
        apply_chat_template=True,
        verifier=MathVerifier(
            verbose=False,
            gold_extraction_target=[math_verify.LatexExtractionConfig()],
            guess_extraction_target=[
                math_verify.ExprExtractionConfig(),
                math_verify.LatexExtractionConfig(boxed_match_priority=0),
            ],
        ),
        schema=TypedDict[{"problem": str, "solution": str}],
    )

    register_prompt_dataset(
        "amc",
        path="knoveleng/AMC-23",
        prompt_keymap="problem",
        answer_keymap="answer",
        split="train",
        assistant_message=None,
        prompt_transform=None,
        apply_chat_template=True,
        verifier=MathVerifier(
            verbose=True,
            gold_extraction_target=[math_verify.ExprExtractionConfig()],
            guess_extraction_target=[
                math_verify.ExprExtractionConfig(),
                math_verify.LatexExtractionConfig(boxed_match_priority=0),
            ],
        ),
        schema=TypedDict[{"problem": str, "answer": str}],
    )

    register_prompt_dataset(
        "aime24",
        path="HuggingFaceH4/aime_2024",
        prompt_keymap="problem",
        answer_keymap="answer",
        split="train",
        assistant_message=None,
        prompt_transform=None,
        apply_chat_template=True,
        verifier=MathVerifier(
            verbose=False,
            gold_extraction_target=[math_verify.ExprExtractionConfig()],
            guess_extraction_target=[
                math_verify.ExprExtractionConfig(),
                math_verify.LatexExtractionConfig(boxed_match_priority=0),
            ],
        ),
        schema=TypedDict[{"problem": str, "answer": str}],
    )
