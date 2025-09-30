import dataclasses
import re

from typing_extensions import TypedDict

from miniseq import cli, recipes
from miniseq import configs as cfg
from miniseq.datasets import Verifier
from miniseq.utils import on_local_rank_zero


def extract_solution(solution_str: str) -> str | None:
    last_row = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"

    matches = re.findall(answer_pattern, last_row)

    if matches:
        return matches[-1].strip()

    return None


def validate_equation(equation_str: str, available_numbers: list[int]) -> bool:
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = list(map(int, re.findall(r"\d+", equation_str)))

        return sorted(numbers_in_eq) == sorted(available_numbers)

    except Exception:
        return False


def evaluate_equation(equation_str: str) -> int | float | None:
    """Safely evaluate the arithmetic equation using eval() with precautions."""

    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"

        if re.match(allowed_pattern, equation_str) is None:
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})

        return result

    except Exception:
        return None


CountdownItem = TypedDict[{"target": int, "nums": list[int]}]


def countdown_verifier(
    *, guess: str, gold: str, document: CountdownItem | None = None
) -> dict[str, float]:
    assert document is not None

    score = format_score = 0.0

    equation = extract_solution(guess)

    if equation is not None:
        # Reward for correct format is 0.1
        format_score += 0.1

        score += format_score

        if validate_equation(equation, document["nums"]):
            result = None

            try:
                result = evaluate_equation(equation)
            except Exception:
                pass

            if result is not None:
                if abs(result - document["target"]) < 1e-5:
                    score += 1.0 - format_score
    return {"correctness": score, "format": format_score}


def qwen_prompt(item: CountdownItem) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant. "
        "You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n"
        f"<|im_start|>user\n Using the numbers {item['nums']}, create an equation that equals {item['target']}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. "
        "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>"
    )


@dataclasses.dataclass
class Config(recipes.OnlineRecipeConfig):
    train_data: cfg.data.PromptDatasetConfig = cfg.data.PromptDatasetConfig(
        path="Jiayi-Pan/Countdown-Tasks-3to4",
        prompt_keymap=qwen_prompt,
        answer_keymap="target",
        assistant_message="\n<|im_start|>assistant\nLet me solve this step by step.\n<think>",
        apply_chat_template=False,
        verifiers=Verifier.from_verifier_map(countdown_verifier),
    )

    lr_scheduler: cfg.LRSchedulerConfig = cfg.scheduler.LinearWarmupCosineConfig(
        warmup_ratio=0.05
    )

    generator: cfg.GeneratorConfig = dataclasses.field(
        default_factory=lambda: cfg.generation.VLLMConfig(
            engine=cfg.generation.VLLMEngineConfig(gpu_memory_utilization=0.45),
            sample=cfg.generation.VLLMSamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=1024,
            ),
        )
    )

    wandb: cfg.WandbConfig | None = cfg.WandbConfig(
        project="miniseq_rl", run_name="qwen3b-flex-prefix", group="countdown"
    )


config = cli.run_default_cli(Config, console_outputs=on_local_rank_zero())
trainer = recipes.create_online_trainer(config)
trainer.run()
