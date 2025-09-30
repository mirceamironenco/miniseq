import re
from dataclasses import dataclass, field
from typing import Literal

from typing_extensions import TypedDict

from miniseq import cli, recipes
from miniseq import configs as cfg
from miniseq.datasets import Verifier
from miniseq.recipes.algorithm.online import GRPOConfig
from miniseq.utils import on_local_rank_zero

GSM8KItem = TypedDict[{"question": str, "answer": str}]


def extract_gsm8k_answer(item: GSM8KItem) -> str:
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", item["answer"])
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def verl_gsm8k_prompt(question: str) -> str:
    # https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py
    return (
        question.strip()
        + " "
        + 'Let\'s think step by step and output the final answer after "####".'
    )


_SOLUTION_CLIP_CHARS = 300


def extract_solution(
    solution_str: str, method: Literal["flexible", "strict"] = "flexible"
) -> str | None:
    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    else:
        raise ValueError(f"method bust me strict/flexible, got {method}")

    return final_answer


def gsm8k_verifier(
    *, guess: str, gold: str, document: GSM8KItem | None = None
) -> dict[str, float]:
    # Source: https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py
    format_score, score = 0.1, 1.0
    answer = extract_solution(solution_str=guess, method="strict")

    if answer is None:
        return {"correctness": 0.0}
    else:
        if answer == gold:
            return {"correctness": score, "format": format_score}
        else:
            return {"correctness": 0.0, "format": format_score}


@dataclass
class Config(recipes.OnlineRecipeConfig):
    train_data: cfg.data.PromptDatasetConfig = cfg.data.PromptDatasetConfig(
        path="openai/gsm8k",
        prompt_keymap="question",
        prompt_transform=verl_gsm8k_prompt,
        answer_keymap=extract_gsm8k_answer,
        system_message=None,
        split="train",
        test_split="test",
        configuration="main",
        apply_chat_template=False,
        verifiers=Verifier.from_verifier_map(gsm8k_verifier),
    )

    valid_data: cfg.data.RegisteredDatasetConfig | None = None

    model: cfg.PretrainedModelConfig = cfg.PretrainedModelConfig(
        name="qwen2.5-0.5b-instruct"
    )

    optimizer: cfg.OptimizerConfig = cfg.optim.AdamWConfig(lr=1e-6, weight_decay=0.01)

    lr_scheduler: cfg.LRSchedulerConfig = cfg.scheduler.LinearWarmupCosineConfig(
        warmup_ratio=0.02
    )

    train: cfg.TrainConfig = cfg.TrainConfig(
        ac=True,
        max_epochs=None,
        max_steps=300,
        device_batch_size=128,
        rollout_batch_size=64,
        micro_batch_size=4,
        no_sync=False,
        max_grad_norm=1.0,
        checkpoint_every=50,
        validate_at_start=False,
        validate_every=25,
        publish_metrics_every=2,
    )

    grpo: GRPOConfig = GRPOConfig(
        group_size=8,
        mu=1,
        std_normalize_advantage=False,
        clip_eps_low=0.2,
        clip_eps_high=0.28,
        beta=0.001,
        rollout_correction=True,
    )

    generator: cfg.GeneratorConfig = field(
        default_factory=lambda: cfg.generation.VLLMConfig(
            engine=cfg.generation.VLLMEngineConfig(gpu_memory_utilization=0.45),
            sample=cfg.generation.VLLMSamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=3000,
            ),
            eval_sample=cfg.generation.VLLMSamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=3600,
            ),
        )
    )


config = cli.run_default_cli(Config, console_outputs=on_local_rank_zero())
trainer = recipes.create_online_trainer(config)
trainer.run()
