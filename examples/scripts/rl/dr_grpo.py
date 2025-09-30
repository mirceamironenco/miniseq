from dataclasses import dataclass, field

import math_verify
from math_benchmark import register_eval_datasets
from typing_extensions import TypedDict

from miniseq import cli, recipes
from miniseq import configs as cfg
from miniseq.datasets import MathVerifier
from miniseq.recipes.algorithm.online import GRPOConfig
from miniseq.utils import on_local_rank_zero


def boxify(ground_truth: str) -> str:
    if "\\boxed" not in ground_truth:
        boxed_ground_truth = f"\\boxed{{{ground_truth}}}"
    else:
        boxed_ground_truth = ground_truth

    return boxed_ground_truth


MATH12KItem = TypedDict[{"problem": str, "answer": str}]


def apply_qwen_math_template(problem: str) -> str:
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + problem
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def understand_r1_gt(item: MATH12KItem) -> str:
    ground_truth = item["answer"].strip()

    boxed_ground_truth = boxify(ground_truth)

    assert len(ground_truth) > 0
    assert len(boxed_ground_truth) > 0

    return boxed_ground_truth


@dataclass
class Config(recipes.OnlineRecipeConfig):
    train_data: cfg.data.PromptDatasetConfig = cfg.data.PromptDatasetConfig(
        path="lkevinzc/math-12k",
        prompt_keymap="problem",
        prompt_transform=apply_qwen_math_template,
        answer_keymap=understand_r1_gt,
        system_message=None,
        assistant_message=None,
        split="train",
        apply_chat_template=False,
        filter_map=lambda item: len(item["answer"].strip()) > 0,
        verifiers=MathVerifier(
            gold_extraction_target=[
                math_verify.LatexExtractionConfig(),
                math_verify.ExprExtractionConfig(),
            ],
            guess_extraction_target=[
                math_verify.LatexExtractionConfig(boxed_match_priority=0),
                math_verify.ExprExtractionConfig(),
            ],
        ),
    )

    valid_data: cfg.data.RegisteredDatasetConfig | None = field(
        default_factory=lambda: cfg.data.RegisteredDatasetConfig(
            datasets=("aime24", "amc", "minerva", "math500"), batch_size=100
        )
    )

    model: cfg.PretrainedModelConfig = cfg.PretrainedModelConfig(
        name="qwen2.5-math-1.5b"
    )

    optimizer: cfg.OptimizerConfig = cfg.optim.AdamWConfig(lr=1e-6, weight_decay=0.01)

    lr_scheduler: cfg.LRSchedulerConfig = cfg.scheduler.NoopSchedulerConfig()

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
        validate_every=50,
        publish_metrics_every=2,
    )

    grpo: GRPOConfig = GRPOConfig(
        group_size=8,
        mu=1,
        std_normalize_advantage=False,
        clip_eps_low=0.2,
        clip_eps_high=0.2,
        beta=0.0,
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
                temperature=0.0,
                top_p=1.0,
                max_tokens=3000,
            ),
        )
    )

    wandb: cfg.WandbConfig | None = cfg.WandbConfig(
        project="miniseq_repro", run_name="qwen_math1.5b_drgrpo"
    )

    packed: bool = True

    prefix_share: bool = True


def main() -> None:
    register_eval_datasets()

    config = cli.run_default_cli(Config, console_outputs=on_local_rank_zero())

    if config.valid_data is not None:
        config.valid_data.apply_chat_template = False
        config.valid_data.system_message = None
        config.valid_data.assistant_message = None
        config.valid_data.prompt_transform = apply_qwen_math_template

    trainer = recipes.create_online_trainer(config)

    trainer.run()


if __name__ == "__main__":
    main()
