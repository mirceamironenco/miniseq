import dataclasses

from math_benchmark import register_eval_datasets

from miniseq import cli, recipes
from miniseq.utils import on_local_rank_zero


def r1_system() -> str:
    return "Please reason step by step, and put your final answer within \\boxed{{}}.\n"


def r1_prompt(question: str) -> str:
    return f"<|User|>{question}\n"


def r1_assistant() -> str:
    return "<|Assistant|>"


def qwen_system_standard() -> str:
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"


def qwen_prompt(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"


def qwen_assistant() -> str:
    return "<|im_start|>assistant\n"


def simplerl_simple_prompt(prompt: str) -> str:
    return f"Question:\n{prompt.strip()}\n"


def simple_rl_simple_assistant() -> str:
    return "Answer:\nLet's think step by step.\n"


@dataclasses.dataclass
class Config(recipes.EvalRecipeConfig):
    apply_chat_template: bool | None = None


def main() -> None:
    register_eval_datasets()

    config = cli.run_default_cli(Config, console_outputs=on_local_rank_zero())

    if config.apply_chat_template is not None:
        config.data.apply_chat_template = config.apply_chat_template

    if config.model.name.startswith("deepseek-r1-distill-qwen"):
        # Results for temperature 0.6, top_p 0.95, max_tokens 30k, avg=4:
        # aime24: 0.2917, amc: 0.6875, minerva: 0.2858, olympiad: 0.4904, math500: 0.8280
        config.data.apply_chat_template = False
        config.data.system_message = r1_system()
        config.data.assistant_message = r1_assistant()
        config.data.prompt_transform = r1_prompt

        config.generator.extra_stop_tokens = [151645, 151643]

    if config.model.name.startswith("qwen2.5-") and "math" not in config.model.name:
        # qwen2.5-3b: math500 = 0.49, amc = 0.25, aime24 = 0.03, minerva = 0.14, olymp = 0.18
        # qwen2.5-1.5b (qwen prompt): math500 = 0.08, amc = 0.02, aime24 = 0.0, minerva = 0.02, olymp = 0.01
        # qwen2.5-1.5b (simple prompt): math500 = 0.23, amc = 0.05, aime24 = 0.0, minerva = 0.044 , olymp = 0.05
        # qwen2.5-1.5b-instruct: math500 = 0.544, amc = 0.225, aime24 = 0.0, minerva = 0.20, olymp = 0.21

        config.data.apply_chat_template = False
        if "instruct" in config.model.name:
            config.data.system_message = qwen_system_standard()
            config.data.assistant_message = qwen_assistant()
            config.data.prompt_transform = qwen_prompt
        else:
            config.data.system_message = None
            config.data.assistant_message = simple_rl_simple_assistant()
            config.data.prompt_transform = simplerl_simple_prompt

    # qwen2.5-math-1.5b (no custom template etc), greedy temp 0, top_p 1.0
    # aime24: 0.1667, amc: 0.4250, minerva: 0.1103, math500: 0.3820, olympiad: 0.2267

    evaluator = recipes.create_evaluator(config)

    evaluator.run()


if __name__ == "__main__":
    main()
