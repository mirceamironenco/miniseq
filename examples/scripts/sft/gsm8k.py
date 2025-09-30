import dataclasses

from typing_extensions import TypedDict

from miniseq import cli, recipes
from miniseq import configs as cfg
from miniseq.data import InstructionDict
from miniseq.utils import on_local_rank_zero


def verl_gsm8k_prompt(question: str) -> str:
    # https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py
    return (
        question.strip()
        + " "
        + 'Let\'s think step by step and output the final answer after "####".'
    )


GSM8KItem = TypedDict[{"question": str, "answer": str}]


# The TypedDict type hint is optional.
def gsm8k_instruction(item: GSM8KItem) -> InstructionDict:
    instruction = verl_gsm8k_prompt(item["question"])
    completion = item["answer"]
    return {"instruction": instruction, "completion": completion}


@dataclasses.dataclass
class Config(recipes.SFTRecipeConfig):
    train_data: cfg.data.SFTDatasetConfig = cfg.data.SFTDatasetConfig(
        name="openai/gsm8k",
        configuration="main",
        split="train",
        completions_only=True,
        packed_seqlen=4097,
        max_seqlen=2048,
        apply_chat_template=False,
        instruct_map=gsm8k_instruction,
    )


config = cli.run_default_cli(Config, console_outputs=on_local_rank_zero())
trainer = recipes.create_finetune_trainer(config)
trainer.run()
