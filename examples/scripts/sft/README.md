## Supervised Fine-Tuning

```bash
python gsm8k.py --model.name=qwen2.5-1.5b-instruct --model.flash_attention2=True
```

The above command will fine-tune `qwen2.5-1.5b-isntruct` on the [gsm8k dataset](https://huggingface.co/datasets/openai/gsm8k), using a prompt format similar to the one used for RL training.

The standard SFT recipe requires that we specify a way to transform any dataset item into an `InstructionDict`, i.e. a `dict` that at least has the keys `instruction` and `completion` present:

```py
class InstructionDict(TypedDict):
    system: NotRequired[str]
    instruction: str
    input: NotRequired[str]
    completion: str
```

We can directly reference the item schema at https://huggingface.co/datasets/openai/gsm8k, and simply pass a callable to the config as shown in [gsm8k.py](./gsm8k.py):

```py
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
```