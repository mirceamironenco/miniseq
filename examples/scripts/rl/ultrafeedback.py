import dataclasses

from typing_extensions import TypedDict

from miniseq import cli, recipes
from miniseq import configs as cfg
from miniseq.data import Message, PreferenceDict
from miniseq.utils import on_local_rank_zero

# Schema on huggingface (optional).
UFItem = TypedDict[{"prompt": str, "chosen": list[Message], "rejected": list[Message]}]


def uf_transform(example: UFItem) -> PreferenceDict:
    # Assume single-turn
    chosen, rejected = filter(
        lambda x: x["role"] == "assistant", example["chosen"] + example["rejected"]
    )

    return {
        "prompt": example["prompt"],
        "chosen": chosen["content"],
        "rejected": rejected["content"],
    }


@dataclasses.dataclass
class Config(recipes.PreferenceRecipeConfig):
    train_data: cfg.data.PreferenceDatasetConfig = cfg.data.PreferenceDatasetConfig(
        name="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        preference_map=uf_transform,
        max_seqlen=2049,
    )


config = cli.run_default_cli(Config, console_outputs=on_local_rank_zero())
trainer = recipes.create_preference_trainer(config)
trainer.run()
