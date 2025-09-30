import collections
import itertools
import sys
from typing import Annotated

import tyro
from rich.table import Table

from miniseq import cli
from miniseq.configs import save_config
from miniseq.logging import get_console
from miniseq.models import (
    all_registered_models,
    get_model_family,
)
from miniseq.recipes import (
    EvalRecipeConfig,
    OnlineRecipeConfig,
    PreferenceRecipeConfig,
    SFTRecipeConfig,
    create_evaluator,
    create_finetune_trainer,
    create_online_trainer,
    create_preference_trainer,
)
from miniseq.utils import get_local_rank


def online(config: Annotated[OnlineRecipeConfig, tyro.conf.arg(name="")]) -> None:
    """Run standard online training recipe."""

    trainer = create_online_trainer(config)

    trainer.run()


def preference(
    config: Annotated[PreferenceRecipeConfig, tyro.conf.arg(name="")],
) -> None:
    """Run standard preference training recipe."""

    trainer = create_preference_trainer(config)

    trainer.run()


def sft(
    config: Annotated[SFTRecipeConfig, tyro.conf.arg(name="")],
) -> None:
    """Run standard finetune training recipe."""

    trainer = create_finetune_trainer(config)

    save_config(config.cache_dir, config, name="sft_config")

    trainer.run()


def evaluate(cfg: Annotated[EvalRecipeConfig, tyro.conf.arg(name="")]) -> None:
    """Run standard evaluation pipeline."""

    evaluator = create_evaluator(cfg)

    evaluator.run()


def model_registry() -> None:
    """Display all registered models."""

    families = collections.defaultdict(list)

    for model in all_registered_models():
        families[get_model_family(model)].append(model)

    table = Table(*families.keys(), title="Registered models")

    for items in itertools.zip_longest(*families.values(), fillvalue=""):
        table.add_row(*items)

    CONSOLE = get_console()

    CONSOLE.print(table)


def main() -> None:
    local_rank = get_local_rank()

    runnable_recipes = {
        "tune": sft,
        "preference": preference,
        "rl": online,
        "evaluate": evaluate,
    }

    other_recipes = {"model_registry": model_registry}

    all_recipes = runnable_recipes | other_recipes

    args = list(sys.argv[1:])

    if args and args[0] in runnable_recipes:
        cli_item = runnable_recipes[args[0]]

        args = args[1:]

        return cli.run_default_cli(cli_item, console_outputs=local_rank == 0, args=args)

    tyro.extras.subcommand_cli_from_dict(
        all_recipes,
        args=args,
        use_underscores=True,
        console_outputs=local_rank == 0,
    )


if __name__ == "__main__":
    main()
