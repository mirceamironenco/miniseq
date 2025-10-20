import collections
import itertools
import os
import sys
from typing import Annotated

import tyro
from rich.table import Table

from miniseq import cli
from miniseq.logging import get_console


def _online():
    """Run standard online training recipe."""

    from miniseq.recipes import OnlineRecipeConfig, create_online_trainer

    def online(config: Annotated[OnlineRecipeConfig, tyro.conf.arg(name="")]) -> None:
        trainer = create_online_trainer(config)

        trainer.run()

    return online


def _preference():
    """Run standard preference training recipe."""

    from miniseq.recipes import PreferenceRecipeConfig, create_preference_trainer

    def preference(
        config: Annotated[PreferenceRecipeConfig, tyro.conf.arg(name="")],
    ) -> None:
        trainer = create_preference_trainer(config)

        trainer.run()

    return preference


def _sft():
    """Run standard finetune training recipe."""

    from miniseq.configs import save_config
    from miniseq.recipes import SFTRecipeConfig, create_finetune_trainer

    def sft(
        config: Annotated[SFTRecipeConfig, tyro.conf.arg(name="")],
    ) -> None:
        trainer = create_finetune_trainer(config)

        save_config(config.cache_dir, config, name="sft_config")

        trainer.run()

    return sft


def _evaluate():
    """Run standard evaluation pipeline."""
    from miniseq.recipes import EvalRecipeConfig, create_evaluator

    def evaluate(cfg: Annotated[EvalRecipeConfig, tyro.conf.arg(name="")]) -> None:
        evaluator = create_evaluator(cfg)

        evaluator.run()

    return evaluate


def model_registry() -> None:
    """Display all registered models."""

    families = collections.defaultdict(list)

    # Lazy-load to not clog CLI.
    from miniseq.models import all_registered_models, get_model_family
    from miniseq.runtime import register_models

    register_models()

    for model in all_registered_models():
        families[get_model_family(model)].append(model)

    table = Table(*families.keys(), title="Registered models")

    for items in itertools.zip_longest(*families.values(), fillvalue=""):
        table.add_row(*items)

    CONSOLE = get_console()

    CONSOLE.print(table)


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    runnable_recipes = {
        "tune": _sft,
        "preference": _preference,
        "rl": _online,
        "evaluate": _evaluate,
    }

    other_recipes = {"model_registry": model_registry}

    all_recipes = runnable_recipes | other_recipes

    args = list(sys.argv[1:])

    if args and args[0] in runnable_recipes:
        cli_item = runnable_recipes[args[0]]()

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
