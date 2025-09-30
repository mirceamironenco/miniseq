import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torchdata.nodes import BaseNode

from miniseq import cli
from miniseq import configs as cfg
from miniseq.data import PipelineBuilder, SequenceBatch, SequenceExample
from miniseq.logging import get_logger, log_config, setup_logging
from miniseq.machine import Machine, setup_default_machine
from miniseq.models import LoRAConfig, log_model
from miniseq.trainer import Trainer
from miniseq.training import set_seed
from miniseq.utils import default_dtype, on_local_rank_zero

# isort: split
from model import ModelConfig, TransformerBuilder
from tasks import MQARConfig, TaskConfig

from miniseq.recipes._common import calculate_total_steps, setup_torch
from miniseq.recipes._setup_model import setup_model
from miniseq.recipes.algorithm import (
    AcuracyEvalUnit,
    InstructionEvalUnit,
    InstructionUnit,
)


@dataclass(kw_only=True)
class RecipeConfig(cfg.TrainRecipeConfig):
    optimizer: cfg.OptimizerConfig = cfg.optim.AdamWConfig(lr=1e-3, weight_decay=0.1)

    lr_scheduler: cfg.LRSchedulerConfig = cfg.scheduler.LinearWarmupCosineConfig(
        warmup_ratio=0.03
    )

    train: cfg.TrainConfig = cfg.TrainConfig(
        micro_batch_size=256,
        device_batch_size=256,
        no_sync=False,
        max_grad_norm=3.0,
        publish_metrics_every=50,
        validate_every=100,
        max_steps=3000,
        max_epochs=None,
    )

    task: TaskConfig = MQARConfig(input_seq_len=64, vocab_size=8192, num_kv_pairs=8)

    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(model_dim=128, num_layers=2)
    )

    dp: cfg.DPStrategyConfig = cfg.dp.DDPConfig()

    compile: cfg.CompileConfig = cfg.CompileConfig(
        model=True, loss=False, dynamic=False
    )

    lora: LoRAConfig | None = None

    wandb: cfg.WandbConfig | None = cfg.WandbConfig(
        project="miniseq_icl", run_name="mqar"
    )

    profiler: cfg.TorchProfilerConfig | None = None

    packed: bool = True

    seed: int = 2

    def __post_init__(self) -> None:
        self.model.input_dim = self.task.input_proj_dim()

        self.model.output_dim = self.task.output_proj_dim()

        assert self.model.input_dim is None or self.model.output_dim is None

        if self.task.input_proj_dim() is not None:
            self.model.embedding = "projection"

        if self.task.output_proj_dim() is not None:
            self.model.embedding = "index"


class IterableRngTensorNode(BaseNode[tuple[torch.Tensor, torch.Tensor]]):
    """A restartable node that yields a (potentially infinite) stream of tensors."""

    _rng_tensor_map: Callable[
        [torch.Generator | None], tuple[torch.Tensor, torch.Tensor]
    ]
    _seed: int
    _size: int | None
    _rng: torch.Generator
    _it: Iterator[tuple[torch.Tensor, torch.Tensor]] | None
    _num_yielded: int

    RNG_STATE_KEY: str = "rng_state"
    YIELDED_KEY: str = "num_yielded"

    def __init__(
        self,
        tensor_map: Callable[
            [torch.Generator | None], tuple[torch.Tensor, torch.Tensor]
        ],
        seed: int,
        size: int | None = None,
    ) -> None:
        super().__init__()
        assert seed >= 0 and seed < 1 << 32

        self._rng_tensor_map = tensor_map
        self._seed = seed
        self._size = size
        self._rng = torch.Generator()
        self._it = None
        self._num_yielded = 0

    def _make_iterator(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield self._rng_tensor_map(self._rng)

    def reset(self, initial_state: dict[str, Any] | None = None) -> None:
        super().reset(initial_state)

        if initial_state is not None:
            rng_state = initial_state[self.RNG_STATE_KEY]
            self._rng.set_state(rng_state)

            self._num_yielded = initial_state[self.YIELDED_KEY]
        else:
            self._rng.manual_seed(self._seed)
            self._num_yielded = 0

        self._it = self._make_iterator()

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._it is None:
            raise RuntimeError("Iterator is not initialized. Call reset() first.")

        if self._size is not None and self._num_yielded == self._size:
            raise StopIteration()

        item = next(self._it)

        self._num_yielded += 1

        return item

    def get_state(self) -> dict[str, Any]:
        return {
            self.RNG_STATE_KEY: self._rng.get_state(),
            self.YIELDED_KEY: self._num_yielded,
        }


def create_trainer(
    config: RecipeConfig, log: logging.Logger | None = None
) -> Trainer[SequenceBatch]:
    if log is None:
        setup_logging(debug=False)

        log = get_logger()

    log_config(log, config)

    setup_torch(expandable_segments=True)

    machine: Machine = setup_default_machine(
        dp_replicate=config.dp.replicate,
        dp_shard=config.dp.shard,
        cpu_offloading=config.dp.cpu_offloading,
    )

    seed = config.seed

    set_seed(seed, torch.device("cpu"), machine.device)

    with default_dtype(config.model.dtype), torch.device("meta"):
        model = TransformerBuilder(config.model, config.task.loss).build_model()

    model = setup_model(
        model,
        machine,
        mp_dtype=config.model.dtype,
        dp_config=config.dp,
        ac=config.train.ac,
        ac_freq=config.train.ac_freq,
        lora_config=config.lora,
        load_on_cpu=False,
        compile_config=config.compile,
    )

    optimizer = config.optimizer.build(params=model.parameters())

    if config.task.loss in ("mse", "bce"):
        train_unit = InstructionUnit(model, name=config.task.loss)
    else:
        train_unit = InstructionUnit(model)

    train_loader: Iterable[SequenceBatch] = (
        PipelineBuilder.from_source_node(
            IterableRngTensorNode(config.task.make_map(), seed=seed + machine.rank)
        )
        .map(lambda seqs_target: SequenceExample(*seqs_target), num_parallel=4)
        .batch(batch_size=config.train.micro_batch_size, drop_last=False)
        .collate(SequenceExample.collate, num_parallel=4)
        .pin_memory()
        .prefetch(prefetch_factor=64)
        .as_loader()
    )

    if config.task.loss == "mse":
        eval_unit = InstructionEvalUnit(model, name="eval_mse")
    else:
        eval_unit = AcuracyEvalUnit(model)

    # Make a different seed for the eval iterator
    eval_seed = seed + 1024

    assert eval_seed != seed
    assert config.task.eval_size > 0

    eval_loader: Iterable[SequenceBatch] = (
        PipelineBuilder.from_source_node(
            IterableRngTensorNode(
                config.task.make_map(),
                seed=eval_seed + machine.rank,
                size=config.task.eval_size // machine.size,
            )
        )
        .map(lambda seqs_target: SequenceExample(*seqs_target))
        .batch(batch_size=1000 // machine.size, drop_last=False)
        .collate(SequenceExample.collate)
        .pin_memory()
        .as_loader()
    )

    total_train_steps = calculate_total_steps(
        loader=train_loader,
        max_steps=config.train.max_steps,
        max_epochs=config.train.max_epochs,
        grad_accum_steps=config.train.grad_accum_steps,
    )

    lr_scheduler = config.lr_scheduler.build(
        optimizer=optimizer, max_num_steps=total_train_steps
    )

    log_model(log, model)

    trainer: Trainer[SequenceBatch] = Trainer.from_configs(
        recipe_config=config,
        train_unit=train_unit,
        train_loader=train_loader,
        valid_units=[eval_unit],
        valid_loaders=[eval_loader],
        generator=None,
        machine=machine,
        seed=seed,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        requires_rollout=False,
        total_steps=total_train_steps,
        log=log,
        compile_optimizer_step=config.compile.optimizer_step,
    )

    return trainer


def main() -> None:
    config = cli.run_default_cli(RecipeConfig, console_outputs=on_local_rank_zero())

    trainer = create_trainer(config)

    trainer.run()


if __name__ == "__main__":
    main()
