from dataclasses import dataclass, field
from pathlib import Path

from miniseq.configs._data import RegisteredDatasetConfig
from miniseq.configs._dp import DDPConfig, DPStrategyConfig
from miniseq.configs._generation import GeneratorConfig
from miniseq.configs._optimizer import AdamWConfig, OptimizerConfig
from miniseq.configs._scheduler import LinearWarmupCosineConfig, LRSchedulerConfig
from miniseq.configs._training import (
    CompileConfig,
    PretrainedModelConfig,
    TorchProfilerConfig,
    TrainConfig,
    WandbConfig,
)
from miniseq.models import LoRAConfig


@dataclass(kw_only=True, frozen=True)
class EvalTaskConfig:
    data: RegisteredDatasetConfig

    generator: GeneratorConfig


@dataclass(kw_only=True)
class TrainRecipeConfig:
    cache_dir: Path = Path("./local_data")
    """Model/data cache dir."""

    tensorboard: bool = False
    """Whether to use tensorboard."""

    optimizer: OptimizerConfig = field(default_factory=lambda: AdamWConfig())

    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: LinearWarmupCosineConfig()
    )

    train: TrainConfig = field(default_factory=lambda: TrainConfig())

    dp: DPStrategyConfig = DDPConfig()

    compile: CompileConfig = CompileConfig()

    wandb: WandbConfig | None = None

    lora: LoRAConfig | None = None

    profiler: TorchProfilerConfig | None = None

    eval_avg_n: int = 1

    eval_pass_k: int = 1

    seed: int = 2


@dataclass(kw_only=True)
class EvalRecipeConfig:
    cache_dir: Path = Path("./local_data")
    """Model/data cache dir."""

    tensorboard: bool = False

    data: RegisteredDatasetConfig

    model: PretrainedModelConfig

    generator: GeneratorConfig

    profiler: TorchProfilerConfig | None = None

    compile: CompileConfig = CompileConfig(model=True, dynamic=False)

    wandb: WandbConfig | None = None

    dp: DPStrategyConfig = DDPConfig()

    eval_avg_n: int = 1

    eval_pass_k: int = 1

    seed: int = 2
