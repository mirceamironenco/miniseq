from __future__ import annotations

import logging
import os
import typing
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from typing_extensions import override

import miniseq.training.data_parallel as data_parallel
from miniseq.generation._base import Generator
from miniseq.logging import get_logger
from miniseq.machine import Machine
from miniseq.models import (
    CheckpointConverter,
    ModelConfig,
    get_model_family,
    get_to_hf_checkpoint_converter,
    merged_named_parameters,
)
from miniseq.training import StopWatch, manual_seed, maybe_unwrap_ac_checkpoint

# Silence vLLM import logging.
logging.disable(logging.WARNING)

# Lazy import since vllm has significant import time.
if TYPE_CHECKING:
    from vllm import LLM, LLMEngine, RequestOutput, SamplingParams, TokensPrompt

logging.disable(logging.NOTSET)

_log = get_logger()


@dataclass(kw_only=True)
class VLLMEngineConfig:
    tensor_parallel_size: int | None = None
    """Number of GPUs for TP. None defaults to world_size."""

    gpu_memory_utilization: float = 0.4
    """GPU memory ratio to reserve."""

    enforce_eager: bool = False
    """Disable CUDA graph & use eager mode."""

    max_seq_len_to_capture: int = 8192
    """Max seqlen covered by CUDA graphs."""

    max_model_len: int = field(init=False, default=8192)

    def __post_init__(self) -> None:
        if self.max_seq_len_to_capture > self.max_model_len:
            self.max_seq_len_to_capture = self.max_model_len

    def set_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len

        self.max_seq_len_to_capture = min(
            self.max_seq_len_to_capture, self.max_model_len
        )


@dataclass(kw_only=True)
class VLLMSamplingConfig:
    n: int = field(init=False, default=1)
    """Number of generations per input prompt."""

    detokenize: bool = field(init=False, default=False)
    """Whether to detokanize the output."""

    temperature: float = 1.0

    top_p: float = 1.0
    """Cumulative prob. of top tokens."""

    top_k: int = -1
    """0 or -1 considers all tokens."""

    max_tokens: int | None = 16
    """Max tokens generated per output sequence."""

    skip_special_tokens: bool = field(init=False, default=True)
    """Whether to skip special tokens in the output."""

    stop_token_ids: list[int] | None = field(init=False, default=None)
    """List of tokens that stop the generation when they are generated.  The returned
    output will contain the stop tokens unless the stop tokens are special tokens."""


class VLLMGenerator(Generator):
    _machine: Machine
    _model_config: ModelConfig
    _online_model: nn.Module
    _dtype: torch.dtype
    _seed: int
    _config: VLLMEngineConfig
    _model_family: str
    _model_repo_id: str
    _sampling_config: VLLMSamplingConfig
    _to_hf_ckpt_converter: CheckpointConverter
    _sync_watch: StopWatch

    _llm: LLM
    _sampling_params: SamplingParams
    _val_sampling_params: SamplingParams
    _validation_mode: bool

    def __init__(
        self,
        *,
        machine: Machine,
        model_config: ModelConfig,
        online_model: nn.Module,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 0,
        cache_dir: Path | None = None,
        config: VLLMEngineConfig = VLLMEngineConfig(),
        sampling_config: VLLMSamplingConfig = VLLMSamplingConfig(),
        val_sampling_config: VLLMSamplingConfig | None = None,
        distributed_executor_backend: str | None = None,
        verbose: bool = True,
    ) -> None:
        self._machine = machine
        self._model_config = model_config
        self._online_model = online_model
        self._dtype = dtype
        self._seed = seed
        self._config = config
        self._model_family = get_model_family(model_config.model)
        self._model_repo_id = model_config.repo_id
        self._sampling_config = sampling_config
        self._to_hf_ckpt_converter = get_to_hf_checkpoint_converter(self._model_family)
        self._sync_watch = StopWatch(device=self._machine.device)
        self._verbose = verbose

        model_path = self._model_repo_id

        if cache_dir is not None:
            # Load from existing cache_dir.
            # TODO: Make more robust to model location.
            model_name = model_path.split("/")[-1]

            download_dir = cache_dir / model_name

            model_path = str(download_dir)

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        if config.tensor_parallel_size is None:
            config.tensor_parallel_size = 1

        # Lazy import since vllm has significant import time.
        from vllm import LLM, SamplingParams

        # Build vLLM engine.
        self._llm = LLM(
            model=model_path,
            dtype=str(dtype).split(".")[-1],  # type: ignore
            seed=seed,
            swap_space=4,
            enable_sleep_mode=True,
            enable_prefix_caching=torch.cuda.get_device_capability()[0] >= 8,
            distributed_executor_backend=distributed_executor_backend,
            max_num_batched_tokens=4096,
            **asdict(config),
        )

        self._sampling_params = SamplingParams(**asdict(sampling_config), logprobs=0)
        if val_sampling_config is not None:
            self._val_sampling_params = SamplingParams(
                **asdict(val_sampling_config), logprobs=0
            )
        else:
            self._val_sampling_params = self._sampling_params

        self._validation_mode = False

        self._machine.barrier()

    @property
    def using_tensor_parallel(self) -> bool:
        if self._machine.size == 1:
            return False

        if (tp_size := self._config.tensor_parallel_size) is None or tp_size == 1:
            return False

        # Limitation, for now.
        assert self._machine.size == self._config.tensor_parallel_size

        return True

    def generation_context(self) -> AbstractContextManager[None]:
        if not self.using_tensor_parallel:
            return nullcontext()

        # vllm tensor parallel requires same seed on all ranks; setting the seed in the
        # engine init is not enough, as vllm seems to use default device generators, so
        # we make sure to set them manually again to the same seed.

        return manual_seed(self._seed, self._machine.device, torch.device("cpu"))

    @property
    def num_gens(self) -> int:
        return self._sampling_config.n

    def _get_llm_engine(self) -> LLMEngine:
        return self._llm.llm_engine

    def _engine(self) -> LLM:
        return self._llm

    def _sleep(self, level: int = 1) -> None:
        self._llm.sleep(level=level)

    def _wake_up(self, tags: list[str] | None = None) -> None:
        if self._get_llm_engine().is_sleeping():
            self._llm.wake_up(tags=tags)

    @override
    def prepare_for_generation(self, **kwargs) -> None:
        self._wake_up(tags=None)

    @override
    def after_generation(self, **kwargs) -> None:
        self._sleep(level=1)

    @override
    def generate(
        self, input_ids: list[list[int]]
    ) -> tuple[list[list[int]], list[list[float]] | None]:
        with self.generation_context():
            if self.using_tensor_parallel:
                input_prompts = [None for _ in range(self._machine.size)]

                self._machine.all_gather_object(input_prompts, input_ids)

                input_prompts = typing.cast(list[list[list[int]]], input_prompts)

                # e.g. [15, 16] (so rank 0 has 15 prompts, rank 1 has 16
                num_prompts_per_rank = list(map(len, input_prompts))

                assert len(num_prompts_per_rank) == self._machine.size

                all_prompts = []
                for prompts in input_prompts:
                    all_prompts.extend(prompts)
            else:
                all_prompts = input_ids
                num_prompts_per_rank = [len(input_ids)]

            if self._validation_mode:
                sampling_params = self._val_sampling_params
            else:
                sampling_params = self._sampling_params

            # list[TokensPrompt]
            token_prompts: list[TokensPrompt] = [
                {"prompt_token_ids": tokens} for tokens in all_prompts
            ]

            self._machine.barrier()

            use_tqdm: Callable | bool = self._machine.rank == 0 and self._verbose

            if not self._verbose:
                # Log in case no tqdm is used.
                _log.info(f"Running rollout on {len(token_prompts)} sequences..")

            outputs: list[RequestOutput] = self._llm.generate(
                token_prompts,
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )

            self._machine.barrier()

            completions = [
                list(completion.token_ids)
                for req_output in outputs
                for completion in req_output.outputs
            ]

            logprobs = [
                [next(iter(pdict.values())).logprob for pdict in completion.logprobs]
                for req_output in outputs
                for completion in req_output.outputs
            ]

            if self.using_tensor_parallel:
                start_slice = sum(num_prompts_per_rank[: self._machine.rank])
                end_slice = start_slice + num_prompts_per_rank[self._machine.rank]
                completions = completions[start_slice:end_slice]
                logprobs = logprobs[start_slice:end_slice]
                self._machine.barrier()

            return completions, logprobs

    @property
    @override
    def model(self) -> nn.Module:
        return self._online_model

    @override
    def update_model(self, *, only_trainable: bool = False) -> None:
        self._wake_up()

        self._update_model_iterative(only_trainable=only_trainable)

    def _update_model_iterative(self, only_trainable: bool = False) -> None:
        with self._sync_watch:
            base_module = data_parallel.base_module(self._online_model)

            for name, parameter in merged_named_parameters(base_module):
                if only_trainable and not parameter.requires_grad:
                    continue

                if isinstance(parameter, DTensor):
                    parameter = cast(DTensor, parameter.detach()).full_tensor()
                else:
                    parameter = parameter.detach()

                param_sd = maybe_unwrap_ac_checkpoint({name: parameter})

                param_sd = self._to_hf_ckpt_converter(param_sd, self._model_config)

                llm_model = (
                    self._llm.llm_engine.model_executor.driver_worker.model_runner.model  # type: ignore
                )

                llm_model.load_weights(param_sd.items())

                torch.cuda.synchronize()

                del parameter

        _log.info(
            f"Finished vLLM weight sync in {self._sync_watch.get_elapsed_time():.3f}s."
        )

        self._sync_watch.reset()

        self._machine.barrier()

    def _update_model_summon(self, only_trainable: bool = False) -> None:
        # TODO: If used, must undo summon in evaluator.
        with data_parallel.summon_full_parameters(self._online_model):
            with self._sync_watch:
                base_module = data_parallel.base_module(self._online_model)

                state_dict = {
                    k: v
                    for k, v in merged_named_parameters(base_module)
                    if not only_trainable or v.requires_grad
                }

                for name, value in state_dict.items():
                    if not isinstance(value, torch.Tensor):
                        raise ValueError(
                            f"Expected params to be torch.Tensor, got {type(value)} for {name}."
                        )

                state_dict = maybe_unwrap_ac_checkpoint(state_dict)

                state_dict = self._to_hf_ckpt_converter(state_dict, self._model_config)

                llm_model = (
                    self._llm.llm_engine.model_executor.driver_worker.model_runner.model  # type: ignore
                )

                llm_model.load_weights(state_dict.items())

                torch.cuda.synchronize()

        _log.info(
            f"Finished vLLM weight sync in {self._sync_watch.get_elapsed_time():.3f}s."
        )

        self._sync_watch.reset()

        self._machine.barrier()

    def validation_mode(self, validation: bool) -> None:
        self._validation_mode = validation
