import dataclasses
import functools
import itertools
from pathlib import Path
from typing import Any, Callable, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.axes import Axes
from tabulate import tabulate
from torch.utils.benchmark import Timer
from tqdm import tqdm
from typing_extensions import TypedDict

from miniseq import cli
from miniseq import configs as cfg
from miniseq.data import (
    PackedSequenceExample,
    SequenceBatch,
    SequenceExample,
    UnfinishedPack,
    UnfinishedPrefixPack,
)
from miniseq.logging import get_logger, log_config, setup_logging
from miniseq.machine import setup_default_machine
from miniseq.models import (
    ModelConfig,
    build_model,
    infer_device,
    load_model_checkpoint,
    log_model,
)
from miniseq.recipes import setup_model, setup_torch
from miniseq.training import manual_seed
from miniseq.transformer import CausalTransformerModel

log = get_logger()


torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.recompile_limit = 1000


PAD_IDX = -100

num_threads = torch.get_num_threads()


def clear_grads(model: CausalTransformerModel) -> None:
    for param in model.parameters():
        param.grad = None


def benchmark_model(
    f: Callable[[CausalTransformerModel, SequenceBatch, SequenceBatch], Any],
    warmup: int,
    model: CausalTransformerModel,
    input_batch: SequenceBatch,
    target_batch: SequenceBatch,
) -> TypedDict[{"time": float, "peak_memory": int}]:
    # Note: warmup at least 1 step to take care of recompilation steps.
    if warmup > 0:
        log.info(f"Running {warmup} warmup steps.")
        for _ in range(warmup):
            f(model, input_batch, target_batch)
            clear_grads(model)

    log.info(f"Benchmarking on {num_threads} threads.")

    clear_grads(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(input_batch.seqs.device)
    torch.cuda.synchronize()

    t0 = Timer(
        stmt="f(model, input_batch, target_batch)",
        globals={
            "f": f,
            "model": model,
            "input_batch": input_batch,
            "target_batch": target_batch,
        },
        num_threads=num_threads,
    )

    time = t0.blocked_autorange().mean * 1e3

    peak_memory = torch.cuda.max_memory_allocated(input_batch.seqs.device)

    clear_grads(model)

    del input_batch, target_batch

    return {"time": time, "peak_memory": peak_memory}


def make_examples(
    batch_size: int,
    prompt_len: list[int],
    completion_len: list[int],
    completions_only: bool,
    device: torch.device,
) -> list[SequenceExample]:
    assert len(prompt_len) == batch_size
    assert len(completion_len) == batch_size

    def make_seqs(length: int) -> torch.Tensor:
        return torch.arange(end=length, device=device, dtype=torch.long)

    examples: list[SequenceExample] = []

    for index in range(batch_size):
        prompt = make_seqs(prompt_len[index])
        completion = make_seqs(completion_len[index])

        examples.append(
            SequenceExample.from_instruction(prompt, completion, completions_only)
        )

    return examples


def make_padded_batch(
    batch_size: int,
    prompt_len: int | list[int],
    completion_len: int | list[int],
    completions_only: bool,
    device: torch.device,
) -> SequenceBatch:
    if isinstance(prompt_len, int):
        prompt_len = [prompt_len] * batch_size

    if isinstance(completion_len, int):
        completion_len = [completion_len] * batch_size

    assert len(prompt_len) == batch_size
    assert len(completion_len) == batch_size

    examples = make_examples(
        batch_size, prompt_len, completion_len, completions_only, device
    )

    return SequenceExample.collate(examples, pad_idx=PAD_IDX)


def make_packed_batch(
    batch_size: int,
    prompt_len: int | list[int],
    completion_len: int | list[int],
    completions_only: bool,
    device: torch.device,
    pad_amount: int = 0,
) -> SequenceBatch:
    if isinstance(prompt_len, int):
        prompt_len = [prompt_len] * batch_size

    if isinstance(completion_len, int):
        completion_len = [completion_len] * batch_size

    assert pad_amount >= 0
    assert len(prompt_len) == batch_size
    assert len(completion_len) == batch_size

    examples = make_examples(
        batch_size, prompt_len, completion_len, completions_only, device
    )

    packer = UnfinishedPack(max_seq_len=1 << 24)

    packer.batch_update(examples)

    final_length = packer.length + pad_amount

    packed_example = packer.finish(pad_idx=PAD_IDX, max_seq_len=final_length)

    return PackedSequenceExample.collate([packed_example], pad_idx=PAD_IDX)


def make_prefix_packed_batch(
    batch_size: int,
    prompt_len: int | list[int],
    completion_len: int | list[int],
    completions_only: bool,
    device: torch.device,
    prefix_group_size: int,
    pad_amount: int = 0,
) -> SequenceBatch:
    assert batch_size % prefix_group_size == 0

    num_groups = batch_size // prefix_group_size

    if isinstance(prompt_len, int):
        prompt_len = [prompt_len] * batch_size
    else:
        assert len(prompt_len) in (num_groups, batch_size)

        if len(prompt_len) == batch_size:
            # Check for correctness.
            for index in range(batch_size):
                assert prompt_len[index] == (prompt_len[index // prefix_group_size])
        else:
            new_prompt_len = []
            for plen in prompt_len:
                new_prompt_len.extend([plen] * prefix_group_size)

            prompt_len = new_prompt_len

    if isinstance(completion_len, int):
        completion_len = [completion_len] * batch_size

    assert pad_amount >= 0
    assert prefix_group_size >= 2
    assert len(prompt_len) == batch_size
    assert len(completion_len) == batch_size

    examples = make_examples(
        batch_size, prompt_len, completion_len, completions_only, device
    )

    packer = UnfinishedPrefixPack(max_seq_len=1 << 24, share_count=prefix_group_size)

    batch = []
    for index, example in enumerate(examples, start=1):
        batch.append(example)

        if index % prefix_group_size == 0:
            packer.maybe_update(batch)
            batch.clear()

    final_length = packer.length + pad_amount

    packed_example = packer.finish(pad_idx=PAD_IDX, max_seq_len=final_length)

    return PackedSequenceExample.collate([packed_example], pad_idx=PAD_IDX)


def run_nll_forward_backward(
    model: CausalTransformerModel,
    input_batch: SequenceBatch,
    target_batch: SequenceBatch,
) -> None:
    seqs, input_pos, attn_mask = input_batch.as_input()

    loss = model(
        seqs,
        target_batch.seqs,
        attn_mask=attn_mask,
        input_pos=input_pos,
        reduction="sum",
        target_mask=target_batch.target_mask,
        num_valids=target_batch.num_target_elements,
    )

    loss.backward()


def run_grpo_forward_backward(
    model: CausalTransformerModel,
    input_batch: SequenceBatch,
    target_batch: SequenceBatch,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
) -> None:
    seqs, input_pos, attn_mask = input_batch.as_input()

    nll = model(
        seqs,
        target_batch.seqs,
        attn_mask=attn_mask,
        input_pos=input_pos,
        target_mask=target_batch.target_mask,
        reduction="none",
        temperature=1.0,
        num_valids=target_batch.num_target_elements,
    )

    pi_logps = -nll

    old_logps = pi_logps.detach()

    is_packed = input_batch.is_packed

    ip_ratio = (pi_logps - old_logps).exp()

    if is_packed:
        lengths, seqlen = input_batch.full_lengths(), pi_logps.size(1)

        advantage = advantage.flatten()

        # (1, packed_seqlen)
        advantage = advantage.repeat_interleave(lengths, output_size=seqlen)[None, ...]
    else:
        # (bsz * group_size, 1)
        advantage = advantage.flatten()[..., None]

    policy_adv = ip_ratio * advantage

    if clip_eps > 0.0:
        clamped_ratio = ip_ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
        clamped_policy_adv = clamped_ratio * advantage
        policy_adv = torch.min(policy_adv, clamped_policy_adv)

    if target_mask := target_batch.target_mask is not None:
        policy_adv = policy_adv * target_mask

    loss = -policy_adv.sum()

    loss.backward()


MethodName: TypeAlias = Literal["nll", "grpo"]
CPU = torch.device("cpu")


def to_cpu(model: CausalTransformerModel) -> CausalTransformerModel:
    model = model.to(device=CPU)

    assert infer_device(model) == CPU

    return model


def to_device(
    model: CausalTransformerModel, device: torch.device
) -> CausalTransformerModel:
    model = model.to(device=device)

    assert infer_device(model) == device

    return model


def run_benchmark(
    model: CausalTransformerModel,
    sdpa_model: CausalTransformerModel,
    model_name: str,
    result_file: Path,
    batch_size: int,
    num_completions: int,
    seqlens: list[tuple[int, int]],
    device: torch.device,
    benchmark_func: MethodName = "nll",
) -> None:
    """Total batch size = batch_size * num_completions."""

    assert batch_size >= 1
    assert num_completions >= 1
    assert device.type == "cuda"

    batch_kwargs = {
        "completions_only": True,
        "device": device,
        "batch_size": batch_size * num_completions,
    }

    table = []

    peak_memory = -1

    match benchmark_func:
        case "nll":
            fwd_bwd_method = run_nll_forward_backward
        case "grpo":
            advantage = torch.randn(
                (batch_size, num_completions), dtype=torch.float, device=device
            )
            fwd_bwd_method = functools.partial(
                run_grpo_forward_backward, advantage=advantage, clip_eps=0.2
            )
        case _:
            raise ValueError()

    for prompt_len, completion_len in tqdm(seqlens, total=len(seqlens)):
        batch_kwargs["prompt_len"] = prompt_len
        batch_kwargs["completion_len"] = [
            completion_len for _ in range(batch_kwargs["batch_size"])
        ]

        padded_batch = make_padded_batch(**batch_kwargs)

        packed_batch = make_packed_batch(pad_amount=0, **batch_kwargs)

        prefix_batch = make_prefix_packed_batch(
            pad_amount=0, prefix_group_size=num_completions, **batch_kwargs
        )

        assert (
            padded_batch.num_examples
            == packed_batch.num_examples
            == prefix_batch.num_examples
        )

        padded_tokens, packed_tokens, prefix_tokens = (
            padded_batch.num_elements,
            packed_batch.num_elements,
            prefix_batch.num_elements,
        )

        log.info(
            f"Running prompt_len={prompt_len}, completion_len={completion_len} num_compl={num_completions}"
            f", padded_tokens: {padded_tokens} packed_tokens: {packed_tokens}"
            f", prefix_tokens: {prefix_tokens} peak_all_time_memory: {peak_memory}"
        )

        sdpa_model = to_device(sdpa_model, device)

        result_padded = benchmark_model(
            fwd_bwd_method,
            1,
            sdpa_model,
            *padded_batch.to(device=device).as_auto_regressive(),
        )

        sdpa_model = to_cpu(sdpa_model)

        model = to_device(model, device)

        result_packed = benchmark_model(
            fwd_bwd_method,
            1,
            model,
            *packed_batch.to(device=device).as_auto_regressive(),
        )

        result_prefix = benchmark_model(
            fwd_bwd_method,
            1,
            model,
            *prefix_batch.to(device=device).as_auto_regressive(),
        )

        model = to_cpu(model)

        row = dict(
            model=model_name,
            method=benchmark_func,
            prompt_len=prompt_len,
            completion_len=completion_len,
            num_completions=num_completions,
            batch_size=batch_size,
            padded_ms=result_padded["time"],
            packed_ms=result_packed["time"],
            prefix_ms=result_prefix["time"],
            padded_gb=result_padded["peak_memory"] / 1e9,
            packed_gb=result_packed["peak_memory"] / 1e9,
            prefix_gb=result_prefix["peak_memory"] / 1e9,
            speedup_pad_pack=result_padded["time"] / result_packed["time"],
            speedup_pack_prefix=result_packed["time"] / result_prefix["time"],
            speedup_pad_prefix=result_padded["time"] / result_prefix["time"],
            padded_tokens=padded_tokens,
            packed_tokens=packed_tokens,
            prefix_tokens=prefix_tokens,
            num_threads=num_threads,
        )

        log.info(
            {k: v for (k, v) in row.items() if (k.endswith("_ms") or k.endswith("_gb"))}
        )

        peak_memory = max(
            peak_memory, row["packed_gb"], row["padded_gb"], row["prefix_gb"]
        )

        table.append(row)

    df = pd.DataFrame(table).sort_values(by=["prompt_len", "completion_len"])
    df.to_csv(result_file, index=False)

    log.info(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f"))  # type: ignore


def generate_plots(
    benchmark_dir: Path, results_file: Path, model: str, num_completions: int
) -> None:
    df = pd.read_csv(results_file)
    speedup_df = df.pivot(
        index="prompt_len", columns="completion_len", values="speedup_pad_prefix"
    )

    df["saved_gb"] = df["padded_gb"] - df["prefix_gb"]
    mem_df = df.pivot(index="prompt_len", columns="completion_len", values="saved_gb")

    def format_seqlen(x):
        if x < 1024:
            return str(x)
        return f"{x // 1024}k"

    fig, (ax1, ax2) = cast(
        tuple[Any, tuple[Axes, Axes]], plt.subplots(1, 2, figsize=(16, 6))
    )

    fig.suptitle(f"Prefix Caching ({model}, completions={num_completions})")

    for dfi in [speedup_df, mem_df]:
        dfi.columns.name = "Response Length"
        dfi.index.name = "Prefix Length"

    x_labels = [format_seqlen(x) for x in speedup_df.columns]
    y_labels = [format_seqlen(y) for y in speedup_df.index]

    ax1.set_title("Performance Speedup")
    sns.heatmap(
        speedup_df,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=None,
        annot=True,
        fmt=".2f",
        ax=ax1,
    )

    ax2.set_title("Memory Saved (GB)")
    sns.heatmap(
        mem_df,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap="mako",
        annot=True,
        fmt=".2f",
        ax=ax2,
    )

    run_name = results_file.name.removesuffix(".csv")

    save_path = benchmark_dir / f"{run_name}.png"
    plt.savefig(save_path)


@dataclasses.dataclass
class BenchmarkConfig:
    model: cfg.PretrainedModelConfig = cfg.PretrainedModelConfig(
        flex_attention=True, name="qwen2.5-7b-instruct"
    )

    batch_size: int = 3

    cache_dir: Path = Path("./local_data")

    result_file: str = "result.csv"

    compile: cfg.CompileConfig = cfg.CompileConfig(model=True, dynamic=False)

    ac: bool = True

    seed: int = 2

    seq_range: tuple[int, int] = (7, 15)

    num_completions: int = 2

    method: MethodName = "nll"


def main(config: BenchmarkConfig) -> None:
    assert config.model.flex_attention

    setup_torch(expandable_segments=True)

    seqlens = list(
        itertools.product([(1 << i) for i in range(*config.seq_range)], repeat=2)
    )

    assert len(seqlens) == len(set(seqlens))

    # Sort from largest to smallest to trigger potential OOM faster.
    seqlens.sort(reverse=True)

    # For now assume single device.
    machine = setup_default_machine(dp_replicate=False, dp_shard=False)

    assert machine.size == 1

    model_config = ModelConfig.from_model_name(
        config.model.name,
        flex_attention=config.model.flex_attention,
        cut_cross_entropy=config.model.use_cce,
        finetune_repo_id=config.model.finetune_repo_id,
    )

    model = build_model(
        model_config, device=torch.device("meta"), dtype=config.model.dtype
    )

    model = setup_model(
        model,
        machine,
        mp_dtype=config.model.dtype,
        state_dict=load_model_checkpoint(
            model_config, cache_dir=config.cache_dir, machine=machine
        ),
        ac=config.ac,
        compile_config=config.compile,
    )

    model = to_cpu(model)

    # Non-flex model for padded benchmark - uses torch Flash Attention SDPA
    sdpa_model_config = ModelConfig.from_model_name(
        config.model.name,
        flex_attention=False,
        cut_cross_entropy=config.model.use_cce,
        finetune_repo_id=config.model.finetune_repo_id,
    )

    sdpa_model = build_model(
        sdpa_model_config, device=torch.device("meta"), dtype=config.model.dtype
    )

    sdpa_model = setup_model(
        sdpa_model,
        machine,
        mp_dtype=config.model.dtype,
        state_dict=load_model_checkpoint(
            sdpa_model_config, cache_dir=config.cache_dir, machine=machine
        ),
        ac=config.ac,
        compile_config=config.compile,
    )

    sdpa_model = to_cpu(sdpa_model)

    log_config(log, config)

    benchmark_dir = config.cache_dir.joinpath("benchmarks")

    benchmark_dir.mkdir(parents=True, exist_ok=True)

    result_file = benchmark_dir / config.result_file

    log_model(log, model)

    run_benchmark(
        model,
        sdpa_model,
        config.model.name,
        result_file,
        config.batch_size,
        num_completions=config.num_completions,
        seqlens=seqlens,  # type: ignore
        device=machine.device,
    )

    cfg.save_config(benchmark_dir, config)

    generate_plots(
        benchmark_dir, result_file, model_config.model, config.num_completions
    )

    machine.close()


if __name__ == "__main__":
    setup_logging()

    config = cli.run_default_cli(BenchmarkConfig)

    with manual_seed(config.seed, torch.device("cpu"), torch.device("cuda:0")):
        main(config)
