from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from miniseq import cli


@cli.make_union_registry(name="task")
@dataclass(frozen=True)
class TaskConfig:
    loss: Literal["cross_entropy", "mse", "bce"] = field(
        default="cross_entropy", init=False
    )

    eval_size: int = 10000

    def make_map(
        self,
    ) -> Callable[[torch.Generator | None], tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def input_proj_dim(self) -> int | None: ...

    def output_proj_dim(self) -> int | None: ...


def interleave_tokens(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Interleave tokens independent of batch dimensions.

    Args:
        x (Tensor): Input features, shaped (*batch_dims, seq_len, dim).
        y (Tensor): Target values, shaped (*batch_dims, seq_len).

    Returns:
        Tensor: Interleaved tensor, shaped (*batch_dims, seq_len * 2, dim).
    """
    assert x.shape[:-1] == y.shape

    y_padded = F.pad(y.unsqueeze(-1), (0, x.shape[-1] - 1), value=0.0)

    # (*batch_dims, seq_len, 2, dim)
    interleaved = torch.stack((x, y_padded), dim=-2)

    # (*batch_dims, seq_len * 2, dim)
    return interleaved.reshape(*x.shape[:-2], -1, x.shape[-1])


@cli.union_struct_choice("task", command="regression")
@dataclass(kw_only=True, frozen=True)
class RegressionConfig(TaskConfig):
    loss: Literal["cross_entropy", "mse", "bce"] = field(default="mse", init=False)
    input_seq_len: int
    input_dim: int
    scale: float = 1.0
    sparsity: int = 5
    noise_std: float = 1.0
    renormalize_noisy: bool = False
    outlier_prob: float = 0.9
    regression_type: Literal[
        "linear", "quadratic", "sparse_linear", "noisy", "outlier"
    ] = "linear"

    def make_map(
        self,
    ) -> Callable[[torch.Generator | None], tuple[torch.Tensor, torch.Tensor]]:
        return functools.partial(regression_map, config=self)

    def input_proj_dim(self) -> int | None:
        return self.input_dim


def regression_map(
    generator: torch.Generator | None = None, *, config: RegressionConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(
        (config.input_seq_len, config.input_dim),
        dtype=torch.float32,
        generator=generator,
    )

    w = torch.randn((config.input_dim, 1), dtype=torch.float32, generator=generator)

    match config.regression_type:
        case "linear":
            y = (x @ w).squeeze(-1) * config.scale
        case "quadratic":
            y = ((x**2) @ w).squeeze(-1)
            y = y / math.sqrt(3.0)
            y = y * config.scale
        case "sparse_linear":
            assert config.sparsity < config.input_dim
            mask = torch.randn(w.size(), dtype=w.dtype, generator=generator)
            mask = mask.argsort(dim=0).squeeze(-1) < config.sparsity
            w = w.masked_fill(mask[..., None], 0.0)
            y = (x @ w).squeeze(-1) * config.scale
        case "noisy":
            y = (x @ w).squeeze(-1) * config.scale
            noise = torch.randn(y.size(), dtype=y.dtype, generator=generator)
            y = y + noise * config.noise_std

            if config.renormalize_noisy:
                y = y * math.sqrt(config.input_dim) / y.std()
        case "outlier":
            y = (x @ w).squeeze(-1) * config.scale
            drop_mask = torch.rand(size=(config.input_seq_len,), generator=generator)
            drop_mask = drop_mask < config.outlier_prob
            x = x.masked_fill(drop_mask[..., None].expand(-1, config.input_dim), 1.0)
            y = y.masked_fill(drop_mask, 1.0)
        case _:
            raise ValueError(
                f"Got unrecognized regression type: {config.regression_type}"
            )

    x = interleave_tokens(x, y)

    mask = torch.zeros(config.input_seq_len * 2, dtype=torch.bool, device=x.device)
    mask[1::2] = True  # Set ODD indices to True

    return x, mask


@cli.union_struct_choice("task", command="mqar")
@dataclass(kw_only=True, frozen=True)
class MQARConfig(TaskConfig):
    loss: Literal["cross_entropy", "mse", "bce"] = field(
        default="cross_entropy", init=False
    )
    input_seq_len: int
    vocab_size: int
    power_a: float = 0.01
    num_kv_pairs: int = 8
    random_non_queries: bool = True

    def make_map(
        self,
    ) -> Callable[[torch.Generator | None], tuple[torch.Tensor, torch.Tensor]]:
        return functools.partial(multiquery_ar_single, config=self)

    def output_proj_dim(self) -> int | None:
        return self.vocab_size


def multiquery_ar_single(
    generator: torch.Generator | None = None, *, config: MQARConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a single example for the multiquery associative recall task.

    Tests a model's ability to recall values based on keys provided in a context.

    Reference: https://github.com/HazyResearch/zoology

    Args:
        generator (torch.Generator): A PyTorch generator for all random operations.
        input_seq_len (int): The length of the input sequence. Must be even.
        vocab_size (int): The total size of the vocabulary.
        power_a (float, optional): The exponent for the power-law distribution
            used to sample query positions. Defaults to 0.01.
        num_kv_pairs (int, optional): The number of key-value pairs in the
            initial context. Defaults to 8.
        random_non_queries (bool, optional): If True, fills all non-key/value/query
            positions with random tokens. Otherwise, they are left as 0s.
            Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - sequence (torch.Tensor): The full sequence of shape (input_seq_len + 1,).
            - target_mask (torch.Tensor): A boolean mask of shape (input_seq_len + 1,).
    """
    input_seq_len, vocab_size, num_kv_pairs, power_a = (
        config.input_seq_len,
        config.vocab_size,
        config.num_kv_pairs,
        config.power_a,
    )

    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len, "vocab_size must be larger than sequence length"
    assert num_kv_pairs * 4 <= input_seq_len, "sequence length is too short"

    context_size = num_kv_pairs * 2

    # --- 1. Create and Sample Unique Keys and Values ---
    key_vocab_size = vocab_size // 2
    key_choices = torch.arange(1, key_vocab_size)
    value_choices = torch.arange(key_vocab_size, vocab_size)

    # Shuffle choices and take the first `num_kv_pairs` for a unique sample
    keys = key_choices[torch.randperm(key_choices.size(0), generator=generator)][
        :num_kv_pairs
    ]
    values = value_choices[torch.randperm(value_choices.size(0), generator=generator)][
        :num_kv_pairs
    ]

    # --- 2. Construct the Initial Key-Value Context ---
    kvs = torch.zeros(context_size, dtype=torch.long)
    kvs[0::2] = keys
    kvs[1::2] = values

    # --- 3. Sample Query Positions using a Power-Law Distribution ---
    space = (input_seq_len - context_size) // 2

    p = power_a * torch.arange(1, space + 1, dtype=torch.float32) ** (power_a - 1)

    gaps = torch.multinomial(
        p, num_samples=num_kv_pairs, replacement=False, generator=generator
    )

    # --- 4. Construct the Final Sequence Tensor ---
    sequence = torch.zeros(input_seq_len + 1, dtype=torch.long)
    target_mask = torch.zeros(input_seq_len + 1, dtype=torch.bool)

    # Place the context at the beginning
    sequence[:context_size] = kvs

    # Place the queries (keys) and answers (values) at their sampled positions
    query_indices = context_size + (gaps * 2)
    answer_indices = query_indices + 1
    sequence.scatter_(0, query_indices, keys)
    sequence.scatter_(0, answer_indices, values)

    # Set the mask to True only at the positions of the answers
    target_mask.scatter_(0, answer_indices, True)

    # --- 5. Fill Remaining Positions with Random Tokens if specified ---
    if config.random_non_queries:
        mask = sequence == 0
        num_zeros = int(mask.sum())
        if num_zeros > 0:
            random_fill = torch.randint(
                low=1,
                high=vocab_size,
                size=(num_zeros,),
                generator=generator,
                dtype=torch.long,
            )
            sequence[mask] = random_fill

    return sequence, target_mask


@cli.union_struct_choice("task", command="disjoint")
@dataclass(kw_only=True, frozen=True)
class DisjointSetsConfig(TaskConfig):
    loss: Literal["cross_entropy", "mse", "bce"] = field(
        default="cross_entropy", init=False
    )
    vocab_size: int
    short_length: int
    long_length: int

    def make_map(
        self,
    ) -> Callable[[torch.Generator | None], tuple[torch.Tensor, torch.Tensor]]:
        return functools.partial(disjoint_sets_single, config=self)

    def output_proj_dim(self) -> int | None:
        return self.vocab_size


def disjoint_sets_single(
    generator: torch.Generator | None = None, *, config: DisjointSetsConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a single example for the disjoint sets task.

    The task is to identify the single token present in two otherwise disjoint
    sets of tokens. This function returns a sequence and a boolean mask, which
    can be sliced to form inputs, labels, and a loss mask.

    Reference: https://github.com/HazyResearch/prefix-linear-attention

    Args:
        vocab_size (int): The total size of the vocabulary.
        short_length (int): The number of tokens in the first set.
        long_length (int): The number of tokens in the second set.
        generator (torch.Generator, optional): A PyTorch generator for random ops.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - sequence (torch.Tensor): The full token sequence.
            - target_mask (torch.Tensor): A boolean mask where True indicates
              the position of the target token for loss calculation.
    """
    vocab_size, short_length, long_length = (
        config.vocab_size,
        config.short_length,
        config.long_length,
    )
    # --- 1. Setup Tokens and Device ---
    device = generator.device if generator else torch.device("cpu")
    prefix_tok, sep_lists_tok, sep_ans_tok = 1, 2, 3
    num_special_tokens = 4

    # --- 2. Create and Sample Two Disjoint Sets of Tokens ---
    half_vocab = vocab_size // 2
    all_idx = torch.arange(num_special_tokens, vocab_size, device=device)
    all_idx_shuffled = all_idx[torch.randperm(all_idx.size(0), generator=generator)]

    all_short_choices = all_idx_shuffled[:half_vocab]
    all_long_choices = all_idx_shuffled[half_vocab:]

    # Sample unique tokens for each list
    short_tokens = all_short_choices[
        torch.randperm(all_short_choices.size(0), generator=generator)
    ][:short_length]
    long_tokens = all_long_choices[
        torch.randperm(all_long_choices.size(0), generator=generator)
    ][:long_length]

    # --- 3. Create Overlap and Define the Answer ---
    # Randomly select one token from the short list to be the answer
    overlap_idx = int(torch.randint(short_length, (1,), generator=generator))
    answer_tok = short_tokens[overlap_idx]

    # Place the answer token at a random position in the long list
    long_tokens[torch.randint(long_length, (1,), generator=generator)] = answer_tok

    # --- 4. Construct the Final Sequence and Mask ---
    sequence = torch.cat(
        [
            torch.tensor([prefix_tok], device=device),
            short_tokens,
            torch.tensor([sep_lists_tok], device=device),
            long_tokens,
            torch.tensor([sep_ans_tok], device=device),
            torch.tensor([answer_tok], device=device),
        ],
        dim=0,
    ).long()

    # The mask is True only at the position of the final answer token.
    # When sliced with [1:], this aligns with the label for the final input token.
    target_mask = torch.zeros_like(sequence, dtype=torch.bool)
    target_mask[-1] = True

    return sequence, target_mask


@cli.union_struct_choice("task", command="boolean")
@dataclass(kw_only=True, frozen=True)
class BooleanTaskConfig(TaskConfig):
    loss: Literal["cross_entropy", "mse", "bce"] = field(default="bce", init=False)
    input_seq_len: int
    input_dim: int
    boolean_type: Literal[
        "dnf3",
        "cnf3",
        "int_halfspace",
        "sparse_threshold",
        "parity",
        "sparse_parity",
        "disjunction",
        "conjunction",
    ]
    k: int = 2

    def make_map(
        self,
    ) -> Callable[[torch.Generator | None], tuple[torch.Tensor, torch.Tensor]]:
        return functools.partial(boolean_map, config=self)

    def input_proj_dim(self) -> int | None:
        return self.input_dim


def boolean_map(
    generator: torch.Generator | None = None, *, config: BooleanTaskConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a single example for various boolean function learning tasks.

    This function consolidates multiple boolean tasks into a single interface.
    It returns an interleaved sequence and a boolean mask suitable for training
    autoregressive models.

    Args:
        input_seq_len (int): The length of the boolean input sequence.
        input_dim (int): The dimensionality of the boolean vectors.
        boolean_type (Literal[...]): The specific boolean task to generate.
        generator (torch.Generator, optional): A PyTorch generator for random ops.
        k (int, optional): The sparsity parameter for the "sparse_parity" task.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - sequence (torch.Tensor): The interleaved sequence.
            - target_mask (torch.Tensor): A boolean mask for loss calculation.
    """

    # Generate integers {0, 1} and map them to floats {-1.0, 1.0}
    x = torch.randint(
        0, 2, (config.input_seq_len, config.input_dim), generator=generator
    )
    x = x.float() * 2.0 - 1.0

    y = None
    if config.boolean_type in ["dnf3", "cnf3", "disjunction", "conjunction"]:
        y = _generate_dnf_cnf_style(x, config.input_dim, config.boolean_type, generator)  # type: ignore

    elif config.boolean_type == "int_halfspace":
        weights = torch.randint(
            -3, 4, (config.input_dim, 1), generator=generator
        ).float()
        y = (x @ weights).squeeze(-1) - 0.5
        y = y.sign()

    elif config.boolean_type == "sparse_threshold":
        probs = torch.tensor([0.7, 0.15, 0.15])
        choices = torch.tensor([0, 1, -1])
        weights_indices = torch.multinomial(
            probs, config.input_dim, replacement=True, generator=generator
        )
        weights = choices[weights_indices].float().unsqueeze(-1)

        kw = torch.randint(-3, 3, (1,), generator=generator).float() + 0.5
        y = (x @ weights).squeeze(-1) - kw
        y = y.sign()

    elif config.boolean_type == "parity":
        # Randomly select a subset of indices to define the parity function
        func_idx = int(torch.randint(0, 2**config.input_dim, (1,), generator=generator))
        subset = [j for j in range(config.input_dim) if (func_idx & (1 << j))]
        weights = torch.zeros(config.input_dim)
        if subset:
            weights[subset] = 1.0

        y = (x @ weights.unsqueeze(-1)).squeeze(-1) % 2
        y = (y * 2.0 - 1.0).sign()

    elif config.boolean_type == "sparse_parity":
        assert config.k <= config.input_dim
        # Select k random indices for the parity mask
        mask_indices = torch.randperm(config.input_dim, generator=generator)[: config.k]
        weights = torch.zeros(config.input_dim)
        weights[mask_indices] = 1.0

        y = (x @ weights.unsqueeze(-1)).squeeze(-1) % 2
        y = (y * 2.0 - 1.0).sign()

    if y is None:
        raise ValueError(
            f"Boolean type '{config.boolean_type}' not recognized or implemented."
        )

    sequence = interleave_tokens(x, y)
    mask = torch.zeros(
        config.input_seq_len * 2, dtype=torch.bool, device=sequence.device
    )
    mask[1::2] = True

    return sequence, mask


def _generate_dnf_cnf_style(
    x: torch.Tensor,
    input_dim: int,
    boolean_type: Literal["dnf3", "cnf3", "disjunction", "conjunction"],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Helper to generate data for DNF, CNF, Disjunction, and Conjunction."""
    device = x.device
    input_seq_len = x.shape[0]

    # Define parameters based on the boolean type
    if boolean_type in ["dnf3", "cnf3"]:
        num_clauses = 3
        probs = torch.tensor([0.8, 0.1, 0.1], device=device)
        bias_prob = 0.35
    else:  # disjunction, conjunction
        num_clauses = 1
        probs = torch.tensor([0.7, 0.15, 0.15], device=device)
        bias_prob = 0.3

    choices = torch.tensor([0, 1, -1], device=device)
    weights_list = []
    for _ in range(num_clauses):
        indices = torch.multinomial(
            probs, input_dim, replacement=True, generator=generator
        )
        weights = choices[indices].float().unsqueeze(-1)
        weights_list.append(weights)

    # Vectorized data biasing (replaces the slow inner loops)
    if boolean_type in ["dnf3", "cnf3"]:
        # For DNF/CNF, one clause is chosen to bias the data
        clause_idx = int(torch.randint(0, num_clauses, (1,), generator=generator))
        wb = weights_list[clause_idx]
        pidx = (wb == 1.0).squeeze()
        nidx = (wb == -1.0).squeeze()

        # Decide which sequence elements to bias
        bias_mask = (
            torch.rand(input_seq_len, generator=generator, device=device) < bias_prob
        )

        if boolean_type == "dnf3":
            x[bias_mask, pidx] = 1.0
            x[bias_mask, nidx] = -1.0
        else:  # cnf3
            x[bias_mask, pidx] = -1.0
            x[bias_mask, nidx] = 1.0

    # Calculate ys for all clauses
    ys_list = []
    for w in weights_list:
        kw = torch.norm(w, p=1) - 1
        if boolean_type in ["dnf3", "conjunction"]:
            y = (x @ w).squeeze(-1) - kw
        else:  # cnf3, disjunction
            y = (x @ w).squeeze(-1) + kw
        ys_list.append(y.sign())

    # Combine clause results
    ys_stack = torch.stack(ys_list, dim=1)
    if boolean_type in ["dnf3", "disjunction"]:
        y = ys_stack.max(dim=1)[0]
    else:  # cnf3, conjunction
        y = ys_stack.min(dim=1)[0]

    return y.sign()
