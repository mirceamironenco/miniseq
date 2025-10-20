# miniseq

A small LLM post-training & fine-tuning library designed primarily to allow for simplified custom component integration.

## Setup / Installation

1. Clone the repository:

```bash
git clone https://github.com/mirceamironenco/miniseq.git
cd miniseq
```

2. Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Install dependencies:
```bash
uv sync
```

Recommended: install with all optional dependencies (flash-attn, flashinfer, etc..):
```bash
uv sync --all-extras
```

With a specific python variant:
```bash
uv sync --python 3.12 --all-extras
```

## Features

miniseq supports SFT and RL (DPO, GRPO, GSPO, etc.) training algorithms, as well as an easy to configure evaluation pipeline.

- **Supervised Fine-Tuning:**
  - Full Weights Fine-Tuning
  - Parameter-Efficient Fine-Tuning (PEFT) with LoRA/Q-LoRA Layers.
- **Preference Fine-Tuning:**
  - Direct Preference Optimization (DPO)
- **Reinforcement Learning (RL):**
  - Group Relative Policy Optimization (GRPO)
  - Token-level Group Sequence Policy Optimization (GSPO-token)
- **Efficiency:**
  - Native support of common model sharding strategies such as DP and FSDP2.
  - Sequence packing for FAv2 and Flex Attention. Additional prefix sharing option (during training) for Flex Attention.
  - Inference backend via [vLLM](https://github.com/vllm-project/vllm/) (with tensor/pipeline parallel via `torchrun`).

## Examples

See [examples](./examples/) for a more comprehensive set of training/evaluation recipes.

Once installed, a `miniseq_recipe` CLI command is available which can run default recipes.

1. Run `uv run miniseq_recipe --help` to see the default recipes:

```
usage: miniseq_recipe [-h] {tune,preference,rl,evaluate,model_registry}

╭─ options ──────────────────────────────────────────────────╮
│ -h, --help        show this help message and exit          │
╰────────────────────────────────────────────────────────────╯
╭─ subcommands ──────────────────────────────────────────────╮
│ {tune,preference,rl,evaluate,model_registry}               │
│     tune          Run standard finetune training recipe.   │
│     preference    Run standard preference training recipe. │
│     rl            Run standard online training recipe.     │
│     evaluate      Run standard evaluation pipeline.        │
│     model_registry                                         │
│                   Display all registered models.           │
╰────────────────────────────────────────────────────────────╯
```

2. Run e.g. `uv run miniseq_recipe tune --help` to see the default argument choices.

3. Run `uv run miniseq_recipe tune --choices` to see the multiple-choice arguments.

4. Running `uv run miniseq_recipe tune` will fine-tune [Qwen 2.5-1.5b-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on the [alpaca cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset. 

5. Multi-GPU training is supported via `torchrun`  e.g.:
```bash
uv run torchrun --nproc-per-node=gpu --no-python miniseq_recipe tune
```

## License

MIT

<details>

<summary><h2>Acknowledgements</a></h2></summary>

- The project originally started as a fork of [fairseq2](https://github.com/facebookresearch/fairseq2/branches) so it shares some of it's design philosophy and components (with the appropriate citations).

</details>
