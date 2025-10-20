# ICL Benchmark

We implement a benchmark for evaluating the ICL capabilities of sequence modeling architectures. It allows for the creation and training of transformer-based models on various synthetic tasks designed to test different aspects of in-context learning.

## Overview

- [tasks.py](./tasks.py) - Defines the synthetic tasks used to evaluate the models. Each task is defined by a configuration class that specifies the task's parameters, such as the loss function, input sequence length, and vocabulary size. The available tasks include:
    - Regression: A family of regression tasks, including linear, quadratic, and sparse linear regression.

    - MQAR (Multi-Query Associative Recall): Tests a model's ability to recall values based on keys provided in a context.

    - Disjoint Sets: Requires the model to identify the single token present in two otherwise disjoint sets of tokens.

    - Boolean Tasks: A collection of tasks that involve learning various boolean functions, such as dnf3, cnf3, and parity. 
- [model.py](./model.py) - defines the transformer-based model used in the benchmark. 
- [layers.py](./layers.py) - contains the building blocks of the transformer model, including various sequence and state mixer configurations. It provides implementations for different attention mechanisms, such as standard multi-head attention, linear attention, and gated linear attention, as well as different feed-forward network architectures like MLP, GLU, and SwiGLU.
- [recipe.py](./recipe.py) - main recipe entrypoint creating and running our trainer.

## Running the benchmark

The following assumes `miniseq` is installed.

First, install [fla](https://github.com/fla-org/flash-linear-attention), as we rely on it to implement certain sequence mixers (e.g. [Gated Linear Attention](https://arxiv.org/abs/2312.06635)):

```sh
uv pip install flash-linear-attention
```
(or `pip install flash-linear-attention` in the corresponding environment)

For the following, if not using `uv` replace `uv run` with `python`.

Run `uv run recipe.py --choices` to see the standard CLI options and understand what **tasks**, **sequence mixers** and **state mixers** are available:

```
╭─ task choices ─────────────────────────────────────────────────────────────╮
│ (default: task:mqar)                                                       │
│ ────────────────────────────────────────────────────────────────────────── │
│ [{task:boolean,task:disjoint,task:mqar,task:regression}]                   │
│     task:boolean                                                           │
│     task:disjoint                                                          │
│     task:mqar                                                              │
│     task:regression                                                        │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ model.seq_mixer choices ──────────────────────────────────────────────────╮
│ (default: model.seq_mixer:attention)                                       │
│ ────────────────────────────────────────────────────────────────────────── │
│ [{model.seq_mixer:gated_linear_attn,model.seq_mixer:linear_attention,mode… │
│     model.seq_mixer:gated_linear_attn                                      │
│     model.seq_mixer:linear_attention                                       │
│     model.seq_mixer:attention                                              │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ model.state_mixer choices ────────────────────────────────────────────────╮
│ (default: model.state_mixer:mlp)                                           │
│ ────────────────────────────────────────────────────────────────────────── │
│ [{model.state_mixer:swiglu,model.state_mixer:glu,model.state_mixer:mlp}]   │
│     model.state_mixer:swiglu                                               │
│     model.state_mixer:glu                                                  │
│     model.state_mixer:mlp                                                  │
╰────────────────────────────────────────────────────────────────────────────╯
```

These options need to be specified before any other overrides, and once specified allow for the customization of individual layer and task parameters. The following is equivalent to running the default command (`uv run recipe.py --help`): 

```sh
uv run recipe.py task:mqar model.seq_mixer:attention model.state_mixer:mlp --help
```

Running the above without the `--help` option will train a small 2-layer standard transformer (with multi-head attention and MLPs) on the [MQAR](https://arxiv.org/abs/2312.04927) task, which is standard for evaluation ICL capabilities. We also turn off wandb logging (controlled by `wandb:on` and `wandb:None`) for a dry run:


```sh
uv run recipe.py wandb:None task:mqar model.seq_mixer:attention model.state_mixer:mlp
```

The model should reach close to ~98% accuracy after ~1500 steps (4-5 minutes on a single 5090).