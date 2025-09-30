# Features

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
  - Inference backend via vLLM (with tensor/pipeline parallel via `torchrun`).
- **In Progress**
  - Proximal Policy Optimization (PPO)
  - SIMPO
  - Q-LoRA
  - Knowledge Distillation

## Customization

Examples of **custom recipes** can be found in the [SFT](./sft/) and [RL](./rl/) folders. The simplest way to get up and running is to mimic the existing example implementations. Note that it possible to register and use custom components for almost all parts of a training pipeline as we largely follow a depdenency injection design where the main trainer object has all of it's requirements built before it is created, and the same trainer object is used for SFT, RL, preference-tuning, etc. 

The CLI also functions based on a registry system allowing users to implement their own optimizer, scheduler, model, etc, and have it show up as a possible choice.

The default SFT, preference and RL recipes entry points and how to run them is described below.

## Default recipe usage

Once installed, a `miniseq_recipe` CLI command is available. The command (described below) points to `miniseq/entry.py`, so it's also possible to run the entrypoint via e.g. `python -m miniseq.entry ...` from the root directory.

To start, simply run `miniseq_recipe --help` to list the standard subcommands:

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

Running `miniseq_recipe model_registry` will list the currently supported models:

```
                                  Registered models                                   
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ qwen                       ┃ deepseek-ai                   ┃ meta-llama            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ qwen2.5-3b-instruct        │ deepseek-r1-distill-qwen-1.5b │ llama-3.2-1b-instruct │
│ qwen2.5-0.5b               │                               │ llama-3.2-1b          │
│ qwen3-1.7b                 │                               │ llama-3.2-3b-instruct │
│ qwen2.5-7b-instruct        │                               │ llama-3.2-3b          │
│ qwen2.5-math-1.5b-instruct │                               │ llama-3.1-8b-instruct │
│ qwen2.5-math-1.5b          │                               │ llama-3.1-8b          │
│ qwen2.5-0.5b-instruct      │                               │ llama-3-8b            │
│ qwen2.5-1.5b-instruct      │                               │                       │
│ qwen2.5-1.5b               │                               │                       │
│ qwen2.5-3b                 │                               │                       │
└────────────────────────────┴───────────────────────────────┴───────────────────────┘

```

The standard recieps can be customized using either YAML files (for simple changes that do not require custom datasets, reward functions, etc.) or via a small script which overrides the default configs (see [RL](./rl/), [SFT](./sft/) for examples). Both options integrate with the existing CLI via [tyro](https://github.com/brentyi/tyro), so one can additionally directly modify and run any standard or custom recipe via command line. 

### Example usage via CLI

The standard supervised fine-tuning recipe exposes the following options (seen by running `miniseq_recipe tune --help`):


<details>
<summary>Fine-tuning recipe options</summary>

```
usage: miniseq_recipe optimizer:adamw lr_scheduler:cosine_decay dp:fsdp2 
wandb:None 
lora:None profiler:None validate:None
       [-h] [VALIDATE:NONE OPTIONS]

╭─ options ──────────────────────────────────────────────────────────────────╮
│ -h, --help              show this help message and exit                    │
│ --cache_dir PATH        Model/data cache dir. (default: local_data)        │
│ --tensorboard {True,False}                                                 │
│                         Whether to use tensorboard. (default: False)       │
│ --eval_avg_n INT        (default: 1)                                       │
│ --eval_pass_k INT       (default: 1)                                       │
│ --seed INT              (default: 2)                                       │
│ --packed {True,False}   (default: True)                                    │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ dp options ───────────────────────────────────────────────────────────────╮
│ --dp.fsdp2_reshard_fwd {True,False}                                        │
│                         Reshard params after forward. (default: True)      │
│ --dp.fsdp2_cpu_offload {True,False}                                        │
│                         (default: False)                                   │
│ --dp.fsdp2_reshard_outer_fwd {True,False}                                  │
│                         Reshard after fwd for outer-most module. (default: │
│                         True)                                              │
│ --dp.fsdp2_fp32_reduce {True,False}                                        │
│                         (default: False)                                   │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ lr_scheduler options ─────────────────────────────────────────────────────╮
│ --lr_scheduler.warmup_steps INT                                            │
│                         Takes precedence over warmup_ratio. (default: 5)   │
│ --lr_scheduler.warmup_ratio FLOAT                                          │
│                         (default: 0.01)                                    │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ optimizer options ────────────────────────────────────────────────────────╮
│ --optimizer.lr FLOAT    (default: 0.0002)                                  │
│ --optimizer.weight_decay FLOAT                                             │
│                         (default: 0.01)                                    │
│ --optimizer.betas FLOAT FLOAT                                              │
│                         (default: 0.9 0.999)                               │
│ --optimizer.eps FLOAT   (default: 1e-08)                                   │
│ --optimizer.amsgrad {True,False}                                           │
│                         (default: False)                                   │
│ --optimizer.fused {None,True,False}                                        │
│                         (default: True)                                    │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ train options ────────────────────────────────────────────────────────────╮
│ --train.max_steps {None}|INT                                               │
│                         Max number of steps to train. (default: 10000)     │
│ --train.max_epochs {None}|INT                                              │
│                         Max number of epochs to train. (default: 15)       │
│ --train.micro_batch_size INT                                               │
│                         Per-GPU fwd+bwd pass batch size. (default: 1)      │
│ --train.device_batch_size INT                                              │
│                         Per-GPU optimizer step batch size. (default: 1)    │
│ --train.rollout_batch_size {None}|INT                                      │
│                         (default: None)                                    │
│ --train.no_sync {True,False}                                               │
│                         (default: False)                                   │
│ --train.anomaly {True,False}                                               │
│                         Turns on anomaly detection. (default: False)       │
│ --train.max_grad_norm {None}|FLOAT                                         │
│                         (default: 3.0)                                     │
│ --train.ac {True,False}                                                    │
│                         Enable activation checkpointing. (default: False)  │
│ --train.ac_freq INT     Apply activ. ckpt. every 'ac_freq'-th layer.       │
│                         (default: 1)                                       │
│ --train.checkpoint_every INT                                               │
│                         (default: 100)                                     │
│ --train.checkpoint_last_n {None}|INT                                       │
│                         Number of ckpts. kept while training. (default: 3) │
│ --train.publish_metrics_every INT                                          │
│                         (default: 3)                                       │
│ --train.validate_every INT                                                 │
│                         (default: 25)                                      │
│ --train.validate_at_start {True,False}                                     │
│                         (default: False)                                   │
│ --train.save_model_only {True,False}                                       │
│                         (default: True)                                    │
│ --train.resume_checkpoint {True,False}                                     │
│                         Resume from local ckpt. (default: False)           │
│ --train.resume_model_only {True,False}                                     │
│                         (default: False)                                   │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ compile options ──────────────────────────────────────────────────────────╮
│ --compile.model {True,False}                                               │
│                         Whether to compile the model. (default: True)      │
│ --compile.loss {True,False}                                                │
│                         Compile the loss function. (default: False)        │
│ --compile.optimizer_step {True,False}                                      │
│                         Compile optim+scheduler step. (default: False)     │
│ --compile.fullgraph {True,False}                                           │
│                         Compile using fullgraph=True. (default: False)     │
│ --compile.dynamic {True,False}                                             │
│                         Compile using dynamic=True. (default: False)       │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ train_data options ───────────────────────────────────────────────────────╮
│ --train_data.name STR   (default: yahma/alpaca-cleaned)                    │
│ --train_data.configuration {None}|STR                                      │
│                         (default: None)                                    │
│ --train_data.data_files {None}|STR                                         │
│                         (default: None)                                    │
│ --train_data.split {None}|STR                                              │
│                         (default: None)                                    │
│ --train_data.test_split {None}|STR                                         │
│                         (default: None)                                    │
│ --train_data.test_split_ratio FLOAT                                        │
│                         (default: 0.0)                                     │
│ --train_data.completions_only {True,False}                                 │
│                         (default: True)                                    │
│ --train_data.packed_seqlen {None}|INT                                      │
│                         (default: 4097)                                    │
│ --train_data.apply_chat_template {True,False}                              │
│                         (default: True)                                    │
│ --train_data.max_seqlen {None}|INT                                         │
│                         (default: 2048)                                    │
│ --train_data.columns {None}|{STR STR {None}|STR {None}|STR}                │
│                         instruction, completion, input, system (default:   │
│                         instruction output input None)                     │
╰────────────────────────────────────────────────────────────────────────────╯
╭─ model options ────────────────────────────────────────────────────────────╮
│ --model.name {qwen2.5-3b-instruct,qwen2.5-0.5b,qwen3-1.7b,...}             │
│                         (default: ['qwen2.5-1.5b-instruct'], run entry.py  │
│                         model_registry)                                    │
│ --model.dtype {bfloat16,float32,float64}                                   │
│                         (default: bfloat16)                                │
│ --model.flex_attention {True,False}                                        │
│                         (default: False)                                   │
│ --model.flash_attention2 {True,False}                                      │
│                         (default: False)                                   │
│ --model.reduce_flex {True,False}                                           │
│                         (default: False)                                   │
│ --model.use_cce {True,False}                                               │
│                         (default: True)                                    │
│ --model.load_on_cpu {True,False}                                           │
│                         (default: True)                                    │
│ --model.finetune_repo_id {None}|STR                                        │
│                         (default: None)                                    │
╰────────────────────────────────────────────────────────────────────────────╯

```

</details>

By default running `miniseq_recipe tune` will train [Qwen 2.5-1.5b-instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on the [alpaca cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset. Overriding options is done via the standard syntax:

```bash
miniseq_recipe tune --model.name=qwen2.5-3b-instruct
```

miniseq supports FSDP2/DDP (with TP/CP in progress) training via `torchrun`:

```bash
torchrun --nproc-per-node=$GPU --no-python miniseq_recipe tune --model.name=qwen2.5-3b-instruct
```