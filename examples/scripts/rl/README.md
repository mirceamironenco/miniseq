<details>

<summary><h3>Running the scripts</a></h3></summary>

All runner scripts in this folder use a shared `parse.sh` helper that cleanly separates the **launcher / runner** from the **script arguments**.

```bash
bash <script>.sh [<RUNNER...>] [-- <SCRIPT_ARGS...>]
```

- **`<RUNNER...>` (optional):** Tokens for the command that launches the Python file  
  (e.g., `python`, `uv run`, `torchrun --nproc-per-node=2`).  
  If omitted, the default is `uv run`.
- **`--` (optional but recommended):** End-of-options delimiter to separate the runner from your script’s own flags.
- **`<SCRIPT_ARGS...>` (optional):** Arguments passed to the underlying Python script (after the file name).

Simply run any of the below examples and the fully-resolved commands will be printed.

</details>

## RL Examples

[GRPO](https://arxiv.org/abs/2402.03300) is a variant of [PPO](https://arxiv.org/abs/1707.06347) which replaces the critic network with a Monte Carlo estimate of the value function. 

<summary><h3>Countdown GRPO</h3></summary>

To run the default GRPO recipe on Countdown:

```bash
bash run_countdown_qwen2.5-3b.sh
```

For distirbuted training pass the launcher command after the filename additionally ending with the end-of-options delimiter `--`. For example, to run on 2 GPUs:

```bash
bash run_countdown_qwen2.5-3b.sh uv run torchrun --nproc-per-node=2 --
```

We compare the effect of different data packing strategies and attention implementation choices:

<img width="949" height="663" alt="image" src="https://github.com/user-attachments/assets/032d1b68-0f95-410c-8507-08a93a69bb9c" />


The above plots  can be reproduced with the following comands (full wandb logs [here](https://wandb.ai/mirceam/miniseq_rl)):

<table>
<thead>
<tr>
<th>Command</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>
<details>
<summary><code>bash run_countdown_qwen2.5-3b.sh ...</code></summary>
<pre><code>bash run_countdown_qwen2.5-3b.sh --prefix_share=True --model.flex_attention=True</code></pre>
</details>
</td>
<td>Uses <b>prefix sharing</b> and <b>Flex Attention</b>.</td>
</tr>
<tr>
<td>
<details>
<summary><code>bash run_countdown_qwen2.5-3b.sh ...</code></summary>
<pre><code>bash run_countdown_qwen2.5-3b.sh --prefix_share=False --model.flash_attention2=True --wandb.run_name=qwen3b-fa2</code></pre>
</details>
</td>
<td>Uses <b>Flash Attention v2</b> and packed batches.</td>
</tr>
<tr>
<td>
<details>
<summary><code>bash run_countdown_qwen2.5-3b.sh ...</code></summary>
<pre><code>bash run_countdown_qwen2.5-3b.sh --prefix_share=False --model.flex_attention=True --wandb.run_name=qwen3b-flex</code></pre>
</details>
</td>
<td>Uses <b>Flex Attention</b> and packed batches.</td>
</tr>
</tbody>
</table>

To use wandb, make sure the `WANDB_API_KEY` env. variable is set, otherwise remove the `--wandb.run_name=...` overrides in the above commands.

<details>



<summary><h3>Reproducing <a href="https://arxiv.org/abs/2503.20783">dr. GRPO</a></h3></summary>

In [dr. GRPO](https://arxiv.org/abs/2503.20783), the authors posit that the original GRPO objective has a response-level length bias due to the normalization by each completion length. They further argue that in the calculation of the advantage, normalizing the rewards by their standard deviation causes a question-level difficulty bias, leading to disproportionate policy updates for questions with low std (i.e. very easy or very hard). 
The proposed corrections are integrated into our GRPO implementation.

To reproduce the results of Table 4 on Qwen2.5-Math-1.5B run:

```bash
bash run_drgrpo.sh --grpo.std_normalize_advantage=False --valid_data.datasets aime24 amc minerva math500 olympiad
```

</details>

<details>
<summary><h3>GSM8k</h3></summary>

[GSM8k](https://huggingface.co/datasets/openai/gsm8k) is a dataset of ~8.5k high-quality, math word problems designed for the grade school level. It was created to train and evaluate the multi-step mathematical reasoning capabilities of large language models.
We follow [verl](https://github.com/volcengine/verl) implementing the environment as a standard showcase for training LLMs via RL. The table below showcases commands used to apply GRPO to models with sizes ranging from 0.5B to 14B.

<table>
<thead>
<tr>
<th>Model</th>
<th>Method</th>
<th>Command</th>
</tr>
</thead>
<tbody>
<tr>
<td>Qwen2.5-0.5B-Instruct</td>
<td>GRPO</td>
<td><code>bash run_gsm8k.sh</code></td>
</tr>
<tr>
<td>Qwen2.5-1.5B-Instruct</td>
<td>GRPO</td>
<td>
<details>
<summary><code>bash run_gsm8k.sh ...</code></summary>
<pre><code>bash run_gsm8k.sh --model.name=qwen2.5-1.5b-instruct</code></pre>
</details>
</td>
</tr>
<tr>
<td>Qwen2.5-3B-Instruct</td>
<td>GRPO</td>
<td>
<details>
<summary><code>bash run_gsm8k.sh ...</code></summary>
<pre><code>bash run_gsm8k.sh --model.name=qwen2.5-3b-instruct</code></pre>
</details>
</td>
</tr>
<tr>
<td>Qwen2.5-7B-Instruct</td>
<td>GRPO-LoRA</td>
<td>
<details>
<summary><code>bash run_gsm8k.sh ...</code></summary>
<pre><code>bash run_gsm8k.sh uv run torchrun --nproc-per-node=2 -- lora:on --model.name=qwen2.5-7b-instruct --train.device_batch_size=64</code></pre>
</details>
</td>
</tr>
<tr>
<td>Qwen2.5-14B-Instruct</td>
<td>GRPO-LoRA</td>
<td>
<details>
<summary><code>bash run_gsm8k.sh ...</code></summary>
<pre><code>bash run_gsm8k.sh uv run torchrun --nproc-per-node=2 -- lora:on --model.name=qwen2.5-14b-instruct --train.device_batch_size=64</code></pre>
</details>
</td>
</tr>
</tbody>
</table>

</details>

## Preference Optimization Examples

<details>
<summary><h3>DPO on UltraFeedback</a></h3></summary>

In [ultrafeedback.py](./ultrafeedback.py) the standard DPO recipe is showcased. We directly reference and use the schema defined on HF at [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) to process the dataset items:


```py
# Schema on huggingface (optional).
UFItem = TypedDict[{"prompt": str, "chosen": list[Message], "rejected": list[Message]}]


def uf_transform(example: UFItem) -> PreferenceDict:
    # Assume single-turn
    chosen, rejected = filter(
        lambda x: x["role"] == "assistant", example["chosen"] + example["rejected"]
    )

    return {
        "prompt": example["prompt"],
        "chosen": chosen["content"],
        "rejected": rejected["content"],
    }


@dataclasses.dataclass
class Config(recipes.PreferenceRecipeConfig):
    train_data: cfg.data.PreferenceDatasetConfig = cfg.data.PreferenceDatasetConfig(
        name="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        preference_map=uf_transform,
        max_seqlen=2049,
    )


config = cli.run_default_cli(Config, console_outputs=on_local_rank_zero())
trainer = recipes.create_preference_trainer(config)
trainer.run()
```

To run on 4 GPUs:

```bash
uv run torchrun --nproc-per-node=4 ultrafeedback.py --model.name=qwen2.5-7b-instruct --packed=True --model.flash_attention2=True
```


</details>
