#!/bin/bash

set -e
set +x

# --- Allow overrides and both 'torchrun'/'python' as launchers. ---
# Set RUNNER_CMD and SCRIPT_ARGS
source "parse.sh"
parse_arguments "$@"

# --- Conditionally set the wandb argument ---
# export WANDB_API_KEY=<your_key_here>
# then run `wandb login`
if [ -n "$WANDB_API_KEY" ]; then
  wandb_arg="wandb:on"
else
  wandb_arg="wandb:None"
fi

# --- Standard script ---
set -x

FILE="countdown.py"

"${RUNNER_CMD[@]}" "$FILE" \
  "$wandb_arg" \
  --model.name=qwen2.5-3b \
  --train.device_batch_size=64 \
  --train.micro_batch_size=4 \
  --train.rollout_batch_size=64 \
  --train.publish_metrics_every=2 \
  --train.ac=True \
  --train.max_epochs=None \
  --train.max_steps=125 \
  --train.no_sync=False \
  --train.max_grad_norm=1.0 \
  --grpo.group_size=8 \
  --grpo.mu=1 \
  --grpo.std_normalize_advantage=False \
  --grpo.clip_eps_high=0.2 \
  --grpo.beta=0.0 \
  --grpo.rollout_correction=True \
  --packed=True \
  --prefix_share=False \
  "${SCRIPT_ARGS[@]}"  