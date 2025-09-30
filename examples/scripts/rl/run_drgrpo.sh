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

FILE="dr_grpo.py"

"${RUNNER_CMD[@]}" "$FILE" \
  "$wandb_arg" \
  --model.name=qwen2.5-math-1.5b \
  --model.flash_attention2=True \
  --train.device_batch_size=128 \
  --train.micro_batch_size=4 \
  --train.rollout_batch_size=64 \
  --train.publish_metrics_every=2 \
  --packed=True \
  --prefix_share=False \
  "${SCRIPT_ARGS[@]}"