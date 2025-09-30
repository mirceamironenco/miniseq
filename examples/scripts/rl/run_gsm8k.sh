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

FILE="gsm8k.py"

"${RUNNER_CMD[@]}" "$FILE" \
  "$wandb_arg" \
  "${SCRIPT_ARGS[@]}"  