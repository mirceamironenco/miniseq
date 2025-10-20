#!/bin/bash

# A function to parse command-line arguments, separating the runner command
# from the script arguments.
#
# Arguments:
#   All the command-line arguments passed to the parent script ("$@").
#
# Sets the following variables in the calling script's scope:
#   RUNNER_CMD: An array containing the command to run the script. Defaults to "python".
#   SCRIPT_ARGS: An array containing the (override) arguments for the script itself.
parse_arguments() {
  local ALL_ARGS=("$@")
  local SEPARATOR_INDEX=-1

  # RUNNER_CMD=("python")
  # SCRIPT_ARGS=()

  # Only set defaults if not already set by the caller
  if ! declare -p RUNNER_CMD &>/dev/null || [ ${#RUNNER_CMD[@]} -eq 0 ]; then
    RUNNER_CMD=("uv" "run")   # default
  fi
  if ! declare -p SCRIPT_ARGS &>/dev/null; then
    SCRIPT_ARGS=()
  fi

  for i in "${!ALL_ARGS[@]}"; do
    if [[ "${ALL_ARGS[$i]}" == "--" ]]; then
      SEPARATOR_INDEX=$i
      break
    fi
  done

  if [[ $SEPARATOR_INDEX -ne -1 ]]; then
    # If something exists before `--`, that's the explicit runner command.
    if [[ $SEPARATOR_INDEX -gt 0 ]]; then
      RUNNER_CMD=("${ALL_ARGS[@]:0:$SEPARATOR_INDEX}")
    fi
    SCRIPT_ARGS=("${ALL_ARGS[@]:$((SEPARATOR_INDEX + 1))}")
  else
    # No `--`: all args go to the script
    SCRIPT_ARGS=("${ALL_ARGS[@]}")
  fi
}