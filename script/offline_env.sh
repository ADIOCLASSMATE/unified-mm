#!/usr/bin/env bash
# Shared defaults for running training/evaluation on no-network instances.
# Export a variable before invoking a script to override any of these defaults.

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

_UNIFIED_MM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "${_UNIFIED_MM_REPO_ROOT}/.venv/bin/activate" ]; then
  echo "ERROR: virtualenv activation script not found: ${_UNIFIED_MM_REPO_ROOT}/.venv/bin/activate" >&2
  exit 1
fi
source "${_UNIFIED_MM_REPO_ROOT}/.venv/bin/activate"
unset _UNIFIED_MM_REPO_ROOT
