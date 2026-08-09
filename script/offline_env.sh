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
if [ -n "${UNIFIED_MM_VENV:-}" ]; then
  case "${UNIFIED_MM_VENV}" in
    /*) _UNIFIED_MM_VENV_ROOT="${UNIFIED_MM_VENV}" ;;
    *) _UNIFIED_MM_VENV_ROOT="${_UNIFIED_MM_REPO_ROOT}/${UNIFIED_MM_VENV}" ;;
  esac
elif [ -f "${_UNIFIED_MM_REPO_ROOT}/.venv/bin/activate" ]; then
  _UNIFIED_MM_VENV_ROOT="${_UNIFIED_MM_REPO_ROOT}/.venv"
elif [ -f "${_UNIFIED_MM_REPO_ROOT}/.venv-npu/bin/activate" ]; then
  _UNIFIED_MM_VENV_ROOT="${_UNIFIED_MM_REPO_ROOT}/.venv-npu"
else
  _UNIFIED_MM_VENV_ROOT="${_UNIFIED_MM_REPO_ROOT}/.venv"
fi
if [ ! -f "${_UNIFIED_MM_VENV_ROOT}/bin/activate" ]; then
  echo "ERROR: virtualenv activation script not found: ${_UNIFIED_MM_VENV_ROOT}/bin/activate" >&2
  exit 1
fi
source "${_UNIFIED_MM_VENV_ROOT}/bin/activate"
unset _UNIFIED_MM_REPO_ROOT _UNIFIED_MM_VENV_ROOT
