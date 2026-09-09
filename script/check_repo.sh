#!/usr/bin/env bash
set -euo pipefail
CHECK_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${CHECK_REPO_ROOT}"
export PYTHONPATH="${CHECK_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_COMPILE_DISABLE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
ruff check caption_farm models utils pretrain scripts tests
.venv/bin/python -m pytest -q tests "$@"
