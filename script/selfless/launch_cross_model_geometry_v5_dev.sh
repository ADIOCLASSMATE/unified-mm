#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=".:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
geometry_root="output/evaluation/research/cross-model-geometry-v5-20260907"
nohup .venv/bin/python -u scripts/run_cross_model_geometry_v5.py "$@" \
  >> "$geometry_root/driver.log" 2>&1 < /dev/null &
geometry_pid=$!
echo "Started V5 supervisor PID $geometry_pid"
