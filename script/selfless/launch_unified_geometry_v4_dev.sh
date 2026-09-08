#!/usr/bin/env bash
set -eo pipefail
if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash $0 <output-dir>" >&2
  exit 2
fi
geometry_output="$1"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=".:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup .venv/bin/python -u scripts/run_unified_geometry_v4.py \
  --output-dir "$geometry_output" >> "$geometry_output/driver.log" 2>&1 < /dev/null &
geometry_pid=$!
echo "Started geometry supervisor PID $geometry_pid"
