#!/usr/bin/env bash
set -eo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: bash $0 <final-ema-model-dir> <prepared-diagnostic-output-dir>" >&2
  exit 2
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=".:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_COMPILE_DISABLE=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1

.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=16 \
  scripts/probe_unified_representations.py extract \
  --model-source "$1" --output-dir "$2" --batch-size 16 \
  2>&1 | tee "$2/extraction.log"
