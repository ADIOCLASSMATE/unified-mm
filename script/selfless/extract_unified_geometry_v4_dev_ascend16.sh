#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=".:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_COMPILE_DISABLE=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=16 \
  scripts/extract_unified_geometry_v4.py "$@"
