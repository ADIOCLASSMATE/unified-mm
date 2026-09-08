#!/usr/bin/env bash
set -eo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: bash $0 <output-dir> <state> <profiles> [extract options...]" >&2
  exit 2
fi
repr_output="$1"
repr_state="$2"
repr_profiles="$3"
shift 3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=".:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_COMPILE_DISABLE=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=16 \
  scripts/probe_unified_semantics_v2.py extract --output-dir "$repr_output" \
  --state "$repr_state" --profiles "$repr_profiles" --batch-size 16 "$@"
