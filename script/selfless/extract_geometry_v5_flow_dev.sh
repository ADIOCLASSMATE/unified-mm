#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="public/models/_dependencies/geometry-v5-flow-tf447:public/models/_dependencies/geometry-v5-common:.:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_COMPILE_DISABLE=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
geometry_world="${GEOMETRY_V5_WORLD_SIZE:-16}"
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$geometry_world" \
  scripts/extract_geometry_v5_flow.py "$@"
