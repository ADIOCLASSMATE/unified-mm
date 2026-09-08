#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_COMPILE_DISABLE=1 WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*' no_proxy='*'
geometry_common_path="public/models/_dependencies/geometry-v5-common:.:${PYTHONPATH:-}"
geometry_models=(b f qwen_text dinov2 mae siglip janusflow showo2)
geometry_pids=()
for geometry_device in "${!geometry_models[@]}"; do
  geometry_name="${geometry_models[$geometry_device]}"
  geometry_imports="$geometry_common_path"
  if [[ "$geometry_name" == janusflow || "$geometry_name" == showo2 ]]; then
    geometry_imports="public/models/_dependencies/geometry-v5-flow-tf447:$geometry_imports"
  fi
  PYTHONPATH="$geometry_imports" .venv/bin/python -u scripts/check_geometry_v5_batch_layers.py \
    --model "$geometry_name" --device "$geometry_device" \
    > "output/evaluation/research/cross-model-geometry-v5-20260907/logs/cal-batch-layers-$geometry_name.log" 2>&1 &
  geometry_pids+=("$!")
done
geometry_failed=0
for geometry_pid in "${geometry_pids[@]}"; do
  wait "$geometry_pid" || geometry_failed=1
done
exit "$geometry_failed"
