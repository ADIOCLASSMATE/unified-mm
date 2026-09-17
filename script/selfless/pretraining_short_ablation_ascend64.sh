#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=2
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HCCL_INTRA_ROCE_ENABLE=1 HCCL_CONNECT_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF TORCH_DEVICE_BACKEND_AUTOLOAD
for driver_library_dir in /usr/local/Ascend/driver/lib64/driver \
  /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
  if [[ -d "${driver_library_dir}" ]]; then
    export LD_LIBRARY_PATH="${driver_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done
if [[ "${1:-}" == "--generation-smoke" ]]; then
  shift
  exec python -m torch.distributed.run --standalone --nproc_per_node=16 \
    scripts/smoke_training_image_generation.py "$@"
fi
exec python scripts/launch_short_ablation.py "$@"
