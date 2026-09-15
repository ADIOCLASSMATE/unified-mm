#!/usr/bin/env bash
set -euo pipefail
POSITIONWISE_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
source "${POSITIONWISE_SOURCE_ROOT}/script/offline_env.sh"
cd "${POSITIONWISE_SOURCE_ROOT}"
export PYTHONPATH="${POSITIONWISE_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=2
export HCCL_INTRA_ROCE_ENABLE=1 HCCL_CONNECT_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF TORCH_DEVICE_BACKEND_AUTOLOAD
for positionwise_driver_library in /usr/local/Ascend/driver/lib64/driver \
    /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
    if [[ -d "${positionwise_driver_library}" ]]; then
        export LD_LIBRARY_PATH="${positionwise_driver_library}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
exec python scripts/smoke_unified_positionwise_flow_head_scaling.py "$@"
