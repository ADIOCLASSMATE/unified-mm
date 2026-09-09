#!/usr/bin/env bash
set -euo pipefail
LIFECYCLE_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIFECYCLE_REPORT_ROOT="${1:?provide a shared acceptance report directory}"
mkdir -p "${LIFECYCLE_REPORT_ROOT}"
finish_lifecycle() {
  lifecycle_exit_code=$?
  printf '%s\n' "${lifecycle_exit_code}" > "${LIFECYCLE_REPORT_ROOT}/exit-code.txt"
  exit "${lifecycle_exit_code}"
}
trap finish_lifecycle EXIT
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
source "${LIFECYCLE_SOURCE_ROOT}/script/offline_env.sh"
cd "${LIFECYCLE_SOURCE_ROOT}"
export PYTHONPATH="${LIFECYCLE_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=2
export HCCL_INTRA_ROCE_ENABLE=1 HCCL_CONNECT_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF TORCH_DEVICE_BACKEND_AUTOLOAD
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
for lifecycle_driver_library in /usr/local/Ascend/driver/lib64/driver \
    /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
  if [[ -d "${lifecycle_driver_library}" ]]; then
    export LD_LIBRARY_PATH="${lifecycle_driver_library}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done
for lifecycle_depth in 30 16; do
  python scripts/smoke_training_validation_lifecycle.py --depth "${lifecycle_depth}" \
    --label validation-r2 --output-dir "${LIFECYCLE_REPORT_ROOT}/depth${lifecycle_depth}"
done
printf 'passed\n' > "${LIFECYCLE_REPORT_ROOT}/SMOKE_PASSED"
