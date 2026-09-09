#!/usr/bin/env bash
set -euo pipefail
REFACTOR_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REFACTOR_REPORT_ROOT="${1:?provide the shared report directory}"
REFACTOR_STAGE="${2:-all}"
REFACTOR_LABEL="${3:-refactor-r1}"
mkdir -p "${REFACTOR_REPORT_ROOT}"
finish_refactor() {
  refactor_exit_code=$?
  printf '%s\n' "${refactor_exit_code}" > "${REFACTOR_REPORT_ROOT}/${REFACTOR_STAGE}-exit-code.txt"
  exit "${refactor_exit_code}"
}
trap finish_refactor EXIT
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
source "${REFACTOR_SOURCE_ROOT}/script/offline_env.sh"
cd "${REFACTOR_SOURCE_ROOT}"
export PYTHONPATH="${REFACTOR_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=2
export HCCL_INTRA_ROCE_ENABLE=1 HCCL_CONNECT_TIMEOUT=120
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF TORCH_DEVICE_BACKEND_AUTOLOAD
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
for refactor_driver_library in /usr/local/Ascend/driver/lib64/driver \
  /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
  if [[ -d "${refactor_driver_library}" ]]; then
    export LD_LIBRARY_PATH="${refactor_driver_library}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done
if [[ "${REFACTOR_STAGE}" == infra ]]; then
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python -m pytest -q \
    tests/test_caption_farm.py tests/test_caption_farm_lock.py tests/test_checkpoint_transaction.py \
    tests/test_checkpoint_retention.py tests/test_sharded_ema.py tests/test_scheduled_combined_loader.py \
    --basetemp="${REFACTOR_REPORT_ROOT}/pytest" --tb=short > "${REFACTOR_REPORT_ROOT}/pytest.log" 2>&1
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=16 \
    tests/distributed_checkpoint_smoke.py --backend hccl --output "${REFACTOR_REPORT_ROOT}/hccl.json" \
    > "${REFACTOR_REPORT_ROOT}/hccl.log" 2>&1
elif [[ "${REFACTOR_STAGE}" == all ]]; then
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python -m pytest -q tests --tb=short -rs \
    > "${REFACTOR_REPORT_ROOT}/pytest.log" 2>&1
  .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=16 \
    tests/distributed_checkpoint_smoke.py --backend hccl --output "${REFACTOR_REPORT_ROOT}/hccl.json" \
    > "${REFACTOR_REPORT_ROOT}/hccl.log" 2>&1
  .venv/bin/python scripts/smoke_flow_shared_content_npu.py \
    --output "${REFACTOR_REPORT_ROOT}/npu-parity.json" > "${REFACTOR_REPORT_ROOT}/npu-parity.log" 2>&1
  git show 1ad1670:models/modeling_model/modeling_selfless_generation.py \
    > "${REFACTOR_REPORT_ROOT}/generation-before-refactor.py"
  for refactor_depth in 30 16; do
    .venv/bin/python scripts/smoke_training_validation_lifecycle.py --depth "${refactor_depth}" \
      --label "${REFACTOR_LABEL}" --output-dir "${REFACTOR_REPORT_ROOT}/depth${refactor_depth}" \
      --reference-generation-file "${REFACTOR_REPORT_ROOT}/generation-before-refactor.py" \
      > "${REFACTOR_REPORT_ROOT}/depth${refactor_depth}-lifecycle.log" 2>&1
  done
else
  printf 'unknown stage: %s\n' "${REFACTOR_STAGE}" >&2
  exit 2
fi
printf 'passed\n' > "${REFACTOR_REPORT_ROOT}/${REFACTOR_STAGE}-PASSED"
