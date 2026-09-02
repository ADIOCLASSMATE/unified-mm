#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 5 ]]; then
  echo "Usage: $0 <geneval|dpgbench|mjhq> <config> <model-source> <official-repository> <output-root>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCHMARK="$1"
CONFIG="$2"
MODEL_SOURCE="$3"
PROMPT_SOURCE="$4"
OUTPUT_ROOT="$5"
NPU_COUNT=16

if [[ ! "${BENCHMARK}" =~ ^(geneval|dpgbench|mjhq)$ ]]; then
  echo "ERROR: unsupported benchmark: ${BENCHMARK}" >&2
  exit 3
fi

CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 4
fi
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

for required in "${CONFIG}" "${MODEL_SOURCE}" "${PROMPT_SOURCE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: missing generation input: ${required}" >&2
    exit 5
  fi
done

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${VISIBLE_NPUS}" >&2
  exit 6
fi

mkdir -p "${OUTPUT_ROOT}"
STATUS_PATH="${OUTPUT_ROOT}/launcher.status"
printf 'state=RUNNING\nstarted_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
finish() {
  launch_status=$?
  if (( launch_status == 0 )); then launch_state=SUCCEEDED; else launch_state=FAILED; fi
  printf 'state=%s\nexit_code=%s\nfinished_at=%s\n' \
    "${launch_state}" "${launch_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
  exit "${launch_status}"
}
trap finish EXIT

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/generate_official_t2i_benchmarks.py "${BENCHMARK}" \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --prompt_source "${PROMPT_SOURCE}" \
  --output_dir "${OUTPUT_ROOT}/generation" \
  --device npu \
  --model_dtype "${MODEL_DTYPE:-bf16}" \
  --vae_dtype "${VAE_DTYPE:-fp32}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK:-4}" \
  --vae_decode_batch_size "${VAE_DECODE_BATCH_SIZE:-4}" \
  --seed "${SEED:-42}" \
  --sampling_steps "${SAMPLING_STEPS:-10}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --cfg "${CFG:-3.5}" \
  --cfg_schedule "${CFG_SCHEDULE:-constant}" \
  --flow_solver "${FLOW_SOLVER:-heun}" \
  --parallel_rate "${PARALLEL_RATE:-1}" \
  --strategy "${STRATEGY:-spatial_halton}" \
  --image_sigma_order "${IMAGE_SIGMA_ORDER:-random}" \
  2>&1 | tee "${OUTPUT_ROOT}/generation.log"
