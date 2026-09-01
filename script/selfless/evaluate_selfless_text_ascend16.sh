#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <text-evaluation-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
OUTPUT_DIR="$2"
PROFILE="${EVAL_PROFILE:-formal}"
CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
TEXT_DATA_ROOT="${TEXT_DATA_ROOT:-public/benchmarks/selfless_text_v1}"
TEXT_TASKS="${TEXT_TASKS:-arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,openbookqa,mmlu}"
NPU_COUNT=16

if [[ "${PROFILE}" == "smoke" ]]; then
  TEXT_LIMIT="${TEXT_LIMIT:-1}"
  TEXT_BATCH_PER_RANK="${TEXT_BATCH_PER_RANK:-1}"
  LM_HEAD_CHUNK_TOKENS="${LM_HEAD_CHUNK_TOKENS:-32}"
  PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
elif [[ "${PROFILE}" == "formal" ]]; then
  TEXT_LIMIT="${TEXT_LIMIT:-0}"
  TEXT_BATCH_PER_RANK="${TEXT_BATCH_PER_RANK:-8}"
  LM_HEAD_CHUNK_TOKENS="${LM_HEAD_CHUNK_TOKENS:-256}"
  PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
else
  echo "ERROR: EVAL_PROFILE must be smoke or formal; got ${PROFILE}" >&2
  exit 3
fi

for integer in \
  "${TEXT_LIMIT}" \
  "${TEXT_BATCH_PER_RANK}" \
  "${LM_HEAD_CHUNK_TOKENS}" \
  "${PROGRESS_EVERY}"; do
  if [[ ! "${integer}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid text-evaluation integer: ${integer}" >&2
    exit 4
  fi
done

CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 5
fi
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"
for required in \
  "${CONFIG}" \
  "${TEXT_DATA_ROOT}/manifest.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing text-evaluation asset: ${required}" >&2
    exit 6
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
  exit 7
fi

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${OUTPUT_DIR}"

STATUS_PATH="${OUTPUT_DIR}/launcher.status"
printf 'state=RUNNING\nprofile=%s\nstarted_at=%s\n' \
  "${PROFILE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_PATH}"
finish() {
  launch_status=$?
  if (( launch_status == 0 )); then
    launch_state=SUCCEEDED
  else
    launch_state=FAILED
  fi
  printf 'state=%s\nprofile=%s\nexit_code=%s\nfinished_at=%s\n' \
    "${launch_state}" "${PROFILE}" "${launch_status}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_PATH}"
  exit "${launch_status}"
}
trap finish EXIT

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_selfless_text_benchmarks.py \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --data_root "${TEXT_DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --tasks "${TEXT_TASKS}" \
  --batch_size_per_rank "${TEXT_BATCH_PER_RANK}" \
  --lm_head_chunk_tokens "${LM_HEAD_CHUNK_TOKENS}" \
  --max_length 4096 \
  --limit "${TEXT_LIMIT}" \
  --seed 42 \
  --device npu \
  --model_dtype bf16 \
  --progress_every "${PROGRESS_EVERY}"
