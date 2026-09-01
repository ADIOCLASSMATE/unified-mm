#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <evaluation-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
EVAL_ROOT="$2"
PROFILE="${EVAL_PROFILE:-formal}"
CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
TEXT_DATA_ROOT="${TEXT_DATA_ROOT:-public/benchmarks/selfless_text_v1}"
NPU_COUNT=16

if [[ "${PROFILE}" == "smoke" ]]; then
  TEXT_LIMIT="${TEXT_LIMIT:-16}"
  T2I_GLOBAL_BATCH="${T2I_GLOBAL_BATCH:-32}"
elif [[ "${PROFILE}" == "formal" ]]; then
  TEXT_LIMIT="${TEXT_LIMIT:-0}"
  # This shape has already completed the independent ImageNet-val FID50K
  # protocol on the same 16x910B development node.
  T2I_GLOBAL_BATCH="${T2I_GLOBAL_BATCH:-4096}"
else
  echo "ERROR: EVAL_PROFILE must be smoke or formal; got ${PROFILE}" >&2
  exit 3
fi

for integer in "${TEXT_LIMIT}" "${T2I_GLOBAL_BATCH}"; do
  if [[ ! "${integer}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid evaluation integer: ${integer}" >&2
    exit 4
  fi
done
if (( T2I_GLOBAL_BATCH < NPU_COUNT || T2I_GLOBAL_BATCH % NPU_COUNT != 0 )); then
  echo "ERROR: T2I_GLOBAL_BATCH must be divisible by ${NPU_COUNT}" >&2
  exit 5
fi

CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 6
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
    echo "ERROR: missing full-evaluation asset: ${required}" >&2
    exit 7
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
  exit 8
fi

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${EVAL_ROOT}/text"

STATUS_PATH="${EVAL_ROOT}/launcher.status"
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

# Run the new adapter first.  A task-complete shard is resumable by readable
# checkpoint/task fields, so a later image-side failure does not repeat text work.
EVAL_PROFILE="${PROFILE}" \
TEXT_LIMIT="${TEXT_LIMIT}" \
CONFIG="${CONFIG}" \
TEXT_DATA_ROOT="${TEXT_DATA_ROOT}" \
  "${REPO_ROOT}/script/selfless/evaluate_selfless_text_ascend16.sh" \
  "${MODEL_SOURCE}" "${EVAL_ROOT}/text" \
  2>&1 | tee "${EVAL_ROOT}/text.log"

EVAL_PROFILE="${PROFILE}" \
T2I_GLOBAL_BATCH="${T2I_GLOBAL_BATCH}" \
CONFIG="${CONFIG}" \
  "${REPO_ROOT}/script/selfless/evaluate_unified_checkpoint_ascend16.sh" \
  "${MODEL_SOURCE}" "${EVAL_ROOT}"

python scripts/summarize_unified_full_evaluation.py \
  --checkpoint "${MODEL_SOURCE}" \
  --output_root "${EVAL_ROOT}" \
  --profile "${PROFILE}" \
  | tee "${EVAL_ROOT}/full-summary.log"
