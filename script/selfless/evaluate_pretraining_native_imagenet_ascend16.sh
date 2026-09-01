#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <imagenet-native-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
OUTPUT_DIR="$2"
PROFILE="${EVAL_PROFILE:-formal}"
CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
CACHE_SHARD_DIR="${CACHE_SHARD_DIR:-public/datasets/imagenet_full/vae_posterior_mar_kl16/val_shards}"
TASKS="${TASKS:-retrieval_1k,retrieval_5k}"
SCORING_BACKEND="${SCORING_BACKEND:-cached_prefix}"
NPU_COUNT=16

if [[ "${PROFILE}" == "smoke" ]]; then
  LIMIT="${LIMIT:-4}"
  BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-2}"
  LM_HEAD_CHUNK_TOKENS="${LM_HEAD_CHUNK_TOKENS:-32}"
elif [[ "${PROFILE}" == "formal" ]]; then
  LIMIT="${LIMIT:-0}"
  BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-32}"
  LM_HEAD_CHUNK_TOKENS="${LM_HEAD_CHUNK_TOKENS:-256}"
else
  echo "ERROR: EVAL_PROFILE must be smoke or formal; got ${PROFILE}" >&2
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
for required in \
  "${CONFIG}" \
  public/datasets/imagenet_full/manifest_val.jsonl \
  public/datasets/imagenet1k_synthetic_v1/captions/imagenet1k_val_visual_descriptions.jsonl \
  public/datasets/imagenet1k_synthetic_v1/t2i/classes.json; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing pretraining-native evaluation asset: ${required}" >&2
    exit 5
  fi
done
if [[ "$(find "${CACHE_SHARD_DIR}" -maxdepth 1 -name 'shard-*-of-*.pt' | wc -l)" != "16" ]]; then
  echo "ERROR: expected 16 ImageNet-val posterior cache shards" >&2
  exit 6
fi

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
printf 'state=RUNNING\nprofile=%s\nstarted_at=%s\nruntime_hashing_enabled=false\n' \
  "${PROFILE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
finish() {
  launch_status=$?
  if (( launch_status == 0 )); then launch_state=SUCCEEDED; else launch_state=FAILED; fi
  printf 'state=%s\nprofile=%s\nexit_code=%s\nfinished_at=%s\nruntime_hashing_enabled=false\n' \
    "${launch_state}" "${PROFILE}" "${launch_status}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
  exit "${launch_status}"
}
trap finish EXIT

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_imagenet_pretraining_native.py \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --cache_shard_dir "${CACHE_SHARD_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --tasks "${TASKS}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --request_chunk_size 128 \
  --lm_head_chunk_tokens "${LM_HEAD_CHUNK_TOKENS}" \
  --max_length 2048 \
  --limit "${LIMIT}" \
  --seed 424242 \
  --device npu \
  --model_dtype bf16 \
  --scoring_backend "${SCORING_BACKEND}" \
  --image_sigma_order auto
