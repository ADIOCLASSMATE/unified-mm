#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 3 ]]; then
  echo "Usage: $0 <model-source-dir> <retrieval-asset-root> <output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
ASSET_ROOT="$2"
OUTPUT_DIR="$3"
PROFILE="${EVAL_PROFILE:-formal}"
CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
CACHE_SHARD_DIR="${CACHE_SHARD_DIR:-${ASSET_ROOT}/vae_posterior_mar_kl16/shards}"
CACHE_COMPLETE_PATH="${CACHE_COMPLETE_PATH:-${ASSET_ROOT}/vae_posterior_mar_kl16/cache.complete.json}"
SCORING_BACKEND="${SCORING_BACKEND:-cached_prefix}"
QUERY_PARTITION_INDEX="${QUERY_PARTITION_INDEX:-0}"
QUERY_PARTITION_COUNT="${QUERY_PARTITION_COUNT:-1}"
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
if ! [[ "${QUERY_PARTITION_COUNT}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${QUERY_PARTITION_INDEX}" =~ ^[0-9]+$ ]] || \
   (( QUERY_PARTITION_INDEX >= QUERY_PARTITION_COUNT )); then
  echo "ERROR: invalid query partition ${QUERY_PARTITION_INDEX}/${QUERY_PARTITION_COUNT}" >&2
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
  "${ASSET_ROOT}/manifest.json" \
  "${ASSET_ROOT}/retrieval.jsonl" \
  "${CACHE_COMPLETE_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing cross-dataset retrieval asset: ${required}" >&2
    exit 5
  fi
done

python - "${ASSET_ROOT}/manifest.json" "${CACHE_COMPLETE_PATH}" <<'PY'
import json
from pathlib import Path
import sys

asset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cache = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if asset.get("complete") is not True or cache.get("status") != "ok":
    raise RuntimeError("retrieval assets or posterior cache are incomplete")
if asset.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError("retrieval assets violate the no-hash contract")
if cache.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError("retrieval cache violates the no-hash contract")
if int(cache.get("records", -1)) != int(asset.get("images", -2)):
    raise RuntimeError("retrieval cache cardinality does not match the asset split")
PY

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
  scripts/evaluate_cross_dataset_retrieval.py \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --asset_root "${ASSET_ROOT}" \
  --cache_shard_dir "${CACHE_SHARD_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --request_chunk_size 128 \
  --lm_head_chunk_tokens "${LM_HEAD_CHUNK_TOKENS}" \
  --max_length 2048 \
  --limit "${LIMIT}" \
  --query_partition_index "${QUERY_PARTITION_INDEX}" \
  --query_partition_count "${QUERY_PARTITION_COUNT}" \
  --seed 424242 \
  --progress_every 1 \
  --device npu \
  --model_dtype bf16 \
  --scoring_backend "${SCORING_BACKEND}" \
  --image_sigma_order auto
