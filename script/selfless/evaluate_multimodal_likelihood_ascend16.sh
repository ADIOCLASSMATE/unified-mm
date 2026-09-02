#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <multimodal-likelihood-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
OUTPUT_DIR="$2"
PROFILE="${EVAL_PROFILE:-formal}"
CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
ASSET_ROOT="${ASSET_ROOT:-public/benchmarks/selfless_multimodal_likelihood_v1}"
CACHE_ROOT="${CACHE_ROOT:-${ASSET_ROOT}/vae_posterior_mar_kl16_v2}"
CACHE_SHARD_DIR="${CACHE_SHARD_DIR:-${CACHE_ROOT}/shards}"
CACHE_COMPLETE_PATH="${CACHE_COMPLETE_PATH:-${CACHE_ROOT}/cache.complete.json}"
TASKS="${TASKS:-}"
MC="${MC:-64}"
NPU_COUNT=16

if [[ "${PROFILE}" == "smoke" ]]; then
  LIMIT="${LIMIT:-2}"
  BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-1}"
  LM_HEAD_CHUNK_TOKENS="${LM_HEAD_CHUNK_TOKENS:-32}"
  PROGRESS_EVERY="${PROGRESS_EVERY:-2}"
  PROTOCOL_ARGS=()
elif [[ "${PROFILE}" == "formal" ]]; then
  LIMIT=0
  MC=64
  BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-4}"
  LM_HEAD_CHUNK_TOKENS="${LM_HEAD_CHUNK_TOKENS:-256}"
  PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
  PROTOCOL_ARGS=(--require_formal_protocol)
else
  echo "ERROR: EVAL_PROFILE must be smoke or formal; got ${PROFILE}" >&2
  exit 3
fi

for integer in \
  "${LIMIT}" \
  "${BATCH_SIZE_PER_RANK}" \
  "${MC}" \
  "${LM_HEAD_CHUNK_TOKENS}" \
  "${PROGRESS_EVERY}"; do
  if [[ ! "${integer}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid multimodal likelihood integer: ${integer}" >&2
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
  "${ASSET_ROOT}/manifest.json" \
  "${CACHE_COMPLETE_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing multimodal likelihood evaluation asset: ${required}" >&2
    exit 6
  fi
done

if [[ -z "${TASKS}" ]]; then
  TASKS="$(python - "${ASSET_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
preferred = (
    "mmbench_dev_en",
    "seed_bench_image",
    "sugarcrepe",
    "aro_vg_relation",
    "aro_vg_attribution",
)
print(",".join(task for task in preferred if task in manifest.get("tasks", {})))
PY
)"
fi
if [[ -z "${TASKS}" ]]; then
  echo "ERROR: asset manifest contains no supported likelihood tasks" >&2
  exit 6
fi

python - "${ASSET_ROOT}/manifest.json" "${ASSET_ROOT}/image_manifest.jsonl" \
  "${CACHE_COMPLETE_PATH}" <<'PY'
import json
from pathlib import Path
import sys

asset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
image_manifest = Path(sys.argv[2]).resolve()
cache = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
stat = image_manifest.stat()
expected_source = {
    "path": str(image_manifest),
    "bytes": int(stat.st_size),
    "mtime_ns": int(stat.st_mtime_ns),
}
if asset.get("schema") != "selfless_multimodal_likelihood_assets_v2":
    raise RuntimeError("multimodal likelihood assets use an obsolete schema")
if cache.get("schema") != "selfless_image_posterior_cache_v2":
    raise RuntimeError("posterior cache uses an obsolete schema")
if cache.get("asset_schema") != asset.get("schema"):
    raise RuntimeError("posterior cache asset schema mismatch")
if cache.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError("posterior cache violates the no-hash contract")
if int(cache.get("records", -1)) != int(asset.get("images", -2)):
    raise RuntimeError("posterior cache record count does not match assets")
if cache.get("source_manifest") != expected_source:
    raise RuntimeError("posterior cache was built for a different readable manifest revision")
if cache.get("language_prior_null_image_ids") != [9000000000, 9000000001, 9000000002]:
    raise RuntimeError("posterior cache lacks the formal null-image contract")
PY

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
printf 'state=RUNNING\nprofile=%s\nmc=%s\nstarted_at=%s\nruntime_hashing_enabled=false\n' \
  "${PROFILE}" "${MC}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
finish() {
  launch_status=$?
  if (( launch_status == 0 )); then
    launch_state=SUCCEEDED
  else
    launch_state=FAILED
  fi
  printf 'state=%s\nprofile=%s\nmc=%s\nexit_code=%s\nfinished_at=%s\nruntime_hashing_enabled=false\n' \
    "${launch_state}" "${PROFILE}" "${MC}" "${launch_status}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
  exit "${launch_status}"
}
trap finish EXIT

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_multimodal_likelihood_benchmarks.py \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --asset_root "${ASSET_ROOT}" \
  --cache_shard_dir "${CACHE_SHARD_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --tasks "${TASKS}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --mc "${MC}" \
  --lm_head_chunk_tokens "${LM_HEAD_CHUNK_TOKENS}" \
  --max_length 2048 \
  --limit "${LIMIT}" \
  --seed 424242 \
  --device npu \
  --model_dtype fp32 \
  --image_sigma_order auto \
  --progress_every "${PROGRESS_EVERY}" \
  "${PROTOCOL_ARGS[@]}"
