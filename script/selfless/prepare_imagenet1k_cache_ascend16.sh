#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

IMAGE_ROOT="${IMAGE_ROOT:-/inspire/sj-ssd3/project/high-dimensionaldata/public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train}"
MANIFEST="${MANIFEST:-public/datasets/imagenet_full/manifest.jsonl}"
SHARD_DIR="${SHARD_DIR:-public/datasets/imagenet_full/vae_posterior_mar_kl16/shards}"
CACHE_PATH="${CACHE_PATH:-public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_train_fp16.pt}"
LOG_DIR="${LOG_DIR:-public/datasets/imagenet_full/vae_posterior_mar_kl16/logs}"
STATUS_PATH="${STATUS_PATH:-public/datasets/imagenet_full/vae_posterior_mar_kl16/preparation.status}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NPU_COUNT=16

if [[ ! -d "${IMAGE_ROOT}" || ! -f "${MANIFEST}" ]]; then
  echo "ERROR: missing ImageNet train root or canonical manifest" >&2
  exit 2
fi
if [[ -e "${CACHE_PATH}" ]]; then
  echo "ERROR: refusing to overwrite existing cache: ${CACHE_PATH}" >&2
  exit 3
fi

mkdir -p "${SHARD_DIR}" "${LOG_DIR}" "$(dirname "${CACHE_PATH}")"
printf 'RUNNING\n' >"${STATUS_PATH}"
record_exit() {
  exit_code=$?
  if [[ "${exit_code}" == "0" ]]; then
    printf 'SUCCEEDED\n' >"${STATUS_PATH}"
  else
    printf 'FAILED exit_code=%s\n' "${exit_code}" >"${STATUS_PATH}"
  fi
}
trap record_exit EXIT

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${VISIBLE_NPUS}" >&2
  exit 4
fi

pids=()
terminate_children() {
  for child_pid in "${pids[@]:-}"; do
    kill "${child_pid}" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

echo "Launching ${NPU_COUNT} ImageNet-1K KL16 cache shards."
for local_rank in $(seq 0 $((NPU_COUNT - 1))); do
  log_path="${LOG_DIR}/encode-shard-$(printf '%05d' "${local_rank}")-of-00016.log"
  python scripts/imagenet_encode_kl16_vae.py \
    --source_mode manifest_jsonl \
    --source_manifest_jsonl "${MANIFEST}" \
    --source_image_root "${IMAGE_ROOT}" \
    --vae_path public/vae/mar-kl16/kl16.ckpt \
    --vae_module_root public/code/mar \
    --cache_shard_dir "${SHARD_DIR}" \
    --device "npu:${local_rank}" \
    --vae_dtype fp16 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor 2 \
    --num_shards "${NPU_COUNT}" \
    --shard_index "${local_rank}" \
    >"${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for child_pid in "${pids[@]}"; do
  if ! wait "${child_pid}"; then
    failed=1
  fi
done
trap - INT TERM
if [[ "${failed}" != "0" ]]; then
  echo "ERROR: at least one encoder failed; inspect ${LOG_DIR}" >&2
  exit 5
fi

python pretrain/merge_flow_latent_shards.py \
  --shard_dir "${SHARD_DIR}" \
  --output_path "${CACHE_PATH}" \
  --manifest_jsonl "${MANIFEST}" \
  --mmap

python - "${CACHE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

from scripts.validate_ascend_imagenet1k_pretraining import validate_cache

cache_path = Path(sys.argv[1])
report = validate_cache(cache_path, deep_scan=True)
completion_path = cache_path.with_suffix(cache_path.suffix + ".complete.json")
completion_path.write_text(
    json.dumps({"status": "ok", "cache": str(cache_path), **report}, indent=2)
    + "\n",
    encoding="utf-8",
)
print(f"PASS ImageNet-1K cache: {cache_path}")
PY
