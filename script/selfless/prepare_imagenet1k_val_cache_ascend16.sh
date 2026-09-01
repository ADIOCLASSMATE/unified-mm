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
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MANIFEST="${MANIFEST:-public/datasets/imagenet_full/manifest_val.jsonl}"
SHARD_DIR="${SHARD_DIR:-public/datasets/imagenet_full/vae_posterior_mar_kl16/val_shards}"
CACHE_PATH="${CACHE_PATH:-public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_val_fp16.pt}"
LOG_DIR="${LOG_DIR:-public/datasets/imagenet_full/vae_posterior_mar_kl16/val_logs}"
STATUS_PATH="${STATUS_PATH:-public/datasets/imagenet_full/vae_posterior_mar_kl16/val_preparation.status}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NPU_COUNT=16

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: missing official ImageNet val manifest: ${MANIFEST}" >&2
  exit 2
fi
if [[ -e "${CACHE_PATH}" ]]; then
  echo "ERROR: refusing to overwrite existing val cache: ${CACHE_PATH}" >&2
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

echo "Launching ${NPU_COUNT} no-hash ImageNet-1K val KL16 cache shards."
for local_rank in $(seq 0 $((NPU_COUNT - 1))); do
  log_path="${LOG_DIR}/encode-shard-$(printf '%05d' "${local_rank}")-of-00016.log"
  python scripts/imagenet_encode_kl16_vae.py \
    --source_mode manifest_jsonl \
    --source_manifest_jsonl "${MANIFEST}" \
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
    --no_hash \
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
  echo "ERROR: at least one val encoder failed; inspect ${LOG_DIR}" >&2
  exit 5
fi

python pretrain/merge_flow_latent_shards.py \
  --shard_dir "${SHARD_DIR}" \
  --output_path "${CACHE_PATH}" \
  --manifest_jsonl "${MANIFEST}" \
  --mmap \
  --no_hash

python - "${CACHE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

import torch

cache_path = Path(sys.argv[1])
payload = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=True)
stats = payload.get("posterior_stats")
img_ids = payload.get("img_ids")
metadata = payload.get("metadata", {})
if not torch.is_tensor(stats) or tuple(stats.shape) != (50_000, 256, 32):
    raise RuntimeError(f"invalid val posterior shape: {getattr(stats, 'shape', None)}")
if stats.dtype != torch.float16:
    raise RuntimeError(f"invalid val posterior dtype: {stats.dtype}")
if not torch.equal(img_ids, torch.arange(1, 50_001, dtype=torch.int64)):
    raise RuntimeError("val cache ids must be exactly 1..50000")
if metadata.get("runtime_hashing_enabled") is not False:
    raise RuntimeError("val cache was not prepared under the no-hash contract")
for start in range(0, 50_000, 512):
    chunk = stats[start : start + 512]
    if not bool(torch.isfinite(chunk).all()) or bool((chunk[..., 16:] < 0).any()):
        raise RuntimeError(f"invalid val posterior values at row {start}")
completion_path = cache_path.with_suffix(cache_path.suffix + ".complete.json")
completion_path.write_text(
    json.dumps(
        {
            "status": "ok",
            "split": "val",
            "records": 50_000,
            "shape": list(stats.shape),
            "dtype": str(stats.dtype),
            "runtime_hashing_enabled": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(f"PASS no-hash ImageNet-1K val cache: {cache_path}")
PY
