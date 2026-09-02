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

ASSET_ROOT="${ASSET_ROOT:-public/benchmarks/selfless_multimodal_likelihood_v1}"
CACHE_ROOT="${CACHE_ROOT:-${ASSET_ROOT}/vae_posterior_mar_kl16_v2}"
MANIFEST="${MANIFEST:-${ASSET_ROOT}/image_manifest.jsonl}"
SHARD_DIR="${SHARD_DIR:-${CACHE_ROOT}/shards}"
LOG_DIR="${LOG_DIR:-${CACHE_ROOT}/logs}"
STATUS_PATH="${STATUS_PATH:-${CACHE_ROOT}/cache.status}"
COMPLETE_PATH="${CACHE_COMPLETE_PATH:-${CACHE_ROOT}/cache.complete.json}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NPU_COUNT=16

for required in "${ASSET_ROOT}/manifest.json" "${MANIFEST}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing multimodal likelihood asset: ${required}" >&2
    exit 2
  fi
done

python - "${ASSET_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
schema = payload.get("schema")
if schema not in {
    "selfless_multimodal_likelihood_assets_v2",
    "selfless_cross_dataset_retrieval_assets_v1",
}:
    raise RuntimeError(f"image benchmark assets use an unsupported schema: {schema!r}")
if not payload.get("complete", False):
    raise RuntimeError("multimodal likelihood asset manifest is incomplete")
if payload.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError("multimodal likelihood assets violate the no-hash contract")
if schema == "selfless_multimodal_likelihood_assets_v2":
    prior = payload.get("language_prior_null_images", {})
    if (
        prior.get("count") != 3
        or [row.get("image_id") for row in prior.get("images", [])]
        != [9000000000, 9000000001, 9000000002]
    ):
        raise RuntimeError("multimodal likelihood assets have the wrong null-image contract")
PY

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${VISIBLE_NPUS}" >&2
  exit 3
fi

mkdir -p "${SHARD_DIR}" "${LOG_DIR}" "$(dirname "${STATUS_PATH}")"
printf 'state=RUNNING\nstarted_at=%s\nruntime_hashing_enabled=false\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
record_exit() {
  exit_code=$?
  if [[ "${exit_code}" == "0" ]]; then
    state=SUCCEEDED
  else
    state=FAILED
  fi
  printf 'state=%s\nexit_code=%s\nfinished_at=%s\nruntime_hashing_enabled=false\n' \
    "${state}" "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"${STATUS_PATH}"
}
trap record_exit EXIT

pids=()
terminate_children() {
  for child_pid in "${pids[@]:-}"; do
    kill "${child_pid}" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

if [[ "${CACHE_VALIDATE_ONLY:-0}" != "1" ]]; then
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
      --overwrite \
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
    echo "ERROR: at least one cache encoder failed; inspect ${LOG_DIR}" >&2
    exit 4
  fi
fi

python - "${ASSET_ROOT}/manifest.json" "${MANIFEST}" "${SHARD_DIR}" \
  "${COMPLETE_PATH}" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

asset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest_path = Path(sys.argv[2])
shard_dir = Path(sys.argv[3])
complete_path = Path(sys.argv[4])
expected_ids = []
with manifest_path.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            expected_ids.append(int(json.loads(line)["img_id"]))
actual_ids = []
shards = sorted(shard_dir.glob("shard-*-of-*.pt"))
if len(shards) != 16:
    raise RuntimeError(f"expected 16 cache shards, got {len(shards)}")
for path in shards:
    # torch_npu's torch.load wrapper requires a string filename with mmap=True.
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    stats = payload.get("posterior_stats")
    img_ids = payload.get("img_ids")
    metadata = payload.get("metadata", {})
    if not torch.is_tensor(stats) or tuple(stats.shape[1:]) != (256, 32):
        raise RuntimeError(f"invalid posterior shape in {path}: {getattr(stats, 'shape', None)}")
    if stats.dtype != torch.float16:
        raise RuntimeError(f"invalid posterior dtype in {path}: {stats.dtype}")
    if metadata.get("runtime_hashing_enabled", True) is not False:
        raise RuntimeError(f"cache shard violates no-hash contract: {path}")
    if not bool(torch.isfinite(stats).all()) or bool((stats[..., 16:] < 0).any()):
        raise RuntimeError(f"cache shard contains invalid values: {path}")
    actual_ids.extend(int(value) for value in img_ids.tolist())
if sorted(actual_ids) != sorted(expected_ids):
    raise RuntimeError("cache image IDs do not exactly match the asset image manifest")
payload = {
    "schema": "selfless_image_posterior_cache_v2",
    "status": "ok",
    "asset_schema": asset.get("schema"),
    "records": len(actual_ids),
    "shards": len(shards),
    "token_shape": [256, 32],
    "storage_dtype": "float16",
    "source_manifest": {
        "path": str(manifest_path.resolve()),
        "bytes": int(manifest_path.stat().st_size),
        "mtime_ns": int(manifest_path.stat().st_mtime_ns),
    },
    "runtime_hashing_enabled": False,
    "language_prior_null_image_ids": [
        int(row["image_id"])
        for row in asset.get("language_prior_null_images", {}).get("images", [])
    ],
}
complete_path.parent.mkdir(parents=True, exist_ok=True)
temporary = None
try:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=complete_path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, complete_path)
    temporary = None
finally:
    if temporary is not None:
        temporary.unlink(missing_ok=True)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
