#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-public/benchmarks/selfless_multimodal_likelihood_v1}"
CACHE_ROOT="${CACHE_ROOT:-${ASSET_ROOT}/vae_posterior_mar_kl16_v2}"
COMPLETE_PATH="${CACHE_COMPLETE_PATH:-${CACHE_ROOT}/cache.complete.json}"
STATUS_PATH="${CACHE_STATUS_PATH:-${CACHE_ROOT}/cache.status}"
WAIT_TIMEOUT_SECONDS="${CACHE_WAIT_TIMEOUT_SECONDS:-21600}"
WAIT_INTERVAL_SECONDS="${CACHE_WAIT_INTERVAL_SECONDS:-15}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

for value in "${WAIT_TIMEOUT_SECONDS}" "${WAIT_INTERVAL_SECONDS}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
    echo "ERROR: cache wait values must be positive integers" >&2
    exit 2
  fi
done

cd "${REPO_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: missing executable Python interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi
started="$(date +%s)"
last_reported=0
while true; do
  if [[ -f "${COMPLETE_PATH}" ]] && "${PYTHON_BIN}" - \
    "${ASSET_ROOT}/manifest.json" "${ASSET_ROOT}/image_manifest.jsonl" \
    "${COMPLETE_PATH}" <<'PY'
import json
from pathlib import Path
import sys

asset = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = Path(sys.argv[2]).resolve()
cache = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
stat = manifest.stat()
expected_source = {
    "path": str(manifest),
    "bytes": int(stat.st_size),
    "mtime_ns": int(stat.st_mtime_ns),
}
if asset.get("schema") != "selfless_multimodal_likelihood_assets_v2":
    raise SystemExit(1)
if cache.get("schema") != "selfless_image_posterior_cache_v2":
    raise SystemExit(1)
if cache.get("asset_schema") != asset.get("schema"):
    raise SystemExit(1)
if cache.get("runtime_hashing_enabled", True) is not False:
    raise SystemExit(1)
if int(cache.get("records", -1)) != int(asset.get("images", -2)):
    raise SystemExit(1)
if cache.get("source_manifest") != expected_source:
    raise SystemExit(1)
if cache.get("language_prior_null_image_ids") != [9000000000, 9000000001, 9000000002]:
    raise SystemExit(1)
PY
  then
    echo "multimodal likelihood cache is ready: ${COMPLETE_PATH}"
    exit 0
  fi
  if [[ -f "${STATUS_PATH}" ]] && grep -q '^state=FAILED$' "${STATUS_PATH}"; then
    echo "ERROR: multimodal likelihood cache builder reported failure" >&2
    exit 3
  fi
  now="$(date +%s)"
  elapsed=$((now - started))
  if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
    echo "ERROR: timed out waiting for multimodal likelihood cache" >&2
    exit 4
  fi
  if (( elapsed - last_reported >= 300 || last_reported == 0 )); then
    echo "waiting for multimodal likelihood cache: elapsed=${elapsed}s"
    last_reported="${elapsed}"
  fi
  sleep "${WAIT_INTERVAL_SECONDS}"
done
