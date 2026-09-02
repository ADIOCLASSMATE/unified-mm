#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <multimodal-likelihood-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-public/benchmarks/selfless_multimodal_likelihood_v1}"
CACHE_ROOT="${CACHE_ROOT:-${ASSET_ROOT}/vae_posterior_mar_kl16_v2}"
COMPLETE_PATH="${CACHE_COMPLETE_PATH:-${CACHE_ROOT}/cache.complete.json}"

cache_ready=0
if [[ -f "${COMPLETE_PATH}" ]]; then
  if python - "${ASSET_ROOT}/manifest.json" "${ASSET_ROOT}/image_manifest.jsonl" \
    "${COMPLETE_PATH}" <<'PY'
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
if (
    asset.get("schema") != "selfless_multimodal_likelihood_assets_v2"
    or cache.get("schema") != "selfless_image_posterior_cache_v2"
    or cache.get("asset_schema") != asset.get("schema")
    or cache.get("runtime_hashing_enabled", True) is not False
    or int(cache.get("records", -1)) != int(asset.get("images", -2))
    or cache.get("source_manifest") != expected_source
    or cache.get("language_prior_null_image_ids")
    != [9000000000, 9000000001, 9000000002]
):
    raise SystemExit(1)
PY
  then
    cache_ready=1
  fi
fi

if [[ "${cache_ready}" != "1" ]]; then
  CACHE_ROOT="${CACHE_ROOT}" CACHE_COMPLETE_PATH="${COMPLETE_PATH}" \
    "${REPO_ROOT}/script/selfless/prepare_multimodal_likelihood_cache_ascend16.sh"
fi

CACHE_ROOT="${CACHE_ROOT}" CACHE_COMPLETE_PATH="${COMPLETE_PATH}" \
  "${REPO_ROOT}/script/selfless/evaluate_multimodal_likelihood_ascend16.sh" "$1" "$2"
