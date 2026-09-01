#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 1 ]]; then
  echo "Usage: $0 <cross-dataset-retrieval-asset-root>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RETRIEVAL_ASSET_ROOT="$1"
RETRIEVAL_CACHE_ROOT="${CACHE_ROOT:-${RETRIEVAL_ASSET_ROOT}/vae_posterior_mar_kl16}"

ASSET_ROOT="${RETRIEVAL_ASSET_ROOT}" \
CACHE_ROOT="${RETRIEVAL_CACHE_ROOT}" \
MANIFEST="${RETRIEVAL_ASSET_ROOT}/image_manifest.jsonl" \
  "${REPO_ROOT}/script/selfless/prepare_multimodal_likelihood_cache_ascend16.sh"
