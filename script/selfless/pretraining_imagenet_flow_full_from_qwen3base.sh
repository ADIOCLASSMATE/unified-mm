#!/usr/bin/env bash
set -euo pipefail

# The finalized base is caption-conditioned with pure-2D flow-head RoPE.
# Keep the formerly used entrypoint as a thin redirect so the remembered
# command cannot accidentally launch the retired class-conditioned recipe.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${REPO_ROOT}/script/selfless/pretraining_imagenet100_caption_base_80ep.sh" "$@"
