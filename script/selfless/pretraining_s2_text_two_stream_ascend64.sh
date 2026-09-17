#!/usr/bin/env bash
set -euo pipefail
S2_TEXT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--smoke-suite" ]]; then
  shift
  exec bash "${S2_TEXT_REPO_ROOT}/script/selfless/pretraining_showo2_unified_ascend64.sh" \
    --text-two-stream-smoke-suite "$@"
fi
exec bash "${S2_TEXT_REPO_ROOT}/script/selfless/pretraining_short_ablation_ascend64.sh" \
  --arm s2-text2stream "$@"
