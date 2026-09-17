#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--smoke-suite" ]]; then
  shift
  exec bash "${REPO_ROOT}/script/selfless/pretraining_z_ascend64.sh" --smoke-suite --experiment z-b "$@"
fi
exec bash "${REPO_ROOT}/script/selfless/pretraining_short_ablation_ascend64.sh" --arm z-b "$@"
