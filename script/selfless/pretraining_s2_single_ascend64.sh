#!/usr/bin/env bash
set -euo pipefail
S2_SINGLE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${S2_SINGLE_REPO_ROOT}/script/selfless/pretraining_short_ablation_ascend64.sh" --arm s2-single "$@"
