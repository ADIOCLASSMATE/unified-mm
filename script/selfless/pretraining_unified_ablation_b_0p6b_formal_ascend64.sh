#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Baseline b is the selected main method: content-stream attention includes
# sigma_kv == sigma_q while the query stream remains strict.
export ABLATION="b"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
