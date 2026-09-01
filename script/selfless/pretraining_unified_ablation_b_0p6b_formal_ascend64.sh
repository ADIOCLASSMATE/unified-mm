#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Controlled comparison with baseline a: only the content-stream attention
# mask gains sigma_kv == sigma_q (the query stream remains strict).
export ABLATION="b"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
