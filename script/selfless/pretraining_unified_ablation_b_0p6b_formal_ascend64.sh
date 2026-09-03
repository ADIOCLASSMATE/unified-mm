#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Baseline b is the selected main method: content-stream attention includes
# sigma_kv == sigma_q while the query stream remains strict. Flow query AdaLN
# uses XT hidden and flow content AdaLN uses X0 hidden.
export ABLATION="b"
export RUN_PROJECT="unified-b-x0content-0p6b-100b-imagenet-split-s42-r1"
export RUN_NAME="unified-b-x0content-qwen3-0.6b-100b-imagenet-split-s42-r1"
export ALLOW_FORMAL_RESUME="false"
export FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
