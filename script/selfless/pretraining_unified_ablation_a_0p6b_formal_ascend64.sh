#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# A keeps strict sigma attention in both backbone and contextual flow head.
# Its new content AdaLN condition comes from backbone X0 while query AdaLN
# continues to use the strict XT/query hidden state.
export ABLATION="a"
export RUN_PROJECT="unified-a-x0content-0p6b-100b-imagenet-split-s42-r1"
export RUN_NAME="unified-a-x0content-qwen3-0.6b-100b-imagenet-split-s42-r1"
export ALLOW_FORMAL_RESUME="false"
export FLOW_HEAD_ATTENTION_CONTRACT="selfless_strict"
export FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
