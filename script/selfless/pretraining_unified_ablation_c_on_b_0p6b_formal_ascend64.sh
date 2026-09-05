#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# C-on-B changes only the text path to conventional single-stream next-token
# AR. The image path inherits baseline b's XLNet-style content diagonal and
# separate backbone XT-query/X0-content flow conditions. This experiment
# starts from Qwen3-0.6B-Base at optimizer step zero and must not resume the
# retired C-on-A or legacy shared-condition C-on-B runs.
export ABLATION="c"
export RUN_PROJECT="${RUN_PROJECT:-unified-c-on-b-x0content-0p6b-100b-imagenet-split-s42-r1}"
export RUN_NAME="${RUN_NAME:-unified-c-on-b-x0content-qwen3-0.6b-100b-imagenet-split-s42-r1}"
export RUN_ROOT="${RUN_ROOT:-output/${RUN_PROJECT}}"
export FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"
export ALLOW_FORMAL_RESUME="false"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
