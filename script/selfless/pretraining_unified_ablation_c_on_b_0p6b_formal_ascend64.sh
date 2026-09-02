#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# C-on-B changes only the text path to conventional single-stream next-token
# AR. The image path inherits baseline b's XLNet-style content diagonal. This
# experiment must start from Qwen3-0.6B-Base at optimizer step zero and must not
# resume the retired C-on-A run.
export ABLATION="c"
export RUN_PROJECT="${RUN_PROJECT:-unified-c-on-b-0p6b-100b-imagenet-split-s42-r1}"
export RUN_NAME="${RUN_NAME:-unified-c-on-b-qwen3-0.6b-100b-imagenet-split-s42-r1}"
export RUN_ROOT="${RUN_ROOT:-output/${RUN_PROJECT}}"
export ALLOW_FORMAL_RESUME="false"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
