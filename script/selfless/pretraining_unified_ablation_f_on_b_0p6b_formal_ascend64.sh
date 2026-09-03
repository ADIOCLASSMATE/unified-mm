#!/usr/bin/env bash
set -euo pipefail

# F is isolated from every other model: it keeps B's backbone, dual-stream
# mask, random order and mul=4 estimator, but loads the dedicated parameter-
# matched position-wise flow model and generation implementation.
export ABLATION="f"
export RUN_PROJECT="unified-f-on-b-0p6b-100b-imagenet-split-s42-r1"
export RUN_NAME="unified-f-on-b-qwen3-0.6b-100b-imagenet-split-s42-r1"
export ALLOW_FORMAL_RESUME="false"
export IMAGE_FLOW_BATCH_MUL="4"
export FLOW_HEAD_ATTENTION_CONTRACT="not_applicable"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
