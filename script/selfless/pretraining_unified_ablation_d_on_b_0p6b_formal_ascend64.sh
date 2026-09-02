#!/usr/bin/env bash
set -euo pipefail

# Ablation D is a fresh B-based run: the only configured model switch is the
# dedicated Dynamic-XT architecture.  In particular, flow batch multiplication
# remains four and the retired A-based Dynamic-XT checkpoint is never resumed.
export ABLATION="d"
export RUN_PROJECT="unified-d-on-b-0p6b-100b-imagenet-split-s42-r1"
export RUN_NAME="unified-d-on-b-qwen3-0.6b-100b-imagenet-split-s42-r1"
export ALLOW_FORMAL_RESUME="false"
export IMAGE_FLOW_BATCH_MUL="4"
export DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="true"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
