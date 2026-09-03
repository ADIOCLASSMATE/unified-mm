#!/usr/bin/env bash
set -euo pipefail

# E keeps baseline B's model and optimizer exactly, replacing random image
# sigma/reveal order with deterministic serialized left-to-right order in both
# training and generation. Text is already serialized left-to-right.
export ABLATION="e"
export RUN_PROJECT="unified-e-on-b-x0content-0p6b-100b-imagenet-split-s42-r1"
export RUN_NAME="unified-e-on-b-x0content-qwen3-0.6b-100b-imagenet-split-s42-r1"
export ALLOW_FORMAL_RESUME="false"
export IMAGE_FLOW_BATCH_MUL="4"
export FLOW_HEAD_ATTENTION_CONTRACT="xlnet_content_diagonal"
export FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
