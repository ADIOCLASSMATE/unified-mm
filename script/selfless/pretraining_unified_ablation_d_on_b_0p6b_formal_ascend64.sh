#!/usr/bin/env bash
set -euo pipefail

# Ablation D is a fresh B-based run: the only configured model switch is the
# dedicated Dynamic-XT architecture.  In particular, flow batch multiplication
# remains four and the retired A-based Dynamic-XT checkpoint is never resumed.
export ABLATION="d"
export RUN_PROJECT="unified-d-on-b-0p6b-100b-imagenet-split-s42-r4"
export RUN_NAME="unified-d-on-b-qwen3-0.6b-100b-imagenet-split-s42-r4"
export ALLOW_FORMAL_RESUME="false"
export IMAGE_FLOW_BATCH_MUL="4"
export FLOW_HEAD_ATTENTION_CONTRACT="xlnet_content_diagonal"
export DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="true"
# The 64-NPU startup validation showed that one DeepSpeed BF16 overflow scan
# at the first optimizer boundary is sufficient. The trainer disables the
# scan immediately afterward, so this does not tax steady-state training.
export DEEPSPEED_BF16_OVERFLOW_CHECK_UNTIL_STEP="1"
# Cache loss scalars on device and inspect them once before each of the first
# ten optimizer-boundary backward calls. This diagnoses any repeat of the r1
# failure without synchronizing individual microbatches or taxing steady state.
export DEBUG_LOSS_TRACE_UNTIL_STEP="10"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
