#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EVAL_ONLY="t2i"
export CONFIG="configs/selfless/imagenet1k_t2i_baseline_80ep_ascend_64npu_bs1024.yaml"
export RUN_PROJECT="selfless-flow-imagenet1k-t2i-baseline-ascend64-b1024-80ep"
export MODEL_SUBDIR="hf_model-final-ema"
export EVAL_SUBDIR="${EVAL_SUBDIR:-generation-evaluation/heun10}"
exec "${REPO_ROOT}/script/selfless/evaluate_imagenet1k_caption_joint_ascend16.sh" "$@"
