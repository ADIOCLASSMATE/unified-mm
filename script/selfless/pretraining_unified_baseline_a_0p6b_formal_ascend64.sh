#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Historical 0.6B sweep winner, frozen as readable constants. The formal run
# always initializes from Qwen3-0.6B-Base at step zero.
export CONFIG="configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
export ACCELERATE_CONFIG="accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml"
export RUN_PROJECT="${RUN_PROJECT:-unified-a-0p6b-100b-imagenet-split-s42-r1}"
export RUN_NAME="${RUN_NAME:-unified-a-qwen3-0.6b-100b-imagenet-split-s42-r1}"
export RUN_ROOT="output/${RUN_PROJECT}"
export OUTPUT_DIR_BASE="output"
export BACKBONE_LR="3.0e-4"
export FLOW_LR="5.0e-5"
export ABLATION="a"
export PRESERVE_MODEL_CONTRACT="false"
export RESUME_FROM="none"
export STOP_AFTER_STEPS="95415"
export SAVE_EVERY="2000"
export CHECKPOINTS_TOTAL_LIMIT="3"
export CHECKPOINT_MILESTONE_EVERY="0"
export VAL_EVERY="2000"
export VALIDATION_IMAGE_EVERY="2000"
export VALIDATION_I2T_EVERY="2000"
export VALIDATION_I2T_SAMPLES="2"
export VALIDATION_I2T_MAX_NEW_TOKENS="64"
export SAVE_EMA_EVAL_EVERY="12510"
export SAVE_FINAL="true"
export SAVE_FINAL_CHECKPOINT="true"
export WANDB_MODE="disabled"

exec bash script/selfless/pretraining_unified_baseline_ascend_64npu_100b.sh
