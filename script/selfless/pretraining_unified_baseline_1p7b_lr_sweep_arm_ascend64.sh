#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ARM_ID="${ARM_ID:-}"
case "${ARM_ID}" in
  b18e5-f4e5) BACKBONE_LR="1.8e-4"; FLOW_LR="4.0e-5" ;;
  b18e5-f5e5) BACKBONE_LR="1.8e-4"; FLOW_LR="5.0e-5" ;;
  b18e5-f6e5) BACKBONE_LR="1.8e-4"; FLOW_LR="6.0e-5" ;;
  b21e5-f4e5) BACKBONE_LR="2.1e-4"; FLOW_LR="4.0e-5" ;;
  b21e5-f5e5) BACKBONE_LR="2.1e-4"; FLOW_LR="5.0e-5" ;;
  b21e5-f6e5) BACKBONE_LR="2.1e-4"; FLOW_LR="6.0e-5" ;;
  b24e5-f4e5) BACKBONE_LR="2.4e-4"; FLOW_LR="4.0e-5" ;;
  b24e5-f5e5) BACKBONE_LR="2.4e-4"; FLOW_LR="5.0e-5" ;;
  b24e5-f6e5) BACKBONE_LR="2.4e-4"; FLOW_LR="6.0e-5" ;;
  *)
    echo "ERROR: ARM_ID must name one frozen 1.7B sweep arm, got ${ARM_ID:-<unset>}" >&2
    exit 2
    ;;
esac

export CONFIG="configs/selfless/unified_baseline_1p7b_100b_ascend_64npu.yaml"
export ACCELERATE_CONFIG="accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml"
export RUN_PROJECT="unified-b-1p7b-lr-sweep-1b-s42/${ARM_ID}"
export RUN_NAME="unified-b-1p7b-${ARM_ID}-1b-s42"
export RUN_ROOT="output/${RUN_PROJECT}"
export BACKBONE_LR FLOW_LR
export ABLATION="b"
export PRESERVE_MODEL_CONTRACT="false"
export RESUME_FROM="none"
export STOP_AFTER_STEPS="955"
export SAVE_EVERY="955"
export CHECKPOINTS_TOTAL_LIMIT="3"
export CHECKPOINT_MILESTONE_EVERY="0"
export VAL_EVERY="955"
export VALIDATION_IMAGE_EVERY="1000000000"
export VALIDATION_I2T_EVERY="1000000000"
export SAVE_EMA_EVAL_EVERY="0"
export SAVE_FINAL="false"
export SAVE_FINAL_CHECKPOINT="true"
export WANDB_MODE="disabled"

exec bash script/selfless/pretraining_unified_baseline_1p7b_ascend_64npu_100b.sh
