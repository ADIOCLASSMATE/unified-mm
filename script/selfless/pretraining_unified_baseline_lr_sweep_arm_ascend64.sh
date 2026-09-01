#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ARM_ID="${ARM_ID:-}"
case "${ARM_ID}" in
  b20e5-f3e5) BACKBONE_LR="2.0e-4"; FLOW_LR="3.0e-5" ;;
  b20e5-f4e5) BACKBONE_LR="2.0e-4"; FLOW_LR="4.0e-5" ;;
  b20e5-f5e5) BACKBONE_LR="2.0e-4"; FLOW_LR="5.0e-5" ;;
  b24e5-f3e5) BACKBONE_LR="2.4e-4"; FLOW_LR="3.0e-5" ;;
  b24e5-f4e5) BACKBONE_LR="2.4e-4"; FLOW_LR="4.0e-5" ;;
  b24e5-f5e5) BACKBONE_LR="2.4e-4"; FLOW_LR="5.0e-5" ;;
  b30e5-f3e5) BACKBONE_LR="3.0e-4"; FLOW_LR="3.0e-5" ;;
  b30e5-f4e5) BACKBONE_LR="3.0e-4"; FLOW_LR="4.0e-5" ;;
  b30e5-f5e5) BACKBONE_LR="3.0e-4"; FLOW_LR="5.0e-5" ;;
  *)
    echo "ERROR: ARM_ID must name one frozen sweep arm, got ${ARM_ID:-<unset>}" >&2
    exit 2
    ;;
esac

export CONFIG="configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
export RUN_PROJECT="unified-a-lr-sweep-1b-s42/${ARM_ID}"
export RUN_NAME="unified-a-${ARM_ID}-1b-s42"
export RUN_ROOT="output/${RUN_PROJECT}"
export BACKBONE_LR FLOW_LR
export ABLATION="a"
export RESUME_FROM="none"
export STOP_AFTER_STEPS="955"
export SAVE_EVERY="955"
export VAL_EVERY="955"
export VALIDATION_IMAGE_EVERY="1000000000"
export SAVE_EMA_EVAL_EVERY="0"
export SAVE_FINAL="false"
export WANDB_MODE="disabled"

exec bash script/selfless/pretraining_unified_baseline_ascend_64npu_100b.sh
