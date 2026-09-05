#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ABLATION="${ABLATION:-}"
if [[ ! "${ABLATION}" =~ ^[abcdef]$ ]]; then
  echo "ERROR: ABLATION must be a, b, c, d, e, or f" >&2
  exit 2
fi

# Freeze every formal-training field to the baseline-b contract. E changes
# only the order policy; F selects its own isolated model/generation files.
# Every arm keeps image_flow_batch_mul=4 and starts from
# Qwen3-0.6B-Base at optimizer step zero. A failed run may only resume a
# checkpoint produced by the same B-based run identity.
export CONFIG="configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
export ACCELERATE_CONFIG="accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml"
export ABLATION
case "${ABLATION}" in
  a) ARM_NAME="a-x0content"; RUN_REVISION="r1" ;;
  b) ARM_NAME="b-x0content"; RUN_REVISION="r1" ;;
  c) ARM_NAME="c-on-b"; RUN_REVISION="r1" ;;
  d) ARM_NAME="d-on-b"; RUN_REVISION="r4" ;;
  e) ARM_NAME="e-on-b-x0content"; RUN_REVISION="r1" ;;
  f) ARM_NAME="f-on-b"; RUN_REVISION="r1" ;;
esac
DEFAULT_RUN_PROJECT="unified-${ARM_NAME}-0p6b-100b-imagenet-split-s42-${RUN_REVISION}"
DEFAULT_RUN_NAME="unified-${ARM_NAME}-qwen3-0.6b-100b-imagenet-split-s42-${RUN_REVISION}"
export RUN_PROJECT="${RUN_PROJECT:-${DEFAULT_RUN_PROJECT}}"
export RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
export RUN_ROOT="${RUN_ROOT:-output/${RUN_PROJECT}}"
export OUTPUT_DIR_BASE="output"
export BACKBONE_LR="3.0e-4"
export FLOW_LR="5.0e-5"
export IMAGE_FLOW_BATCH_MUL="4"
export PRESERVE_MODEL_CONTRACT="false"
export RESUME_FROM="none"
ALLOW_FORMAL_RESUME="${ALLOW_FORMAL_RESUME:-false}"
if [[ "${ALLOW_FORMAL_RESUME}" == "true" ]]; then
  RESUME_FROM="${FORMAL_RESUME_FROM:-}"
  if [[ -z "${RESUME_FROM}" ]]; then
    echo "ERROR: FORMAL_RESUME_FROM is required for formal resume" >&2
    exit 3
  fi
  if [[ "${RESUME_FROM}" != "${RUN_ROOT}"/checkpoint-[0-9]* ]]; then
    echo "ERROR: formal resume checkpoint must belong to ${RUN_ROOT}" >&2
    exit 3
  fi
  if [[ ! -f "${RESUME_FROM}/checkpoint_complete.json" || ! -f "${RESUME_FROM}/metadata.json" ]]; then
    echo "ERROR: formal resume checkpoint is incomplete: ${RESUME_FROM}" >&2
    exit 3
  fi
elif [[ "${ALLOW_FORMAL_RESUME}" != "false" ]]; then
  echo "ERROR: ALLOW_FORMAL_RESUME must be true or false" >&2
  exit 3
fi
export RESUME_FROM
export STOP_AFTER_STEPS="95415"
export SAVE_EVERY="2000"
export CHECKPOINTS_TOTAL_LIMIT="3"
export CHECKPOINT_MILESTONE_EVERY="125100"
export VAL_EVERY="2000"
export VALIDATION_IMAGE_EVERY="2000"
export VALIDATION_I2T_EVERY="2000"
export VALIDATION_I2T_SAMPLES="2"
export VALIDATION_I2T_MAX_NEW_TOKENS="64"
# Project default: permanent full raw/EMA exports every 20 image epochs.
export SAVE_EMA_EVAL_EVERY="25020"
export SAVE_FINAL="true"
export SAVE_FINAL_CHECKPOINT="true"
export WANDB_MODE="disabled"

exec bash script/selfless/pretraining_unified_baseline_ascend_64npu_100b.sh
