#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ABLATION="${ABLATION:-}"
if [[ ! "${ABLATION}" =~ ^[abc]$ ]]; then
  echo "ERROR: ABLATION must be one of a, b, c" >&2
  exit 2
fi

# Freeze every formal-training field to the current baseline-a contract.  For
# ablation b changes only the content attention contract; ablation c changes
# only architecture_variant to the isolated single-stream text-AR model.  All
# arms start from Qwen3-0.6B-Base at optimizer step zero; no sweep checkpoint
# is resumed.  A failed formal run may opt into its own validated checkpoint
# through the dedicated ALLOW_FORMAL_RESUME contract below.
export CONFIG="configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
export ACCELERATE_CONFIG="accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml"
export ABLATION
export RUN_PROJECT="${RUN_PROJECT:-unified-${ABLATION}-0p6b-100b-imagenet-split-s42-r1}"
export RUN_NAME="${RUN_NAME:-unified-${ABLATION}-qwen3-0.6b-100b-imagenet-split-s42-r1}"
export RUN_ROOT="${RUN_ROOT:-output/${RUN_PROJECT}}"
export OUTPUT_DIR_BASE="output"
export BACKBONE_LR="3.0e-4"
export FLOW_LR="5.0e-5"
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
export CHECKPOINT_MILESTONE_EVERY="0"
export VAL_EVERY="2000"
export VALIDATION_IMAGE_EVERY="2000"
export VALIDATION_I2T_EVERY="2000"
export VALIDATION_I2T_SAMPLES="2"
export VALIDATION_I2T_MAX_NEW_TOKENS="64"
# Match the immutable a/b launches: periodic paired evaluation exports were
# disabled; normal checkpoints, final current/EMA export and evaluation stay
# unchanged.
export SAVE_EMA_EVAL_EVERY="0"
export SAVE_FINAL="true"
export SAVE_FINAL_CHECKPOINT="true"
export WANDB_MODE="disabled"

exec bash script/selfless/pretraining_unified_baseline_ascend_64npu_100b.sh
