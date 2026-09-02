#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ABLATION="${ABLATION:-}"
if [[ ! "${ABLATION}" =~ ^[bcd]$ ]]; then
  echo "ERROR: ABLATION must be b, c, or d" >&2
  exit 2
fi

# Freeze every formal-training field to the baseline-b contract. Ablation c
# selects the isolated single-stream text-AR model. Ablation d selects the
# isolated Dynamic-XT model and keeps B's image_flow_batch_mul=4. Every fresh
# arm starts from Qwen3-0.6B-Base at optimizer step zero. A failed run may only
# resume a checkpoint produced by the same B-based run identity.
export CONFIG="configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
export ACCELERATE_CONFIG="accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml"
export ABLATION
if [[ "${ABLATION}" == "c" ]]; then
  DEFAULT_RUN_PROJECT="unified-c-on-b-0p6b-100b-imagenet-split-s42-r1"
  DEFAULT_RUN_NAME="unified-c-on-b-qwen3-0.6b-100b-imagenet-split-s42-r1"
elif [[ "${ABLATION}" == "d" ]]; then
  DEFAULT_RUN_PROJECT="unified-d-on-b-0p6b-100b-imagenet-split-s42-r1"
  DEFAULT_RUN_NAME="unified-d-on-b-qwen3-0.6b-100b-imagenet-split-s42-r1"
else
  DEFAULT_RUN_PROJECT="unified-b-0p6b-100b-imagenet-split-s42-r1"
  DEFAULT_RUN_NAME="unified-b-qwen3-0.6b-100b-imagenet-split-s42-r1"
fi
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
export CHECKPOINT_MILESTONE_EVERY="0"
export VAL_EVERY="2000"
export VALIDATION_IMAGE_EVERY="2000"
export VALIDATION_I2T_EVERY="2000"
export VALIDATION_I2T_SAMPLES="2"
export VALIDATION_I2T_MAX_NEW_TOKENS="64"
# Match the immutable baseline-b launch: periodic paired evaluation exports were
# disabled; normal checkpoints, final current/EMA export and evaluation stay
# unchanged.
export SAVE_EMA_EVAL_EVERY="0"
export SAVE_FINAL="true"
export SAVE_FINAL_CHECKPOINT="true"
export WANDB_MODE="disabled"

exec bash script/selfless/pretraining_unified_baseline_ascend_64npu_100b.sh
