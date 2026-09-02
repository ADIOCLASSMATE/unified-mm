#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 1
fi
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/16_npus_1node_deepspeed_zero2.yaml}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"
SAVE_EVERY="${SAVE_EVERY:-${SMOKE_STEPS}}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-3}"
CHECKPOINT_MILESTONE_EVERY="${CHECKPOINT_MILESTONE_EVERY:-0}"
VAL_EVERY="${VAL_EVERY:-1000000000}"
VALIDATION_IMAGE_EVERY="${VALIDATION_IMAGE_EVERY:-1000000000}"
VALIDATION_I2T_EVERY="${VALIDATION_I2T_EVERY:-${VALIDATION_IMAGE_EVERY}}"
VALIDATION_I2T_SAMPLES="${VALIDATION_I2T_SAMPLES:-2}"
VALIDATION_I2T_MAX_NEW_TOKENS="${VALIDATION_I2T_MAX_NEW_TOKENS:-32}"
VALIDATION_MAX_BATCHES="${VALIDATION_MAX_BATCHES:-1}"
VALIDATION_IMAGE_SAMPLES="${VALIDATION_IMAGE_SAMPLES:-2}"
VALIDATION_SINGLE_STREAM_PARALLEL_RATE="${VALIDATION_SINGLE_STREAM_PARALLEL_RATE:-1}"
RESUME_FROM="${RESUME_FROM:-none}"
SAVE_FINAL="${SAVE_FINAL:-false}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"
ABLATION="${ABLATION:-b}"
BACKBONE_LR="${BACKBONE_LR:-3.0e-4}"
FLOW_LR="${FLOW_LR:-5.0e-5}"
IMAGE_FLOW_BATCH_MUL="${IMAGE_FLOW_BATCH_MUL:-4}"
FORMAL_WORLD_SIZE="${FORMAL_WORLD_SIZE:-64}"
NPROC_PER_NODE=16

case "${ABLATION}" in
  b)
    ARCHITECTURE_VARIANT="selfless_contextual"
    TRAINING_OBJECTIVE="selfless_dual_stream"
    DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_RUN_PROJECT="unified-b-qwen3-0.6b-smoke-ascend16"
    TRAIN_ENTRY="pretrain/train_selfless_flow.py"
    DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="false"
    ;;
  c)
    ARCHITECTURE_VARIANT="single_stream_text_ar"
    TRAINING_OBJECTIVE="selfless_dual_stream"
    DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_RUN_PROJECT="unified-c-on-b-qwen3-0.6b-smoke-ascend16"
    TRAIN_ENTRY="pretrain/train_selfless_flow.py"
    DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="false"
    ;;
  d)
    ARCHITECTURE_VARIANT="dynamic_xt"
    TRAINING_OBJECTIVE="selfless_dual_stream"
    DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_RUN_PROJECT="unified-d-on-b-qwen3-0.6b-smoke-ascend16"
    TRAIN_ENTRY="pretrain/train_selfless_flow_dynamic_xt.py"
    # D keeps four RF query states for T2I. Checkpoint only those dynamic
    # decoder-layer activations so the formal B16 image microbatch fits 910B.
    DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="true"
    ;;
  *)
    echo "ERROR: ABLATION must be b, c, or d; got ${ABLATION}" >&2
    exit 2
    ;;
esac

DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING:-${DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}}"

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
# The parent process tokenizes during model/dataset construction.  Keep its
# Rayon pool disabled before DataLoader forks; the ClimbMix worker enables its
# own fresh two-thread pool after the fork.
export TOKENIZERS_PARALLELISM=false
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

if [[ ! "${SMOKE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SMOKE_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ "${IMAGE_FLOW_BATCH_MUL}" != "4" ]]; then
  echo "ERROR: B-based unified ablations require IMAGE_FLOW_BATCH_MUL=4" >&2
  exit 2
fi
if [[ "${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}" != "true" && "${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}" != "false" ]]; then
  echo "ERROR: DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING must be true or false" >&2
  exit 2
fi
if [[ "${ABLATION}" == "d" && "${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}" != "true" ]]; then
  echo "ERROR: ablation D requires T2I activation checkpointing at formal B16" >&2
  exit 2
fi
if [[ "${FORMAL_WORLD_SIZE}" != "64" && "${FORMAL_WORLD_SIZE}" != "128" ]]; then
  echo "ERROR: FORMAL_WORLD_SIZE must be 64 or 128" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" || ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: missing CONFIG=${CONFIG} or ACCELERATE_CONFIG=${ACCELERATE_CONFIG}" >&2
  exit 3
fi

read -r NPU_AVAILABLE LOCAL_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${LOCAL_NPUS}" != "${NPROC_PER_NODE}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${LOCAL_NPUS}" >&2
  exit 4
fi

RUN_PROJECT="${RUN_PROJECT:-${DEFAULT_RUN_PROJECT}}"
RUN_ROOT="${RUN_ROOT:-output/${RUN_PROJECT}}"
AUDIT_DIR="${AUDIT_DIR:-${RUN_ROOT}/prelaunch_audit}"
mkdir -p "${AUDIT_DIR}"

python scripts/validate_unified_baseline.py \
  --config "${CONFIG}" \
  --formal-world-size "${FORMAL_WORLD_SIZE}" \
  --require-npu-count "${NPROC_PER_NODE}" \
  --tokenizer-probe \
  --run-project "${RUN_PROJECT}" \
  --backbone-lr "${BACKBONE_LR}" \
  --flow-lr "${FLOW_LR}" \
  --save-ema-eval-every 0 \
  --ablation "${ABLATION}" \
  --dynamic-xt-t2i-gradient-checkpointing "${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}" \
  >"${AUDIT_DIR}/asset_preflight.json"

COMMAND=(
  accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_machines 1
  --num_processes "${NPROC_PER_NODE}"
  "${TRAIN_ENTRY}"
  "config=${CONFIG}"
  "experiment.project=${RUN_PROJECT}"
  "experiment.name=${RUN_PROJECT}-steps${SMOKE_STEPS}"
  "experiment.resume_from_checkpoint=${RESUME_FROM}"
  "experiment.save_every=${SAVE_EVERY}"
  "experiment.checkpoints_total_limit=${CHECKPOINTS_TOTAL_LIMIT}"
  "experiment.checkpoint_milestone_every=${CHECKPOINT_MILESTONE_EVERY}"
  "experiment.log_every=1"
  "experiment.val_every=${VAL_EVERY}"
  "experiment.validation_image_every=${VALIDATION_IMAGE_EVERY}"
  "experiment.validation_i2t_every=${VALIDATION_I2T_EVERY}"
  "experiment.validation_i2t_samples=${VALIDATION_I2T_SAMPLES}"
  "experiment.validation_i2t_max_new_tokens=${VALIDATION_I2T_MAX_NEW_TOKENS}"
  "experiment.validation_max_batches=${VALIDATION_MAX_BATCHES}"
  "experiment.validation_image_samples=${VALIDATION_IMAGE_SAMPLES}"
  "experiment.validation_single_stream_parallel_rate=${VALIDATION_SINGLE_STREAM_PARALLEL_RATE}"
  "experiment.save_ema_eval_every=0"
  "experiment.save_final=${SAVE_FINAL}"
  "experiment.save_final_checkpoint=${SAVE_FINAL_CHECKPOINT}"
  "optimizer.params.learning_rate=${BACKBONE_LR}"
  "optimizer.params.backbone_learning_rate=${BACKBONE_LR}"
  "optimizer.params.special_token_learning_rate=${BACKBONE_LR}"
  "optimizer.params.projector_learning_rate=${FLOW_LR}"
  "optimizer.params.flow_learning_rate=${FLOW_LR}"
  "model.architecture_variant=${ARCHITECTURE_VARIANT}"
  "model.training_objective=${TRAINING_OBJECTIVE}"
  "model.dual_stream_attention_contract=${DUAL_STREAM_ATTENTION_CONTRACT}"
  "model.image_flow_batch_mul=${IMAGE_FLOW_BATCH_MUL}"
  "training.use_gradient_checkpointing=false"
  "model.dynamic_xt_t2i_gradient_checkpointing=${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}"
  "model.showo_mask_schedule=cosine"
  "model.showo_min_masking_rate=0.0"
  "training.stop_after_steps=${SMOKE_STEPS}"
)
printf '%q ' "${COMMAND[@]}" >"${AUDIT_DIR}/launch_command.sh"
printf '\n' >>"${AUDIT_DIR}/launch_command.sh"

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  WANDB_MODE="${WANDB_MODE}" \
  "${COMMAND[@]}" 2>&1 | tee "${AUDIT_DIR}/training.log"
