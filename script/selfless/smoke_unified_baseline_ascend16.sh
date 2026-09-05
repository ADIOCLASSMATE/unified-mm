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
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-output}"
DEBUG_LOSS_TRACE_UNTIL_STEP="${DEBUG_LOSS_TRACE_UNTIL_STEP:-0}"
DEEPSPEED_BF16_OVERFLOW_CHECK_UNTIL_STEP="${DEEPSPEED_BF16_OVERFLOW_CHECK_UNTIL_STEP:-0}"
ABLATION="${ABLATION:-b}"
BACKBONE_LR="${BACKBONE_LR:-3.0e-4}"
FLOW_LR="${FLOW_LR:-5.0e-5}"
IMAGE_FLOW_BATCH_MUL="${IMAGE_FLOW_BATCH_MUL:-4}"
FORMAL_WORLD_SIZE="${FORMAL_WORLD_SIZE:-64}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"

case "${ABLATION}" in
  a)
    ARCHITECTURE_VARIANT="selfless_contextual"
    TRAINING_OBJECTIVE="selfless_dual_stream"
    DUAL_STREAM_ATTENTION_CONTRACT="selfless_strict"
    DEFAULT_RUN_PROJECT="unified-a-x0content-qwen3-0.6b-smoke-ascend16"
    TRAIN_ENTRY="pretrain/train_selfless_flow.py"
    DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="false"
    DEFAULT_IMAGE_SIGMA_ORDER="random"
    DEFAULT_VALIDATION_ORDER_STRATEGY="spatial_halton"
    DEFAULT_FLOW_HEAD_WIDTH="1280"
    DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT="selfless_strict"
    DEFAULT_FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"
    ;;
  b)
    ARCHITECTURE_VARIANT="selfless_contextual"
    TRAINING_OBJECTIVE="selfless_dual_stream"
    DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_RUN_PROJECT="unified-b-x0content-qwen3-0.6b-smoke-ascend16"
    TRAIN_ENTRY="pretrain/train_selfless_flow.py"
    DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="false"
    DEFAULT_IMAGE_SIGMA_ORDER="random"
    DEFAULT_VALIDATION_ORDER_STRATEGY="spatial_halton"
    DEFAULT_FLOW_HEAD_WIDTH="1280"
    DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"
    ;;
  c)
    ARCHITECTURE_VARIANT="single_stream_text_ar"
    TRAINING_OBJECTIVE="selfless_dual_stream"
    DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_RUN_PROJECT="unified-c-on-b-x0content-qwen3-0.6b-smoke-ascend16"
    TRAIN_ENTRY="pretrain/train_selfless_flow.py"
    DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="false"
    DEFAULT_IMAGE_SIGMA_ORDER="random"
    DEFAULT_VALIDATION_ORDER_STRATEGY="spatial_halton"
    DEFAULT_FLOW_HEAD_WIDTH="1280"
    DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"
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
    DEFAULT_IMAGE_SIGMA_ORDER="random"
    DEFAULT_VALIDATION_ORDER_STRATEGY="spatial_halton"
    DEFAULT_FLOW_HEAD_WIDTH="1280"
    DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"
    ;;
  e)
    ARCHITECTURE_VARIANT="selfless_contextual"
    TRAINING_OBJECTIVE="selfless_dual_stream"
    DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_RUN_PROJECT="unified-e-on-b-x0content-qwen3-0.6b-smoke-ascend16"
    TRAIN_ENTRY="pretrain/train_selfless_flow.py"
    DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="false"
    DEFAULT_IMAGE_SIGMA_ORDER="sequential"
    DEFAULT_VALIDATION_ORDER_STRATEGY="sequential"
    DEFAULT_FLOW_HEAD_WIDTH="1280"
    DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_FLOW_CONDITION_CONTRACT="backbone_xt_query_backbone_x0_content"
    ;;
  f)
    ARCHITECTURE_VARIANT="positionwise_flow_head_on_b"
    TRAINING_OBJECTIVE="selfless_dual_stream"
    DUAL_STREAM_ATTENTION_CONTRACT="xlnet_content_diagonal"
    DEFAULT_RUN_PROJECT="unified-f-on-b-qwen3-0.6b-smoke-ascend16"
    TRAIN_ENTRY="pretrain/train_selfless_flow.py"
    DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="false"
    DEFAULT_IMAGE_SIGMA_ORDER="random"
    DEFAULT_VALIDATION_ORDER_STRATEGY="spatial_halton"
    DEFAULT_FLOW_HEAD_WIDTH="1936"
    DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT="not_applicable"
    DEFAULT_FLOW_CONDITION_CONTRACT="not_applicable"
    ;;
  *)
    echo "ERROR: ABLATION must be a, b, c, d, e, or f; got ${ABLATION}" >&2
    exit 2
    ;;
esac

DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING="${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING:-${DEFAULT_DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}}"
IMAGE_SIGMA_ORDER="${IMAGE_SIGMA_ORDER:-${DEFAULT_IMAGE_SIGMA_ORDER}}"
VALIDATION_ORDER_STRATEGY="${VALIDATION_ORDER_STRATEGY:-${DEFAULT_VALIDATION_ORDER_STRATEGY}}"
FLOW_HEAD_WIDTH="${FLOW_HEAD_WIDTH:-${DEFAULT_FLOW_HEAD_WIDTH}}"
FLOW_HEAD_ATTENTION_CONTRACT="${FLOW_HEAD_ATTENTION_CONTRACT:-${DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT}}"
FLOW_CONDITION_CONTRACT="${FLOW_CONDITION_CONTRACT:-${DEFAULT_FLOW_CONDITION_CONTRACT}}"

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
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NPROC_PER_NODE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${DEBUG_LOSS_TRACE_UNTIL_STEP}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: DEBUG_LOSS_TRACE_UNTIL_STEP must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${DEEPSPEED_BF16_OVERFLOW_CHECK_UNTIL_STEP}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: DEEPSPEED_BF16_OVERFLOW_CHECK_UNTIL_STEP must be a non-negative integer" >&2
  exit 2
fi
if [[ "${IMAGE_FLOW_BATCH_MUL}" != "4" ]]; then
  echo "ERROR: B-based unified ablations require IMAGE_FLOW_BATCH_MUL=4" >&2
  exit 2
fi
if [[ "${IMAGE_SIGMA_ORDER}" != "${DEFAULT_IMAGE_SIGMA_ORDER}" ]]; then
  echo "ERROR: ablation ${ABLATION} requires IMAGE_SIGMA_ORDER=${DEFAULT_IMAGE_SIGMA_ORDER}" >&2
  exit 2
fi
if [[ "${VALIDATION_ORDER_STRATEGY}" != "${DEFAULT_VALIDATION_ORDER_STRATEGY}" ]]; then
  echo "ERROR: ablation ${ABLATION} requires VALIDATION_ORDER_STRATEGY=${DEFAULT_VALIDATION_ORDER_STRATEGY}" >&2
  exit 2
fi
if [[ "${FLOW_HEAD_WIDTH}" != "${DEFAULT_FLOW_HEAD_WIDTH}" ]]; then
  echo "ERROR: ablation ${ABLATION} requires FLOW_HEAD_WIDTH=${DEFAULT_FLOW_HEAD_WIDTH}" >&2
  exit 2
fi
if [[ "${FLOW_HEAD_ATTENTION_CONTRACT}" != "${DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT}" ]]; then
  echo "ERROR: ablation ${ABLATION} requires FLOW_HEAD_ATTENTION_CONTRACT=${DEFAULT_FLOW_HEAD_ATTENTION_CONTRACT}" >&2
  exit 2
fi
if [[ "${FLOW_CONDITION_CONTRACT}" != "${DEFAULT_FLOW_CONDITION_CONTRACT}" ]]; then
  echo "ERROR: ablation ${ABLATION} requires FLOW_CONDITION_CONTRACT=${DEFAULT_FLOW_CONDITION_CONTRACT}" >&2
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
if [[ "${ABLATION}" != "d" && "${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}" != "false" ]]; then
  echo "ERROR: T2I-only Dynamic-XT checkpointing is valid only for ablation D" >&2
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
RUN_ROOT="${RUN_ROOT:-${OUTPUT_DIR_BASE}/${RUN_PROJECT}}"
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
  --image-sigma-order "${IMAGE_SIGMA_ORDER}" \
  --validation-order-strategy "${VALIDATION_ORDER_STRATEGY}" \
  --flow-head-width "${FLOW_HEAD_WIDTH}" \
  --flow-head-attention-contract "${FLOW_HEAD_ATTENTION_CONTRACT}" \
  --flow-condition-contract "${FLOW_CONDITION_CONTRACT}" \
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
  "experiment.output_dir=${OUTPUT_DIR_BASE}"
  "experiment.resume_from_checkpoint=${RESUME_FROM}"
  "experiment.save_every=${SAVE_EVERY}"
  "experiment.checkpoints_total_limit=${CHECKPOINTS_TOTAL_LIMIT}"
  "experiment.checkpoint_milestone_every=${CHECKPOINT_MILESTONE_EVERY}"
  "experiment.log_every=1"
  "experiment.debug_loss_trace_until_step=${DEBUG_LOSS_TRACE_UNTIL_STEP}"
  "experiment.deepspeed_bf16_overflow_check_until_step=${DEEPSPEED_BF16_OVERFLOW_CHECK_UNTIL_STEP}"
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
  "model.flow_head_attention_contract=${FLOW_HEAD_ATTENTION_CONTRACT}"
  "model.flow_condition_contract=${FLOW_CONDITION_CONTRACT}"
  "model.image_flow_batch_mul=${IMAGE_FLOW_BATCH_MUL}"
  "training.use_gradient_checkpointing=false"
  "model.dynamic_xt_t2i_gradient_checkpointing=${DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING}"
  "model.training_image_sigma_order=${IMAGE_SIGMA_ORDER}"
  "model.image_flow_width=${FLOW_HEAD_WIDTH}"
  "model.image_flow_grad_checkpointing=false"
  "dataset.params.image.image_sigma_order=${IMAGE_SIGMA_ORDER}"
  "experiment.validation_single_stream_order_strategies=[${VALIDATION_ORDER_STRATEGY}]"
  "evaluation.strategies=${VALIDATION_ORDER_STRATEGY}"
  "model.showo_mask_schedule=cosine"
  "model.showo_min_masking_rate=0.0"
  "training.stop_after_steps=${SMOKE_STEPS}"
)
if [[ "${ABLATION}" == "f" ]]; then
  COMMAND+=(
    "model.positionwise_reference_flow_width=1280"
    "model.positionwise_reference_flow_depth=8"
    "model.positionwise_max_parameter_relative_error=0.005"
  )
fi
printf '%q ' "${COMMAND[@]}" >"${AUDIT_DIR}/launch_command.sh"
printf '\n' >>"${AUDIT_DIR}/launch_command.sh"

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  WANDB_MODE="${WANDB_MODE}" \
  "${COMMAND[@]}" 2>&1 | tee "${AUDIT_DIR}/training.log"
