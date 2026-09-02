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

SOURCE_TASK="${SOURCE_TASK:-${1:-}}"
case "${SOURCE_TASK}" in
  climbmix|text)
    SOURCE_TASK="climbmix"
    DEFAULT_CONFIG="configs/selfless/unified_single_text_0p6b_100b_ascend16.yaml"
    DEFAULT_RUN_PROJECT="unified-b-0p6b-text-only-100bphys-s42-r1"
    FORMAL_STEPS=95368
    DEFAULT_SAVE_EMA_EVAL_EVERY=12510
    ;;
  i2t|caption)
    SOURCE_TASK="i2t"
    DEFAULT_CONFIG="configs/selfless/unified_single_caption_0p6b_100b_ascend16.yaml"
    DEFAULT_RUN_PROJECT="unified-b-0p6b-caption-only-100bphys-s42-r1"
    FORMAL_STEPS=190736
    DEFAULT_SAVE_EMA_EVAL_EVERY=12510
    ;;
  t2i|image)
    SOURCE_TASK="t2i"
    DEFAULT_CONFIG="configs/selfless/unified_single_t2i_0p6b_100b_ascend16.yaml"
    DEFAULT_RUN_PROJECT="unified-b-0p6b-t2i-only-100bphys-s42-r1"
    FORMAL_STEPS=190736
    DEFAULT_SAVE_EMA_EVAL_EVERY=12510
    ;;
  *)
    echo "ERROR: set SOURCE_TASK to climbmix, i2t, or t2i" >&2
    exit 2
    ;;
esac

CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
PROTOCOL="${PROTOCOL:-configs/protocols/unified_single_source_0p6b_100b_ascend16.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/16_npus_1node_deepspeed_zero2.yaml}"
RESUME_FROM="${RESUME_FROM:-none}"
STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-${FORMAL_STEPS}}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-3}"
CHECKPOINT_MILESTONE_EVERY="${CHECKPOINT_MILESTONE_EVERY:-0}"
VAL_EVERY="${VAL_EVERY:-2000}"
VALIDATION_IMAGE_EVERY="${VALIDATION_IMAGE_EVERY:-${VAL_EVERY}}"
VALIDATION_I2T_EVERY="${VALIDATION_I2T_EVERY:-${VALIDATION_IMAGE_EVERY}}"
VALIDATION_I2T_SAMPLES="${VALIDATION_I2T_SAMPLES:-2}"
VALIDATION_I2T_MAX_NEW_TOKENS="${VALIDATION_I2T_MAX_NEW_TOKENS:-64}"
VALIDATION_MAX_BATCHES="${VALIDATION_MAX_BATCHES:-4}"
VALIDATION_IMAGE_SAMPLES="${VALIDATION_IMAGE_SAMPLES:-2}"
SAVE_EMA_EVAL_EVERY="${SAVE_EMA_EVAL_EVERY:-${DEFAULT_SAVE_EMA_EVAL_EVERY}}"
SAVE_FINAL="${SAVE_FINAL:-true}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"
BACKBONE_LR="${BACKBONE_LR:-3.0e-4}"
FLOW_LR="${FLOW_LR:-5.0e-5}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-output}"
RUN_PROJECT="${RUN_PROJECT:-${DEFAULT_RUN_PROJECT}}"
RUN_NAME="${RUN_NAME:-${RUN_PROJECT}}"

NODE_RANK="${PET_NODE_RANK:-0}"
NUM_MACHINES="${PET_NNODES:-1}"
PLATFORM_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-0}"
MAIN_PROCESS_IP="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
MAIN_PROCESS_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
NPROC_PER_NODE=16
EXPECTED_WORLD_SIZE=16

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
# The parent process tokenizes during dataset construction.  The ClimbMix
# worker creates its own bounded Rayon pool after fork.
export TOKENIZERS_PARALLELISM=false
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

if [[ "${NODE_RANK}" != "0" || "${NUM_MACHINES}" != "1" ]]; then
  echo "ERROR: this launcher requires exactly one node with rank 0; got rank=${NODE_RANK}, nodes=${NUM_MACHINES}" >&2
  exit 3
fi
if [[ "${PLATFORM_NPROC_PER_NODE}" != "0" && "${PLATFORM_NPROC_PER_NODE}" != "16" ]]; then
  echo "ERROR: expected PET_NPROC_PER_NODE=0 or 16, got ${PLATFORM_NPROC_PER_NODE}" >&2
  exit 4
fi
if [[ ! "${STOP_AFTER_STEPS}" =~ ^[1-9][0-9]*$ ]] || (( STOP_AFTER_STEPS > FORMAL_STEPS )); then
  echo "ERROR: STOP_AFTER_STEPS must be in [1,${FORMAL_STEPS}], got ${STOP_AFTER_STEPS}" >&2
  exit 5
fi
if [[ ! -f "${CONFIG}" || ! -f "${PROTOCOL}" || ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: missing CONFIG=${CONFIG}, PROTOCOL=${PROTOCOL}, or ACCELERATE_CONFIG=${ACCELERATE_CONFIG}" >&2
  exit 6
fi
if [[ "${HCCL_INTRA_ROCE_ENABLE}" != "1" ]]; then
  echo "ERROR: HCCL_INTRA_ROCE_ENABLE must equal 1" >&2
  exit 7
fi

read -r NPU_AVAILABLE LOCAL_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${LOCAL_NPUS}" != "${NPROC_PER_NODE}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${LOCAL_NPUS}" >&2
  exit 8
fi

RUN_ROOT="${RUN_ROOT:-${OUTPUT_DIR_BASE}/${RUN_PROJECT}}"
AUDIT_DIR="${AUDIT_DIR:-${RUN_ROOT}/prelaunch_audit}"
if [[ "${RESUME_FROM}" == "none" && -f "${RUN_ROOT}/config.yaml" ]]; then
  echo "ERROR: refusing a fresh run over existing ${RUN_ROOT}/config.yaml" >&2
  exit 9
fi
mkdir -p "${AUDIT_DIR}"

PREFLIGHT=(
  python scripts/validate_unified_single_source.py
  --config "${CONFIG}"
  --protocol "${PROTOCOL}"
  --source "${SOURCE_TASK}"
  --formal-world-size "${EXPECTED_WORLD_SIZE}"
  --require-npu-count "${NPROC_PER_NODE}"
  --tokenizer-probe
  --run-project "${RUN_PROJECT}"
  --backbone-lr "${BACKBONE_LR}"
  --flow-lr "${FLOW_LR}"
  --save-ema-eval-every "${SAVE_EMA_EVAL_EVERY}"
)
"${PREFLIGHT[@]}" >"${AUDIT_DIR}/asset_preflight.json"

COMMAND=(
  python scripts/launch_accelerate_multinode.py launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_machines "${NUM_MACHINES}"
  --num_processes "${EXPECTED_WORLD_SIZE}"
  --machine_rank "${NODE_RANK}"
  --main_process_ip "${MAIN_PROCESS_IP}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  --rdzv_backend static
  --same_network
  pretrain/train_selfless_flow.py
  "config=${CONFIG}"
  "experiment.project=${RUN_PROJECT}"
  "experiment.name=${RUN_NAME}"
  "experiment.output_dir=${OUTPUT_DIR_BASE}"
  "experiment.resume_from_checkpoint=${RESUME_FROM}"
  "experiment.save_every=${SAVE_EVERY}"
  "experiment.checkpoints_total_limit=${CHECKPOINTS_TOTAL_LIMIT}"
  "experiment.checkpoint_milestone_every=${CHECKPOINT_MILESTONE_EVERY}"
  "experiment.val_every=${VAL_EVERY}"
  "experiment.validation_image_every=${VALIDATION_IMAGE_EVERY}"
  "experiment.validation_i2t_every=${VALIDATION_I2T_EVERY}"
  "experiment.validation_i2t_samples=${VALIDATION_I2T_SAMPLES}"
  "experiment.validation_i2t_max_new_tokens=${VALIDATION_I2T_MAX_NEW_TOKENS}"
  "experiment.validation_max_batches=${VALIDATION_MAX_BATCHES}"
  "experiment.validation_image_samples=${VALIDATION_IMAGE_SAMPLES}"
  "experiment.save_ema_eval_every=${SAVE_EMA_EVAL_EVERY}"
  "experiment.save_final=${SAVE_FINAL}"
  "experiment.save_final_checkpoint=${SAVE_FINAL_CHECKPOINT}"
  "optimizer.params.learning_rate=${BACKBONE_LR}"
  "optimizer.params.backbone_learning_rate=${BACKBONE_LR}"
  "optimizer.params.special_token_learning_rate=${BACKBONE_LR}"
  "optimizer.params.projector_learning_rate=${FLOW_LR}"
  "optimizer.params.flow_learning_rate=${FLOW_LR}"
  "training.stop_after_steps=${STOP_AFTER_STEPS}"
)
printf '%q ' "${COMMAND[@]}" >"${AUDIT_DIR}/launch_command.sh"
printf '\n' >>"${AUDIT_DIR}/launch_command.sh"

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  WANDB_MODE="${WANDB_MODE}" \
  "${COMMAND[@]}" 2>&1 | tee "${AUDIT_DIR}/training.log"
