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

CONFIG="${CONFIG:-configs/selfless/unified_baseline_1p7b_100b_ascend_64npu.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml}"
RESUME_FROM="${RESUME_FROM:-none}"
STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-955}"
SAVE_EVERY="${SAVE_EVERY:-955}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-3}"
CHECKPOINT_MILESTONE_EVERY="${CHECKPOINT_MILESTONE_EVERY:-125100}"
VAL_EVERY="${VAL_EVERY:-10000}"
SAVE_EMA_EVAL_EVERY="${SAVE_EMA_EVAL_EVERY:-25020}"
SAVE_FINAL="${SAVE_FINAL:-false}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-true}"
WANDB_MODE="${WANDB_MODE:-disabled}"
RUN_PROJECT="${RUN_PROJECT:-unified-b-qwen3-1.7b-100b-s42-r1}"
RUN_NAME="${RUN_NAME:-${RUN_PROJECT//\//-}}"
BACKBONE_LR="${BACKBONE_LR:-2.4e-4}"
FLOW_LR="${FLOW_LR:-6.0e-5}"
ABLATION="${ABLATION:-b}"
PRESERVE_MODEL_CONTRACT="${PRESERVE_MODEL_CONTRACT:-false}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-output}"

if [[ "${ABLATION}" != "b" ]]; then
  echo "ERROR: the Qwen3-1.7B experiment is frozen to baseline b" >&2
  exit 2
fi
if [[ "${PRESERVE_MODEL_CONTRACT}" != "false" && "${PRESERVE_MODEL_CONTRACT}" != "true" ]]; then
  echo "ERROR: PRESERVE_MODEL_CONTRACT must be true or false" >&2
  exit 2
fi

NODE_RANK="${PET_NODE_RANK:-}"
NUM_MACHINES="${PET_NNODES:-}"
PLATFORM_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-0}"
MAIN_PROCESS_IP="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
MAIN_PROCESS_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-}}"
NPROC_PER_NODE=16
EXPECTED_NUM_MACHINES=4
EXPECTED_WORLD_SIZE=64

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
# B4xL2048/B16xL512 fits numerically, but the default NPU caching allocator
# reserved about 61.6 GiB/rank and one multi-node rank exhausted its remaining
# contiguous space during the first matmul backward.  Expandable segments keep
# the frozen batch/GA contract while avoiding that fragmentation failure.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

echo "NPU allocator: PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF}"

if [[ ! "${NODE_RANK}" =~ ^[0-3]$ ]]; then
  echo "ERROR: expected PET_NODE_RANK in [0,3], got ${NODE_RANK:-<unset>}" >&2
  exit 2
fi
if [[ "${NUM_MACHINES}" != "${EXPECTED_NUM_MACHINES}" ]]; then
  echo "ERROR: expected PET_NNODES=4, got ${NUM_MACHINES:-<unset>}" >&2
  exit 3
fi
if [[ "${PLATFORM_NPROC_PER_NODE}" != "0" && "${PLATFORM_NPROC_PER_NODE}" != "16" ]]; then
  echo "ERROR: expected PET_NPROC_PER_NODE=0 or 16, got ${PLATFORM_NPROC_PER_NODE}" >&2
  exit 4
fi
if [[ -z "${MAIN_PROCESS_IP}" || -z "${MAIN_PROCESS_PORT}" ]]; then
  echo "ERROR: platform master address/port is missing" >&2
  exit 5
fi
if [[ ! -f "${CONFIG}" || ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: missing CONFIG=${CONFIG} or ACCELERATE_CONFIG=${ACCELERATE_CONFIG}" >&2
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

RUN_ROOT="${RUN_ROOT:-output/${RUN_PROJECT}}"
AUDIT_DIR="${AUDIT_DIR:-${RUN_ROOT}/prelaunch_audit/node-${NODE_RANK}}"
if [[ "${NODE_RANK}" == "0" && "${RESUME_FROM}" == "none" && -f "${RUN_ROOT}/config.yaml" ]]; then
  echo "ERROR: refusing a fresh run over existing ${RUN_ROOT}/config.yaml" >&2
  exit 9
fi
mkdir -p "${AUDIT_DIR}"

PREFLIGHT=(
  python scripts/validate_unified_baseline.py
  --config "${CONFIG}"
  --formal-world-size "${EXPECTED_WORLD_SIZE}"
  --require-npu-count "${NPROC_PER_NODE}"
  --run-project "${RUN_PROJECT}"
  --backbone-lr "${BACKBONE_LR}"
  --flow-lr "${FLOW_LR}"
  --save-ema-eval-every "${SAVE_EMA_EVAL_EVERY}"
  --ablation b
)
if [[ "${NODE_RANK}" == "0" ]]; then
  PREFLIGHT+=(--tokenizer-probe)
fi
"${PREFLIGHT[@]}" >"${AUDIT_DIR}/asset_preflight.json"

COMMAND=(
  python scripts/launch_accelerate_multinode.py launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_machines "${EXPECTED_NUM_MACHINES}"
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
if [[ "${PRESERVE_MODEL_CONTRACT}" == "false" ]]; then
  COMMAND+=(
    "model.training_objective=selfless_dual_stream"
    "model.dual_stream_attention_contract=xlnet_content_diagonal"
    "model.showo_mask_schedule=cosine"
    "model.showo_min_masking_rate=0.0"
  )
fi
printf '%q ' "${COMMAND[@]}" >"${AUDIT_DIR}/launch_command.sh"
printf '\n' >>"${AUDIT_DIR}/launch_command.sh"

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  WANDB_MODE="${WANDB_MODE}" \
  "${COMMAND[@]}" 2>&1 | tee "${AUDIT_DIR}/training.log"
