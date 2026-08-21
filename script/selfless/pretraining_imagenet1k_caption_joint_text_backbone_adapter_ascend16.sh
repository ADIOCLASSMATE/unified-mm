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

CONFIG="${CONFIG:-configs/selfless/imagenet1k_caption_joint_text_backbone_adapter_10ep_ascend16_b1024.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/16_npus_1node_deepspeed_zero2.yaml}"
RUN_PROJECT="${RUN_PROJECT:-selfless-flow-imagenet1k-caption-joint-text-backbone-adapter-b2e6-f2e5-lt0p4}"
BACKBONE_LR="${BACKBONE_LR:-2e-6}"
FLOW_LR="${FLOW_LR:-2e-5}"
LAMBDA_TEXT="${LAMBDA_TEXT:-0.4}"
STOP_AFTER_STEPS="${STOP_AFTER_STEPS:-2404}"
RESUME_FROM="${RESUME_FROM:-none}"
WANDB_MODE="${WANDB_MODE:-offline}"

NODE_RANK="${PET_NODE_RANK:-}"
NUM_MACHINES="${PET_NNODES:-}"
PLATFORM_NPROC_PER_NODE="${PET_NPROC_PER_NODE:-0}"
MAIN_PROCESS_IP="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
MAIN_PROCESS_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-}}"
NPROC_PER_NODE=16
EXPECTED_NUM_MACHINES=1
EXPECTED_WORLD_SIZE=16

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

if [[ "${NODE_RANK}" != "0" || "${NUM_MACHINES}" != "${EXPECTED_NUM_MACHINES}" ]]; then
  echo "ERROR: split-init training requires one platform node with rank 0" >&2
  exit 2
fi
if [[ "${PLATFORM_NPROC_PER_NODE}" != "0" && "${PLATFORM_NPROC_PER_NODE}" != "16" ]]; then
  echo "ERROR: expected PET_NPROC_PER_NODE=0 or 16, got ${PLATFORM_NPROC_PER_NODE}" >&2
  exit 3
fi
if [[ -z "${MAIN_PROCESS_IP}" || -z "${MAIN_PROCESS_PORT}" ]]; then
  echo "ERROR: platform master address/port is missing" >&2
  exit 4
fi
if [[ ! "${RUN_PROJECT}" =~ ^selfless-flow-imagenet1k-caption-joint-text-backbone-adapter-[a-z0-9.-]+$ ]]; then
  echo "ERROR: invalid RUN_PROJECT=${RUN_PROJECT}" >&2
  exit 5
fi
if [[ "${STOP_AFTER_STEPS}" != "1202" && "${STOP_AFTER_STEPS}" != "2404" && "${STOP_AFTER_STEPS}" != "4808" && "${STOP_AFTER_STEPS}" != "12020" ]]; then
  echo "ERROR: invalid STOP_AFTER_STEPS=${STOP_AFTER_STEPS}" >&2
  exit 6
fi
if [[ ! -f "${CONFIG}" || ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: missing training or Accelerate config" >&2
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

RUN_ROOT="output/${RUN_PROJECT}"
AUDIT_DIR="${RUN_ROOT}/prelaunch_audit/stage-${STOP_AFTER_STEPS}"
mkdir -p "${AUDIT_DIR}"

python scripts/validate_ascend_imagenet1k_caption_joint_split_init.py \
  --config "${CONFIG}" \
  --backbone_lr "${BACKBONE_LR}" \
  --flow_lr "${FLOW_LR}" \
  --lambda_text "${LAMBDA_TEXT}" \
  --stop_after_steps "${STOP_AFTER_STEPS}" \
  --world_size "${EXPECTED_WORLD_SIZE}" \
  --require_npu_count "${NPROC_PER_NODE}" \
  --require_hccl_intra_roce \
  >"${AUDIT_DIR}/asset_preflight.json"

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
  "experiment.name=${RUN_PROJECT}-seed42-16x910b-b16ga4-b1024"
  "experiment.resume_from_checkpoint=${RESUME_FROM}"
  "model.lambda_text=${LAMBDA_TEXT}"
  "model.lambda_image=1.0"
  "optimizer.params.learning_rate=${FLOW_LR}"
  "optimizer.params.backbone_learning_rate=${BACKBONE_LR}"
  "optimizer.params.special_token_learning_rate=${BACKBONE_LR}"
  "optimizer.params.projector_learning_rate=${FLOW_LR}"
  "optimizer.params.flow_learning_rate=${FLOW_LR}"
  "training.stop_after_steps=${STOP_AFTER_STEPS}"
  "evaluation.checkpoint=${RUN_ROOT}/hf_model-final-ema"
)
printf '%q ' "${COMMAND[@]}" >"${AUDIT_DIR}/launch_command.sh"
printf '\n' >>"${AUDIT_DIR}/launch_command.sh"

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  WANDB_MODE="${WANDB_MODE}" \
  "${COMMAND[@]}" 2>&1 | tee "${AUDIT_DIR}/training.log"
