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

CONFIG="${CONFIG:-configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml}"
RESUME_FROM="${RESUME_FROM:-none}"
WANDB_MODE="${WANDB_MODE:-offline}"

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
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

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

RUN_ROOT="output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep"
AUDIT_DIR="${RUN_ROOT}/prelaunch_audit/node-${NODE_RANK}"
mkdir -p "${AUDIT_DIR}"

python scripts/validate_ascend_imagenet1k_pretraining.py \
  --config "${CONFIG}" \
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
  "experiment.resume_from_checkpoint=${RESUME_FROM}"
)
printf '%q ' "${COMMAND[@]}" >"${AUDIT_DIR}/launch_command.sh"
printf '\n' >>"${AUDIT_DIR}/launch_command.sh"

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  WANDB_MODE="${WANDB_MODE}" \
  "${COMMAND[@]}" 2>&1 | tee "${AUDIT_DIR}/training.log"
