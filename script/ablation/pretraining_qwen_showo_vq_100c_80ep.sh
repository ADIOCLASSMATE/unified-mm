#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-8}"
CONFIG="${CONFIG:-configs/ablation/qwen_showo_vq_100c_80ep.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/8_gpus_deepspeed_zero2.yaml}"
PORT="${PORT:-8892}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --main_process_port "${PORT}" \
  --num_processes "${NUM_GPUS}" \
  pretrain/train_qwen_showo.py \
  config="${CONFIG}" \
  "$@"
