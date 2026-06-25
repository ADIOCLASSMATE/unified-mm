#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
CONFIG="${CONFIG:-configs/selfless/imagenet_flow_refine_full.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/4_gpus_deepspeed_zero2.yaml}"
PORT="${PORT:-8892}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
uv run accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --main_process_port "${PORT}" \
  --num_processes "${NUM_GPUS}" \
  pretrain/train_selfless_flow.py \
  config="${CONFIG}" \
  "$@"
