#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
CONFIG="${CONFIG:-configs/selfless/imagenet_diffusion_warmup_full.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/4_gpus_deepspeed_zero2.yaml}"
PORT="${PORT:-8888}"
WANDB_MODE="${WANDB_MODE:-online}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
WANDB_MODE="${WANDB_MODE}" \
uv run accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --main_process_port "${PORT}" \
  --num_processes "${NUM_GPUS}" \
  pretrain/train_selfless_flow.py \
  config="${CONFIG}" \
  "$@"
