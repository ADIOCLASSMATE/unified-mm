#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd/global_user/wanjiaxin-253108030048/code/unified-mm
source "./script/offline_env.sh"

NUM_GPUS="${NUM_GPUS:-8}"
CONFIG="${CONFIG:-configs/ablation/imagenet_flow_full_from_qwen3base_image_embedder_nonorm.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/8_gpus_deepspeed_zero2.yaml}"
PORT="${PORT:-8891}"
WANDB_MODE="${WANDB_MODE:-offline}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
WANDB_MODE="${WANDB_MODE}" \
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --main_process_port "${PORT}" \
  --num_processes "${NUM_GPUS}" \
  pretrain/train_selfless_flow.py \
  config="${CONFIG}" \
  "$@"
