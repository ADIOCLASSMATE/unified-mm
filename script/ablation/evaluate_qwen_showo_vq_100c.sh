#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/ablation/qwen_showo_vq_100c_80ep.yaml}"
CHECKPOINT="${CHECKPOINT:-output/qwen-showo-vq-ablation-imagenet100-80ep/hf_model-final}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen-showo-vq-ablation-imagenet100-80ep/fid_is_selected_w12p75_s11p75}"
NUM_GPUS="${NUM_GPUS:-8}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/evaluate_qwen_showo_fid_is.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --real_stats "public/datasets/imagenet_ablation_100c_balanced/fid_stats/inception_v3_2048_original_256.pt" \
  --batch_size "${BATCH_SIZE:-8}" \
  --timesteps "${TIMESTEPS:-12}" \
  --guidance_scale "${GUIDANCE_SCALE:-11.75}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --seed "${SEED:-42}" \
  --split_seed 42 \
  --val_samples_per_class 100 \
  "$@"
