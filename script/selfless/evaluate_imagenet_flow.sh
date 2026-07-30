#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/selfless/imagenet_flow_full_from_qwen3base.yaml}"
CHECKPOINT="${CHECKPOINT:-output/selfless-flow-stage0-imagenet-full-from-qwen3base/hf_model-final-ema}"
OUTPUT_DIR="${OUTPUT_DIR:-output/selfless-flow-evaluation}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-512}"
BATCH_SIZE="${BATCH_SIZE:-$((NUM_GPUS * BATCH_SIZE_PER_GPU))}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/evaluate_single_stream_fid_is.py \
  --config "${CONFIG}" \
  --model_path_override "${CHECKPOINT}" \
  --model_dtype "${MODEL_DTYPE:-bf16}" \
  --samples "${SAMPLES:-10000}" \
  --split val \
  --batch_size "${BATCH_SIZE}" \
  --sampling_steps "${SAMPLING_STEPS:-100}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --cfg "${CFG:-3.5}" \
  --cfg_schedule "${CFG_SCHEDULE:-constant}" \
  --flow_solver "${FLOW_SOLVER:-heun}" \
  --parallel_rate "${PARALLEL_RATE:-1}" \
  --strategies "${STRATEGIES:-spatial_halton}" \
  --seed "${SEED:-42}" \
  --fid_feature 2048 \
  --is_splits 10 \
  --vae_dtype fp32 \
  --real_stats_path "${REAL_STATS_PATH:-}" \
  --inception_weights_path \
    "${INCEPTION_WEIGHTS_PATH:-output/cache/inception/weights-inception-2015-12-05-6726825d.pth}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
