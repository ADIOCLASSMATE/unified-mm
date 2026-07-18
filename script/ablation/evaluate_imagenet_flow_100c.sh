#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/ablation/imagenet_flow_100c_80ep.yaml}"
CHECKPOINT="${CHECKPOINT:-output/selfless-flow-ablation-imagenet100-80ep/hf_model-final-ema}"
OUTPUT_DIR="${OUTPUT_DIR:-output/selfless-flow-ablation-imagenet100-80ep/fid_is_selected_cfg3p5_ema}"
NUM_GPUS="${NUM_GPUS:-8}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
if [[ "${MODEL_DTYPE}" != "bf16" && "${MODEL_DTYPE}" != "fp32" ]]; then
  echo "MODEL_DTYPE must be bf16 or fp32, got ${MODEL_DTYPE}" >&2
  exit 2
fi
for arg in "$@"; do
  if [[ "${arg}" == "--model_dtype" || "${arg}" == --model_dtype=* ]]; then
    echo "Pass model precision through MODEL_DTYPE, not an extra --model_dtype argument" >&2
    exit 2
  fi
done
PROTOCOL_ARGS=()
if [[ "${REQUIRE_OFFICIAL_PROTOCOL:-1}" == "1" ]]; then
  PROTOCOL_ARGS+=(--require_official_protocol)
fi

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/evaluate_single_stream_fid_is.py \
  --config "${CONFIG}" \
  --model_path_override "${CHECKPOINT}" \
  --model_dtype "${MODEL_DTYPE}" \
  --samples "${SAMPLES:-10000}" \
  --split val \
  --batch_size "${BATCH_SIZE:-512}" \
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
  --real_stats_path \
    "public/datasets/imagenet_ablation_100c_balanced/fid_stats/inception_v3_2048_original_256.pt" \
  --inception_weights_path \
    "output/cache/inception/weights-inception-2015-12-05-6726825d.pth" \
  --output_dir "${OUTPUT_DIR}" \
  "${PROTOCOL_ARGS[@]}" \
  "$@"
