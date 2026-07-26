#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:?Set CONFIG to one generated backbone-flow-head joint YAML}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-output/backbone_flow_head_joint_ablation/evidence/runtime_source_manifest.json}"
NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" != "8" ]]; then
  echo "Formal joint ablation requires NUM_GPUS=8, got ${NUM_GPUS}" >&2
  exit 2
fi

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/8_gpus_deepspeed_zero2.yaml}"
TRAIN_PORT="${TRAIN_PORT:-8891}"
EVAL_PORT="${EVAL_PORT:-29531}"
WANDB_MODE="${WANDB_MODE:-offline}"

python scripts/backbone_flow_head_joint_ablation.py \
  validate-runtime --manifest "${RUNTIME_MANIFEST}" >/dev/null
python scripts/backbone_flow_head_joint_ablation.py \
  validate-config --config "${CONFIG}" >/dev/null

readarray -t RUN_FIELDS < <(
  python scripts/backbone_flow_head_joint_ablation.py \
    inspect --config "${CONFIG}"
)
PROJECT="${RUN_FIELDS[0]}"
TRAINING_SEED="${RUN_FIELDS[1]}"
CELL_ID="${RUN_FIELDS[2]}"
CHECKPOINT="${RUN_FIELDS[3]}"
OUTPUT_DIR="${RUN_FIELDS[4]}"
RUN_ROOT="output/${PROJECT}"
PREFLIGHT="${RUN_ROOT}/joint_ablation_preflight.json"
METRICS="${OUTPUT_DIR}/metrics.json"
VALIDATED_RESULT="${RUN_ROOT}/joint_ablation_validated_result.json"

if [[ -e "${PREFLIGHT}" || -e "${CHECKPOINT}" || -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing run artifacts for ${CELL_ID}" >&2
  exit 3
fi
mkdir -p "${RUN_ROOT}"
python scripts/backbone_flow_head_joint_ablation.py write-preflight \
  --config "${CONFIG}" \
  --manifest "${RUNTIME_MANIFEST}" \
  --output "${PREFLIGHT}" >/dev/null

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
WANDB_MODE="${WANDB_MODE}" \
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --main_process_port "${TRAIN_PORT}" \
  --num_processes "${NUM_GPUS}" \
  pretrain/train_selfless_flow.py \
  --config "${CONFIG}"

python scripts/backbone_flow_head_joint_ablation.py \
  validate-runtime --manifest "${RUNTIME_MANIFEST}" >/dev/null
if [[ ! -s "${CHECKPOINT}/model.safetensors" ]]; then
  echo "Training did not produce ${CHECKPOINT}/model.safetensors" >&2
  exit 4
fi

EVAL_PROCESS_GROUP_TIMEOUT_SECONDS="${EVAL_PROCESS_GROUP_TIMEOUT_SECONDS:-7200}" \
MASTER_PORT="${EVAL_PORT}" \
torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/evaluate_single_stream_fid_is.py \
  --config "${CONFIG}" \
  --model_path_override "${CHECKPOINT}" \
  --model_dtype bf16 \
  --samples 10000 \
  --split val \
  --batch_size 512 \
  --sampling_steps 100 \
  --temperature 1.0 \
  --cfg 3.5 \
  --cfg_schedule constant \
  --flow_solver heun \
  --parallel_rate 1 \
  --strategies spatial_halton \
  --seed 42 \
  --fid_feature 2048 \
  --is_splits 10 \
  --vae_dtype fp32 \
  --real_stats_path \
    public/datasets/imagenet_ablation_100c_balanced/fid_stats/inception_v3_2048_original_256.pt \
  --inception_weights_path \
    output/cache/inception/weights-inception-2015-12-05-6726825d.pth \
  --require_official_protocol \
  --output_dir "${OUTPUT_DIR}"

python scripts/backbone_flow_head_joint_ablation.py validate-metrics \
  --config "${CONFIG}" \
  --metrics "${METRICS}" \
  --output "${VALIDATED_RESULT}" >/dev/null

echo "Validated ${CELL_ID} seed ${TRAINING_SEED}: ${METRICS}"
