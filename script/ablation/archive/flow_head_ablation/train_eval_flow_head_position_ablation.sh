#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:?Set CONFIG to one generated FH run YAML}"
NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" != "8" ]]; then
  echo "Formal FH training/evaluation requires NUM_GPUS=8, got ${NUM_GPUS}" >&2
  exit 2
fi
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/8_gpus_deepspeed_zero2.yaml}"
TRAIN_PORT="${TRAIN_PORT:-8891}"
EVAL_PORT="${EVAL_PORT:-29531}"
WANDB_MODE="${WANDB_MODE:-offline}"

readarray -t RUN_FIELDS < <(
  python - "${CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf
from scripts.flow_head_position_ablation import validate_ablation_config

config = OmegaConf.load(sys.argv[1])
validate_ablation_config(config)
print(config.experiment.project)
print(config.training.seed)
PY
)
PROJECT="${RUN_FIELDS[0]}"
TRAINING_SEED="${RUN_FIELDS[1]}"
CHECKPOINT="${REPO_ROOT}/output/${PROJECT}/hf_model-final-ema"
OUTPUT_DIR="${REPO_ROOT}/output/${PROJECT}/fid_is_cfg3p5_10k_ema"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
WANDB_MODE="${WANDB_MODE}" \
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --main_process_port "${TRAIN_PORT}" \
  --num_processes "${NUM_GPUS}" \
  pretrain/train_selfless_flow.py \
  --config "${CONFIG}"

if [[ ! -s "${CHECKPOINT}/model.safetensors" ]]; then
  echo "Training did not produce ${CHECKPOINT}/model.safetensors" >&2
  exit 3
fi

# Evaluation begins immediately for this cell when its training finishes; it
# does not wait for any other FH cell.
MASTER_PORT="${EVAL_PORT}" torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
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
  --require_flow_head_position_ablation_protocol \
  --output_dir "${OUTPUT_DIR}"

python - "${OUTPUT_DIR}/metrics.json" "${PROJECT}" "${TRAINING_SEED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload["architecture"]["ablation_id"].lower() not in sys.argv[2]:
    raise SystemExit("evaluation architecture ID does not match run slug")
if int(payload["training_protocol"]["flow_head_position"]["provenance"]["training_seed"]) != int(sys.argv[3]):
    raise SystemExit("evaluation training seed does not match run config")
print(path.resolve())
PY
