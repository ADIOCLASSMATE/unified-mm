#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/selfless/imagenet100_class_base_80ep.yaml}"
CHECKPOINT="${CHECKPOINT:-output/selfless-flow-im100-class-lr80-b32ga2-nogc-b4e5-f1e4/hf_model-final-ema}"
OUTPUT_DIR="${OUTPUT_DIR:-output/selfless-flow-im100-class-lr80-b32ga2-nogc-b4e5-f1e4-fid-is}"
NUM_GPUS="${NUM_GPUS:-8}"
SAMPLES="${SAMPLES:-10000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-100}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-384}"

if [[ "${NUM_GPUS}" != "8" ]]; then
  echo "ERROR: formal ImageNet-100 FID/IS requires NUM_GPUS=8, got ${NUM_GPUS}" >&2
  exit 2
fi
if [[ "${SAMPLES}" != "10000" ]]; then
  echo "ERROR: formal ImageNet-100 FID/IS requires SAMPLES=10000, got ${SAMPLES}" >&2
  exit 3
fi
if [[ "${SAMPLING_STEPS}" != "100" ]]; then
  echo "ERROR: formal ImageNet-100 FID/IS requires SAMPLING_STEPS=100, got ${SAMPLING_STEPS}" >&2
  exit 4
fi
if [[ "${BATCH_SIZE_PER_GPU}" != "384" ]]; then
  echo "ERROR: validated H100 evaluation batch is 384/GPU, got ${BATCH_SIZE_PER_GPU}" >&2
  exit 5
fi

PYTHONPATH=. python - "${CONFIG}" "${CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
checkpoint = Path(sys.argv[2])
if str(config.dataset.params.conditioning_mode) != "class":
    raise RuntimeError("formal class evaluation requires conditioning_mode=class")
if not checkpoint.is_dir():
    raise FileNotFoundError(f"missing EMA checkpoint directory: {checkpoint}")
for filename in ("config.json", "model.safetensors", "tokenizer.json"):
    if not (checkpoint / filename).is_file():
        raise FileNotFoundError(f"incomplete EMA checkpoint: {checkpoint / filename}")
real_stats = Path(config.evaluation.real_stats_path)
if not real_stats.is_file():
    raise FileNotFoundError(f"missing real-stat cache: {real_stats}")
print(f"Evaluation preflight OK: checkpoint={checkpoint}, real_stats={real_stats}")
PY

export CONFIG CHECKPOINT OUTPUT_DIR NUM_GPUS SAMPLES SAMPLING_STEPS BATCH_SIZE_PER_GPU
export REAL_STATS_PATH="${REAL_STATS_PATH:-public/datasets/imagenet_ablation_100c_balanced/fid_stats/inception_v3_2048_original_256.pt}"
exec bash "${REPO_ROOT}/script/selfless/evaluate_imagenet_flow.sh" \
  --require_official_protocol \
  "$@"
