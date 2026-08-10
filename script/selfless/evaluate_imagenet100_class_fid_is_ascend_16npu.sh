#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 1
fi
source "${CANN_SET_ENV}"
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv-npu}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/selfless/imagenet100_class_base_80ep_ascend_16npu.yaml}"
CHECKPOINT="${CHECKPOINT:-output/selfless-flow-im100-class-ascend16-b16ga2-b4e5-f1e4/hf_model-final-ema}"
OUTPUT_DIR="${OUTPUT_DIR:-output/selfless-flow-im100-class-ascend16-b16ga2-b4e5-f1e4-fid-is}"
DEVICE="${DEVICE:-npu}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
SAMPLES="${SAMPLES:-10000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-100}"
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-16}"
VAE_DECODE_BATCH_SIZE="${VAE_DECODE_BATCH_SIZE:-16}"
REAL_STATS_PATH="${REAL_STATS_PATH:-public/datasets/imagenet_ablation_100c_balanced/fid_stats/inception_v3_2048_original_256.pt}"
INCEPTION_WEIGHTS_PATH="${INCEPTION_WEIGHTS_PATH:-output/cache/inception/weights-inception-2015-12-05-6726825d.pth}"
EVALUATION_ACCEPTANCE_JSON="${EVALUATION_ACCEPTANCE_JSON:-/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/npu-parity-audit/ASCEND_EVALUATION_FINAL_ACCEPTANCE.json}"

if [[ "${DEVICE}" != "npu" ]]; then
  echo "ERROR: Ascend formal evaluation requires DEVICE=npu, got ${DEVICE}" >&2
  exit 2
fi
if [[ "${NUM_PROCESSES}" != "16" ]]; then
  echo "ERROR: Ascend formal evaluation requires 16 NPUs, got ${NUM_PROCESSES}" >&2
  exit 3
fi
if [[ "${SAMPLES}" != "10000" ]]; then
  echo "ERROR: formal ImageNet-100 FID/IS requires 10000 samples, got ${SAMPLES}" >&2
  exit 4
fi
if [[ "${SAMPLING_STEPS}" != "100" ]]; then
  echo "ERROR: formal ImageNet-100 FID/IS requires 100 sampling steps, got ${SAMPLING_STEPS}" >&2
  exit 5
fi
if [[ "${BATCH_SIZE_PER_DEVICE}" != "16" ]]; then
  echo "ERROR: validated Ascend evaluation batch is 16/NPU, got ${BATCH_SIZE_PER_DEVICE}" >&2
  exit 6
fi

PYTHONPATH=. python - \
  "${CONFIG}" "${CHECKPOINT}" "${REAL_STATS_PATH}" \
  "${INCEPTION_WEIGHTS_PATH}" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

from scripts.evaluate_single_stream_fid_is import (
    load_shared_original_real_stats,
)

config = OmegaConf.load(sys.argv[1])
checkpoint = Path(sys.argv[2])
real_stats = Path(sys.argv[3])
inception_weights = Path(sys.argv[4])
if str(config.dataset.params.conditioning_mode) != "class":
    raise RuntimeError("formal class evaluation requires conditioning_mode=class")
if not checkpoint.is_dir():
    raise FileNotFoundError(f"missing EMA checkpoint directory: {checkpoint}")
for filename in ("config.json", "model.safetensors", "tokenizer.json"):
    if not (checkpoint / filename).is_file():
        raise FileNotFoundError(f"incomplete EMA checkpoint: {checkpoint / filename}")
if not real_stats.is_file():
    raise FileNotFoundError(f"missing real-stat cache: {real_stats}")
if not inception_weights.is_file():
    raise FileNotFoundError(f"missing Inception weights: {inception_weights}")
payload = load_shared_original_real_stats(
    str(real_stats),
    config=config,
    fid_feature=2048,
    real_image_size=256,
    inception_weights_path=str(inception_weights),
)
if int(payload["stats"]["count"]) != 10000:
    raise RuntimeError(
        "formal real-stat cache must contain 10000 images, got "
        f"{payload['stats']['count']}"
    )
print(
    "Evaluation preflight OK: "
    f"checkpoint={checkpoint}, real_stats={real_stats}, "
    f"inception_weights={inception_weights}"
)
PY

export CONFIG CHECKPOINT OUTPUT_DIR DEVICE NUM_PROCESSES SAMPLES
export SAMPLING_STEPS BATCH_SIZE_PER_DEVICE VAE_DECODE_BATCH_SIZE
export REAL_STATS_PATH INCEPTION_WEIGHTS_PATH
bash "${REPO_ROOT}/script/selfless/evaluate_imagenet_flow.sh" \
  --require_official_protocol \
  "$@"

python scripts/validate_ascend_imagenet100_evaluation.py \
  --metrics "${OUTPUT_DIR}/metrics.json" \
  --checkpoint "${CHECKPOINT}" \
  --real_stats "${REAL_STATS_PATH}" \
  --inception_weights "${INCEPTION_WEIGHTS_PATH}" \
  --output_json "${EVALUATION_ACCEPTANCE_JSON}"
