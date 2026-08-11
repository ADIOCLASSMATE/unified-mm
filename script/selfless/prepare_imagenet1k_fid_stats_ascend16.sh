#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

VAL_ROOT="${VAL_ROOT:-/inspire/sj-ssd3/project/high-dimensionaldata/public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/val}"
INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt}"
STATUS_PATH="${STATUS_PATH:-public/datasets/imagenet_full/fid_stats/preparation.status}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
NPU_COUNT=16

if [[ ! -d "${VAL_ROOT}" || ! -f "${INCEPTION_WEIGHTS}" ]]; then
  echo "ERROR: missing ImageNet validation root or fixed Inception weights" >&2
  exit 2
fi
if [[ -e "${OUTPUT_PATH}" ]]; then
  echo "ERROR: refusing to overwrite existing real stats: ${OUTPUT_PATH}" >&2
  exit 3
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
printf 'RUNNING\n' >"${STATUS_PATH}"
record_exit() {
  exit_code=$?
  if [[ "${exit_code}" == "0" ]]; then
    printf 'SUCCEEDED\n' >"${STATUS_PATH}"
  else
    printf 'FAILED exit_code=%s\n' "${exit_code}" >"${STATUS_PATH}"
  fi
}
trap record_exit EXIT

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${VISIBLE_NPUS}" >&2
  exit 4
fi

torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/precompute_imagenet_fid_stats.py \
  --class_image_root "${VAL_ROOT}" \
  --inception_weights_path "${INCEPTION_WEIGHTS}" \
  --output "${OUTPUT_PATH}" \
  --device npu \
  --split validation \
  --expected_samples 50000 \
  --expected_classes 1000 \
  --expected_samples_per_class 50 \
  --feature 2048 \
  --image_size 256 \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --dataloader_workers "${DATALOADER_WORKERS}"

python - "${OUTPUT_PATH}" <<'PY'
import sys
from pathlib import Path

from scripts.validate_ascend_imagenet1k_pretraining import validate_real_stats

stats_path = Path(sys.argv[1])
report = validate_real_stats(stats_path)
print(f"PASS ImageNet-val 50K FID real stats: {stats_path} {report}")
PY
