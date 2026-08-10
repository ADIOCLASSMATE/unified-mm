#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
source "${CANN_SET_ENV}"
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv-npu}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

NUM_NPUS="${NUM_NPUS:-16}"
BATCH_SIZE_PER_NPU="${BATCH_SIZE_PER_NPU:-16}"
OUTPUT="${OUTPUT:-public/datasets/imagenet_ablation_100c_balanced/fid_stats/inception_v3_2048_original_256.pt}"
INCEPTION_WEIGHTS_PATH="${INCEPTION_WEIGHTS_PATH:-output/cache/inception/weights-inception-2015-12-05-6726825d.pth}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"

if [[ "${NUM_NPUS}" != "16" ]]; then
  echo "ERROR: canonical real-stat build requires 16 NPUs, got ${NUM_NPUS}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "ERROR: refusing to overwrite real-stat cache: ${OUTPUT}" >&2
  exit 3
fi
if [[ ! -f "${INCEPTION_WEIGHTS_PATH}" ]]; then
  echo "ERROR: missing Inception weights: ${INCEPTION_WEIGHTS_PATH}" >&2
  exit 4
fi

torchrun --standalone --nproc_per_node="${NUM_NPUS}" \
  scripts/precompute_imagenet_fid_stats.py \
  --manifest public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl \
  --split_manifest public/datasets/imagenet_ablation_100c_balanced/split_seed42_val100.jsonl \
  --imagenet_train_dir public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train \
  --inception_weights_path "${INCEPTION_WEIGHTS_PATH}" \
  --output "${OUTPUT}" \
  --device npu \
  --split validation \
  --expected_samples 10000 \
  --expected_classes 100 \
  --expected_samples_per_class 100 \
  --feature 2048 \
  --image_size 256 \
  --batch_size_per_rank "${BATCH_SIZE_PER_NPU}" \
  --dataloader_workers 0
