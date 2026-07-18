#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-4}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train}"
MANIFEST="${MANIFEST:-public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl}"
SYNSET_MAPPING="${SYNSET_MAPPING:-public/datasets/imagenet/LOC_synset_mapping.txt}"
OUTPUT="${OUTPUT:-public/datasets/imagenet_ablation_100c_balanced/fid_stats/inception_v3_2048_original_256.pt}"
INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-output/cache/inception/weights-inception-2015-12-05-6726825d.pth}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/cache_imagenet100_real_stats.py \
  --manifest "${MANIFEST}" \
  --synset_mapping "${SYNSET_MAPPING}" \
  --imagenet_train_dir "${IMAGENET_ROOT}" \
  --output "${OUTPUT}" \
  --inception_weights_path "${INCEPTION_WEIGHTS}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --split_seed 42 \
  --val_samples_per_class 100 \
  "$@"
