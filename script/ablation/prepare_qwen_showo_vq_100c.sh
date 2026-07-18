#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-4}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train}"
MANIFEST="${MANIFEST:-public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-public/datasets/imagenet_ablation_100c_balanced/vq_tokens_magvit2_showo_8192}"
SHOWO_REPO="${SHOWO_REPO:-/inspire/hdd/global_user/wanjiaxin-253108030048/code/Show-o}"
VQ_MODEL_PATH="${VQ_MODEL_PATH:-public/models/showlab/magvitv2}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/prepare_imagenet100_showo_vq_tokens.py \
  --manifest_jsonl "${MANIFEST}" \
  --imagenet_root "${IMAGENET_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --showo_repo "${SHOWO_REPO}" \
  --vq_model_path "${VQ_MODEL_PATH}" \
  --batch_size "${BATCH_SIZE:-64}" \
  --num_workers "${NUM_WORKERS:-8}" \
  "$@"
