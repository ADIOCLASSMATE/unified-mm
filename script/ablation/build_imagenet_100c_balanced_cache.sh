#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"

SOURCE_CACHE="${SOURCE_CACHE:-public/datasets/imagenet_full/vae_latents_mar_kl16/flow_latents_all_fp16.pt}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-public/datasets/imagenet_full/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-public/datasets/imagenet_ablation_100c_balanced/vae_latents_mar_kl16}"
SEED="${SEED:-42}"

python scripts/build_imagenet_flow_cache_subset.py \
  --source_cache "${SOURCE_CACHE}" \
  --source_manifest "${SOURCE_MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_name "flow_latents_100c_1250pc_fp16.pt" \
  --num_classes 100 \
  --class_selection stratified \
  --min_samples_per_class 1250 \
  --max_samples_per_class 1250 \
  --shuffle_within_class \
  --seed "${SEED}" \
  "$@"

python scripts/build_imagenet100_split_manifest.py \
  --reference_cache "${OUTPUT_DIR}/flow_latents_100c_1250pc_fp16.pt" \
  --manifest "$(dirname "${OUTPUT_DIR}")/manifest.jsonl" \
  --output "$(dirname "${OUTPUT_DIR}")/split_seed42_val100.jsonl" \
  --seed "${SEED}" \
  --val_samples_per_class 100 \
  --overwrite
