#!/usr/bin/env bash
set -euo pipefail

SOURCE_CACHE="${SOURCE_CACHE:-public/datasets/imagenet_full/vae_latents_mar_kl16/flow_latents_all_fp16.pt}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-public/datasets/imagenet_full/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-public/datasets/imagenet_stage0_10c/vae_latents_mar_kl16}"
NUM_CLASSES="${NUM_CLASSES:-10}"
MAX_SAMPLES_PER_CLASS="${MAX_SAMPLES_PER_CLASS:--1}"
SEED="${SEED:-42}"

uv run python scripts/build_imagenet_flow_cache_subset.py \
  --source_cache "${SOURCE_CACHE}" \
  --source_manifest "${SOURCE_MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_classes "${NUM_CLASSES}" \
  --max_samples_per_class "${MAX_SAMPLES_PER_CLASS}" \
  --seed "${SEED}" \
  "$@"
