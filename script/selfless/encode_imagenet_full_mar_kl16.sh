#!/usr/bin/env bash
set -euo pipefail

IMAGENET_TRAIN_DIR="${IMAGENET_TRAIN_DIR:-/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train}"
OUTPUT_DIR="${OUTPUT_DIR:-public/datasets/imagenet_full/vae_latents_mar_kl16}"
MANIFEST_JSONL="${MANIFEST_JSONL:-public/datasets/imagenet_full/manifest.jsonl}"
VAE_PATH="${VAE_PATH:-public/vae/mar-kl16/kl16.ckpt}"
CACHE_SHARD_DIR="${CACHE_SHARD_DIR:-public/datasets/imagenet_full/vae_latents_mar_kl16_cache_shards}"
CACHE_PATH="${CACHE_PATH:-public/datasets/imagenet_full/vae_latents_mar_kl16/flow_latents_all_fp16.pt}"

BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PARALLEL="${PARALLEL:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
MAX_IMAGES="${MAX_IMAGES:--1}"
OVERWRITE="${OVERWRITE:-0}"
SAVE_PER_IMAGE="${SAVE_PER_IMAGE:-0}"
MERGE_CACHE="${MERGE_CACHE:-1}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi
save_args=(--cache_shard_dir "${CACHE_SHARD_DIR}")
if [[ "${SAVE_PER_IMAGE}" != "1" ]]; then
  save_args+=(--skip_per_image)
fi

run_encode_shard() {
  local shard_index="$1"
  local num_shards="$2"
  local gpu_id="${3:-}"
  local manifest_arg="${MANIFEST_JSONL}"

  if [[ -n "${gpu_id}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" uv run python scripts/imagenet_encode_mar_kl16.py \
      --source_mode imagenet_train \
      --imagenet_train_dir "${IMAGENET_TRAIN_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --manifest_jsonl "${manifest_arg}" \
      --vae_path "${VAE_PATH}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --prefetch_factor "${PREFETCH_FACTOR}" \
      --device cuda \
      --num_shards "${num_shards}" \
      --shard_index "${shard_index}" \
      --max_images "${MAX_IMAGES}" \
      "${save_args[@]}" \
      "${overwrite_args[@]}"
  else
    uv run python scripts/imagenet_encode_mar_kl16.py \
      --source_mode imagenet_train \
      --imagenet_train_dir "${IMAGENET_TRAIN_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --manifest_jsonl "${manifest_arg}" \
      --vae_path "${VAE_PATH}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --prefetch_factor "${PREFETCH_FACTOR}" \
      --device "${DEVICE}" \
      --num_shards "${num_shards}" \
      --shard_index "${shard_index}" \
      --max_images "${MAX_IMAGES}" \
      "${save_args[@]}" \
      "${overwrite_args[@]}"
  fi
}

if [[ "${PARALLEL}" == "1" && "${NUM_GPUS}" -gt 1 ]]; then
  IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
  if [[ "${#gpu_ids[@]}" -lt "${NUM_GPUS}" ]]; then
    echo "GPU_IDS=${GPU_IDS} has fewer entries than NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
  fi

  mkdir -p "${LOG_DIR}"
  pids=()
  trap 'for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done' INT TERM

  echo "Launching ${NUM_GPUS} ImageNet MAR-KL16 VAE encode shards."
  echo "Output: ${OUTPUT_DIR}"
  echo "Cache shards: ${CACHE_SHARD_DIR}"
  for shard in $(seq 0 $((NUM_GPUS - 1))); do
    gpu="${gpu_ids[$shard]}"
    log_file="${LOG_DIR}/encode-shard-${shard}-of-${NUM_GPUS}.log"
    echo "  shard ${shard}/${NUM_GPUS} -> GPU ${gpu}, log ${log_file}"
    (
      run_encode_shard "${shard}" "${NUM_GPUS}" "${gpu}"
    ) >"${log_file}" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  trap - INT TERM

  if [[ "${failed}" != "0" ]]; then
    echo "At least one encode shard failed. Check logs in ${LOG_DIR}." >&2
    exit 1
  fi
else
  run_encode_shard "${SHARD_INDEX}" "${NUM_SHARDS}" ""
fi

if [[ "${MERGE_CACHE}" == "1" ]]; then
  uv run python pretrain/merge_flow_latent_shards.py \
    --shard_dir "${CACHE_SHARD_DIR}" \
    --output_path "${CACHE_PATH}" \
    --mmap
fi
