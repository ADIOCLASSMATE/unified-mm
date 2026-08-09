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

SOURCE_MANIFEST="${SOURCE_MANIFEST:-public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl}"
SOURCE_IMAGE_ROOT="${SOURCE_IMAGE_ROOT:-public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train}"
VAE_MODULE_ROOT="${VAE_MODULE_ROOT:-public/code/mar}"
VAE_PATH="${VAE_PATH:-public/vae/mar-kl16/kl16.ckpt}"
CACHE_SHARD_DIR="${CACHE_SHARD_DIR:-public/datasets/imagenet_ablation_100c_balanced/vae_posterior_mar_kl16_cache_shards}"
CACHE_PATH="${CACHE_PATH:-public/datasets/imagenet_ablation_100c_balanced/vae_posterior_mar_kl16/posterior_stats_100c_1250pc_fp16.pt}"
LOG_DIR="${LOG_DIR:-${CACHE_SHARD_DIR}/logs}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
VAE_DTYPE="${VAE_DTYPE:-fp16}"
NUM_NPUS="${NUM_NPUS:-$(python -c 'import torch, torch_npu; print(torch.npu.device_count())')}"
NPU_IDS="${NPU_IDS:-}"
OVERWRITE="${OVERWRITE:-0}"
MERGE_CACHE="${MERGE_CACHE:-1}"
MAX_IMAGES="${MAX_IMAGES:--1}"

if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "ERROR: missing canonical ImageNet-100 manifest: ${SOURCE_MANIFEST}" >&2
  exit 2
fi
if [[ "$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')" != "6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a" ]]; then
  echo "ERROR: ImageNet-100 manifest SHA256 does not match the canonical membership" >&2
  exit 3
fi
if [[ "$(sha256sum "${VAE_MODULE_ROOT}/models/vae.py" | awk '{print $1}')" != "95e9d47d017817cd86858d78587786c931a9ba9596fe3eb6d6dce4136580112b" ]]; then
  echo "ERROR: MAR KL16 VAE source SHA256 mismatch" >&2
  exit 4
fi
if [[ "$(sha256sum "${VAE_PATH}" | awk '{print $1}')" != "34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f" ]]; then
  echo "ERROR: MAR KL16 checkpoint SHA256 mismatch" >&2
  exit 5
fi
if [[ "${NUM_NPUS}" -lt 1 ]]; then
  echo "ERROR: NUM_NPUS must be positive, got ${NUM_NPUS}" >&2
  exit 6
fi

if [[ -z "${NPU_IDS}" ]]; then
  NPU_IDS="$(seq -s, 0 $((NUM_NPUS - 1)))"
fi
IFS=',' read -r -a npu_ids <<< "${NPU_IDS}"
if [[ "${#npu_ids[@]}" -ne "${NUM_NPUS}" ]]; then
  echo "ERROR: NPU_IDS=${NPU_IDS} must contain exactly NUM_NPUS=${NUM_NPUS} entries" >&2
  exit 7
fi

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

mkdir -p "${LOG_DIR}"
pids=()
trap 'for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done' INT TERM
echo "Launching ${NUM_NPUS} deterministic ImageNet-100 KL16 posterior shards."
for shard_index in $(seq 0 $((NUM_NPUS - 1))); do
  npu_id="${npu_ids[$shard_index]}"
  log_file="${LOG_DIR}/encode-shard-${shard_index}-of-${NUM_NPUS}.log"
  echo "  shard ${shard_index}/${NUM_NPUS} -> npu:${npu_id}, log ${log_file}"
  (
    python scripts/imagenet_encode_kl16_vae.py \
      --source_mode manifest_jsonl \
      --source_manifest_jsonl "${SOURCE_MANIFEST}" \
      --source_image_root "${SOURCE_IMAGE_ROOT}" \
      --vae_module_root "${VAE_MODULE_ROOT}" \
      --vae_path "${VAE_PATH}" \
      --cache_shard_dir "${CACHE_SHARD_DIR}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --prefetch_factor "${PREFETCH_FACTOR}" \
      --device "npu:${npu_id}" \
      --vae_dtype "${VAE_DTYPE}" \
      --num_shards "${NUM_NPUS}" \
      --shard_index "${shard_index}" \
      --max_images "${MAX_IMAGES}" \
      "${overwrite_args[@]}"
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
  echo "ERROR: at least one KL16 encode shard failed; inspect ${LOG_DIR}" >&2
  exit 8
fi

if [[ "${MERGE_CACHE}" == "1" ]]; then
  python pretrain/merge_flow_latent_shards.py \
    --shard_dir "${CACHE_SHARD_DIR}" \
    --output_path "${CACHE_PATH}" \
    --manifest_jsonl "${SOURCE_MANIFEST}" \
    --mmap
fi
