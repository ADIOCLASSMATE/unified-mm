#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 1
fi
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to the evaluation checkpoint or HF export}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set to a new or resumable evaluation directory}"
INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth}"
REAL_STATS_PATH="${REAL_STATS_PATH:-public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"
CFG="${CFG:-3.5}"
SEED="${SEED:-42}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4096}"
VAE_DECODE_BATCH_SIZE="${VAE_DECODE_BATCH_SIZE:-16}"
NPU_COUNT=16

mkdir -p "${OUTPUT_DIR}"
STATUS_PATH="${OUTPUT_DIR}/launcher.status"
write_launcher_status() {
  local exit_code=$?
  local state="FAILED"
  trap - EXIT
  if [[ "${exit_code}" -eq 0 ]]; then
    state="SUCCEEDED"
  fi
  {
    printf 'status=%s\n' "${state}"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'updated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${STATUS_PATH}.tmp"
  mv "${STATUS_PATH}.tmp" "${STATUS_PATH}"
  exit "${exit_code}"
}
trap write_launcher_status EXIT

for required in \
  "${CONFIG}" \
  "${MODEL_PATH}/config.json" \
  "${MODEL_PATH}/model.safetensors" \
  "${MODEL_PATH}/tokenizer.json" \
  "${INCEPTION_WEIGHTS}" \
  "${REAL_STATS_PATH}" \
  public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_val_fp16.pt \
  public/datasets/imagenet_full/manifest_val.jsonl \
  public/datasets/imagenet1k_synthetic_v1/captions/imagenet1k_val_visual_descriptions.jsonl \
  public/datasets/imagenet1k_synthetic_v1/indexed/val/manifest.json; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing formal evaluation asset: ${required}" >&2
    exit 2
  fi
done

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401

print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${VISIBLE_NPUS}" >&2
  exit 3
fi

if [[ -f "${OUTPUT_DIR}/metrics.json" ]]; then
  echo "Formal metrics already exist: ${OUTPUT_DIR}/metrics.json"
  exit 0
fi

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_single_stream_fid_is.py \
  --config "${CONFIG}" \
  --model_path_override "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --device npu \
  --model_dtype bf16 \
  --seed "${SEED}" \
  --samples 50000 \
  --batch_size "${GLOBAL_BATCH_SIZE}" \
  --caption_sequence_mode t2i \
  --sampling_steps "${SAMPLING_STEPS}" \
  --temperature 1.0 \
  --cfg "${CFG}" \
  --cfg_schedule constant \
  --flow_solver heun \
  --parallel_rate 1 \
  --strategies spatial_halton \
  --vae_dtype fp32 \
  --vae_decode_batch_size "${VAE_DECODE_BATCH_SIZE}" \
  --inception_weights_path "${INCEPTION_WEIGHTS}" \
  --real_stats_path "${REAL_STATS_PATH}" \
  --require_official_protocol \
  --canonical_pairing \
  --resume_progress \
  --resume_checkpoint_interval_batches 1 \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
