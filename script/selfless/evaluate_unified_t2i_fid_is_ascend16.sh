#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <evaluation-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
OUTPUT_DIR="$2"
PROFILE="${EVAL_PROFILE:-formal}"
CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth}"
REAL_STATS="${REAL_STATS:-public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt}"
NPU_COUNT=16

if [[ "${PROFILE}" == "smoke" ]]; then
  SAMPLES="${T2I_SAMPLES:-32}"
  GLOBAL_BATCH="${T2I_GLOBAL_BATCH:-32}"
  VAE_BATCH_PER_RANK="${T2I_VAE_BATCH_PER_RANK:-2}"
  IS_SPLITS="${T2I_IS_SPLITS:-8}"
  PROTOCOL_ARGS=(--no-resume_progress)
elif [[ "${PROFILE}" == "formal" ]]; then
  SAMPLES="${T2I_SAMPLES:-50000}"
  GLOBAL_BATCH="${T2I_GLOBAL_BATCH:-4096}"
  VAE_BATCH_PER_RANK="${T2I_VAE_BATCH_PER_RANK:-16}"
  IS_SPLITS="${T2I_IS_SPLITS:-10}"
  PROTOCOL_ARGS=(--require_official_protocol --resume_progress --resume_checkpoint_interval_batches 1)
else
  echo "ERROR: EVAL_PROFILE must be smoke or formal; got ${PROFILE}" >&2
  exit 3
fi

for integer in \
  "${SAMPLES}" \
  "${GLOBAL_BATCH}" \
  "${VAE_BATCH_PER_RANK}" \
  "${IS_SPLITS}"; do
  if [[ ! "${integer}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid evaluation integer: ${integer}" >&2
    exit 4
  fi
done
if (( SAMPLES < NPU_COUNT || SAMPLES % NPU_COUNT != 0 )); then
  echo "ERROR: samples must be divisible by ${NPU_COUNT}" >&2
  exit 5
fi
if (( GLOBAL_BATCH < NPU_COUNT || GLOBAL_BATCH % NPU_COUNT != 0 )); then
  echo "ERROR: global batch must be divisible by ${NPU_COUNT}" >&2
  exit 6
fi

CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 7
fi
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"
for required in \
  "${CONFIG}" \
  "${INCEPTION_WEIGHTS}" \
  "${REAL_STATS}" \
  public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_val_fp16.pt \
  public/datasets/imagenet_full/manifest_val.jsonl \
  public/datasets/imagenet1k_synthetic_v1/indexed/val/manifest.json; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing evaluation asset: ${required}" >&2
    exit 8
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
  exit 9
fi

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

if [[ -f "${OUTPUT_DIR}/metrics.json" ]]; then
  echo "Evaluation metrics already exist: ${OUTPUT_DIR}/metrics.json"
  exit 0
fi

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_single_stream_fid_is.py \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --output_dir "${OUTPUT_DIR}" \
  --device npu \
  --model_dtype bf16 \
  --seed 42 \
  --samples "${SAMPLES}" \
  --batch_size "${GLOBAL_BATCH}" \
  --caption_sequence_mode t2i \
  --sampling_steps 10 \
  --temperature 1.0 \
  --cfg 3.5 \
  --cfg_schedule constant \
  --flow_solver heun \
  --parallel_rate 1 \
  --strategies spatial_halton \
  --vae_dtype fp32 \
  --vae_decode_batch_size "${VAE_BATCH_PER_RANK}" \
  --fid_feature 2048 \
  --is_splits "${IS_SPLITS}" \
  --inception_weights_path "${INCEPTION_WEIGHTS}" \
  --real_stats_path "${REAL_STATS}" \
  --canonical_pairing \
  "${PROTOCOL_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
