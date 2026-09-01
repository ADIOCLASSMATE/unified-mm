#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <evaluation-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
EVAL_ROOT="$2"
PROFILE="${EVAL_PROFILE:-smoke}"
CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth}"
REAL_STATS="${REAL_STATS:-public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt}"
NPU_COUNT=16

if [[ "${PROFILE}" == "smoke" ]]; then
  VALIDATION_MAX_BATCHES="${VALIDATION_MAX_BATCHES:-2}"
  I2T_MAX_NEW_TOKENS="${I2T_MAX_NEW_TOKENS:-32}"
  T2I_SAMPLES="${T2I_SAMPLES:-32}"
  T2I_GLOBAL_BATCH="${T2I_GLOBAL_BATCH:-32}"
  T2I_VAE_BATCH_PER_RANK="${T2I_VAE_BATCH_PER_RANK:-2}"
  T2I_IS_SPLITS="${T2I_IS_SPLITS:-8}"
  T2I_PROTOCOL_ARGS=(--no-resume_progress)
elif [[ "${PROFILE}" == "formal" ]]; then
  VALIDATION_MAX_BATCHES="${VALIDATION_MAX_BATCHES:-0}"
  I2T_MAX_NEW_TOKENS="${I2T_MAX_NEW_TOKENS:-96}"
  T2I_SAMPLES="${T2I_SAMPLES:-50000}"
  T2I_GLOBAL_BATCH="${T2I_GLOBAL_BATCH:-256}"
  T2I_VAE_BATCH_PER_RANK="${T2I_VAE_BATCH_PER_RANK:-16}"
  T2I_IS_SPLITS="${T2I_IS_SPLITS:-10}"
  T2I_PROTOCOL_ARGS=(--require_official_protocol --resume_progress --resume_checkpoint_interval_batches 10)
else
  echo "ERROR: EVAL_PROFILE must be smoke or formal; got ${PROFILE}" >&2
  exit 3
fi

for integer in \
  "${VALIDATION_MAX_BATCHES}" \
  "${I2T_MAX_NEW_TOKENS}" \
  "${T2I_SAMPLES}" \
  "${T2I_GLOBAL_BATCH}" \
  "${T2I_VAE_BATCH_PER_RANK}" \
  "${T2I_IS_SPLITS}"; do
  if [[ ! "${integer}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: evaluation integer is invalid: ${integer}" >&2
    exit 4
  fi
done
if (( T2I_SAMPLES < NPU_COUNT )); then
  echo "ERROR: distributed sample counts must be at least ${NPU_COUNT}" >&2
  exit 5
fi
if (( T2I_SAMPLES % NPU_COUNT != 0 || T2I_GLOBAL_BATCH % NPU_COUNT != 0 )); then
  echo "ERROR: T2I samples and global batch must be divisible by ${NPU_COUNT}" >&2
  exit 6
fi
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 8
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
    exit 9
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
  exit 10
fi

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${EVAL_ROOT}/validation" "${EVAL_ROOT}/t2i-fid-is"

distributed_run() {
  env \
    -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
    torchrun --standalone --nproc_per_node="${NPU_COUNT}" "$@"
}

distributed_run \
  scripts/evaluate_unified_validation.py \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --output_dir "${EVAL_ROOT}/validation" \
  --validation_max_batches "${VALIDATION_MAX_BATCHES}" \
  --batch_size_per_rank 16 \
  --num_workers 2 \
  --image_samples 2 \
  --i2t_samples 2 \
  --i2t_max_new_tokens "${I2T_MAX_NEW_TOKENS}" \
  --model_dtype bf16 \
  2>&1 | tee "${EVAL_ROOT}/validation.log"

distributed_run \
  scripts/evaluate_single_stream_fid_is.py \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --output_dir "${EVAL_ROOT}/t2i-fid-is" \
  --device npu \
  --model_dtype bf16 \
  --samples "${T2I_SAMPLES}" \
  --batch_size "${T2I_GLOBAL_BATCH}" \
  --caption_sequence_mode t2i \
  --sampling_steps 10 \
  --temperature 1.0 \
  --cfg 3.5 \
  --cfg_schedule constant \
  --flow_solver heun \
  --parallel_rate 1 \
  --strategies spatial_halton \
  --vae_dtype fp32 \
  --vae_decode_batch_size "${T2I_VAE_BATCH_PER_RANK}" \
  --fid_feature 2048 \
  --is_splits "${T2I_IS_SPLITS}" \
  --inception_weights_path "${INCEPTION_WEIGHTS}" \
  --real_stats_path "${REAL_STATS}" \
  --canonical_pairing \
  "${T2I_PROTOCOL_ARGS[@]}" \
  2>&1 | tee "${EVAL_ROOT}/t2i-fid-is.log"

python scripts/summarize_unified_evaluation.py \
  --checkpoint "${MODEL_SOURCE}" \
  --output_root "${EVAL_ROOT}" \
  --profile "${PROFILE}" \
  | tee "${EVAL_ROOT}/summary.log"
