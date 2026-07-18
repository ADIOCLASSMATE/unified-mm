#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_ROOT="${SWEEP_ROOT:-output/qwen-showo-vq-ablation-imagenet100-80ep/fid_is_cfg_sweep_official}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0}"
CHECKPOINT="${CHECKPOINT:-output/qwen-showo-vq-ablation-imagenet100-80ep/hf_model-final}"
CONFIG="${CONFIG:-configs/ablation/qwen_showo_vq_100c_80ep.yaml}"
CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-2eaf3c5958c36be4f2554ce88f67082cc6e40d67924df945c8b35a3efdec1806}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TIMESTEPS="${TIMESTEPS:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

if [[ "${NUM_GPUS}" != "8" || "${BATCH_SIZE}" != "8" ]]; then
  echo "Formal Show-o CFG sweep requires NUM_GPUS=8 and BATCH_SIZE=8" >&2
  exit 2
fi
if [[ "${TIMESTEPS}" != "12" || "${TEMPERATURE}" != "1.0" ]]; then
  echo "Formal Show-o CFG sweep requires TIMESTEPS=12 and TEMPERATURE=1.0" >&2
  exit 2
fi
if [[ "${SEED}" != "42" ]]; then
  echo "Formal Show-o CFG sweep requires SEED=42" >&2
  exit 2
fi
if [[ "${SAVE_IMAGES}" != "0" && "${SAVE_IMAGES}" != "1" ]]; then
  echo "SAVE_IMAGES must be 0 or 1" >&2
  exit 2
fi

read -r -a GUIDANCE_ARRAY <<< "${GUIDANCE_SCALES}"
if (( ${#GUIDANCE_ARRAY[@]} == 0 )); then
  echo "GUIDANCE_SCALES must contain at least one value" >&2
  exit 2
fi

mkdir -p "${SWEEP_ROOT}"

ensure_sweep_contract() {
  local contract_lock_fd
  exec {contract_lock_fd}>"${SWEEP_ROOT}/.sweep_contract.lock"
  if ! flock -w 60 "${contract_lock_fd}"; then
    echo "Timed out waiting for Show-o sweep contract lock" >&2
    return 4
  fi
  python scripts/ensure_showo_cfg_sweep_contract.py \
    --root "${SWEEP_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
    --config "${CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --local-batch-size "${BATCH_SIZE}" \
    --timesteps "${TIMESTEPS}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --save-images "${SAVE_IMAGES}"
  flock -u "${contract_lock_fd}"
  exec {contract_lock_fd}>&-
}

common_cfg_for_guidance() {
  python - "$1" <<'PY'
from decimal import Decimal
import sys

print(Decimal(sys.argv[1]) + Decimal("1"))
PY
}

ensure_sweep_contract

for guidance_scale in "${GUIDANCE_ARRAY[@]}"; do
(
  if [[ ! "${guidance_scale}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid Show-o guidance scale: ${guidance_scale}" >&2
    exit 2
  fi

  common_cfg="$(common_cfg_for_guidance "${guidance_scale}")"
  guidance_slug="${guidance_scale//./p}"
  common_slug="${common_cfg//./p}"
  output_dir="${SWEEP_ROOT}/cfg_w_${common_slug}_showo_s_${guidance_slug}"
  metrics_path="${output_dir}/metrics.json"
  work_dir="${output_dir}.work"
  work_metrics_path="${work_dir}/metrics.json"
  lock_file="${output_dir}.lock"

  exec {CFG_LOCK_FD}>"${lock_file}"
  if ! flock -n "${CFG_LOCK_FD}"; then
    echo "CFG w=${common_cfg} is already owned by another process" >&2
    exit 4
  fi
  ensure_sweep_contract

  validation_args=(
    --guidance-scale "${guidance_scale}"
    --expected-checkpoint "${CHECKPOINT}"
    --expected-checkpoint-sha256 "${CHECKPOINT_SHA256}"
    --expected-config "${CONFIG}"
    --expected-seed "${SEED}"
  )
  if [[ "${SAVE_IMAGES}" == "1" ]]; then
    validation_args+=(--require-images)
  fi

  if [[ -e "${output_dir}" ]]; then
    if [[ "${SKIP_COMPLETED}" == "1" ]] && \
      python scripts/validate_showo_cfg_metrics.py \
        --metrics "${metrics_path}" "${validation_args[@]}"; then
      echo "Skipping validated CFG w=${common_cfg} (Show-o s=${guidance_scale})"
      exit 0
    fi
    echo "Refusing to overwrite existing output: ${output_dir}" >&2
    exit 3
  fi

  if [[ -e "${work_dir}" ]]; then
    if python scripts/validate_showo_cfg_metrics.py \
      --metrics "${work_metrics_path}" "${validation_args[@]}"; then
      ensure_sweep_contract
      mv -T "${work_dir}" "${output_dir}"
      echo "Recovered CFG w=${common_cfg} from validated work directory"
      exit 0
    fi
    partial_dir="${work_dir}.partial_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    echo "Archiving incomplete work directory: ${partial_dir}"
    mv -T "${work_dir}" "${partial_dir}"
  fi

  eval_args=()
  if [[ "${SAVE_IMAGES}" == "1" ]]; then
    eval_args+=(--save_images)
  fi

  echo "Evaluating common CFG w=${common_cfg} (Show-o s=${guidance_scale})"
  CONFIG="${CONFIG}" \
  CHECKPOINT="${CHECKPOINT}" \
  OUTPUT_DIR="${work_dir}" \
  NUM_GPUS="${NUM_GPUS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  TIMESTEPS="${TIMESTEPS}" \
  GUIDANCE_SCALE="${guidance_scale}" \
  TEMPERATURE="${TEMPERATURE}" \
  SEED="${SEED}" \
  bash script/ablation/evaluate_qwen_showo_vq_100c.sh "${eval_args[@]}"

  ensure_sweep_contract
  python scripts/validate_showo_cfg_metrics.py \
    --metrics "${work_metrics_path}" "${validation_args[@]}"
  mv -T "${work_dir}" "${output_dir}"
  echo "Published common CFG w=${common_cfg} (Show-o s=${guidance_scale})"
)
done
