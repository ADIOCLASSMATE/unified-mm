#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SWEEP_ROOT="${SWEEP_ROOT:-output/selfless-flow-ablation-imagenet100-80ep/fid_is_cfg_sweep}"
CFG_VALUES="${CFG_VALUES:-1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
EXPECTED_MODEL_PATH="${EXPECTED_MODEL_PATH:-${CHECKPOINT:-output/selfless-flow-ablation-imagenet100-80ep/hf_model-final-ema}}"
EXPECTED_CONFIG="${EXPECTED_CONFIG:-${CONFIG:-configs/ablation/imagenet_flow_100c_80ep.yaml}}"
EXPECTED_SEED="${EXPECTED_SEED:-${SEED:-42}}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
EXPECTED_MODEL_SHA256="${EXPECTED_MODEL_SHA256:-}"
if [[ "${MODEL_DTYPE}" != "bf16" && "${MODEL_DTYPE}" != "fp32" ]]; then
  echo "MODEL_DTYPE must be bf16 or fp32, got ${MODEL_DTYPE}" >&2
  exit 2
fi
if [[ -z "${EXPECTED_MODEL_SHA256}" ]]; then
  case "${EXPECTED_MODEL_PATH}" in
    output/selfless-flow-ablation-imagenet100-80ep/hf_model-final-ema)
      EXPECTED_MODEL_SHA256="81f86d1805d732f8c8e377a08cef6a6aad285eb533677405d4867bda90a86203"
      ;;
    output/selfless-flow-ablation-imagenet100-80ep/hf_model-final)
      EXPECTED_MODEL_SHA256="1af7302e4498a8bf4b50c8bd0d8fe3b008487ab2b82f1504eb34b9ac21b2dab1"
      ;;
    *)
      echo "EXPECTED_MODEL_SHA256 is required for checkpoint ${EXPECTED_MODEL_PATH}" >&2
      exit 2
      ;;
  esac
fi

read -r -a CFG_ARRAY <<< "${CFG_VALUES}"
if (( ${#CFG_ARRAY[@]} == 0 )); then
  echo "CFG_VALUES must contain at least one value" >&2
  exit 2
fi

ensure_sweep_contract() {
  local contract_lock_fd
  exec {contract_lock_fd}>"${SWEEP_ROOT}/.sweep_contract.lock"
  if ! flock -w 60 "${contract_lock_fd}"; then
    echo "Timed out waiting for sweep contract lock: ${SWEEP_ROOT}" >&2
    return 4
  fi
  python scripts/ensure_flow_cfg_sweep_contract.py \
    --root "${SWEEP_ROOT}" \
    --model-path "${EXPECTED_MODEL_PATH}" \
    --model-sha256 "${EXPECTED_MODEL_SHA256}" \
    --model-dtype "${MODEL_DTYPE}" \
    --config "${EXPECTED_CONFIG}" \
    --seed "${EXPECTED_SEED}" \
    --batch-size "${BATCH_SIZE:-512}" \
    --samples "${SAMPLES:-10000}" \
    --sampling-steps "${SAMPLING_STEPS:-100}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --cfg-schedule "${CFG_SCHEDULE:-constant}" \
    --flow-solver "${FLOW_SOLVER:-heun}" \
    --parallel-rate "${PARALLEL_RATE:-1}" \
    --strategies "${STRATEGIES:-spatial_halton}" \
    --save-images "${SAVE_IMAGES}"
  flock -u "${contract_lock_fd}"
  exec {contract_lock_fd}>&-
}

mkdir -p "${SWEEP_ROOT}"
ensure_sweep_contract

for cfg in "${CFG_ARRAY[@]}"; do
(
  if [[ ! "${cfg}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid CFG value: ${cfg}" >&2
    exit 2
  fi

  slug="${cfg//./p}"
  output_dir="${SWEEP_ROOT}/cfg_${slug}"
  metrics_path="${output_dir}/metrics.json"
  work_dir="${output_dir}.work"
  work_metrics_path="${work_dir}/metrics.json"
  lock_file="${output_dir}.lock"
  exec {CFG_LOCK_FD}>"${lock_file}"
  if ! flock -n "${CFG_LOCK_FD}"; then
    echo "CFG=${cfg} is already owned by another sweep process: ${lock_file}" >&2
    exit 4
  fi
  ensure_sweep_contract

  validation_args=(
    --cfg "${cfg}"
    --expected-model-path "${EXPECTED_MODEL_PATH}"
    --expected-model-sha256 "${EXPECTED_MODEL_SHA256}"
    --expected-config "${EXPECTED_CONFIG}"
    --expected-seed "${EXPECTED_SEED}"
    --expected-model-dtype "${MODEL_DTYPE}"
  )
  if [[ "${SAVE_IMAGES}" == "1" ]]; then
    validation_args+=(--require-images)
  fi

  if [[ -e "${output_dir}" ]]; then
    if [[ "${SKIP_COMPLETED}" == "1" ]] && \
      python scripts/validate_flow_cfg_metrics.py \
         --metrics "${metrics_path}" "${validation_args[@]}"; then
      echo "Skipping validated CFG=${cfg}: ${metrics_path}"
      exit 0
    fi
    echo "Refusing to overwrite existing CFG=${cfg} output: ${output_dir}" >&2
    exit 3
  fi
  if [[ -e "${work_dir}" ]]; then
    if python scripts/validate_flow_cfg_metrics.py \
       --metrics "${work_metrics_path}" "${validation_args[@]}"; then
      ensure_sweep_contract
      mv -T "${work_dir}" "${output_dir}"
      echo "Recovered and published validated CFG=${cfg}: ${output_dir}"
      exit 0
    fi
    partial_dir="${work_dir}.partial_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    echo "Archiving incomplete CFG=${cfg} work directory: ${partial_dir}"
    mv -T "${work_dir}" "${partial_dir}"
  fi

  eval_args=()
  if [[ "${SAVE_IMAGES}" == "1" ]]; then
    eval_args+=(--save_images)
  fi

  echo "Evaluating CFG=${cfg} -> ${work_dir}"
  CFG="${cfg}" \
  MODEL_DTYPE="${MODEL_DTYPE}" \
  OUTPUT_DIR="${work_dir}" \
  bash script/ablation/evaluate_imagenet_flow_100c.sh "${eval_args[@]}"

  ensure_sweep_contract
  python scripts/validate_flow_cfg_metrics.py \
    --metrics "${work_metrics_path}" "${validation_args[@]}"
  mv -T "${work_dir}" "${output_dir}"
  echo "Published validated CFG=${cfg}: ${output_dir}"
)
done
