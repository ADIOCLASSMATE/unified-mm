#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [output-dir]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-output/evaluation/vae-rfid/mar-kl16-imagenet-val50k}"
PROFILE="${EVAL_PROFILE:-formal}"
NPU_COUNT=16
CACHE_SHARD_DIR="${CACHE_SHARD_DIR:-public/datasets/imagenet_full/vae_posterior_mar_kl16/val_shards}"
VAE_MODULE_ROOT="${VAE_MODULE_ROOT:-public/code/mar}"
VAE_PATH="${VAE_PATH:-public/vae/mar-kl16/kl16.ckpt}"
INCEPTION_WEIGHTS="${INCEPTION_WEIGHTS:-public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth}"
REAL_STATS="${REAL_STATS:-public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt}"

if [[ "${PROFILE}" == "formal" ]]; then
  SAMPLES="${RFID_SAMPLES:-50000}"
  BATCH_SIZE_PER_RANK="${RFID_BATCH_SIZE_PER_RANK:-16}"
  PROTOCOL_ARGS=(--require_full_protocol)
elif [[ "${PROFILE}" == "smoke" ]]; then
  SAMPLES="${RFID_SAMPLES:-32}"
  BATCH_SIZE_PER_RANK="${RFID_BATCH_SIZE_PER_RANK:-2}"
  PROTOCOL_ARGS=()
else
  echo "ERROR: EVAL_PROFILE must be formal or smoke; got ${PROFILE}" >&2
  exit 3
fi

CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 4
fi
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for required in \
  "${VAE_MODULE_ROOT}/models/vae.py" \
  "${VAE_PATH}" \
  "${INCEPTION_WEIGHTS}" \
  "${REAL_STATS}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing rFID asset: ${required}" >&2
    exit 5
  fi
done
for shard_index in $(seq 0 15); do
  shard_path="$(printf '%s/shard-%05d-of-00016.pt' "${CACHE_SHARD_DIR}" "${shard_index}")"
  if [[ ! -f "${shard_path}" ]]; then
    echo "ERROR: missing rFID cache shard: ${shard_path}" >&2
    exit 6
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
  exit 7
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

METRICS_PATH="${OUTPUT_DIR}/metrics.json"
if [[ -f "${METRICS_PATH}" ]]; then
  if python - "${METRICS_PATH}" "${PROFILE}" "${SAMPLES}" <<'PY'
import json
import sys

path, profile, samples = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = json.load(open(path, encoding="utf-8"))
protocol = payload.get("protocol", {})
valid = (
    payload.get("schema") == "mar_kl16_vae_rfid_v2"
    and payload.get("primary_metric", {}).get("path")
    == "metrics.sample.rfid"
    and protocol.get("samples") == samples
    and protocol.get("posterior_modes") == ["sample", "mean"]
    and protocol.get("runtime_hashing_enabled") is False
    and bool(protocol.get("full_protocol")) == (profile == "formal")
)
raise SystemExit(0 if valid else 1)
PY
  then
    echo "Validated rFID metrics already exist: ${METRICS_PATH}"
    exit 0
  fi
  echo "ERROR: existing rFID metrics do not match the requested protocol: ${METRICS_PATH}" >&2
  exit 8
fi

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export EVAL_PROCESS_GROUP_TIMEOUT_SECONDS="${EVAL_PROCESS_GROUP_TIMEOUT_SECONDS:-3600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_vae_rfid.py \
  --cache_shard_dir "${CACHE_SHARD_DIR}" \
  --cache_shards "${NPU_COUNT}" \
  --vae_module_root "${VAE_MODULE_ROOT}" \
  --vae_path "${VAE_PATH}" \
  --inception_weights_path "${INCEPTION_WEIGHTS}" \
  --real_stats_path "${REAL_STATS}" \
  --output "${METRICS_PATH}" \
  --device npu \
  --vae_dtype fp32 \
  --posterior_modes sample mean \
  --samples "${SAMPLES}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --feature 2048 \
  --seed 42 \
  "${PROTOCOL_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
