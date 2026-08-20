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

CONFIG="${CONFIG:-configs/selfless/imagenet1k_caption_joint_sweep_10ep_ascend16_b1024.yaml}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/hf_model-final-ema}"
PROBE_EMA_DIR="${PROBE_EMA_DIR:-}"
PROBE_OUTPUT="${PROBE_OUTPUT:-output/selfless-flow-imagenet1k-caption-joint-gradient-probe/init/probe.json}"
PROBE_BATCHES="${PROBE_BATCHES:-16}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-16}"
PROBE_SEED="${PROBE_SEED:-424242}"

read -r NPU_AVAILABLE LOCAL_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${LOCAL_NPUS}" -lt 1 ]]; then
  echo "ERROR: gradient probe requires at least one visible Ascend NPU" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: missing config: ${CONFIG}" >&2
  exit 3
fi
if [[ -n "${PROBE_EMA_DIR}" ]]; then
  if [[ ! -f "${PROBE_EMA_DIR}/ema_manifest.json" ]]; then
    echo "ERROR: missing sharded EMA checkpoint: ${PROBE_EMA_DIR}" >&2
    exit 3
  fi
  PROBE_SOURCE_ARGS=(--ema_dir "${PROBE_EMA_DIR}")
else
  if [[ ! -f "${PROBE_CHECKPOINT}/model.safetensors" ]]; then
    echo "ERROR: missing HF checkpoint: ${PROBE_CHECKPOINT}" >&2
    exit 3
  fi
  PROBE_SOURCE_ARGS=(--checkpoint "${PROBE_CHECKPOINT}")
fi

AUDIT_DIR="$(dirname "${PROBE_OUTPUT}")"
mkdir -p "${AUDIT_DIR}"
python tests/smoke_npu_joint_gradient_probe.py \
  >"${AUDIT_DIR}/npu_smoke.log" 2>&1

python scripts/probe_imagenet1k_caption_joint_gradients.py \
  --config "${CONFIG}" \
  "${PROBE_SOURCE_ARGS[@]}" \
  --output "${PROBE_OUTPUT}" \
  --num_batches "${PROBE_BATCHES}" \
  --batch_size "${PROBE_BATCH_SIZE}" \
  --seed "${PROBE_SEED}" \
  --device npu:0 \
  >"${AUDIT_DIR}/probe.stdout.json"

jq -e '.status == "complete" and .gradients_cleared == true' "${PROBE_OUTPUT}" >/dev/null
echo "CAPTION/T2I GRADIENT PROBE PASS: ${PROBE_OUTPUT}"
