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

CONFIG="${CONFIG:-configs/selfless/imagenet100_class_base_80ep_ascend_16npu.yaml}"
TRAIN_PORT="${TRAIN_PORT:-29531}"
WANDB_MODE="${WANDB_MODE:-offline}"
BACKBONE_LR="${BACKBONE_LR:-4e-5}"
FLOW_HEAD_LR="${FLOW_HEAD_LR:-1e-4}"
RESUME_FROM="${RESUME_FROM:-none}"
DRY_RUN="${DRY_RUN:-0}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

NUM_NPUS="${NUM_NPUS:-16}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/16_npus_deepspeed_zero2.yaml}"
RUN_PROJECT="${RUN_PROJECT:-selfless-flow-im100-class-ascend16-b16ga2-b4e5-f1e4}"
RUN_NAME="${RUN_NAME:-im100-class-80ep-seed42-16x910b-b16ga2-b512}"
if [[ "${NUM_NPUS}" != "16" ]]; then
  echo "ERROR: formal training requires exactly 16 NPUs, got ${NUM_NPUS}" >&2
  exit 2
fi
for override in "$@"; do
  case "${override}" in
    model.*|dataset.*|optimizer.*|lr_scheduler.*|training.*|experiment.save_every=*|experiment.val_every=*)
      echo "ERROR: formal training rejects semantic override: ${override}" >&2
      exit 3
      ;;
  esac
done

if [[ ! -f "${CONFIG}" || ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: missing config: CONFIG=${CONFIG}, ACCELERATE_CONFIG=${ACCELERATE_CONFIG}" >&2
  exit 6
fi
if [[ "${HCCL_INTRA_ROCE_ENABLE}" != "1" ]]; then
  echo "ERROR: HCCL_INTRA_ROCE_ENABLE must equal 1" >&2
  exit 7
fi

read -r NPU_AVAILABLE ACTUAL_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${ACTUAL_NPUS}" != "${NUM_NPUS}" ]]; then
  echo "ERROR: NPU resource gate failed: available=${NPU_AVAILABLE}, visible=${ACTUAL_NPUS}, required=${NUM_NPUS}" >&2
  exit 8
fi

RUN_ROOT="output/${RUN_PROJECT}"
if [[ ! "${RUN_PROJECT}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: RUN_PROJECT must be a safe single path component: ${RUN_PROJECT}" >&2
  exit 9
fi
if [[ "${RESUME_FROM}" == "none" || "${RESUME_FROM}" == "null" || -z "${RESUME_FROM}" ]]; then
  if [[ -d "${RUN_ROOT}" ]] && find "${RUN_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: refusing to overwrite non-empty fresh run directory: ${RUN_ROOT}" >&2
    exit 11
  fi
else
  if [[ ! -f "${RESUME_FROM}/checkpoint_complete.json" ]]; then
    echo "ERROR: RESUME_FROM is not a complete checkpoint: ${RESUME_FROM}" >&2
    exit 12
  fi
  resume_absolute="$(realpath -m "${RESUME_FROM}")"
  run_root_absolute="$(realpath -m "${RUN_ROOT}")"
  resume_basename="$(basename "${resume_absolute}")"
  if [[ "$(dirname "${resume_absolute}")" != "${run_root_absolute}" \
    || ! "${resume_basename}" =~ ^checkpoint-[0-9]+$ ]]; then
    echo "ERROR: formal resume must use ${RUN_ROOT}/checkpoint-<step>: ${RESUME_FROM}" >&2
    exit 13
  fi
fi

PREFLIGHT_COMMAND=(
  python scripts/validate_ascend_imagenet100_assets.py
  --config "${CONFIG}"
  --world_size "${NUM_NPUS}"
  --require_npu_count "${NUM_NPUS}"
  --require_hccl_intra_roce
)

COMMAND=(
  accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  --main_process_port "${TRAIN_PORT}"
  --num_processes "${NUM_NPUS}"
  pretrain/train_selfless_flow.py
  "config=${CONFIG}"
  "experiment.project=${RUN_PROJECT}"
  "experiment.name=${RUN_NAME}"
  "experiment.resume_from_checkpoint=${RESUME_FROM}"
  "optimizer.params.learning_rate=${FLOW_HEAD_LR}"
  "optimizer.params.backbone_learning_rate=${BACKBONE_LR}"
  "optimizer.params.projector_learning_rate=${FLOW_HEAD_LR}"
  "optimizer.params.flow_learning_rate=${FLOW_HEAD_LR}"
  "optimizer.params.special_token_learning_rate=${BACKBONE_LR}"
)
COMMAND+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PREFLIGHT_COMMAND[@]}"
  printf 'DRY RUN:'
  printf ' %q' env HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE}" WANDB_MODE="${WANDB_MODE}" "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ "${RESUME_FROM}" == "none" || "${RESUME_FROM}" == "null" || -z "${RESUME_FROM}" ]]; then
  AUDIT_DIR="${RUN_ROOT}/prelaunch_audit"
else
  AUDIT_DIR="${RUN_ROOT}/prelaunch_audit_resume_$(basename "${RESUME_FROM}")"
fi
if [[ -e "${AUDIT_DIR}" ]]; then
  echo "ERROR: refusing to overwrite launch audit directory: ${AUDIT_DIR}" >&2
  exit 14
fi
mkdir -p "${AUDIT_DIR}"
"${PREFLIGHT_COMMAND[@]}" >"${AUDIT_DIR}/asset_preflight.json"
git rev-parse HEAD >"${AUDIT_DIR}/git_commit.txt"
git status --short >"${AUDIT_DIR}/git_status.txt"
AUDIT_SOURCE_PATHS=(
  pyproject.toml
  accelerate_configs/16_npus_deepspeed_zero2.yaml
  configs/selfless/imagenet100_class_base_80ep_ascend_16npu.yaml
  models/modeling_model/image_flow_loss.py
  models/modeling_model/modeling_selfless_flow.py
  pretrain/merge_flow_latent_shards.py
  pretrain/train_selfless_flow.py
  script/offline_env.sh
  script/selfless/encode_imagenet100_kl16_vae_ascend.sh
  script/selfless/start_imagenet100_class_ascend16_tmux.sh
  script/selfless/pretraining_imagenet100_class_ascend.sh
  script/selfless/validate_imagenet100_class_ascend16_final.sh
  scripts/build_imagenet100_manifests.py
  scripts/imagenet_encode_kl16_vae.py
  scripts/validate_ascend_imagenet100_assets.py
  scripts/validate_ascend_imagenet100_final_run.py
  scripts/validate_kl16_posterior_decode.py
  tests/test_imagenet_flow_cache_epoch.py
  tests/test_npu_operator_optimizations.py
  tests/test_npu_runtime_primitives.py
  utils/imagenet_flow_dataloaders.py
  utils/utils.py
  utils/selfless_training_runtime.py
)
{
  git diff --binary HEAD -- "${AUDIT_SOURCE_PATHS[@]}"
  for source_path in "${AUDIT_SOURCE_PATHS[@]}"; do
    if [[ -f "${source_path}" ]] && ! git ls-files --error-unmatch "${source_path}" >/dev/null 2>&1; then
      git diff --binary --no-index /dev/null "${source_path}" || true
    fi
  done
} >"${AUDIT_DIR}/git_diff.patch"
npu-smi info >"${AUDIT_DIR}/npu_smi.txt"
python - <<'PY' >"${AUDIT_DIR}/software_versions.json"
import json
import platform

import accelerate
import deepspeed
import torch
import torch_npu

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "accelerate": accelerate.__version__,
    "deepspeed": deepspeed.__version__,
}, indent=2, sort_keys=True))
PY
{
  printf 'HCCL_INTRA_ROCE_ENABLE=%q WANDB_MODE=%q' "${HCCL_INTRA_ROCE_ENABLE}" "${WANDB_MODE}"
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} >"${AUDIT_DIR}/launch_command.sh"

set -o pipefail
WANDB_MODE="${WANDB_MODE}" "${COMMAND[@]}" 2>&1 | tee "${AUDIT_DIR}/training.log"
