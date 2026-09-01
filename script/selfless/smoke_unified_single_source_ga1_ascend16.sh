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
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SOURCE_TASK="${SOURCE_TASK:-${1:-}}"
CANDIDATE_GA="${CANDIDATE_GA:-1}"
if [[ ! "${CANDIDATE_GA}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CANDIDATE_GA must be a positive integer" >&2
  exit 2
fi

repeat_csv() {
  local value="$1"
  local count="$2"
  local result=""
  local index
  for ((index = 0; index < count; index++)); do
    if [[ -n "${result}" ]]; then
      result+=","
    fi
    result+="${value}"
  done
  printf '%s' "${result}"
}

case "${SOURCE_TASK}" in
  climbmix|text)
    SOURCE_TASK="climbmix"
    CONFIG="${CONFIG:-configs/selfless/unified_single_text_0p6b_100b_ascend16.yaml}"
    if (( 32 % CANDIDATE_GA != 0 )); then
      echo "ERROR: text CANDIDATE_GA must divide 32" >&2
      exit 2
    fi
    MICRO_BATCH=$((32 / CANDIDATE_GA))
    TOTAL_ROWS=512
    SOURCE_SCHEDULE="$(repeat_csv climbmix "${CANDIDATE_GA}")"
    SOURCE_OVERRIDES=(
      "dataset.params.schedule=[${SOURCE_SCHEDULE}]"
      "dataset.params.sources.climbmix.micro_batch_size=${MICRO_BATCH}"
    )
    ;;
  i2t|caption)
    SOURCE_TASK="i2t"
    CONFIG="${CONFIG:-configs/selfless/unified_single_caption_0p6b_100b_ascend16.yaml}"
    if (( 64 % CANDIDATE_GA != 0 )); then
      echo "ERROR: image CANDIDATE_GA must divide 64" >&2
      exit 2
    fi
    MICRO_BATCH=$((64 / CANDIDATE_GA))
    TOTAL_ROWS=1024
    SOURCE_SCHEDULE="$(repeat_csv i2t "${CANDIDATE_GA}")"
    PAD_SCHEDULE="$(repeat_csv 512 "${CANDIDATE_GA}")"
    SOURCE_OVERRIDES=(
      "dataset.params.schedule=[${SOURCE_SCHEDULE}]"
      "dataset.params.sources.i2t.micro_batch_size=${MICRO_BATCH}"
      "dataset.params.sources.i2t.pad_to_length_schedule=[${PAD_SCHEDULE}]"
    )
    ;;
  t2i|image)
    SOURCE_TASK="t2i"
    CONFIG="${CONFIG:-configs/selfless/unified_single_t2i_0p6b_100b_ascend16.yaml}"
    if (( 64 % CANDIDATE_GA != 0 )); then
      echo "ERROR: image CANDIDATE_GA must divide 64" >&2
      exit 2
    fi
    MICRO_BATCH=$((64 / CANDIDATE_GA))
    TOTAL_ROWS=1024
    SOURCE_SCHEDULE="$(repeat_csv t2i "${CANDIDATE_GA}")"
    PAD_SCHEDULE="$(repeat_csv 512 "${CANDIDATE_GA}")"
    SOURCE_OVERRIDES=(
      "dataset.params.schedule=[${SOURCE_SCHEDULE}]"
      "dataset.params.sources.t2i.micro_batch_size=${MICRO_BATCH}"
      "dataset.params.sources.t2i.pad_to_length_schedule=[${PAD_SCHEDULE}]"
    )
    ;;
  *)
    echo "ERROR: set SOURCE_TASK to climbmix, i2t, or t2i" >&2
    exit 2
    ;;
esac

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/16_npus_1node_deepspeed_zero2.yaml}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"
RUN_PROJECT="${RUN_PROJECT:-smoke-ga${CANDIDATE_GA}-${SOURCE_TASK}-ascend16}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-output}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_DIR_BASE}/${RUN_PROJECT}}"
AUDIT_DIR="${AUDIT_DIR:-${RUN_ROOT}/prelaunch_audit}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29611}"
NPROC_PER_NODE=16

if [[ ! "${SMOKE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: SMOKE_STEPS must be a positive integer" >&2
  exit 3
fi
if [[ ! -f "${CONFIG}" || ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: missing CONFIG=${CONFIG} or ACCELERATE_CONFIG=${ACCELERATE_CONFIG}" >&2
  exit 4
fi
if [[ -f "${RUN_ROOT}/config.yaml" ]]; then
  echo "ERROR: refusing to overwrite existing smoke output ${RUN_ROOT}" >&2
  exit 5
fi

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

read -r NPU_AVAILABLE LOCAL_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${LOCAL_NPUS}" != "${NPROC_PER_NODE}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${LOCAL_NPUS}" >&2
  exit 6
fi

mkdir -p "${AUDIT_DIR}"
COMMAND=(
  python scripts/launch_accelerate_multinode.py launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_machines 1
  --num_processes "${NPROC_PER_NODE}"
  --machine_rank 0
  --main_process_ip 127.0.0.1
  --main_process_port "${MAIN_PROCESS_PORT}"
  --rdzv_backend static
  --same_network
  pretrain/train_selfless_flow.py
  "config=${CONFIG}"
  "experiment.project=${RUN_PROJECT}"
  "experiment.name=${RUN_PROJECT}"
  "experiment.output_dir=${OUTPUT_DIR_BASE}"
  "experiment.resume_from_checkpoint=none"
  "experiment.save_every=1000000000"
  "experiment.save_hfmodel_every=1000000000"
  "experiment.save_ema_eval_every=0"
  "experiment.val_every=1000000000"
  "experiment.validation_image_every=1000000000"
  "experiment.validation_i2t_every=1000000000"
  "experiment.log_every=1"
  "experiment.log_grad_norm_every=1"
  "experiment.flow_stats_every=0"
  "experiment.save_final=false"
  "experiment.save_final_checkpoint=false"
  "training.batch_size=${MICRO_BATCH}"
  "training.total_batch_size=${TOTAL_ROWS}"
  "training.gradient_accumulation_steps=${CANDIDATE_GA}"
  "training.stop_after_steps=${SMOKE_STEPS}"
  "${SOURCE_OVERRIDES[@]}"
)
printf '%q ' "${COMMAND[@]}" >"${AUDIT_DIR}/launch_command.sh"
printf '\n' >>"${AUDIT_DIR}/launch_command.sh"

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  WANDB_MODE=disabled \
  "${COMMAND[@]}" 2>&1 | tee "${AUDIT_DIR}/training.log"
