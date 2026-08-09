#!/usr/bin/env bash
set -euo pipefail

# The only supported formal-run entrypoint.  It verifies the full 16-rank
# resource/data/HCCL gate before creating a detached persistent training session.

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

RUN_PROJECT="${RUN_PROJECT:-selfless-flow-im100-class-ascend16-b16ga2-b4e5-f1e4}"
RUN_NAME="${RUN_NAME:-im100-class-80ep-seed42-16x910b-b16ga2-b512}"
TRAIN_PORT="${TRAIN_PORT:-29531}"
RESUME_FROM="${RESUME_FROM:-none}"
TMUX_SOCKET_NAME="${TMUX_SOCKET_NAME:-unified-mm-formal16}"
if [[ -n "${SESSION_NAME:-}" ]]; then
  SESSION_NAME="${SESSION_NAME}"
elif [[ "${RESUME_FROM}" == "none" || "${RESUME_FROM}" == "null" || -z "${RESUME_FROM}" ]]; then
  SESSION_NAME="${RUN_PROJECT}"
else
  SESSION_NAME="${RUN_PROJECT}-resume-$(basename "${RESUME_FROM}")"
fi
WANDB_MODE="${WANDB_MODE:-offline}"
ASCEND_AUDIT_ROOT="${ASCEND_AUDIT_ROOT:-/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/npu-parity-audit}"
FINAL_ACCEPTANCE_JSON="${FINAL_ACCEPTANCE_JSON:-${ASCEND_AUDIT_ROOT}/ASCEND_TRAINING_FINAL_ACCEPTANCE.json}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"

for pair in \
  "RUN_PROJECT:${RUN_PROJECT}" \
  "SESSION_NAME:${SESSION_NAME}" \
  "TMUX_SOCKET_NAME:${TMUX_SOCKET_NAME}"; do
  label="${pair%%:*}"
  value="${pair#*:}"
  if [[ ! "${value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: ${label} must contain only letters, digits, dot, underscore or hyphen: ${value}" >&2
    exit 2
  fi
done
if [[ ! "${TRAIN_PORT}" =~ ^[0-9]+$ \
  || "${TRAIN_PORT}" -lt 1024 \
  || "${TRAIN_PORT}" -gt 65535 ]]; then
  echo "ERROR: TRAIN_PORT must be an integer in [1024, 65535]: ${TRAIN_PORT}" >&2
  exit 3
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is required for the formal persistent run" >&2
  exit 4
fi
TMUX=(tmux -L "${TMUX_SOCKET_NAME}")
if "${TMUX[@]}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "ERROR: refusing duplicate tmux session: ${SESSION_NAME}" >&2
  exit 5
fi

LAUNCHER=(
  bash script/selfless/pretraining_imagenet100_class_ascend.sh
  "$@"
)
COMMON_ENV=(
  env
  RUN_PROJECT="${RUN_PROJECT}"
  RUN_NAME="${RUN_NAME}"
  TRAIN_PORT="${TRAIN_PORT}"
  RESUME_FROM="${RESUME_FROM}"
  WANDB_MODE="${WANDB_MODE}"
  HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE}"
)

# This is intentionally evaluated before creating either the run directory or
# a tmux session so resource and asset failures leave no partial formal run.
DRY_RUN_OUTPUT="$(
  "${COMMON_ENV[@]}" DRY_RUN=1 "${LAUNCHER[@]}"
)"

gate_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
FORMAL_GATE_DIR="${ASCEND_AUDIT_ROOT}/formal-gates/${RUN_PROJECT}-${gate_stamp}"
if [[ -e "${FORMAL_GATE_DIR}" ]]; then
  echo "ERROR: refusing to overwrite formal gate audit: ${FORMAL_GATE_DIR}" >&2
  exit 6
fi
mkdir -p "${FORMAL_GATE_DIR}"
printf '%s\n' "${DRY_RUN_OUTPUT}" >"${FORMAL_GATE_DIR}/asset_and_command_preflight.log"
npu-smi info >"${FORMAL_GATE_DIR}/npu_smi_before_hccl.txt"

HCCL_GATE_COMMAND=(
  torchrun --standalone --nproc_per_node=16
  tests/test_npu_runtime_primitives.py
)
{
  printf 'HCCL_INTRA_ROCE_ENABLE=%q PYTHONPATH=%q' \
    "${HCCL_INTRA_ROCE_ENABLE}" "${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"
  printf ' %q' "${HCCL_GATE_COMMAND[@]}"
  printf '\n'
} >"${FORMAL_GATE_DIR}/hccl_gate_command.sh"
PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}" \
  "${HCCL_GATE_COMMAND[@]}" 2>&1 | tee "${FORMAL_GATE_DIR}/hccl_gate.log"
if ! grep -Fq \
  'PASS world=16 memory_dtype=torch.int64 elapsed_dtype=torch.float32 window_dtype=torch.float32 hccl_all_reduce_sum=136' \
  "${FORMAL_GATE_DIR}/hccl_gate.log"; then
  echo "ERROR: 16-rank HCCL gate did not emit the exact success contract" >&2
  exit 7
fi

TMUX_LAUNCH=(
  "${COMMON_ENV[@]}"
  DRY_RUN=0
  "${LAUNCHER[@]}"
)
printf -v tmux_command '%q ' "${TMUX_LAUNCH[@]}"
FINAL_VALIDATION=(
  bash script/selfless/validate_imagenet100_class_ascend16_final.sh
  --run_root "output/${RUN_PROJECT}"
  --audit_root "${ASCEND_AUDIT_ROOT}"
  --output_json "${FINAL_ACCEPTANCE_JSON}"
)
printf -v validation_command '%q ' "${FINAL_VALIDATION[@]}"
tmux_workflow_command="set -o pipefail; ${tmux_command} && ${validation_command} 2>&1 | tee $(printf '%q' "${FORMAL_GATE_DIR}/final_acceptance.log")"
printf '%s\n' "${tmux_workflow_command}" >"${FORMAL_GATE_DIR}/tmux_launch_command.sh"
printf '%s\n' "${SESSION_NAME}" >"${FORMAL_GATE_DIR}/tmux_session.txt"
printf '%s\n' "${TMUX_SOCKET_NAME}" >"${FORMAL_GATE_DIR}/tmux_socket_name.txt"

"${TMUX[@]}" new-session -d -s "${SESSION_NAME}" -c "${REPO_ROOT}" "${tmux_workflow_command}"
"${TMUX[@]}" set-window-option -t "${SESSION_NAME}:0" remain-on-exit on >/dev/null
if ! "${TMUX[@]}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "ERROR: tmux session disappeared immediately: ${SESSION_NAME}" >&2
  exit 8
fi

echo "Started formal 16-NPU run in tmux session: ${SESSION_NAME}"
echo "Dedicated tmux socket: ${TMUX_SOCKET_NAME}"
echo "Formal gate audit: ${FORMAL_GATE_DIR}"
if [[ "${RESUME_FROM}" == "none" || "${RESUME_FROM}" == "null" || -z "${RESUME_FROM}" ]]; then
  training_audit_dir="prelaunch_audit"
else
  training_audit_dir="prelaunch_audit_resume_$(basename "${RESUME_FROM}")"
fi
echo "Training log: output/${RUN_PROJECT}/${training_audit_dir}/training.log"
echo "Final acceptance JSON: ${FINAL_ACCEPTANCE_JSON}"
