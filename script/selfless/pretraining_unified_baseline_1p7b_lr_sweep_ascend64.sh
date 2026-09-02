#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NODE_RANK="${PET_NODE_RANK:-}"
NUM_MACHINES="${PET_NNODES:-}"
SUITE_RUN_ID="${SUITE_RUN_ID:-r1}"
SWEEP_ROOT="output/unified-b-1p7b-lr-sweep-1b-s42"
STATE_ROOT="${SWEEP_ROOT}/suite_state/${SUITE_RUN_ID}"
MANIFEST="configs/protocols/unified_baseline_1p7b_lr_sweep_1b_ascend64.yaml"
BARRIER_TIMEOUT_SECONDS="${BARRIER_TIMEOUT_SECONDS:-900}"
ARMS=(
  b18e5-f4e5 b18e5-f5e5 b18e5-f6e5
  b21e5-f4e5 b21e5-f5e5 b21e5-f6e5
  b24e5-f4e5 b24e5-f5e5 b24e5-f6e5
)

if [[ ! "${NODE_RANK}" =~ ^[0-3]$ || "${NUM_MACHINES}" != "4" ]]; then
  echo "ERROR: this suite requires PET_NODE_RANK in [0,3] and PET_NNODES=4" >&2
  exit 2
fi
if [[ ! "${SUITE_RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: invalid SUITE_RUN_ID" >&2
  exit 2
fi
if [[ ! "${BARRIER_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: BARRIER_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi

wait_for_file() {
  local path="$1"
  local label="$2"
  local started now
  started="$(date +%s)"
  while [[ ! -f "${path}" ]]; do
    now="$(date +%s)"
    if (( now - started >= BARRIER_TIMEOUT_SECONDS )); then
      echo "ERROR: timed out waiting for ${label}" >&2
      return 1
    fi
    sleep 2
  done
}

wait_for_all_nodes() {
  local directory="$1"
  local started now count
  started="$(date +%s)"
  while true; do
    count="$(find "${directory}" -maxdepth 1 -type f -name 'node-*.done' | wc -l)"
    if [[ "${count}" == "4" ]]; then
      return 0
    fi
    now="$(date +%s)"
    if (( now - started >= BARRIER_TIMEOUT_SECONDS )); then
      echo "ERROR: timed out waiting for all four nodes; observed ${count}" >&2
      return 1
    fi
    sleep 2
  done
}

mkdir -p "${STATE_ROOT}"
for ARM_ID in "${ARMS[@]}"; do
  ARM_ROOT="${SWEEP_ROOT}/${ARM_ID}"
  ARM_CHECKPOINT="${ARM_ROOT}/checkpoint-955"
  ARM_STATE="${STATE_ROOT}/${ARM_ID}"
  mkdir -p "${ARM_STATE}"

  if [[ -f "${ARM_CHECKPOINT}/checkpoint_complete.json" ]]; then
    echo "Sweep arm ${ARM_ID} is already complete; validating it during selection."
  else
    if [[ "${NODE_RANK}" == "0" && -f "${ARM_ROOT}/config.yaml" ]]; then
      echo "ERROR: ${ARM_ID} has a partial run without checkpoint-955; refusing overwrite" >&2
      exit 3
    fi
    echo "Starting 1.7B sweep arm ${ARM_ID} on node rank ${NODE_RANK}."
    ARM_ID="${ARM_ID}" WANDB_MODE=disabled \
      bash script/selfless/pretraining_unified_baseline_1p7b_lr_sweep_arm_ascend64.sh
    if [[ ! -f "${ARM_CHECKPOINT}/checkpoint_complete.json" ]]; then
      echo "ERROR: ${ARM_ID} returned without a complete checkpoint-955" >&2
      exit 4
    fi
  fi

  printf 'done\n' >"${ARM_STATE}/node-${NODE_RANK}.done"
  if [[ "${NODE_RANK}" == "0" ]]; then
    wait_for_all_nodes "${ARM_STATE}"
    printf 'ready\n' >"${ARM_STATE}/all-nodes.done"
  else
    wait_for_file "${ARM_STATE}/all-nodes.done" "${ARM_ID} node barrier"
  fi
done

SELECTION="${SWEEP_ROOT}/lr_selection.json"
if [[ "${NODE_RANK}" == "0" ]]; then
  python scripts/select_unified_lr_sweep.py \
    --manifest "${MANIFEST}" \
    --output "${SELECTION}" \
    >"${STATE_ROOT}/selection_report.json"
  printf 'ready\n' >"${STATE_ROOT}/selection.done"
else
  wait_for_file "${STATE_ROOT}/selection.done" "LR selection"
fi

if [[ ! -f "${SELECTION}" ]]; then
  echo "ERROR: LR selection is missing after the sweep barrier" >&2
  exit 5
fi
echo "All nine 1.7B/64-card arms are complete and the winner is selected."
echo "Formal 1.7B continuation is intentionally not auto-started."
