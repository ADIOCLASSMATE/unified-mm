#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RESUME_STEP="${RESUME_STEP:-}"
if [[ ! "${RESUME_STEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: RESUME_STEP must be a positive integer" >&2
  exit 2
fi
RESUME_ATTEMPT="${RESUME_ATTEMPT:-r1}"
if [[ ! "${RESUME_ATTEMPT}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: RESUME_ATTEMPT contains unsupported characters" >&2
  exit 2
fi

export ABLATION="c"
export RUN_PROJECT="unified-c-0p6b-100b-imagenet-split-s42-r1"
export RUN_NAME="unified-c-qwen3-0.6b-100b-imagenet-split-s42-r1"
export RUN_ROOT="output/${RUN_PROJECT}"
export ALLOW_FORMAL_RESUME="true"
export FORMAL_RESUME_FROM="${RUN_ROOT}/checkpoint-${RESUME_STEP}"
export AUDIT_DIR="${RUN_ROOT}/prelaunch_audit/resume-step-${RESUME_STEP}-${RESUME_ATTEMPT}/node-${PET_NODE_RANK:-unknown}"

exec bash script/selfless/pretraining_unified_ablation_100b_ascend64.sh
