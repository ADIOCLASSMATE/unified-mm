#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export SOURCE_TASK=i2t
export CONFIG=configs/selfless/unified_b_i2t_only_matched_ascend16.yaml
export PROTOCOL=configs/protocols/unified_b_image_only_matched_ascend16.yaml
export RUN_PROJECT="${RUN_PROJECT:-unified-b-x0content-0p6b-i2t-only-bmatched-s42-r1}"
export RUN_NAME="${RUN_NAME:-${RUN_PROJECT}}"
export RESUME_FROM=none
export WANDB_MODE=disabled
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
exec bash script/selfless/pretraining_unified_single_source_0p6b_ascend16.sh
