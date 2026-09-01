#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CONFIG="configs/selfless/unified_baseline_1p7b_100b_ascend_64npu.yaml"
export ACCELERATE_CONFIG="accelerate_configs/16_npus_1node_deepspeed_zero2.yaml"
export FORMAL_WORLD_SIZE="64"
export RUN_PROJECT="${RUN_PROJECT:-unified-a-qwen3-1.7b-smoke-ascend16}"
export RUN_ROOT="${RUN_ROOT:-output/${RUN_PROJECT}}"
export ABLATION="a"
export BACKBONE_LR="${BACKBONE_LR:-2.4e-4}"
export FLOW_LR="${FLOW_LR:-6.0e-5}"
export WANDB_MODE="disabled"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

exec bash script/selfless/smoke_unified_baseline_ascend16.sh
