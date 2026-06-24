#!/usr/bin/env bash
# Run smoke training for separate lm_head and unified lm_head on the same
# preprocessed OmniCorpus smoke Arrow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

SMOKE_DEVICES="${SMOKE_DEVICES:-0,1,2,3}"
SMOKE_NUM_GPUS="${SMOKE_NUM_GPUS:-}"
RUN="${RUN:-both}"  # both | dual | unified
WANDB_MODE="${WANDB_MODE:-offline}"
PORT_BASE="${PORT_BASE:-8890}"

DUAL_CONFIG="${DUAL_CONFIG:-configs/selfless/omnicorpus_smoke.yaml}"
UNIFIED_CONFIG="${UNIFIED_CONFIG:-configs/selfless/omnicorpus_smoke_unified.yaml}"

if [ -z "$SMOKE_NUM_GPUS" ]; then
    IFS=',' read -r -a _smoke_device_list <<< "$SMOKE_DEVICES"
    SMOKE_NUM_GPUS="${#_smoke_device_list[@]}"
fi

if [ "$SMOKE_NUM_GPUS" = "1" ]; then
    SMOKE_ACCELERATE_CONFIG="${SMOKE_ACCELERATE_CONFIG:-accelerate_configs/1_gpu.yaml}"
else
    SMOKE_ACCELERATE_CONFIG="${SMOKE_ACCELERATE_CONFIG:-accelerate_configs/${SMOKE_NUM_GPUS}_gpus_deepspeed_zero2.yaml}"
fi

run_one() {
    local name="$1"
    local config="$2"
    local port="$3"

    echo "=== Smoke run: ${name} ==="
    echo "config: ${config}"
    echo "devices: ${SMOKE_DEVICES}"
    echo "num_gpus: ${SMOKE_NUM_GPUS}"
    echo "accelerate_config: ${SMOKE_ACCELERATE_CONFIG}"
    echo "wandb_mode: ${WANDB_MODE}"
    echo

    CUDA_VISIBLE_DEVICES="${SMOKE_DEVICES}" \
    NUM_GPUS="${SMOKE_NUM_GPUS}" \
    ACCELERATE_CONFIG="${SMOKE_ACCELERATE_CONFIG}" \
    CONFIG="${config}" \
    WANDB_MODE="${WANDB_MODE}" \
    PORT="${port}" \
    bash script/selfless/pretraining_omnicorpus.sh
}

case "$RUN" in
    both)
        run_one "dual-head" "$DUAL_CONFIG" "$PORT_BASE"
        run_one "unified-head" "$UNIFIED_CONFIG" "$((PORT_BASE + 1))"
        ;;
    dual)
        run_one "dual-head" "$DUAL_CONFIG" "$PORT_BASE"
        ;;
    unified)
        run_one "unified-head" "$UNIFIED_CONFIG" "$PORT_BASE"
        ;;
    *)
        echo "Unknown RUN=${RUN}. Use RUN=both, RUN=dual, or RUN=unified." >&2
        exit 1
        ;;
esac
