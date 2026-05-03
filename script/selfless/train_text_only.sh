#!/bin/bash
# =============================================================================
# Text-only Selfless Attention Training Script
# =============================================================================
# Usage:
#   bash train_text_only.sh [NUM_GPUS] [PORT]
# =============================================================================

set -e

NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
PORT=${PORT:-8888}

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_DIR"

if [ "$NUM_GPUS" = "1" ]; then
    echo "=== Single-GPU training mode ==="
    ACCELERATE_CONFIG="accelerate_configs/1_gpu.yaml"
else
    echo "=== ${NUM_GPUS}-GPU DeepSpeed Zero-2 training mode ==="
    ACCELERATE_CONFIG="accelerate_configs/${NUM_GPUS}_gpus_deepspeed_zero2.yaml"
fi

echo "Config: $ACCELERATE_CONFIG"
echo "Port: $PORT"
echo "Project dir: $SCRIPT_DIR"

uv run accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --main_process_port="$PORT" \
    pretrain/train_selfless.py \
    config=configs/selfless/pretraining.yaml
