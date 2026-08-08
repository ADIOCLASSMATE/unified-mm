#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CONFIG="${CONFIG:-configs/selfless/imagenet100_caption_base_40ep_wsd.yaml}"
export CHECKPOINT="${CHECKPOINT:-output/selfless-flow-base-imagenet100-caption-30ep-flowhead-to200ep/hf_model-final-ema}"
export OUTPUT_DIR="${OUTPUT_DIR:-output/selfless-flow-base-imagenet100-caption-200ep-flowhead-fid-is}"

exec "${REPO_ROOT}/script/selfless/evaluate_imagenet_flow.sh" "$@"
