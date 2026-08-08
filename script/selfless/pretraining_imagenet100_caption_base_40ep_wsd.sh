#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CONFIG="${CONFIG:-configs/selfless/imagenet100_caption_base_40ep_wsd.yaml}"
export EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-40}"
export EXPECTED_TRAINABLE_SCOPE="${EXPECTED_TRAINABLE_SCOPE:-full}"
export RUN_PROJECT="${RUN_PROJECT:-selfless-flow-base-imagenet100-caption-40ep-wsd}"
export RUN_NAME="${RUN_NAME:-pure2d-caption-qwen3base-imagenet100-seed42-8xh100-b256-40ep-wsd}"

exec "${REPO_ROOT}/script/selfless/pretraining_imagenet100_caption_base_80ep.sh" "$@"
