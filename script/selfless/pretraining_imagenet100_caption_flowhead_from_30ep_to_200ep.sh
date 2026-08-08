#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export CONFIG="${CONFIG:-configs/selfless/imagenet100_caption_flowhead_from_30ep_to_200ep.yaml}"
export EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-170}"
export EXPECTED_TRAINABLE_SCOPE="${EXPECTED_TRAINABLE_SCOPE:-image_flow_head}"
export RUN_PROJECT="${RUN_PROJECT:-selfless-flow-base-imagenet100-caption-30ep-flowhead-to200ep}"
export RUN_NAME="${RUN_NAME:-pure2d-caption-imagenet100-flowhead-only-30to200ep-8xh100-b256}"

exec "${REPO_ROOT}/script/selfless/pretraining_imagenet100_caption_base_80ep.sh" "$@"
