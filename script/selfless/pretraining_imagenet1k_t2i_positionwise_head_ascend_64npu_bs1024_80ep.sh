#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-configs/selfless/imagenet1k_t2i_positionwise_head_80ep_ascend_64npu_bs1024.yaml}"
export RUN_PROJECT="${RUN_PROJECT:-selfless-flow-imagenet1k-t2i-positionwise-head-ascend64-b1024-80ep}"
export T2I_VARIANT="positionwise_head"
exec "${REPO_ROOT}/script/selfless/pretraining_imagenet1k_t2i_baseline_ascend_64npu_bs1024_80ep.sh" "$@"
