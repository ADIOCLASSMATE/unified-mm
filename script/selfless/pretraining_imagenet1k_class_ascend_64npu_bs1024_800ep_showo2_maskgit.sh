#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_showo2_maskgit.yaml}"
export RUN_ROOT="${RUN_ROOT:-output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-showo2-maskgit}"

exec bash "${REPO_ROOT}/script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep.sh"
