#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_positionwise_head.yaml}"
export PROJECT="${PROJECT:-selfless-flow-positionwise-head-ascend16-smoke}"
export GENERATION_STRATEGY="spatial_halton"
export REPORT_PATH="${REPORT_PATH:-public/datasets/imagenet_full/preparation/positionwise_head_smoke_report.json}"
export STATUS_PATH="${STATUS_PATH:-public/datasets/imagenet_full/preparation/positionwise_head_smoke.status}"
export MASTER_PORT="${MASTER_PORT:-29631}"

exec bash "${REPO_ROOT}/script/selfless/smoke_imagenet1k_train_val_eval_ascend16.sh"
