#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_showo2_maskgit.yaml}"
export PROJECT="${PROJECT:-selfless-flow-showo2-maskgit-ascend16-smoke}"
export GENERATION_STRATEGY="maskgit"
export REPORT_PATH="${REPORT_PATH:-public/datasets/imagenet_full/preparation/showo2_maskgit_smoke_report.json}"
export STATUS_PATH="${STATUS_PATH:-public/datasets/imagenet_full/preparation/showo2_maskgit_smoke.status}"
export MASTER_PORT="${MASTER_PORT:-29632}"

exec bash "${REPO_ROOT}/script/selfless/smoke_imagenet1k_train_val_eval_ascend16.sh"
