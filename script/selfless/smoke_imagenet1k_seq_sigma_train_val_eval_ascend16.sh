#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONFIG="${CONFIG:-configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_seq_sigma.yaml}"
export PROJECT="${PROJECT:-selfless-flow-imagenet1k-seq-sigma-ascend16-train-val-eval-smoke-v2}"
export GENERATION_STRATEGY="sequential"
export REPORT_PATH="${REPORT_PATH:-public/datasets/imagenet_full/preparation/seq_sigma_train_val_eval_smoke_report_v2.json}"
export STATUS_PATH="${STATUS_PATH:-public/datasets/imagenet_full/preparation/seq_sigma_train_val_eval_smoke_v2.status}"
export MASTER_PORT="${MASTER_PORT:-29628}"

exec bash "${REPO_ROOT}/script/selfless/smoke_imagenet1k_train_val_eval_ascend16.sh"
