#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/ablation/imagenet_flow_token_mlp_param_matched_100c_80ep.yaml}" \
CHECKPOINT="${CHECKPOINT:-output/selfless-flow-token-mlp-param-matched-ablation-imagenet100-80ep/hf_model-final-ema}" \
OUTPUT_DIR="${OUTPUT_DIR:-output/selfless-flow-token-mlp-param-matched-ablation-imagenet100-80ep/fid_is_selected_cfg3p5_ema}" \
CFG="3.5" \
  bash script/ablation/evaluate_imagenet_flow_100c.sh "$@"
