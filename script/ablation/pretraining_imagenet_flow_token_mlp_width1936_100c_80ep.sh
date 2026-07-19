#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/ablation/imagenet_flow_token_mlp_width1936_100c_80ep.yaml}" \
  bash script/ablation/pretraining_imagenet_flow_100c_80ep.sh "$@"
