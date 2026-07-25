#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

PYTHONPATH=. pytest -q \
  tests/test_flow_head_position_ablation.py \
  tests/test_flow_head_position_protocol.py \
  --disable-warnings

PYTHONPATH=. python scripts/smoke_flow_head_position_ablation.py \
  --output output/flow_head_position_ablation/smoke/cuda_smoke.json
