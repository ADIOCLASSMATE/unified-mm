#!/usr/bin/env bash
set -euo pipefail
POSITIONWISE_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${POSITIONWISE_SOURCE_ROOT}/script/selfless/pretraining_unified_flow_head_scaling_ascend64.sh" --ablation f "$@"
