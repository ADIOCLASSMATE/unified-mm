#!/usr/bin/env bash
# Historical launcher: retained for audit only.
set -euo pipefail
echo "Backbone evaluation launcher is archived; use existing metrics." >&2
exit 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ABLATION_ID="${1:?Usage: $0 E0|E1|E2a|E2b|E2|E3|E4a|E4b|E4|E5|E6a|E6b|E6|E7a|E7b|E7 [training_seed]}"
TRAINING_SEED="${2:-${TRAINING_SEED:-42}}"
ID_LOWER="$(printf '%s' "${ABLATION_ID}" | tr '[:upper:]' '[:lower:]')"
RUN_SLUG="selfless-flow-image-embedder-${ID_LOWER}-seed${TRAINING_SEED}"
RUN_DIR="output/${RUN_SLUG}"
CONFIG="${CONFIG:-${RUN_DIR}/config.yaml}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/hf_model-final-ema}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/fid_is_selected_cfg3p5_ema}"

if [[ -f "${OUTPUT_DIR}/metrics.json" && "${ALLOW_EXISTING_METRICS:-0}" != "1" ]]; then
  echo "Refusing to reuse existing formal metrics: ${OUTPUT_DIR}/metrics.json" >&2
  echo "Move the old result aside, or set ALLOW_EXISTING_METRICS=1 deliberately." >&2
  exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing resolved training config: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Missing EMA checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

CONFIG="${CONFIG}" CHECKPOINT="${CHECKPOINT}" OUTPUT_DIR="${OUTPUT_DIR}" \
  script/ablation/evaluate_imagenet_flow_100c.sh \
    --require_image_embedder_ablation_protocol \
    "${@:3}"
