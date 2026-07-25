#!/usr/bin/env bash
# Historical launcher: retained only to audit completed Q-factor evaluations.
set -euo pipefail
echo "Q-factor evaluation is complete; reuse existing metrics." >&2
exit 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ABLATION_ID="${1:?Usage: $0 E2b-Q1|E2b-Q0|E2-Q1|E2-Q0 43|44|45 [evaluation overrides...]}"
TRAINING_SEED="${2:?Usage: $0 E2b-Q1|E2b-Q0|E2-Q1|E2-Q0 43|44|45 [evaluation overrides...]}"
shift 2

ID_LOWER="$(printf '%s' "${ABLATION_ID}" | tr '[:upper:]' '[:lower:]')"
case "${ID_LOWER}" in
  e2b-q1|e2b-q0|e2-q1|e2-q0) ;;
  *)
    echo "Unknown Q-factor ID: ${ABLATION_ID}" >&2
    exit 2
    ;;
esac
case "${TRAINING_SEED}" in
  43|44|45) ;;
  *)
    echo "Q-factor training seed must be 43, 44, or 45; got ${TRAINING_SEED}" >&2
    exit 2
    ;;
esac

RUN_SLUG="selfless-flow-image-embedder-qf-${ID_LOWER}-seed${TRAINING_SEED}"
RUN_DIR="output/${RUN_SLUG}"
CONFIG="${CONFIG:-${RUN_DIR}/config.yaml}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/hf_model-final-ema}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/fid_is_selected_cfg3p5_ema}"

if [[ -f "${OUTPUT_DIR}/metrics.json" && "${ALLOW_EXISTING_METRICS:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing Q-factor metrics: ${OUTPUT_DIR}/metrics.json" >&2
  echo "Move the result aside, or set ALLOW_EXISTING_METRICS=1 deliberately." >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing resolved Q-factor training config: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Missing Q-factor EMA checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

python scripts/image_mask_position_ablation_matrix.py \
  --id "${ABLATION_ID}" \
  --seed "${TRAINING_SEED}" \
  --validate-config "${CONFIG}"

CONFIG="${CONFIG}" CHECKPOINT="${CHECKPOINT}" OUTPUT_DIR="${OUTPUT_DIR}" SEED=42 \
  script/ablation/evaluate_imagenet_flow_100c.sh \
    --require_image_embedder_ablation_protocol \
    "$@"
