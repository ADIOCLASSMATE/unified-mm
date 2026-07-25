#!/usr/bin/env bash
# Historical launcher: Q1 must be reused and no backbone runs may be resubmitted.
set -euo pipefail
echo "Q-factor training is complete; reuse existing Q1/Q0 runs instead of resubmitting." >&2
exit 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/script/offline_env.sh"

ABLATION_ID="${1:?Usage: $0 E2b-Q1|E2b-Q0|E2-Q1|E2-Q0 43|44|45 [training overrides...]}"
TRAINING_SEED="${2:?Usage: $0 E2b-Q1|E2b-Q0|E2-Q1|E2-Q0 43|44|45 [training overrides...]}"
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
if [[ ( -e "${RUN_DIR}" || -L "${RUN_DIR}" ) && "${ALLOW_EXISTING_RUN_DIR:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing Q-factor training directory: ${RUN_DIR}" >&2
  echo "Move it aside, or set ALLOW_EXISTING_RUN_DIR=1 deliberately." >&2
  exit 2
fi

CONFIG_DIR="${CONFIG_DIR:-output/image_mask_position_ablation/configs}"
CONFIG_PATH="${CONFIG_DIR}/${ID_LOWER}-seed${TRAINING_SEED}.yaml"
BASE_CONFIG="${BASE_CONFIG:-configs/ablation/imagenet_flow_image_embedder_100c_80ep.yaml}"
PARENT_SUMMARY_JSON="${PARENT_SUMMARY_JSON:-output/image_embedder_ablation/confirmation_d1_summary.json}"

python scripts/image_mask_position_ablation_matrix.py \
  --id "${ABLATION_ID}" \
  --seed "${TRAINING_SEED}" \
  --base-config "${BASE_CONFIG}" \
  --parent-summary-json "${PARENT_SUMMARY_JSON}" \
  --output "${CONFIG_PATH}"

CONFIG="${CONFIG_PATH}" \
  script/ablation/pretraining_imagenet_flow_100c_80ep.sh "$@"
