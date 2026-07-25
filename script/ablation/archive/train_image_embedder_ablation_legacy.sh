#!/usr/bin/env bash
# Historical launcher: retained for audit only. Do not submit new backbone runs.
set -euo pipefail
echo "Backbone training matrix is archived; new runs are intentionally disabled." >&2
exit 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/script/offline_env.sh"

ABLATION_ID="${1:?Usage: $0 E0|E1|E2a|E2b|E2|E3|E4a|E4b|E4|E5|E6a|E6b|E6|E7a|E7b|E7 [seed]}"
SEED="${2:-${SEED:-42}}"
ID_LOWER="$(printf '%s' "${ABLATION_ID}" | tr '[:upper:]' '[:lower:]')"
CONFIG_DIR="${CONFIG_DIR:-output/image_embedder_ablation/configs}"
CONFIG_PATH="${CONFIG_DIR}/${ID_LOWER}-seed${SEED}.yaml"
BASE_CONFIG="${BASE_CONFIG:-configs/ablation/imagenet_flow_image_embedder_100c_80ep.yaml}"
CONFIRMATION_SCREEN_JSON="${CONFIRMATION_SCREEN_JSON:-}"

MATRIX_ARGS=(
  --id "${ABLATION_ID}"
  --seed "${SEED}"
  --base-config "${BASE_CONFIG}"
  --output "${CONFIG_PATH}"
)
if [[ -n "${CONFIRMATION_SCREEN_JSON}" ]]; then
  MATRIX_ARGS+=(--confirmation-screen-json "${CONFIRMATION_SCREEN_JSON}")
  RUN_DIR="output/selfless-flow-image-embedder-${ID_LOWER}-seed${SEED}"
  if [[ -d "${RUN_DIR}" && -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite existing confirmation run directory: ${RUN_DIR}" >&2
    exit 2
  fi
fi

python scripts/image_embedder_ablation_matrix.py "${MATRIX_ARGS[@]}"

CONFIG="${CONFIG_PATH}" \
  script/ablation/pretraining_imagenet_flow_100c_80ep.sh "${@:3}"
