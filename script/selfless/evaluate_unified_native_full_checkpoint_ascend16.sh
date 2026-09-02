#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <native-full-evaluation-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
OUTPUT_ROOT="$2"
PROFILE="${EVAL_PROFILE:-formal}"
CORE_OUTPUT_ROOT="${REUSE_CORE_EVAL_ROOT:-${OUTPUT_ROOT}/core}"
NATIVE_OUTPUT_ROOT="${OUTPUT_ROOT}/pretraining-native-understanding"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: missing executable Python interpreter: ${PYTHON_BIN}" >&2
  exit 3
fi
mkdir -p "${OUTPUT_ROOT}"
STATUS_PATH="${OUTPUT_ROOT}/launcher.status"
printf 'state=RUNNING\nprofile=%s\nstarted_at=%s\nruntime_hashing_enabled=false\n' \
  "${PROFILE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
finish() {
  launch_status=$?
  if (( launch_status == 0 )); then launch_state=SUCCEEDED; else launch_state=FAILED; fi
  printf 'state=%s\nprofile=%s\nexit_code=%s\nfinished_at=%s\nruntime_hashing_enabled=false\n' \
    "${launch_state}" "${PROFILE}" "${launch_status}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
  exit "${launch_status}"
}
trap finish EXIT

if [[ -n "${REUSE_CORE_EVAL_ROOT:-}" ]]; then
  "${PYTHON_BIN}" - "${MODEL_SOURCE}" \
    "${CORE_OUTPUT_ROOT}/full_evaluation_summary.json" "${PROFILE}" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1]).resolve()
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
profile = sys.argv[3]
if summary.get("complete") is not True or summary.get("profile") != profile:
    raise RuntimeError("reused core evaluation is not complete for this profile")
if summary.get("schema") != "unified_full_checkpoint_evaluation_summary_v3":
    raise RuntimeError("reused core evaluation uses an obsolete protocol")
if Path(summary["checkpoint"]).resolve() != checkpoint:
    raise RuntimeError("reused core evaluation belongs to another checkpoint")
if summary.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError("reused core evaluation violates the no-hash contract")
contract = summary.get("dataset_contract", {})
if contract.get("training_split") != "imagenet_train":
    raise RuntimeError("reused core training split is not ImageNet train")
if contract.get("evaluation_split") != "imagenet_val":
    raise RuntimeError("reused core evaluation split is not ImageNet val")
generation = summary.get("generation", {}).get("imagenet_val_t2i", {})
expected_generation = {
    "leaderboard_comparable_to_adm_dit": False,
    "protocol_name": "imagenet_val_fid50k_torch_fidelity_stratified_is",
    "reference_distribution": "imagenet_val_50000",
    "comparison_scope": "same_protocol_only",
}
if any(generation.get(key) != value for key, value in expected_generation.items()):
    raise RuntimeError("reused core generation result uses an obsolete protocol")
if profile == "formal" and (
    generation.get("project_formal_protocol") is not True
    or generation.get("samples") != 50_000
):
    raise RuntimeError("reused core generation result is not formal FID50K")
PY
else
  EVAL_PROFILE="${PROFILE}" \
    "${REPO_ROOT}/script/selfless/evaluate_unified_full_checkpoint_ascend16.sh" \
    "${MODEL_SOURCE}" "${CORE_OUTPUT_ROOT}" \
    2>&1 | tee "${OUTPUT_ROOT}/core.log"
fi

EVAL_PROFILE="${PROFILE}" \
  "${REPO_ROOT}/script/selfless/evaluate_pretraining_native_understanding_ascend16.sh" \
  "${MODEL_SOURCE}" "${NATIVE_OUTPUT_ROOT}" \
  2>&1 | tee "${OUTPUT_ROOT}/pretraining-native-understanding.log"

"${PYTHON_BIN}" scripts/summarize_unified_native_full_evaluation.py \
  --checkpoint "${MODEL_SOURCE}" \
  --core_output_root "${CORE_OUTPUT_ROOT}" \
  --native_output_root "${NATIVE_OUTPUT_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --profile "${PROFILE}" \
  | tee "${OUTPUT_ROOT}/native-full-summary.log"
