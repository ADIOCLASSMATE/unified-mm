#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 9 ]]; then
  echo "Usage: $0 <checkpoint> <core-root> <native-root> <coco-asset-root> <coco-partition-0> <coco-partition-1> <coco-output-root> <flickr-output-root> <combined-output-root>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="$1"
CORE_ROOT="$2"
NATIVE_ROOT="$3"
COCO_ASSET_ROOT="$4"
COCO_PARTITION_0="$5"
COCO_PARTITION_1="$6"
COCO_OUTPUT_ROOT="$7"
FLICKR_OUTPUT_ROOT="$8"
COMBINED_OUTPUT_ROOT="$9"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
STATUS_PATH="${COMBINED_OUTPUT_ROOT}/finalizer.status"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
for required in \
  "${CORE_ROOT}/full_evaluation_summary.json" \
  "${NATIVE_ROOT}/imagenet-retrieval/manifest.json" \
  "${NATIVE_ROOT}/retained-benchmarks/manifest.json" \
  "${COCO_ASSET_ROOT}/manifest.json" \
  "${COCO_PARTITION_0}/partition_summary.json" \
  "${COCO_PARTITION_1}/partition_summary.json" \
  "${FLICKR_OUTPUT_ROOT}/summary.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing finalization input: ${required}" >&2
    exit 3
  fi
done
if [[ -d "${COCO_OUTPUT_ROOT}" ]] && \
   find "${COCO_OUTPUT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  echo "ERROR: COCO merged output is not empty: ${COCO_OUTPUT_ROOT}" >&2
  exit 4
fi

mkdir -p "${COMBINED_OUTPUT_ROOT}"
printf 'state=RUNNING\nstarted_at=%s\nruntime_hashing_enabled=false\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
finish() {
  finalizer_exit_code=$?
  if (( finalizer_exit_code == 0 )); then
    finalizer_state=SUCCEEDED
  else
    finalizer_state=FAILED
  fi
  printf 'state=%s\nexit_code=%s\nfinished_at=%s\nruntime_hashing_enabled=false\n' \
    "${finalizer_state}" "${finalizer_exit_code}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${STATUS_PATH}"
  exit "${finalizer_exit_code}"
}
trap finish EXIT

TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${PYTHON_BIN}" \
  scripts/merge_cross_dataset_retrieval_partitions.py \
  --asset_root "${COCO_ASSET_ROOT}" \
  --partition_root "${COCO_PARTITION_0}" \
  --partition_root "${COCO_PARTITION_1}" \
  --output_dir "${COCO_OUTPUT_ROOT}"

"${PYTHON_BIN}" scripts/summarize_pretraining_native_understanding.py \
  --checkpoint "${CHECKPOINT}" \
  --imagenet_retrieval_root "${NATIVE_ROOT}/imagenet-retrieval" \
  --coco_retrieval_root "${COCO_OUTPUT_ROOT}" \
  --flickr30k_retrieval_root "${FLICKR_OUTPUT_ROOT}" \
  --benchmark_root "${NATIVE_ROOT}/retained-benchmarks" \
  --output_dir "${NATIVE_ROOT}"

"${PYTHON_BIN}" scripts/summarize_unified_native_full_evaluation.py \
  --checkpoint "${CHECKPOINT}" \
  --core_output_root "${CORE_ROOT}" \
  --native_output_root "${NATIVE_ROOT}" \
  --output_root "${COMBINED_OUTPUT_ROOT}" \
  --profile formal
