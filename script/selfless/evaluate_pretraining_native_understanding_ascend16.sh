#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <pretraining-native-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
OUTPUT_ROOT="$2"
PROFILE="${EVAL_PROFILE:-formal}"
ASSET_ROOT="${ASSET_ROOT:-public/benchmarks/selfless_multimodal_likelihood_v1}"
CACHE_ROOT="${CACHE_ROOT:-${ASSET_ROOT}/vae_posterior_mar_kl16}"
BENCHMARK_ROOT="${REUSE_BENCHMARK_EVAL_ROOT:-${OUTPUT_ROOT}/retained-benchmarks}"
COCO_ASSET_ROOT="${COCO_RETRIEVAL_ASSET_ROOT:-public/benchmarks/mscoco_karpathy_retrieval_v1}"
FLICKR30K_ASSET_ROOT="${FLICKR30K_RETRIEVAL_ASSET_ROOT:-public/benchmarks/flickr30k_karpathy_retrieval_v1}"
COCO_RETRIEVAL_ROOT="${REUSE_COCO_RETRIEVAL_ROOT:-${OUTPUT_ROOT}/standard-retrieval/mscoco-5k}"
FLICKR30K_RETRIEVAL_ROOT="${REUSE_FLICKR30K_RETRIEVAL_ROOT:-${OUTPUT_ROOT}/standard-retrieval/flickr30k-1k}"
RETAINED_TASKS="mmbench_dev_en,seed_bench_image,sugarcrepe,aro_vg_relation,aro_vg_attribution"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
REUSE_WAIT_SECONDS="${REUSE_WAIT_SECONDS:-86400}"
REUSE_POLL_SECONDS="${REUSE_POLL_SECONDS:-30}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"

for integer in "${REUSE_WAIT_SECONDS}" "${REUSE_POLL_SECONDS}"; do
  if [[ ! "${integer}" =~ ^[0-9]+$ ]] || (( integer <= 0 )); then
    echo "ERROR: invalid reused-result wait integer: ${integer}" >&2
    exit 3
  fi
done

wait_for_reused_retrieval() {
  local root="$1"
  local task="$2"
  local expected_images="$3"
  local expected_captions="$4"
  local waited=0
  while [[ ! -f "${root}/summary.json" || ! -f "${root}/manifest.json" ]]; do
    if [[ -f "${root}/launcher.status" ]] && grep -q '^state=FAILED$' "${root}/launcher.status"; then
      echo "ERROR: reused retrieval failed before completion: ${root}" >&2
      return 1
    fi
    if (( waited >= REUSE_WAIT_SECONDS )); then
      echo "ERROR: timed out waiting for reused retrieval: ${root}" >&2
      return 1
    fi
    sleep "${REUSE_POLL_SECONDS}"
    waited=$(( waited + REUSE_POLL_SECONDS ))
  done
  "${PYTHON_BIN}" - "${MODEL_SOURCE}" "${root}" "${task}" \
    "${expected_images}" "${expected_captions}" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])
expected_task = sys.argv[3]
expected_images = int(sys.argv[4])
expected_captions = int(sys.argv[5])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if manifest.get("complete") is not True:
    raise RuntimeError(f"reused retrieval manifest is incomplete: {root}")
if summary.get("complete_formal_target") is not True:
    raise RuntimeError(f"reused retrieval did not complete its formal target: {root}")
if Path(summary["checkpoint"]).resolve() != checkpoint:
    raise RuntimeError(f"reused retrieval belongs to another checkpoint: {root}")
if summary.get("task") != expected_task or manifest.get("task") != expected_task:
    raise RuntimeError(f"reused retrieval task mismatch: {root}")
if int(summary.get("images", -1)) != expected_images:
    raise RuntimeError(f"reused retrieval image cardinality mismatch: {root}")
if int(summary.get("captions", -1)) != expected_captions:
    raise RuntimeError(f"reused retrieval caption cardinality mismatch: {root}")
if summary.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError(f"reused retrieval violates the no-hash contract: {root}")
if manifest.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError(f"reused retrieval manifest violates the no-hash contract: {root}")
PY
}

EVAL_PROFILE="${PROFILE}" \
  "${REPO_ROOT}/script/selfless/evaluate_pretraining_native_imagenet_ascend16.sh" \
  "${MODEL_SOURCE}" "${OUTPUT_ROOT}/imagenet-retrieval" \
  2>&1 | tee "${OUTPUT_ROOT}/imagenet-retrieval.log"

if [[ -n "${REUSE_BENCHMARK_EVAL_ROOT:-}" ]]; then
  "${PYTHON_BIN}" - "${MODEL_SOURCE}" "${BENCHMARK_ROOT}" \
    "${RETAINED_TASKS}" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])
required = set(sys.argv[3].split(","))
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if manifest.get("complete") is not True:
    raise RuntimeError("reused benchmark evaluation is incomplete")
if Path(manifest["checkpoint"]).resolve() != checkpoint:
    raise RuntimeError("reused benchmark evaluation belongs to another checkpoint")
if manifest.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError("reused benchmark evaluation violates the no-hash contract")
missing = sorted(required - set(summary.get("tasks", {})))
if missing:
    raise RuntimeError(f"reused benchmark evaluation misses retained tasks: {missing}")
PY
else
  ASSET_ROOT="${ASSET_ROOT}" CACHE_ROOT="${CACHE_ROOT}" \
    "${REPO_ROOT}/script/selfless/wait_for_multimodal_likelihood_cache.sh"
  EVAL_PROFILE="${PROFILE}" ASSET_ROOT="${ASSET_ROOT}" CACHE_ROOT="${CACHE_ROOT}" \
  TASKS="${RETAINED_TASKS}" \
    "${REPO_ROOT}/script/selfless/evaluate_multimodal_likelihood_ascend16.sh" \
    "${MODEL_SOURCE}" "${BENCHMARK_ROOT}" \
    2>&1 | tee "${OUTPUT_ROOT}/retained-benchmarks.log"
fi

if [[ -z "${REUSE_COCO_RETRIEVAL_ROOT:-}" ]]; then
  EVAL_PROFILE="${PROFILE}" \
    "${REPO_ROOT}/script/selfless/evaluate_cross_dataset_retrieval_ascend16.sh" \
    "${MODEL_SOURCE}" "${COCO_ASSET_ROOT}" "${COCO_RETRIEVAL_ROOT}" \
    2>&1 | tee "${OUTPUT_ROOT}/mscoco-retrieval.log"
else
  wait_for_reused_retrieval \
    "${COCO_RETRIEVAL_ROOT}" "mscoco_karpathy_test_5k" 5000 25010
fi

if [[ -z "${REUSE_FLICKR30K_RETRIEVAL_ROOT:-}" ]]; then
  EVAL_PROFILE="${PROFILE}" \
    "${REPO_ROOT}/script/selfless/evaluate_cross_dataset_retrieval_ascend16.sh" \
    "${MODEL_SOURCE}" "${FLICKR30K_ASSET_ROOT}" "${FLICKR30K_RETRIEVAL_ROOT}" \
    2>&1 | tee "${OUTPUT_ROOT}/flickr30k-retrieval.log"
else
  wait_for_reused_retrieval \
    "${FLICKR30K_RETRIEVAL_ROOT}" "flickr30k_karpathy_test_1k" 1000 5000
fi

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/summarize_pretraining_native_understanding.py" \
  --checkpoint "${MODEL_SOURCE}" \
  --imagenet_retrieval_root "${OUTPUT_ROOT}/imagenet-retrieval" \
  --coco_retrieval_root "${COCO_RETRIEVAL_ROOT}" \
  --flickr30k_retrieval_root "${FLICKR30K_RETRIEVAL_ROOT}" \
  --benchmark_root "${BENCHMARK_ROOT}" \
  --output_dir "${OUTPUT_ROOT}" \
  | tee "${OUTPUT_ROOT}/summary.log"
