#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <smoke-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE="$1"
OUTPUT_ROOT="$2"
ASSET_ROOT="${ASSET_ROOT:-public/benchmarks/selfless_multimodal_likelihood_v1}"
CONFIG="${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_COUNT=16

set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs" >&2
  exit 3
fi

CACHE_DIR="${OUTPUT_ROOT}/tiny-cache"
EVAL_DIR="${OUTPUT_ROOT}/evaluation"
mkdir -p "${CACHE_DIR}" "${EVAL_DIR}"
TINY_MANIFEST="${OUTPUT_ROOT}/tiny-image-manifest.jsonl"
python - "${ASSET_ROOT}/tasks/mmbench_dev_en.jsonl" \
  "${ASSET_ROOT}/image_manifest.jsonl" "${TINY_MANIFEST}" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

task_path = Path(sys.argv[1])
image_manifest_path = Path(sys.argv[2])
destination = Path(sys.argv[3])
task_rows = [
    json.loads(line)
    for line in task_path.read_text(encoding="utf-8").splitlines()[:2]
]
wanted = {int(row["image_id"]) for row in task_rows}
selected = []
with image_manifest_path.open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            if int(row["img_id"]) in wanted:
                selected.append(row)
if {int(row["img_id"]) for row in selected} != wanted:
    raise RuntimeError("could not resolve smoke task images")
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = None
try:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    temporary = None
finally:
    if temporary is not None:
        temporary.unlink(missing_ok=True)
PY
python scripts/imagenet_encode_kl16_vae.py \
  --source_mode manifest_jsonl \
  --source_manifest_jsonl "${TINY_MANIFEST}" \
  --vae_path public/vae/mar-kl16/kl16.ckpt \
  --vae_module_root public/code/mar \
  --cache_shard_dir "${CACHE_DIR}" \
  --device npu:0 \
  --vae_dtype fp16 \
  --batch_size 2 \
  --num_workers 2 \
  --num_shards 1 \
  --shard_index 0 \
  --overwrite \
  --no_hash

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_multimodal_likelihood_benchmarks.py \
  --config "${CONFIG}" \
  --model_source "${MODEL_SOURCE}" \
  --asset_root "${ASSET_ROOT}" \
  --cache_shard_dir "${CACHE_DIR}" \
  --output_dir "${EVAL_DIR}" \
  --tasks mmbench_dev_en \
  --batch_size_per_rank 1 \
  --lm_head_chunk_tokens 32 \
  --max_length 2048 \
  --limit 2 \
  --seed 424242 \
  --device npu \
  --model_dtype bf16 \
  --image_sigma_order auto \
  --progress_every 2

python - "${EVAL_DIR}/manifest.json" "${EVAL_DIR}/summary.json" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if manifest.get("complete") is not True:
    raise RuntimeError("smoke evaluation is incomplete")
if manifest.get("runtime_hashing_enabled", True) is not False:
    raise RuntimeError("smoke evaluation violates the no-hash contract")
task = summary["tasks"]["mmbench_dev_en"]
if int(task["metrics"]["records"]) != 2:
    raise RuntimeError("smoke evaluation did not score two MMBench rows")
print(
    json.dumps(
        {
            "status": "ok",
            "checkpoint_step": summary["checkpoint_step"],
            "records": task["metrics"]["records"],
            "runtime_hashing_enabled": False,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
