#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/selfless/imagenet100_caption_base_80ep.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/8_gpus_deepspeed_zero2.yaml}"
TRAIN_PORT="${TRAIN_PORT:-29531}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_PROJECT="${RUN_PROJECT:-selfless-flow-base-imagenet100-caption-80ep}"
RUN_NAME="${RUN_NAME:-pure2d-caption-qwen3base-imagenet100-seed42-8xh100-b256-80ep}"
RESUME_FROM="${RESUME_FROM:-none}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-80}"
EXPECTED_TRAINABLE_SCOPE="${EXPECTED_TRAINABLE_SCOPE:-full}"

if [[ "${NUM_GPUS}" != "8" ]]; then
  echo "ERROR: canonical ImageNet-100 base training requires NUM_GPUS=8, got ${NUM_GPUS}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: missing config: ${CONFIG}" >&2
  exit 3
fi
if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "ERROR: missing Accelerate config: ${ACCELERATE_CONFIG}" >&2
  exit 4
fi

PYTHONPATH=. python - "${CONFIG}" "${EXPECTED_EPOCHS}" "${EXPECTED_TRAINABLE_SCOPE}" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
epochs = int(sys.argv[2])
expected_trainable_scope = str(sys.argv[3])
params = config.dataset.params

required = {
    "posterior_cache": Path(params.cache_path),
    "membership": Path(params.manifest_jsonl),
    "split": Path(params.split_manifest_jsonl),
    "captions": Path(params.caption_jsonl),
}
for label, path in required.items():
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")

digest = hashlib.sha256()
with required["captions"].open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual_caption_sha = digest.hexdigest()
expected_caption_sha = str(params.caption_manifest_sha256)
if actual_caption_sha != expected_caption_sha:
    raise RuntimeError(
        f"caption SHA mismatch: {actual_caption_sha} != {expected_caption_sha}"
    )

cache = torch.load(
    required["posterior_cache"],
    map_location="cpu",
    mmap=True,
    weights_only=True,
)
posterior = cache["posterior_stats"]
img_ids = cache["img_ids"]
if tuple(posterior.shape) != (125000, 256, 32):
    raise RuntimeError(f"unexpected posterior shape: {tuple(posterior.shape)}")
if tuple(img_ids.shape) != (125000,):
    raise RuntimeError(f"unexpected img_ids shape: {tuple(img_ids.shape)}")

split_counts = Counter()
with required["split"].open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            split = str(json.loads(line)["split"]).lower()
            split_counts["validation" if split == "val" else split] += 1
if split_counts != Counter({"train": 115000, "validation": 10000}):
    raise RuntimeError(f"unexpected split counts: {dict(split_counts)}")

global_batch = int(config.training.total_batch_size)
steps_per_epoch = split_counts["train"] // global_batch
expected_steps = epochs * steps_per_epoch
if int(config.training.max_train_steps) != expected_steps:
    raise RuntimeError(
        f"max_train_steps={config.training.max_train_steps}, expected {expected_steps}"
    )
if str(params.conditioning_mode) != "caption":
    raise RuntimeError("canonical base training must use caption conditioning")
if str(config.model.backbone_attention_output_gate) != "none":
    raise RuntimeError("canonical base training must not enable an ablation gate")
if str(config.training.get("trainable_scope", "full")) != expected_trainable_scope:
    raise RuntimeError(
        "trainable scope mismatch: "
        f"config={config.training.get('trainable_scope', 'full')!r}, "
        f"expected={expected_trainable_scope!r}"
    )

print(
    "Preflight OK: pure-2D caption base, "
    f"train={split_counts['train']}, val={split_counts['validation']}, "
    f"global_batch={global_batch}, steps/epoch={steps_per_epoch}, "
    f"epochs={epochs}, max_steps={expected_steps}"
)
PY

RUN_ROOT="output/${RUN_PROJECT}"
if [[ "${RESUME_FROM}" == "none" || "${RESUME_FROM}" == "null" || -z "${RESUME_FROM}" ]]; then
  if [[ -d "${RUN_ROOT}" ]] && find "${RUN_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: refusing to overwrite non-empty fresh run directory: ${RUN_ROOT}" >&2
    exit 5
  fi
else
  if [[ ! -f "${RESUME_FROM}/checkpoint_complete.json" ]]; then
    echo "ERROR: RESUME_FROM is not a complete checkpoint: ${RESUME_FROM}" >&2
    exit 6
  fi
fi

COMMAND=(
  accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  --main_process_port "${TRAIN_PORT}"
  --num_processes "${NUM_GPUS}"
  pretrain/train_selfless_flow.py
  "config=${CONFIG}"
  "experiment.project=${RUN_PROJECT}"
  "experiment.name=${RUN_NAME}"
  "experiment.resume_from_checkpoint=${RESUME_FROM}"
)
COMMAND+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY RUN:'
  printf ' %q' env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" WANDB_MODE="${WANDB_MODE}" "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
WANDB_MODE="${WANDB_MODE}" \
exec "${COMMAND[@]}"
