#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/selfless/imagenet100_class_base_80ep.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/8_gpus_deepspeed_zero2.yaml}"
TRAIN_PORT="${TRAIN_PORT:-29531}"
WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
BACKBONE_LR="${BACKBONE_LR:-4e-5}"
FLOW_HEAD_LR="${FLOW_HEAD_LR:-1e-4}"
RUN_PROJECT="${RUN_PROJECT:-selfless-flow-im100-class-lr80-b32ga2-nogc-b4e5-f1e4}"
RUN_NAME="${RUN_NAME:-im100-class-lr80-b32ga2-nogc-b4e5-f1e4-seed42-8xh100-b512}"
RESUME_FROM="${RESUME_FROM:-none}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_EPOCHS="${EXPECTED_EPOCHS:-80}"

if [[ "${NUM_GPUS}" != "8" ]]; then
  echo "ERROR: ImageNet-100 class LR tuning requires NUM_GPUS=8, got ${NUM_GPUS}" >&2
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

PYTHONPATH=. python - \
  "${CONFIG}" "${EXPECTED_EPOCHS}" "${NUM_GPUS}" \
  "${BACKBONE_LR}" "${FLOW_HEAD_LR}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
epochs = int(sys.argv[2])
world_size = int(sys.argv[3])
backbone_lr = float(sys.argv[4])
flow_head_lr = float(sys.argv[5])
if backbone_lr <= 0 or flow_head_lr <= 0:
    raise ValueError(
        f"learning rates must be positive: backbone={backbone_lr}, flow_head={flow_head_lr}"
    )

params = config.dataset.params
required = {
    "posterior_cache": Path(params.cache_path),
    "membership": Path(params.manifest_jsonl),
    "split": Path(params.split_manifest_jsonl),
    "synset_mapping": Path(params.synset_mapping_path),
}
for label, path in required.items():
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")

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

membership_counts = Counter()
with required["membership"].open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            membership_counts[str(json.loads(line)["synset"])] += 1
if len(membership_counts) != 100 or set(membership_counts.values()) != {1250}:
    raise RuntimeError(
        "expected exactly 100 ImageNet classes with 1250 images each, got "
        f"classes={len(membership_counts)}, counts={sorted(set(membership_counts.values()))}"
    )

split_counts = Counter()
with required["split"].open(encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            split = str(json.loads(line)["split"]).lower()
            split_counts["validation" if split == "val" else split] += 1
if split_counts != Counter({"train": 115000, "validation": 10000}):
    raise RuntimeError(f"unexpected split counts: {dict(split_counts)}")

batch_size = int(config.training.batch_size)
global_batch = int(config.training.total_batch_size)
if batch_size != 32:
    raise RuntimeError(f"training.batch_size must be 32 per GPU, got {batch_size}")
global_micro_batch = batch_size * world_size
if global_batch % global_micro_batch:
    raise RuntimeError(
        "global batch must be divisible by batch_size * world_size: "
        f"{global_batch} % ({batch_size} * {world_size}) != 0"
    )
gradient_accumulation_steps = global_batch // global_micro_batch
if gradient_accumulation_steps != 2:
    raise RuntimeError(
        "ImageNet-100 class training requires gradient accumulation 2, got "
        f"{gradient_accumulation_steps} from global_batch={global_batch}, "
        f"batch_size={batch_size}, world_size={world_size}"
    )
steps_per_epoch = split_counts["train"] // global_batch
expected_steps = epochs * steps_per_epoch
if int(config.training.max_train_steps) != expected_steps:
    raise RuntimeError(
        f"max_train_steps={config.training.max_train_steps}, expected {expected_steps}"
    )
if str(params.conditioning_mode) != "class":
    raise RuntimeError("ImageNet-100 class LR tuning must use conditioning_mode=class")
if params.get("caption_jsonl", None):
    raise RuntimeError("class LR tuning must not read a caption manifest")
if str(config.model.backbone_attention_output_gate) != "none":
    raise RuntimeError("the canonical architecture must not enable an ablation gate")
if bool(config.training.use_gradient_checkpointing):
    raise RuntimeError(
        "B32 + GA2 class training requires training.use_gradient_checkpointing=false"
    )

print(
    "Preflight OK: ImageNet-100 class mode, "
    f"train={split_counts['train']}, val={split_counts['validation']}, "
    f"batch/GPU={batch_size}, global_batch={global_batch}, "
    f"gradient_accumulation_steps={gradient_accumulation_steps}, "
    f"epochs={epochs}, max_steps={expected_steps}, "
    f"backbone_lr={backbone_lr:g}, flow_head_lr={flow_head_lr:g}"
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
  "optimizer.params.learning_rate=${FLOW_HEAD_LR}"
  "optimizer.params.backbone_learning_rate=${BACKBONE_LR}"
  "optimizer.params.projector_learning_rate=${FLOW_HEAD_LR}"
  "optimizer.params.flow_learning_rate=${FLOW_HEAD_LR}"
  "optimizer.params.special_token_learning_rate=${BACKBONE_LR}"
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
