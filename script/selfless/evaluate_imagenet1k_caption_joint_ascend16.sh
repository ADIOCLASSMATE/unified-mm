#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -f "${CANN_SET_ENV}" ]]; then
  echo "ERROR: missing CANN environment script: ${CANN_SET_ENV}" >&2
  exit 1
fi
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

EVAL_ONLY="${EVAL_ONLY:-both}"
CONFIG="${CONFIG:-configs/selfless/imagenet1k_caption_joint_10ep_ascend16_b1024.yaml}"
RUN_PROJECT="${RUN_PROJECT:-selfless-flow-imagenet1k-caption-joint}"
RUN_ROOT="output/${RUN_PROJECT}"
MODEL_SUBDIR="${MODEL_SUBDIR:-hf_model-final-ema}"
EVAL_SUBDIR="${EVAL_SUBDIR:-generation-evaluation}"
MODEL_PATH="${RUN_ROOT}/${MODEL_SUBDIR}"
EVAL_ROOT="${RUN_ROOT}/${EVAL_SUBDIR}"
I2T_ROOT="${EVAL_ROOT}/i2t-clip"
T2I_ROOT="${EVAL_ROOT}/t2i-fid-is"
CLIP_MODEL="${CLIP_MODEL:-public/models/openai--clip-vit-base-patch32}"
EXPECTED_CLIP_WEIGHT_SHA256="a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
NPU_COUNT=16

if [[ ! "${RUN_PROJECT}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "ERROR: unsafe RUN_PROJECT=${RUN_PROJECT}" >&2
  exit 7
fi
if [[ ! "${MODEL_SUBDIR}" =~ ^[a-zA-Z0-9._/-]+$ || "${MODEL_SUBDIR}" == /* || "${MODEL_SUBDIR}" == *..* ]]; then
  echo "ERROR: unsafe MODEL_SUBDIR=${MODEL_SUBDIR}" >&2
  exit 8
fi
if [[ ! "${EVAL_SUBDIR}" =~ ^[a-zA-Z0-9._/-]+$ || "${EVAL_SUBDIR}" == /* || "${EVAL_SUBDIR}" == *..* ]]; then
  echo "ERROR: unsafe EVAL_SUBDIR=${EVAL_SUBDIR}" >&2
  exit 9
fi
if [[ "${EVAL_ONLY}" != "both" && "${EVAL_ONLY}" != "i2t" && "${EVAL_ONLY}" != "t2i" ]]; then
  echo "ERROR: EVAL_ONLY must be both, i2t, or t2i; got ${EVAL_ONLY}" >&2
  exit 3
fi
for required in \
  "${CONFIG}" \
  "${MODEL_PATH}/config.json" \
  "${MODEL_PATH}/model.safetensors" \
  "${MODEL_PATH}/tokenizer.json" \
  "${CLIP_MODEL}/config.json" \
  "${CLIP_MODEL}/pytorch_model.bin" \
  public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth \
  public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing evaluation asset: ${required}" >&2
    exit 4
  fi
done

ACTUAL_CLIP_WEIGHT_SHA256="$(sha256sum "${CLIP_MODEL}/pytorch_model.bin" | awk '{print $1}')"
if [[ "${ACTUAL_CLIP_WEIGHT_SHA256}" != "${EXPECTED_CLIP_WEIGHT_SHA256}" ]]; then
  echo "ERROR: CLIP weight SHA256 mismatch: ${ACTUAL_CLIP_WEIGHT_SHA256}" >&2
  exit 5
fi

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${VISIBLE_NPUS}" >&2
  exit 6
fi

export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF
mkdir -p "${EVAL_ROOT}/prelaunch_audit"

python - "${RUN_PROJECT}" "${MODEL_PATH}" "${CLIP_MODEL}" >"${EVAL_ROOT}/prelaunch_audit/assets.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

run_project, model_path, clip_model = sys.argv[1:]
payload = {
    "schema": "selfless_imagenet1k_caption_joint_generation_preflight_v1",
    "run_project": run_project,
    "model": {
        "path": model_path,
        "weights_sha256": sha256(Path(model_path) / "model.safetensors"),
        "config_sha256": sha256(Path(model_path) / "config.json"),
    },
    "clip": {
        "path": clip_model,
        "weights_sha256": sha256(Path(clip_model) / "pytorch_model.bin"),
        "config_sha256": sha256(Path(clip_model) / "config.json"),
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

run_i2t() {
  env \
    -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
    torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
    scripts/evaluate_imagenet1k_i2t_clip.py \
    --config "${CONFIG}" \
    --model_path "${MODEL_PATH}" \
    --clip_model_dir "${CLIP_MODEL}" \
    --output_dir "${I2T_ROOT}" \
    --samples 1000 \
    --batch_size_per_rank 4 \
    --clip_batch_size_per_rank 16 \
    --max_new_tokens 96 \
    --temperature 0 \
    --seed 424242 \
    --device npu \
    --model_dtype bf16 \
    2>&1 | tee "${EVAL_ROOT}/i2t-clip.log"
}

run_t2i() {
  env \
    -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
    -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
    torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
    scripts/evaluate_single_stream_fid_is.py \
    --config "${CONFIG}" \
    --model_path_override "${MODEL_PATH}" \
    --output_dir "${T2I_ROOT}" \
    --device npu \
    --model_dtype bf16 \
    --samples 50000 \
    --batch_size 4096 \
    --split val \
    --caption_sequence_mode t2i \
    --sampling_steps 100 \
    --temperature 1.0 \
    --cfg 3.5 \
    --cfg_schedule constant \
    --flow_solver heun \
    --parallel_rate 1 \
    --strategies spatial_halton \
    --vae_dtype fp32 \
    --vae_decode_batch_size 16 \
    --inception_weights_path public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth \
    --real_stats_path public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt \
    --skip_target_decode \
    --require_official_protocol \
    --canonical_pairing \
    --resume_progress \
    --resume_checkpoint_interval_batches 1 \
    2>&1 | tee "${EVAL_ROOT}/t2i-fid-is.log"
}

if [[ "${EVAL_ONLY}" == "both" || "${EVAL_ONLY}" == "i2t" ]]; then
  run_i2t
fi
if [[ "${EVAL_ONLY}" == "both" || "${EVAL_ONLY}" == "t2i" ]]; then
  run_t2i
fi
