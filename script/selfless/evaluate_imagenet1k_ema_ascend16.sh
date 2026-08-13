#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <ema-checkpoint-dir> <evaluation-output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EMA_CHECKPOINT="$1"
EVAL_ROOT="$2"
CONFIG="configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_COUNT=16

set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

if [[ ! -f "${EMA_CHECKPOINT}/checkpoint_complete.json" ]]; then
  echo "ERROR: incomplete checkpoint: ${EMA_CHECKPOINT}" >&2
  exit 3
fi
if [[ ! -f "${EMA_CHECKPOINT}/ema_manifest.json" ]]; then
  echo "ERROR: missing EMA manifest: ${EMA_CHECKPOINT}" >&2
  exit 4
fi

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != 1 || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${VISIBLE_NPUS}" >&2
  exit 5
fi

export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=600
export OMP_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_single_stream_fid_is.py \
  --config "${CONFIG}" \
  --ema_checkpoint "${EMA_CHECKPOINT}" \
  --output_dir "${EVAL_ROOT}" \
  --device npu \
  --model_dtype bf16 \
  --samples 50000 \
  --batch_size 4096 \
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
  --resume_checkpoint_interval_batches 1
