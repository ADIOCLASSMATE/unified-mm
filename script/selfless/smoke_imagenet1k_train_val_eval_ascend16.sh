#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
set +u
source "${CANN_SET_ENV}"
set -u
export UNIFIED_MM_VENV="${UNIFIED_MM_VENV:-.venv}"
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"

CONFIG="configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml"
ACCELERATE_CONFIG="accelerate_configs/64_npus_4nodes_deepspeed_zero2.yaml"
PROJECT="${PROJECT:-selfless-flow-imagenet1k-ascend16-train-val-eval-smoke}"
RUN_ROOT="output/${PROJECT}"
EVAL_ROOT="output/${PROJECT}-fid-is"
REPORT_PATH="${REPORT_PATH:-public/datasets/imagenet_full/preparation/train_val_eval_smoke_report.json}"
STATUS_PATH="${STATUS_PATH:-public/datasets/imagenet_full/preparation/train_val_eval_smoke.status}"
MASTER_PORT="${MASTER_PORT:-29617}"
NPU_COUNT=16

if [[ -e "${RUN_ROOT}" || -e "${EVAL_ROOT}" ]]; then
  echo "ERROR: refusing to overwrite an existing smoke output" >&2
  exit 2
fi
mkdir -p "$(dirname "${REPORT_PATH}")"
printf 'RUNNING\n' >"${STATUS_PATH}"
record_exit() {
  exit_code=$?
  if [[ "${exit_code}" == "0" ]]; then
    printf 'SUCCEEDED\n' >"${STATUS_PATH}"
  else
    printf 'FAILED exit_code=%s\n' "${exit_code}" >"${STATUS_PATH}"
  fi
}
trap record_exit EXIT

read -r NPU_AVAILABLE VISIBLE_NPUS <<< "$(python - <<'PY'
import torch
import torch_npu  # noqa: F401
print(int(torch.npu.is_available()), torch.npu.device_count())
PY
)"
if [[ "${NPU_AVAILABLE}" != "1" || "${VISIBLE_NPUS}" != "${NPU_COUNT}" ]]; then
  echo "ERROR: expected 16 visible NPUs, got available=${NPU_AVAILABLE}, count=${VISIBLE_NPUS}" >&2
  exit 3
fi

python - "${MASTER_PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit(f"smoke rendezvous port is already in use: {port}")
PY

export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=600
export WANDB_MODE=offline
export OMP_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF

env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  python scripts/launch_accelerate_multinode.py launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines 1 \
  --num_processes "${NPU_COUNT}" \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT}" \
  --rdzv_backend static \
  --same_network \
  pretrain/train_selfless_flow.py \
  "config=${CONFIG}" \
  "experiment.project=${PROJECT}" \
  "experiment.name=${PROJECT}" \
  experiment.save_every=1 \
  experiment.log_every=1 \
  experiment.log_grad_norm_every=1 \
  experiment.checkpoints_total_limit=1 \
  experiment.save_ema_eval_every=1 \
  experiment.save_hfmodel_every=1000000000 \
  experiment.save_final=true \
  experiment.val_every=1 \
  experiment.validation_image_every=1 \
  experiment.validation_image_samples=1 \
  experiment.validation_flow_cfg=1.0 \
  experiment.validation_flow_solver=euler \
  experiment.validation_flow_probe_times='[0.5]' \
  experiment.validation_save_debug_images=true \
  experiment.validation_single_stream_parallel_rate=1 \
  model.image_flow_num_sampling_steps=2 \
  model.image_flow_batch_mul=1 \
  model.image_flow_solver=euler \
  dataset.params.max_samples=32 \
  dataset.params.split_strategy=shuffle \
  dataset.params.val_ratio=0.5 \
  dataset.params.val_samples_per_class=null \
  training.batch_size=1 \
  training.total_batch_size=16 \
  training.samples_per_epoch=16 \
  training.optimizer_steps_per_epoch=1 \
  training.num_train_epochs=1 \
  training.max_train_steps=1 \
  lr_scheduler.params.warmup_steps=0 \
  lr_scheduler.params.decay_steps=1

TRAIN_CONFIG="${RUN_ROOT}/config.yaml"
EMA_MODEL="${RUN_ROOT}/hf_model-final-ema"
EMA_EVAL_MODEL="${RUN_ROOT}/hf_model-1-ema-eval"
python - "${RUN_ROOT}" "${EMA_MODEL}" "${EMA_EVAL_MODEL}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
ema_model = Path(sys.argv[2])
ema_eval_model = Path(sys.argv[3])
checkpoint = json.loads((run_root / "checkpoint-1/checkpoint_complete.json").read_text())
runtime = json.loads((run_root / "training_runtime_metrics.json").read_text())
validation = json.loads((run_root / "validation_metrics_step_1.json").read_text())
if checkpoint.get("global_step") != 1 or runtime.get("global_step") != 1:
    raise SystemExit("training/checkpoint smoke did not reach global_step=1")
if runtime.get("steps_this_run") != 1:
    raise SystemExit("training runtime did not record exactly one optimizer step")
if not math.isfinite(float(validation["metrics"]["val/loss_image_flow"])):
    raise SystemExit("validation loss is not finite")
if not (ema_model / "model.safetensors").is_file():
    raise SystemExit("final EMA HF model was not exported")
if not (ema_eval_model / "model.safetensors").is_file():
    raise SystemExit("periodic EMA evaluation model was not exported")
if not (run_root / "image_flow_adapter-final.pt").is_file():
    raise SystemExit("final image-flow adapter was not exported")
if (run_root / "image_flow_adapter-1.pt").exists():
    raise SystemExit("intermediate image-flow adapter should be disabled")
ema_eval_metadata = json.loads(
    (ema_eval_model / "ema_export_metadata.json").read_text()
)
if (
    ema_eval_metadata.get("source_global_step") != 1
    or ema_eval_metadata.get("floating_dtype") != "bfloat16"
    or ema_eval_metadata.get("export_kind") != "evaluation"
):
    raise SystemExit(f"invalid periodic EMA evaluation export: {ema_eval_metadata}")
PY

torchrun --standalone --nproc_per_node="${NPU_COUNT}" \
  scripts/evaluate_single_stream_fid_is.py \
  --config "${TRAIN_CONFIG}" \
  --model_path_override "${EMA_MODEL}" \
  --output_dir "${EVAL_ROOT}" \
  --device npu \
  --model_dtype bf16 \
  --samples 16 \
  --batch_size 16 \
  --sampling_steps 2 \
  --temperature 1.0 \
  --cfg 1.0 \
  --cfg_schedule constant \
  --flow_solver euler \
  --parallel_rate 1 \
  --strategies spatial_halton \
  --vae_dtype fp32 \
  --vae_decode_batch_size 1 \
  --inception_weights_path public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth \
  --real_stats_path public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt \
  --skip_target_decode \
  --allow_nonofficial_fid \
  --canonical_pairing

python - "${RUN_ROOT}" "${EVAL_ROOT}" "${REPORT_PATH}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_root, eval_root, report_path = map(Path, sys.argv[1:])
runtime = json.loads((run_root / "training_runtime_metrics.json").read_text())
validation = json.loads((run_root / "validation_metrics_step_1.json").read_text())
evaluation = json.loads((eval_root / "metrics.json").read_text())
strategy = evaluation["strategies"]["spatial_halton"]
required = {
    "fid": strategy["fid"],
    "inception_score_mean": strategy["inception_score_mean"],
    "inception_score_std": strategy["inception_score_std"],
}
if evaluation.get("samples_evaluated") != 16:
    raise SystemExit("evaluation did not process all 16 smoke samples")
if any(not math.isfinite(float(value)) for value in required.values()):
    raise SystemExit(f"non-finite smoke evaluation metric: {required}")
report = {
    "status": "ok",
    "training": runtime,
    "validation": validation,
    "evaluation": {
        "samples_evaluated": evaluation["samples_evaluated"],
        "real_source": evaluation["real_source"],
        "target_decode_skipped": evaluation["target_decode_skipped"],
        "strategy": "spatial_halton",
        **required,
        "generation_step_max": strategy["generation_step_max"],
    },
    "artifacts": {
        "checkpoint_complete": str(run_root / "checkpoint-1/checkpoint_complete.json"),
        "raw_hf_model": str(run_root / "hf_model-final"),
        "ema_hf_model": str(run_root / "hf_model-final-ema"),
        "periodic_ema_eval_model": str(run_root / "hf_model-1-ema-eval"),
        "final_image_flow_adapter": str(run_root / "image_flow_adapter-final.pt"),
        "evaluation_metrics": str(eval_root / "metrics.json"),
    },
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
