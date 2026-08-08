#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source script/offline_env.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STEPS="${STEPS:-40}"
DATE_TAG="${DATE_TAG:-20260807}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/4_gpus_deepspeed_zero2.yaml}"
ORDER="${ORDER:-b64-first}"

run_case() {
    local case_name="$1"
    local port="$2"
    local batch_size="$3"
    local total_batch_size="$4"
    local gradient_checkpointing="$5"
    local run_name="benchmark-im100-class-${case_name}-4xh100-${DATE_TAG}"

    if [[ -e "output/${run_name}" ]]; then
        echo "ERROR: refusing to reuse output/${run_name}" >&2
        return 2
    fi

    accelerate launch \
        --config_file "${ACCELERATE_CONFIG}" \
        --main_process_port "${port}" \
        --num_processes 4 \
        pretrain/train_selfless_flow.py \
        config=configs/selfless/imagenet100_class_base_80ep.yaml \
        "experiment.project=${run_name}" \
        "experiment.name=${run_name}" \
        experiment.resume_from_checkpoint=none \
        experiment.log_every=5 \
        experiment.log_grad_norm_every=1000000000 \
        experiment.save_every=1000000000 \
        experiment.save_hfmodel_every=1000000000 \
        experiment.val_every=1000000000 \
        experiment.validation_image_every=1000000000 \
        experiment.save_final=false \
        lr_scheduler.params.warmup_steps=0 \
        lr_scheduler.params.decay_steps=0 \
        "training.batch_size=${batch_size}" \
        "training.total_batch_size=${total_batch_size}" \
        "training.max_train_steps=${STEPS}" \
        "training.use_gradient_checkpointing=${gradient_checkpointing}"
}

case "${ORDER}" in
    b64-first)
        run_case b64-gc 29541 64 256 true
        run_case b32-nogc 29542 32 128 false
        ;;
    b32-first)
        run_case b32-nogc 29542 32 128 false
        run_case b64-gc 29541 64 256 true
        ;;
    ga2-only)
        run_case b32-nogc-ga2 29543 32 256 false
        ;;
    *)
        echo "ERROR: ORDER must be b64-first, b32-first, or ga2-only, got ${ORDER}" >&2
        exit 2
        ;;
esac
