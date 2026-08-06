#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/inspire/hdd/global_user/wanjiaxin-253108030048/code/unified-mm"
OUTPUT_ROOT="${REPO_ROOT}/output/final_caption_kv_cache_probe"
PROMPTS="${REPO_ROOT}/scripts/prompts/imagenet100_caption_and_composition_probe.jsonl"
CHECKPOINT="${REPO_ROOT}/output/selfless-flow-base-imagenet100-caption-80ep/hf_model-final-ema"
CONFIG="${REPO_ROOT}/configs/selfless/imagenet100_caption_base_80ep.yaml"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"

mkdir -p "${OUTPUT_ROOT}/cache_on" "${OUTPUT_ROOT}/cache_off"

run_probe() {
    local cache_mode="$1"
    local gpu_index="$2"
    CUDA_VISIBLE_DEVICES="${gpu_index}" "${PYTHON}" "${REPO_ROOT}/scripts/generate_caption_kv_cache_probe.py" \
        --config "${CONFIG}" \
        --checkpoint "${CHECKPOINT}" \
        --prompts_jsonl "${PROMPTS}" \
        --output_dir "${OUTPUT_ROOT}/cache_${cache_mode}" \
        --cache_mode "${cache_mode}" \
        --model_dtype bf16 \
        --vae_dtype fp32 \
        --sampling_steps "${SAMPLING_STEPS}" \
        --temperature 1.0 \
        --cfg 3.5 \
        --cfg_schedule constant \
        --flow_solver heun \
        --order_strategy spatial_halton \
        --seed 42 \
        >"${OUTPUT_ROOT}/cache_${cache_mode}/run.log" 2>&1
}

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [[ "${GPU_COUNT}" -ge 2 ]]; then
    run_probe on 0 &
    CACHE_ON_PID=$!
    run_probe off 1 &
    CACHE_OFF_PID=$!
    wait "${CACHE_ON_PID}"
    CACHE_ON_STATUS=$?
    wait "${CACHE_OFF_PID}"
    CACHE_OFF_STATUS=$?
else
    run_probe on 0
    CACHE_ON_STATUS=$?
    run_probe off 0
    CACHE_OFF_STATUS=$?
fi

if [[ "${CACHE_ON_STATUS}" -eq 0 && "${CACHE_OFF_STATUS}" -eq 0 ]]; then
    touch "${OUTPUT_ROOT}/SUCCESS"
    exit 0
fi

touch "${OUTPUT_ROOT}/FAILED"
exit 1
