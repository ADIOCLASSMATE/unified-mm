#!/usr/bin/env bash
set -euo pipefail
SCALING_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCALING_REPORT_ROOT="${1:?provide the shared study report directory}"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
source "${SCALING_SOURCE_ROOT}/script/offline_env.sh"
cd "${SCALING_SOURCE_ROOT}"
export PYTHONPATH="${SCALING_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF TORCH_DEVICE_BACKEND_AUTOLOAD
for scaling_driver_library in /usr/local/Ascend/driver/lib64/driver \
    /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
    if [[ -d "${scaling_driver_library}" ]]; then
        export LD_LIBRARY_PATH="${scaling_driver_library}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
mkdir -p "${SCALING_REPORT_ROOT}"
python scripts/smoke_flow_shared_content_npu.py \
    --output "${SCALING_REPORT_ROOT}/npu-parity.json" 2>&1 | tee "${SCALING_REPORT_ROOT}/npu-parity.log"
for scaling_depth in 30 16; do
    bash script/selfless/pretraining_unified_flow_head_scaling_ascend64.sh \
        --depth "${scaling_depth}" --smoke --label memory-r1 --steps 12 \
        2>&1 | tee "${SCALING_REPORT_ROOT}/depth${scaling_depth}-training.log"
done
for scaling_depth in 30 16; do
    scaling_run="unified-b-x0content-flowdepth${scaling_depth}-0p6b-100b-imagenet-split-s42-r1-smoke-memory-r1"
    python scripts/smoke_unified_flow_head_generation.py --depth "${scaling_depth}" \
        --checkpoint "output/${scaling_run}/hf_model-final-ema" \
        --output-dir "${SCALING_REPORT_ROOT}/depth${scaling_depth}-generation" \
        2>&1 | tee "${SCALING_REPORT_ROOT}/depth${scaling_depth}-generation.log"
done
printf 'passed\n' > "${SCALING_REPORT_ROOT}/SMOKE_PASSED"
