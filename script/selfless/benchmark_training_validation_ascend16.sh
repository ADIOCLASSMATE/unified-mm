#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 2 ]]; then
  echo "Usage: $0 <model-source-dir> <timing-output-dir>" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
source script/offline_env.sh
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-1}"
export HCCL_CONNECT_TIMEOUT=120
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF
mkdir -p "$2"
TIMING_OUTPUT_DIR="$2"
timing_launch_started=$(date +%s)
finish() {
  timing_exit_code=$?
  printf 'exit_code=%s\nlauncher_wall_seconds=%s\n' \
    "${timing_exit_code}" "$(( $(date +%s) - timing_launch_started ))" \
    > "${TIMING_OUTPUT_DIR}/launcher.status"
  exit "${timing_exit_code}"
}
trap finish EXIT
# Emergency process bound for a failed collective/kernel, separate from the
# runner's 540-second work deadline and 600-second acceptance criterion.
timeout --signal=TERM --kill-after=15s 660s env \
  -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node=16 \
  scripts/benchmark_training_validation.py \
  --config "${CONFIG:-configs/selfless/unified_baseline_100b_ascend_64npu.yaml}" \
  --model_source "$1" --output_dir "$2" --device npu
