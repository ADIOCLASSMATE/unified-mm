#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 1 ]]; then
  echo "Usage: $0 <prepared-qualitative-output-dir>" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QUALITATIVE_ROOT="$1"
if [[ ! -f "${QUALITATIVE_ROOT}/manifest.json" || ! -f "${QUALITATIVE_ROOT}/i2t_latents.pt" ]]; then
  echo "Missing frozen qualitative inputs" >&2
  exit 3
fi
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export WANDB_MODE=disabled
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF TORCH_DEVICE_BACKEND_AUTOLOAD
# Some Job nodes omit the mounted driver's library paths from the image env.
# Only add existing standard driver directories; never substitute a stub library.
for driver_library_dir in \
  /usr/local/Ascend/driver/lib64/driver \
  /usr/local/Ascend/driver/lib64/common \
  /usr/local/Ascend/driver/lib64; do
  if [[ -d "${driver_library_dir}" ]]; then
    export LD_LIBRARY_PATH="${driver_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "Using mounted driver library directory: ${driver_library_dir}"
  fi
done
python -c 'import torch, torch_npu; assert torch.npu.is_available() and torch.npu.device_count() == 16; print("NPU preflight: 16 visible devices; torch", torch.__version__, "torch_npu", torch_npu.__version__, flush=True)'
printf 'RUNNING\n' > "${QUALITATIVE_ROOT}/launcher.status"
finish() {
  local code="$?"
  if (( code == 0 )); then
    printf 'SUCCEEDED\n' > "${QUALITATIVE_ROOT}/launcher.status"
  else
    printf 'FAILED exit=%s\n' "$code" > "${QUALITATIVE_ROOT}/launcher.status"
  fi
}
trap finish EXIT
# Independent inference workers; torchrun supervises failures. No HCCL collective.
env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
  -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node=16 \
    scripts/generate_unified_qualitative.py run --output-dir "${QUALITATIVE_ROOT}" \
    2>&1 | tee "${QUALITATIVE_ROOT}/generation.log"
python scripts/generate_unified_qualitative.py render \
  --output-dir "${QUALITATIVE_ROOT}" --require-complete \
  2>&1 | tee "${QUALITATIVE_ROOT}/render.log"
