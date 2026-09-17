#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STUDY_ROOT="${1:?prepared study directory required}"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
source "${REPO_ROOT}/script/offline_env.sh"
cd "${REPO_ROOT}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled TRANSFORMERS_VERBOSITY=error
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF TORCH_DEVICE_BACKEND_AUTOLOAD
for style_driver_dir in /usr/local/Ascend/driver/lib64/driver /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
  if [[ -d "${style_driver_dir}" ]]; then
    export LD_LIBRARY_PATH="${style_driver_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done
python -c 'import torch,torch_npu; assert torch.npu.is_available() and torch.npu.device_count()==16; print(torch.__version__,torch_npu.__version__)'
printf 'RUNNING\n' > "${STUDY_ROOT}/launcher.status"
finish() {
  local style_exit=$?
  printf 'exit_code=%s\n' "${style_exit}" > "${STUDY_ROOT}/launcher.status"
}
trap finish EXIT
env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node=16 scripts/probe_style_instruction.py run --root "${STUDY_ROOT}" \
  > "${STUDY_ROOT}/generation.log" 2>&1
env -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u GROUP_WORLD_SIZE -u ROLE_RANK -u ROLE_WORLD_SIZE \
  torchrun --standalone --nproc_per_node=16 scripts/probe_style_instruction.py score --root "${STUDY_ROOT}" \
  > "${STUDY_ROOT}/scoring.log" 2>&1
