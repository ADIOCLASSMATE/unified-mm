"""Bounded actual-device check; run only on dev-wjx-ascend."""

import importlib.metadata
import json
import os
import socket

import torch
import torch_npu

from scripts.prepare_geometry_v5_assets import RUN, write_json

assert torch.npu.is_available() and torch.npu.device_count() == 16
torch.npu.set_device(0)
x = torch.ones(8, 8, device="npu:0")
y = x @ x
assert y.sum().item() == 512.0
report = {
    "hostname": socket.gethostname(),
    "device_count": torch.npu.device_count(),
    "device": str(y.device),
    "arithmetic_sum": y.sum().item(),
    "versions": {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torch_npu",
            "transformers",
            "diffusers",
            "timm",
            "safetensors",
        )
    },
    "torch_npu_imported": torch_npu is not None,
    "cann_environment_present": all(
        os.environ.get(k) for k in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH")
    ),
    "public_weights_path_exists": (RUN.parents[1] / "public/models").is_dir(),
}
write_json(RUN / "runtime-check.json", report)
print(json.dumps(report), flush=True)
