"""Cal-only SigLIP batch/single precision diagnostic, never evaluates test semantics."""

import gc
import json

import torch

from scripts.geometry_v5_encoders import EncoderAdapter
from utils.research.geometry_v5_assets import RUN, emit, write_json


def compare(left, right):
    left, right = left.float(), right.float()
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=-1)
    relative = (left - right).norm(dim=-1) / right.norm(dim=-1).clamp_min(1e-10)
    return {
        "cosine": cosine.tolist(),
        "relative_l2": relative.tolist(),
        "minimum_cosine": float(cosine.min()),
        "maximum_relative_l2": float(relative.max()),
    }


def main():
    import torch_npu

    assert torch_npu is not None and torch.npu.is_available()
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    samples = json.loads((RUN / "samples.json").read_text())
    cal = [row for row in samples["imagenet_images"] if row["split"] == "cal"][:32]
    rows = [cal[5], cal[21]]
    report = {
        "selection": "exact failed calibration shard rank5: cal indices 5 and 21; zero test rows",
        "precisions": {},
    }
    for dtype in (torch.bfloat16, torch.float32):
        adapter = EncoderAdapter("siglip", torch.device("npu", 0), dtype)
        a = adapter.pixel_loader.batch(rows, adapter.processor, 384)
        b = adapter.pixel_loader.batch(rows[:1], adapter.processor, 384)
        assert torch.equal(a[:1], b)
        batch, endpoint, _ = adapter.forward(rows, "image")
        singles, native = [], []
        for row in rows:
            one, last, _ = adapter.forward([row], "image")
            singles.append(one)
            native.append(last)
        result = {
            "pixel_batch_exact": True,
            "layers": compare(batch, torch.cat(singles)),
            "native_endpoint": compare(endpoint, torch.cat(native)),
        }
        report["precisions"][str(dtype)] = result
        emit(
            "siglip_precision_checked",
            dtype=str(dtype),
            minimum_layer_cosine=result["layers"]["minimum_cosine"],
            minimum_native_cosine=result["native_endpoint"]["minimum_cosine"],
        )
        write_json(RUN / "siglip-cal-precision.json", report)
        for collector in adapter.collectors.values():
            for handle in collector.handles:
                handle.remove()
        del adapter
        gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
