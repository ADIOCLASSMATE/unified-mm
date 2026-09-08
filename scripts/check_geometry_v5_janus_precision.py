"""Cal-only JanusFlow all-layer batch stability diagnostic."""

import gc
import json

import torch

from scripts.check_geometry_v5_siglip_precision import compare
from scripts.geometry_v5_flow import FlowAdapter
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json


def main():
    import torch_npu

    assert torch_npu is not None and torch.npu.is_available()
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    samples = json.loads((RUN / "samples.json").read_text())
    cal = [row for row in samples["imagenet_images"] if row["split"] == "cal"][:32]
    rows = [cal[1], cal[17]]
    report = {
        "selection": "failed cal rank1; ImageNet cal indices1,17; no test examples",
        "precisions": {},
    }
    for dtype in (torch.bfloat16, torch.float32):
        adapter = FlowAdapter("janusflow", torch.device("npu", 0), dtype)
        batch, upstream = adapter.forward(rows, "image", "understanding")
        singles, input_rows = [], []
        for row in rows:
            one, inputs = adapter.forward([row], "image", "understanding")
            singles.append(one)
            input_rows.append(inputs["understanding_encoder"])
        result = {
            "layers": compare(batch, torch.cat(singles)),
            "upstream": compare(
                upstream["understanding_encoder"], torch.cat(input_rows)
            ),
        }
        report["precisions"][str(dtype)] = result
        write_json(RUN / "janusflow-cal-precision.json", report)
        emit(
            "janus_precision_checked",
            dtype=str(dtype),
            minimum_cosine=result["layers"]["minimum_cosine"],
            upstream_minimum=result["upstream"]["minimum_cosine"],
        )
        for handle in adapter.collector.handles:
            handle.remove()
        del adapter
        gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
