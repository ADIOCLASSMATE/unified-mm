"""All-layer cal-only numerical audit at formal batch sizes, with frozen adapters."""

import argparse
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.check_geometry_v5_siglip_precision import compare
from scripts.prepare_cross_model_geometry_v5 import BASELINE, F_RUN, V4
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json


def layer_comparison(batch, singles):
    assert batch.shape == singles.shape
    report = compare(batch, singles)
    a, b = batch.double(), singles.double()
    difference = (a - b).square().sum((0, 3))
    semantic_energy = (b - b.mean(0, keepdim=True)).square().sum((0, 3))
    relative = (difference / semantic_energy.clamp_min(1e-20)).sqrt()
    report["cal_semantic_relative_rms_by_layer_pool"] = [
        [
            float(relative[i, j]) if semantic_energy[i, j] > 1e-12 else None
            for j in range(relative.shape[1])
        ]
        for i in range(relative.shape[0])
    ]
    flat_cosine = torch.nn.functional.cosine_similarity(a.flatten(1), b.flatten(1))
    report["minimum_flattened_cosine"] = float(flat_cosine.min())
    report["minimum_cosine_by_layer_pool"] = (
        torch.nn.functional.cosine_similarity(a, b, dim=-1).min(0).values.tolist()
    )
    report["formal_flattened_threshold_passed"] = bool(flat_cosine.gt(0.995).all())
    report["per_layer_below_0p995_count"] = int(
        torch.nn.functional.cosine_similarity(a, b, dim=-1).lt(0.995).sum()
    )
    return report


class NativeAdapter:
    def __init__(self, name, root, device):
        from scripts import probe_unified_semantics_v2 as v2

        v2.RUN = BASELINE if name == "b" else F_RUN
        self.protocol = json.loads(
            ((V4 if name == "b" else root / "f-v4") / "protocol.json").read_text()
        )
        self.model, self.tokenizer, self.checks = v2.load_state(
            SimpleNamespace(state="final_ema"), self.protocol, device
        )
        self.collector = v2.Collector(self.model.model)
        self.device, self.v2, self.caches = device, v2, {}

    def forward(self, rows, modality, path, family):
        from utils.evaluation.multimodal_likelihood import PosteriorCache

        if modality == "image" and family not in self.caches:
            self.caches[family] = PosteriorCache(
                Path(self.protocol["cache_roots"][family]),
                expected_image_tokens=256,
                expected_latent_dim=16,
                seed=self.protocol["seed"],
            )
        batch, mask, last, query = self.v2.make_batch(
            rows,
            modality,
            self.tokenizer,
            self.model.config,
            self.caches.get(family) if modality == "image" else None,
            self.device,
            self.protocol["seed"],
            self.protocol["profiles"][path],
        )
        self.collector.begin(mask, last, query)
        self.model.model(**batch)
        return self.collector.finish()


@torch.inference_mode()
def main():
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument(
        "--model",
        choices=[
            "b",
            "f",
            "qwen_text",
            "dinov2",
            "mae",
            "siglip",
            "janusflow",
            "showo2",
        ],
        required=True,
    )
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--diagnostic-dtype", choices=["float32", "bfloat16"])
    parser.add_argument("--diagnostic-storage-dtype", choices=["float32", "bfloat16"])
    args = parser.parse_args()
    assert torch_npu is not None and torch.npu.is_available()
    torch.set_num_threads(2)
    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    root, name = args.output_dir, args.model
    flow = name in {"janusflow", "showo2"}
    native = name in {"b", "f"}
    assert not args.diagnostic_storage_dtype or flow
    assert not (native and args.diagnostic_dtype), (
        "B/F retain their V4 source-loader contract"
    )
    dtype = getattr(torch, args.diagnostic_dtype) if args.diagnostic_dtype else None
    storage_dtype = (
        getattr(torch, args.diagnostic_storage_dtype)
        if args.diagnostic_storage_dtype
        else None
    )
    report_name = f"{name}-{args.diagnostic_dtype}-diagnostic" if dtype else name
    if storage_dtype is not None:
        report_name += f"-storage-{args.diagnostic_storage_dtype}"
    if native:
        adapter = NativeAdapter(name, root, device)
        paths, modalities, size = ("native", "bare", "neutral"), ("image", "text"), 16
    elif flow:
        from scripts.geometry_v5_flow import FlowAdapter

        adapter = FlowAdapter(name, device, dtype, storage_dtype)
        paths, modalities, size = ("understanding", "generation"), ("image", "text"), 8
    else:
        from scripts.geometry_v5_encoders import EncoderAdapter

        adapter = EncoderAdapter(name, device, dtype)
        paths, modalities, size = (
            ("main",),
            tuple(adapter.collectors),
            8 if name == "siglip" else 32,
        )
    if not native and not (args.diagnostic_dtype or args.diagnostic_storage_dtype):
        assert adapter.contract == json.loads(
            (root / "adapter-contracts" / f"{name}.json").read_text()
        )
    samples = json.loads((root / "samples.json").read_text())
    output = {
        "schema": "geometry_v5_all_layer_cal_numerics_1",
        "model": name,
        "hostname": socket.gethostname(),
        "device": str(device),
        "selection": "first 32 cal-view rows per dataset, same rule as original adapter smoke; no test or ARO rows or semantic scoring",
        "semantic_calibration": "numerical cal views only; mapping modality means still use their separate fit-class/scene calibration contract",
        "training_updates": 0,
        "formal_batch_size": size,
        "cal_only_precision_diagnostic": args.diagnostic_dtype,
        "cal_only_storage_diagnostic": args.diagnostic_storage_dtype,
        "rows": [],
    }
    if native:
        output["model_checks"] = adapter.checks
    else:
        output["adapter_contract"] = adapter.contract
    for family in ("imagenet", "coco"):
        for modality in modalities:
            dataset = f"{family}_{'images' if modality == 'image' else 'texts'}"
            rows = [row for row in samples[dataset] if row["split"] == "cal"][:32]
            assert len(rows) == 32
            for path in paths:

                def forward(selected, modality=modality, path=path, family=family):
                    if native:
                        return adapter.forward(selected, modality, path, family)
                    if flow:
                        return adapter.forward(selected, modality, path, "main")[0]
                    return adapter.forward(selected, modality)[0]

                batch = torch.cat(
                    [
                        forward(rows[start : start + size])
                        for start in range(0, len(rows), size)
                    ]
                )
                singles = torch.cat([forward([row]) for row in rows])
                result = {
                    "dataset": dataset,
                    "path": path,
                    "shape": list(batch.shape),
                    "groups": [row["group"] for row in rows],
                    "comparison": layer_comparison(batch, singles),
                }
                output["rows"].append(result)
                write_json(
                    root / "audits" / f"cal-batch-layers-{report_name}.json", output
                )
                emit(
                    "v5_cal_batch_layers_checked",
                    model=name,
                    dataset=dataset,
                    path=path,
                    minimum_cosine=result["comparison"]["minimum_cosine"],
                    flattened_passed=result["comparison"][
                        "formal_flattened_threshold_passed"
                    ],
                )
    output["complete"] = True
    output["all_formal_flattened_checks_passed"] = all(
        row["comparison"]["formal_flattened_threshold_passed"] for row in output["rows"]
    )
    write_json(root / "audits" / f"cal-batch-layers-{report_name}.json", output)
    emit(
        "v5_cal_batch_audit_complete",
        model=name,
        all_formal_flattened_checks_passed=output["all_formal_flattened_checks_passed"],
    )


if __name__ == "__main__":
    main()
