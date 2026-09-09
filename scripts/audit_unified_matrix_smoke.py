#!/usr/bin/env python3
"""Publish per-model smoke gates from verified 16-NPU worker artifacts."""
import argparse
import math
from pathlib import Path
import sys
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_unified_t2i_sampling import read, write, require, now


def audit(root):
    protocol = read(root / "protocol.json")
    ids = list(protocol["models"])
    reports = []
    for rank in range(16):
        folder = root / "smoke" / f"rank-{rank:02d}"
        path = folder / "result.json"
        if not path.exists():
            continue
        result = read(path)
        mid = ids[rank % len(ids)]
        spec = protocol["models"][mid]
        require(result["passed"] and result["rank"] == rank and result["model"] == mid, "smoke worker identity mismatch")
        require(result["generation_contract"] == spec["generation_contract"]
                and Path(result["checkpoint"]["path"]).resolve() == Path(spec["model_source"]).resolve()
                and result["checkpoint"]["global_step"] == spec["checkpoint_step"], "smoke checkpoint mismatch")
        require(result["full_reveals"] == 256 and result["full_decode_batch"] == 2
                and result["capacity_batch"] == 256 and result["capacity_reveals"] == 17
                and result["padded_sequence_length"] == 512 and result["device_count"] == 16
                and result["cfg"] == 2 and result["heun_steps"] == 10, "smoke does not cover actual generation/capacity")
        require(result["matched_probe_shape_relative_rmse"] < .005
                and result["cpu_and_npu_rng_unchanged"] and result["driver_libraries"], "smoke parity/device gate failed")
        load = read(folder / "load.json")
        require(load["full_checkpoint_value_check"]["complete"], "post-load tensor check incomplete")
        required_strategies = {"spatial_halton", "confidence_stability", "confidence_halton"}
        if mid == "e_on_b":
            required_strategies.add("sequential")
        require(set(result["strategies"]) == required_strategies, "smoke strategy coverage mismatch")
        for strategy in result["strategies"]:
            trace = read(folder / f"{strategy}-trace.json")
            require(trace["order_strategy"] == strategy, "smoke trace strategy mismatch")
            for ranks in trace["generation_order"]:
                require(sorted(v for row in ranks for v in row) == list(range(1, 257)), "invalid smoke permutation")
            if strategy.startswith("confidence_"):
                require(all(math.isfinite(v) for sample in trace["order_confidence_proxy"] for row in sample for v in row),
                        "nonfinite smoke proxy")
            for i in range(2):
                with Image.open(folder / strategy / f"{i:02d}.png") as image:
                    require(image.mode == "RGB" and image.size == (256, 256), "invalid smoke PNG")
                    image.verify()
        reports.append(result)
    validated = [mid for mid in ids if all(any(r["rank"] == rank for r in reports)
                 for rank in range(16) if ids[rank % len(ids)] == mid)]
    exit_path = root / "smoke/exit-code"
    exit_code = int(exit_path.read_text()) if exit_path.exists() else None
    result = {"passed": len(reports) == 16 and exit_code == 0, "at": now(), "ranks_complete": len(reports),
        "validated_models": validated, "expected_models": ids, "worker_reports": reports,
        "smoke_exit_code": exit_code, "runtime_hashing_enabled": False, "formal_metrics": False}
    write(root / "smoke/audit.json", result)
    require(exit_code in {None, 0}, f"development smoke exited {exit_code}; inspect smoke/run.log")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        result = audit(args.output_dir.resolve())
        status = (result["passed"], tuple(result["validated_models"]))
        if status != previous:
            print({k: result[k] for k in ["passed", "ranks_complete", "validated_models"]}, flush=True)
            previous = status
        if result["passed"] or not args.watch:
            break
        time.sleep(20)
