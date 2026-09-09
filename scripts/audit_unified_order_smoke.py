#!/usr/bin/env python3
"""Validate completed 16-NPU order smoke before enabling full evaluation Jobs."""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageChops
from sweep_unified_t2i_sampling import read, write, require, now


def audit(root):
    smoke = root / "smoke"
    require((smoke / "EVALUATOR_SUCCEEDED").is_file(), "smoke process has not succeeded")
    protocol, metrics = read(root / "protocol.json"), read(smoke / "metrics.json")
    environment = read(smoke / "environment.json")
    require(environment["device_count"] == 16 and metrics["distributed"]["world_size"] == 16, "smoke must use 16 NPUs")
    require(metrics["samples_evaluated"] == 32 and metrics["cfg"] == 2 and metrics["sampling_steps"] == "10", "wrong smoke parameters")
    require(set(metrics["strategies"]) == set(protocol["strategies"]), "smoke has not covered every order strategy")
    require(metrics["evaluation_model_source"]["global_step"] == protocol["checkpoint_step"]
            and Path(metrics["evaluation_model_source"]["path"]).resolve() == Path(protocol["model_source"]), "smoke checkpoint mismatch")
    require(metrics["implementation_contracts"]["canonical_initial_noise_enabled"], "smoke noise must be paired")
    require(metrics["mechanism_diagnostics"]["generated_latent_finite_rate"] == 1.0, "smoke produced non-finite latents")
    indices = metrics["saved_image_subset"]["global_sample_indices"]
    require(len(indices) == 8, "smoke image subset missing")
    reference = None
    for strategy, result in metrics["strategies"].items():
        # The evaluator intentionally withholds FID for undersampled smoke.
        # Full Jobs still require finite formal 50K FID in validate_metrics.
        require(result["count"] == 32 and result["generation_step_max"] == 256
                and result["fid"] is None and all(math.isfinite(result[k]) for k in
                    ("latent_rms", "inception_score_mean", "inception_score_std", "generation_wall_seconds")), "smoke output invalid")
        records = []
        for index in indices:
            image_path = smoke / strategy / f"{index:08d}.png"
            with Image.open(image_path) as image:
                require(image.size == (256, 256) and image.mode == "RGB", "invalid smoke PNG")
                image.verify()
            records.append(read(image_path.with_suffix(".json")))
            trace = read(image_path.parent / "order_trace" / f"{index:08d}.json")
            ranks = [v for row in trace["generation_order"] for v in row]
            require(sorted(ranks) == list(range(1, 257)), "smoke order must be a permutation")
            require(trace["policy"] == protocol["order_policies"][strategy], "smoke policy mismatch")
            if strategy.startswith("confidence_"):
                base = read(smoke / "spatial_halton/order_trace" / f"{index:08d}.json")
                base_ranks = [v for row in base["generation_order"] for v in row]
                require(all((a - 1) // 16 == (b - 1) // 16 for a, b in zip(ranks, base_ranks)), "probe changed Halton candidate blocks")
                score = [v for row in trace["confidence_proxy"] for v in row]
                require(len(score) == 256 and all(math.isfinite(v) for v in score), "invalid smoke confidence scores")
                ordered = sorted(range(256), key=lambda p: ranks[p])
                if strategy == "confidence_halton":
                    require(ranks == base_ranks, "probe control changed reveal order")
                else:
                    for start in range(0, 256, 16):
                        block = [score[p] for p in ordered[start:start + 16]]
                        require(block == sorted(block, reverse=strategy == "confidence_cfg_reverse"), "smoke score sorting incorrect")
        if reference is None:
            reference = records
        require(reference == records, "smoke images do not share prompt / image ID / canonical noise")
    identical = 0
    for index in indices:
        with Image.open(smoke / "spatial_halton" / f"{index:08d}.png") as a, Image.open(smoke / "confidence_halton" / f"{index:08d}.png") as b:
            identical += ImageChops.difference(a, b).getbbox() is None
    result = {"passed": True, "at": now(), "strategies": protocol["strategies"], "world_size": 16,
              "samples_per_strategy": 32, "images_verified": 8 * len(protocol["strategies"]),
              "order_permutations_and_scores_verified": True, "paired_identities_verified": True,
              "probe_control_identical_images": identical, "probe_control_images_compared": len(indices),
              "smoke_metrics_used_for_strategy_selection": False}
    write(smoke / "audit.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    audit(parser.parse_args().output_dir.resolve())
