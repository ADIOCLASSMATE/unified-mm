"""Artifact audit, VAE baseline, and fixed-endpoint uncertainty for V2."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_unified_representations import (
    normalize,
    ridge_fit,
    ridge_predict,
    top1,
)
from scripts.analyze_unified_semantics_v2 import (
    calibrated,
    labels,
    load_dataset,
    split_mask,
)
from scripts.probe_unified_representations import emit, write_json


def grouped_retrieval_null(scores, image_groups, text_groups, repeats=999):
    """Shuffle image identities as whole groups, never individual captions."""
    if image_groups.unique().numel() != image_groups.numel():
        raise ValueError("This control expects one representation per image")
    predicted_text_group = text_groups[scores.argmax(1)]
    predicted_image_index = scores.argmax(0)

    def accuracy(permuted_groups):
        return (
            torch.stack(
                (
                    predicted_text_group.eq(permuted_groups).float().mean(),
                    permuted_groups[predicted_image_index]
                    .eq(text_groups)
                    .float()
                    .mean(),
                )
            )
            * 100
        )

    observed = accuracy(image_groups)
    generator = torch.Generator().manual_seed(20260907)
    null = torch.stack(
        [
            accuracy(
                image_groups[torch.randperm(len(image_groups), generator=generator)]
            )
            for _ in range(repeats)
        ]
    )
    return {
        direction: {
            "observed_r1": float(observed[i]),
            "shuffled_mean_r1": float(null[:, i].mean()),
            "shuffled_r1_95_interval": null[:, i]
            .quantile(torch.tensor([0.025, 0.975]))
            .tolist(),
            "permutation_tail_p_uncorrected": float(
                (1 + null[:, i].ge(observed[i]).sum()) / (repeats + 1)
            ),
        }
        for i, direction in enumerate(("i2t", "t2i"))
    }


def retrieval_controls(root, protocol, samples):
    results = []
    for state in ("final_ema", "final_raw", "init42", "init43", "init44"):
        profiles = (
            ("bare", "native", "neutral")
            if state == "final_ema"
            else ("bare", "native")
        )
        for profile_name in profiles:
            profile = protocol["profiles"][profile_name]
            image, ir, _ = load_dataset(
                root / state / profile_name,
                "coco_images",
                samples["coco_images"],
                profile,
                state,
            )
            text, tr, _ = load_dataset(
                root / state / profile_name,
                "coco_texts",
                samples["coco_texts"],
                profile,
                state,
            )
            si, st = split_mask(ir, "test"), split_mask(tr, "test")
            gi, gt = labels(ir)[si], labels(tr)[st]
            for pool, readout in ((0, "content_mean"), (2, "query_native")):
                vi, _ = calibrated(image[:, -1, pool].float(), ir)
                vt, _ = calibrated(text[:, -1, pool].float(), tr)
                scores = normalize(vi[si]) @ normalize(vt[st]).T
                control = grouped_retrieval_null(scores, gi, gt)
                reference = json.loads(
                    (
                        root
                        / "analysis"
                        / state
                        / profile_name
                        / f"layer-final_norm-{readout}.json"
                    ).read_text()
                )["coco"]["test"]["centered"]
                for direction in ("i2t", "t2i"):
                    if (
                        abs(
                            control[direction]["observed_r1"]
                            - reference[f"{direction}_r1"]
                        )
                        > 1e-5
                    ):
                        raise AssertionError(
                            "Supplementary control does not reproduce main endpoint"
                        )
                results.append(
                    {
                        "state": state,
                        "profile": profile_name,
                        "layer": "final_norm",
                        "readout": readout,
                        **control,
                    }
                )
    write_json(
        root / "retrieval-permutation-controls.json",
        {
            "rows": results,
            "permutations": 999,
            "note": "Supplementary fixed-endpoint null diagnostic. Entire image identities are permuted, preserving caption groups and the fixed candidate pool. P values are not multiple-comparison adjusted.",
        },
    )
    emit("retrieval_permutation_controls_complete", rows=len(results))


def audit(root, protocol, samples):
    stages = json.loads((root / "execution-plan.json").read_text())["stages"]
    records = []
    for _, state, profiles, smoke in stages:
        if smoke:
            continue
        for profile_name in profiles.split(","):
            profile = protocol["profiles"][profile_name]
            for dataset, rows in samples.items():
                expected = [
                    i
                    for i, row in enumerate(rows)
                    if profile["subset"] == "all" or row["robust"]
                ]
                if not expected:
                    continue
                paths = sorted(
                    (root / state / profile_name).glob(f"{dataset}-rank-*-of-*.pt")
                )
                if len(paths) != 16:
                    raise AssertionError(
                        f"Incomplete ranks: {state}/{profile_name}/{dataset}"
                    )
                indices, ranks, checks = [], [], []
                for path in paths:
                    p = torch.load(
                        str(path), weights_only=True, mmap=True, map_location="cpu"
                    )
                    x = p["features"]
                    if (
                        x.dtype != torch.bfloat16
                        or tuple(x.shape[1:2]) != (30,)
                        or x.shape[-1] != 1024
                    ):
                        raise AssertionError(f"Invalid features: {path}")
                    if not bool(torch.isfinite(x).all()):
                        raise AssertionError(f"Nonfinite persisted features: {path}")
                    indices.extend(p["indices"].tolist())
                    ranks.append(p["rank"])
                    checks.append(p["state_checks"])
                if sorted(indices) != expected or sorted(ranks) != list(range(16)):
                    raise AssertionError("Missing/duplicate sample or rank")
                if (
                    any(check != checks[0] for check in checks)
                    or checks[0]["state"] != state
                ):
                    raise AssertionError("Mixed model source within one feature set")
                records.append(
                    {
                        "state": state,
                        "profile": profile_name,
                        "dataset": dataset,
                        "samples": len(indices),
                        "rank_shards": 16,
                        "finite": True,
                    }
                )
    log_counts = {}
    for name, _, _, smoke in stages:
        if not smoke:
            lines = (root / f"extract-{name}.log").read_text().splitlines()
            log_counts[name] = {
                event: sum(f'"event": "{event}"' in line for line in lines)
                for event in (
                    "model_loaded_v2",
                    "hidden_target_invariance_v2",
                    "all_x0_mask_swap_invariance_v2",
                    "dataset_complete_v2",
                )
            }
    result = {
        "feature_sets": records,
        "shards": sum(r["rank_shards"] for r in records),
        "counts": log_counts,
        "all_passed": True,
    }
    write_json(root / "integrity-audit.json", result)
    emit("integrity_audit_complete", shards=result["shards"])


def vae_baseline(root, protocol, samples):
    from utils.evaluation.multimodal_likelihood import PosteriorCache

    rows = samples["imagenet_images"]
    cache = PosteriorCache(
        Path(protocol["cache_roots"]["imagenet"]),
        expected_image_tokens=256,
        expected_latent_dim=16,
        seed=protocol["seed"],
    )
    values = torch.stack(
        [cache.sample(r["image_id"]).to(torch.bfloat16).float() for r in rows]
    )
    group = labels(rows)
    fit, test = split_mask(rows, "fit"), split_mask(rows, "test")
    target = F.one_hot(group[fit], 1000).float()
    results = {}
    for name, x in (
        ("spatial_mean_16d", values.mean(1)),
        ("flattened_4096d", values.flatten(1)),
    ):
        normalized, _ = calibrated(x, rows)
        fitted = ridge_fit(normalized[fit], target, strength=0.1)
        results[name] = {
            "image_within_top1": top1(
                ridge_predict(fitted, normalized[test]), group[test]
            ),
            "fit_images": int(fit.sum()),
            "test_images": int(test.sum()),
            "classes": 1000,
        }
    write_json(
        root / "vae-baseline.json",
        {
            "results": results,
            "posterior": "same deterministic sample and BF16 input rounding as extraction",
            "note": "Supervised within-image probes only; no direct cross-modal coordinates implied",
        },
    )
    emit("vae_baseline_complete", results=results)


def hard_intervals(root, protocol, samples):
    results = []
    for state in ("final_ema", "final_raw", "init42", "init43", "init44"):
        profiles = (
            ("bare", "native", "neutral")
            if state == "final_ema"
            else ("bare", "native")
        )
        for profile_name in profiles:
            profile = protocol["profiles"][profile_name]
            image, ir, _ = load_dataset(
                root / state / profile_name,
                "hard_images",
                samples["hard_images"],
                profile,
                state,
            )
            text, tr, _ = load_dataset(
                root / state / profile_name,
                "hard_texts",
                samples["hard_texts"],
                profile,
                state,
            )
            index = torch.tensor([i for i, r in enumerate(ir) if r["split"] == "test"])
            pairs = {r["group"]: {} for r in ir}
            for i, r in enumerate(tr):
                pairs[r["group"]][r["positive"]] = i
            pos = torch.tensor([pairs[ir[i]["group"]][True] for i in index])
            neg = torch.tensor([pairs[ir[i]["group"]][False] for i in index])
            cats = [ir[i]["category"] for i in index]
            generator = torch.Generator().manual_seed(20260907)
            shuffled = index.clone()
            strata = []
            for category in sorted(set(cats)):
                where = torch.tensor([i for i, c in enumerate(cats) if c == category])
                shuffled[where] = index[
                    where[torch.randperm(len(where), generator=generator)]
                ]
                strata.append(where)
            generator = torch.Generator().manual_seed(20260907)
            draw = torch.cat(
                [
                    s[torch.randint(len(s), (2000, len(s)), generator=generator)]
                    for s in strata
                ],
                1,
            )
            for pool, name in (
                (0, "content_mean"),
                (1, "content_last_sigma"),
                (2, "query_native"),
            ):
                vi, _ = calibrated(image[:, -1, pool].float(), ir)
                vt, _ = calibrated(text[:, -1, pool].float(), tr)

                def hits(selected, vi=vi, vt=vt, pos=pos, neg=neg):
                    margin = (vi[selected] * vt[pos]).sum(1) - (
                        vi[selected] * vt[neg]
                    ).sum(1)
                    return margin.gt(0).float() + 0.5 * margin.eq(0).float()

                observed, null = hits(index), hits(shuffled)
                score = observed[draw].mean(1) * 100
                difference = (observed - null)[draw].mean(1) * 100
                results.append(
                    {
                        "state": state,
                        "profile": profile_name,
                        "layer": "final_norm",
                        "readout": name,
                        "accuracy": float(observed.mean() * 100),
                        "accuracy_95_interval": score.quantile(
                            torch.tensor([0.025, 0.975])
                        ).tolist(),
                        "image_shuffled_accuracy": float(null.mean() * 100),
                        "paired_advantage_percentage_points": float(
                            (observed - null).mean() * 100
                        ),
                        "paired_advantage_95_interval": difference.quantile(
                            torch.tensor([0.025, 0.975])
                        ).tolist(),
                    }
                )
    write_json(
        root / "hard-negative-intervals.json",
        {
            "rows": results,
            "repeats": 2000,
            "sampling": "unique images, stratified within 7 categories",
            "note": "Fixed final-layer diagnostics; intervals are not a family-wise multiple-testing guarantee",
        },
    )
    emit("hard_intervals_complete", rows=len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actions", default="audit,vae,hard,retrieval")
    args = parser.parse_args()
    torch.set_num_threads(8)
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    samples = json.loads((args.output_dir / "samples.json").read_text())
    actions = {
        "audit": audit,
        "vae": vae_baseline,
        "hard": hard_intervals,
        "retrieval": retrieval_controls,
    }
    for action in args.actions.split(","):
        actions[action](args.output_dir, protocol, samples)
