"""Calibration-only, grouped, layer-wise analysis of the frozen V2 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_unified_representations import (
    linear_cka,
    normalize,
    retrieval,
    ridge_fit,
    ridge_predict,
    top1,
)
from utils.research.representation_protocol import emit, write_json
from scripts.probe_unified_semantics_v2 import LAYERS

DATA = {}
ROWS = {}
CONTEXT = {}


def split_mask(rows, split):
    return torch.tensor([row["split"] == split for row in rows])


def labels(rows):
    return torch.tensor([row["group"] for row in rows])


def calibrated(features, rows):
    mask = split_mask(rows, "cal")
    if not bool(mask.any()):
        raise ValueError("Independent calibration split is required")
    mean = features[mask].float().mean(0)
    return normalize(features.float() - mean), mean


def grouped_means(features, feature_groups, target_groups):
    lookup = {int(value): i for i, value in enumerate(target_groups)}
    index = torch.tensor([lookup[int(value)] for value in feature_groups])
    total = torch.zeros(len(target_groups), features.shape[-1])
    total.index_add_(0, index, features.float())
    count = torch.bincount(index, minlength=len(target_groups)).float()
    if not bool(count.gt(0).all()):
        raise ValueError("Missing group when averaging captions")
    return total / count[:, None]


def ranks_with_ties(values):
    ordered, idx = values.sort()
    _, inverse, counts = ordered.unique_consecutive(
        return_inverse=True, return_counts=True
    )
    end = counts.cumsum(0).float()
    mean_rank = end - (counts.float() - 1) / 2
    result = torch.empty_like(values)
    result[idx] = mean_rank[inverse]
    return result


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    return float((x @ y) / denominator.clamp_min(1e-12))


def geometry(x, y, permutations=99):
    if len(x) != len(y):
        raise ValueError("Geometry requires corresponding groups")
    xc, yc = x - x.mean(0), y - y.mean(0)
    gx, gy = xc @ xc.T, yc @ yc.T
    denom = (gx.norm() * gy.norm()).clamp_min(1e-12)
    observed = float((gx * gy).sum() / denom)
    generator = torch.Generator().manual_seed(20260907)
    null = []
    for _ in range(permutations):
        order = torch.randperm(len(x), generator=generator)
        null.append(float((gx * gy[order][:, order]).sum() / denom))
    upper = torch.triu_indices(len(x), len(x), 1)
    sx, sy = normalize(x) @ normalize(x).T, normalize(y) @ normalize(y).T
    rx, ry = (
        ranks_with_ties(sx[upper[0], upper[1]]),
        ranks_with_ties(sy[upper[0], upper[1]]),
    )
    return {
        "linear_cka": observed,
        "permutation_cka_mean": sum(null) / max(len(null), 1),
        "permutation_cka_q95": float(torch.tensor(null).quantile(0.95)),
        "permutation_tail_p_uncorrected": (1 + sum(v >= observed for v in null))
        / (1 + len(null)),
        "off_diagonal_similarity_rank_correlation": pearson(rx, ry),
        "groups": len(x),
        "permutations": permutations,
        "note": "CKA and rank structure are not independently semantic proof; layer-wise p values are descriptive, not multiplicity-adjusted",
    }


def grouped_bootstrap(scores, gi, gt, repeats=2000):
    positive = gi[:, None].eq(gt[None, :])
    ihit = (
        positive.gather(1, scores.argsort(dim=1, descending=True, stable=True)[:, :1])
        .squeeze(1)
        .float()
    )
    thit = (
        positive.T.gather(
            1, scores.T.argsort(dim=1, descending=True, stable=True)[:, :1]
        )
        .squeeze(1)
        .float()
    )
    lookup = {int(v): i for i, v in enumerate(gi)}
    ti = torch.tensor([lookup[int(v)] for v in gt])
    tc = torch.bincount(ti, minlength=len(gi)).float()
    ts = torch.zeros(len(gi)).index_add_(0, ti, thit)
    generator = torch.Generator().manual_seed(20260907)
    draw = torch.randint(len(gi), (repeats, len(gi)), generator=generator)
    iacc = ihit[draw].mean(1) * 100
    tacc = ts[draw].sum(1) / tc[draw].sum(1) * 100
    return {
        "i2t_r1_interval": iacc.quantile(torch.tensor([0.025, 0.975])).tolist(),
        "t2i_r1_interval": tacc.quantile(torch.tensor([0.025, 0.975])).tolist(),
        "sampling_unit": "image with all captions; fixed candidate pool and calibration means",
        "repeats": repeats,
    }


def alignment_maps(xfit, yfit, xtest, ytest, gi, gt):
    generator = torch.Generator().manual_seed(20260907)
    permutation = torch.randperm(len(xfit), generator=generator)
    result = {}
    for label, target in (("paired", yfit), ("shuffled", yfit[permutation])):
        fit = ridge_fit(xfit, target, strength=0.1)
        predicted = ridge_predict(fit, xtest)
        result[f"{label}_ridge"] = retrieval(predicted, ytest, gi, gt)
        mx, my = xfit.mean(0), target.mean(0)
        u, _, vh = torch.linalg.svd((xfit - mx).T @ (target - my), full_matrices=False)
        predicted = (xtest - mx) @ (u @ vh) + my
        result[f"{label}_orthogonal"] = retrieval(predicted, ytest, gi, gt)
    return result


def coco_metrics(images, texts, image_rows, text_rows, *, layer, maps=True):
    gi, gt = labels(image_rows), labels(text_rows)
    xi, _ = calibrated(images, image_rows)
    xt, _ = calibrated(texts, text_rows)
    result = {}
    for split in ("dev", "test"):
        si, st = split_mask(image_rows, split), split_mask(text_rows, split)
        if not bool(si.any()):
            continue
        result[split] = {
            "images": int(si.sum()),
            "captions": int(st.sum()),
            "raw": retrieval(images[si], texts[st], gi[si], gt[st]),
            "centered": retrieval(xi[si], xt[st], gi[si], gt[st]),
        }
    ti, tt = split_mask(image_rows, "test"), split_mask(text_rows, "test")
    cy = grouped_means(xt[tt], gt[tt], gi[ti])
    result["test"]["geometry"] = geometry(xi[ti], normalize(cy))
    fi, ft = split_mask(image_rows, "fit"), split_mask(text_rows, "fit")
    if maps and bool(fi.any()):
        targets = normalize(grouped_means(xt[ft], gt[ft], gi[fi]))
        result["test"]["learned_maps"] = alignment_maps(
            xi[fi], targets, xi[ti], xt[tt], gi[ti], gt[tt]
        )
    if layer == "final_norm":
        result["test"]["centered_bootstrap"] = grouped_bootstrap(
            xi[ti] @ xt[tt].T, gi[ti], gt[tt]
        )
    return result


def imagenet_metrics(images, texts, image_rows, text_rows, maps=True):
    gi, gt = labels(image_rows), labels(text_rows)
    fi, ft = split_mask(image_rows, "fit"), split_mask(text_rows, "fit")
    ti, tt = split_mask(image_rows, "test"), split_mask(text_rows, "test")
    yi, yt = F.one_hot(gi, 1000).float(), F.one_hot(gt, 1000).float()
    xi, _ = calibrated(images, image_rows)
    xt, _ = calibrated(texts, text_rows)
    result = {"classes": 1000, "image_test": int(ti.sum()), "text_test": int(tt.sum())}
    for label, vi, vt in (
        ("raw", normalize(images), normalize(texts)),
        ("centered", xi, xt),
    ):
        imodel = ridge_fit(vi[fi], yi[fi])
        tmodel = ridge_fit(vt[ft], yt[ft])
        text_to_image = ridge_predict(tmodel, vi[ti])
        result[label] = {
            "image_within_top1": top1(ridge_predict(imodel, vi[ti]), gi[ti]),
            "text_within_top1": top1(ridge_predict(tmodel, vt[tt]), gt[tt]),
            "text_to_image_top1": top1(text_to_image, gi[ti]),
            "image_to_text_top1": top1(ridge_predict(imodel, vt[tt]), gt[tt]),
        }
        permutation = torch.randperm(
            1000, generator=torch.Generator().manual_seed(20260907)
        )
        result[label]["permuted_source_class_labels_top1"] = float(
            permutation[text_to_image.argmax(1)].eq(gi[ti]).float().mean() * 100
        )
        prototypes = normalize(grouped_means(vt[ft], gt[ft], torch.arange(1000)))
        result[label]["text_prototype_to_image_top1"] = top1(
            vi[ti] @ prototypes.T, gi[ti]
        )
    proto_i = grouped_means(xi[ti], gi[ti], torch.arange(1000))
    proto_t = grouped_means(xt[tt], gt[tt], torch.arange(1000))
    result["class_geometry"] = geometry(normalize(proto_i), normalize(proto_t))
    if maps:
        mf_i = fi & torch.tensor([r["mapping_split"] == "fit" for r in image_rows])
        mf_t = ft & torch.tensor([r["mapping_split"] == "fit" for r in text_rows])
        mt_i = ti & torch.tensor([r["mapping_split"] == "test" for r in image_rows])
        mt_t = tt & torch.tensor([r["mapping_split"] == "test" for r in text_rows])
        classes = gi[mf_i].unique(sorted=True)
        px = normalize(grouped_means(xi[mf_i], gi[mf_i], classes))
        py = normalize(grouped_means(xt[mf_t], gt[mf_t], classes))
        result["class_disjoint_maps"] = {
            "fit_classes": 600,
            "test_classes": 200,
            "centered_no_map": retrieval(xi[mt_i], xt[mt_t], gi[mt_i], gt[mt_t]),
            **alignment_maps(px, py, xi[mt_i], xt[mt_t], gi[mt_i], gt[mt_t]),
        }
    return result


def pair_accuracy(positive_scores, negative_scores):
    return float(
        (
            (positive_scores > negative_scores).float()
            + 0.5 * positive_scores.eq(negative_scores).float()
        ).mean()
        * 100
    )


def hard_metrics(images, texts, image_rows, text_rows):
    xi, _ = calibrated(images, image_rows)
    xt, _ = calibrated(texts, text_rows)
    lookup = {r["group"]: i for i, r in enumerate(image_rows)}
    pairs = defaultdict(dict)
    for j, row in enumerate(text_rows):
        pairs[row["group"]][bool(row["positive"])] = j
    result = {}
    for split in ("dev", "test"):
        indices = [i for i, r in enumerate(image_rows) if r["split"] == split]
        positive = torch.tensor([pairs[image_rows[i]["group"]][True] for i in indices])
        negative = torch.tensor([pairs[image_rows[i]["group"]][False] for i in indices])
        selected = torch.tensor(indices)
        categories = [image_rows[i]["category"] for i in indices]
        result[split] = {"count": len(indices)}
        for name, vi, vt in (
            ("raw", normalize(images), normalize(texts)),
            ("centered", xi, xt),
        ):
            ps = (vi[selected] * vt[positive]).sum(1)
            ns = (vi[selected] * vt[negative]).sum(1)
            shuffled_images = selected.clone()
            gen = torch.Generator().manual_seed(20260907)
            by_category = {}
            for category in sorted(set(categories)):
                mask = torch.tensor([c == category for c in categories])
                where = torch.where(mask)[0]
                shuffled_images[where] = selected[
                    where[torch.randperm(len(where), generator=gen)]
                ]
                by_category[category] = pair_accuracy(ps[mask], ns[mask])
            shuffled_ps = (vi[shuffled_images] * vt[positive]).sum(1)
            shuffled_ns = (vi[shuffled_images] * vt[negative]).sum(1)
            result[split][name] = {
                "accuracy": pair_accuracy(ps, ns),
                "by_category": by_category,
                "within_category_image_shuffle_accuracy": pair_accuracy(
                    shuffled_ps, shuffled_ns
                ),
                "mean_positive_minus_negative": float((ps - ns).mean()),
            }
    del lookup
    return result


def mask_change(features, rows, layer_index):
    native, swapped = (
        features[:, layer_index, 2].float(),
        features[:, layer_index, 3].float(),
    )
    vi, mi = calibrated(native, rows)
    vs, ms = calibrated(swapped, rows)
    test = split_mask(rows, "test")
    if len(rows) > 4000:
        # One caption per image; never construct a 25K x 25K matrix.
        seen = set()
        chosen = []
        for i, r in enumerate(rows):
            if r["split"] == "test" and r["group"] not in seen:
                chosen.append(i)
                seen.add(r["group"])
        test = torch.tensor(chosen)
    delta = (swapped[test] - ms) - (native[test] - mi)
    scale = (native[test] - mi).square().mean().sqrt()
    return {
        "calibration_mean_shift_norm": float((mi - ms).norm()),
        "within_sample_centered_cosine": float((vi[test] * vs[test]).sum(1).mean()),
        "centered_change_over_native_spread": float(
            delta.square().mean().sqrt() / scale.clamp_min(1e-8)
        ),
        "linear_cka": linear_cka(vi[test], vs[test]),
    }


def analyze_one(task):
    layer_index, name, ipool, tpool = task
    torch.set_num_threads(CONTEXT["threads"])
    layer = LAYERS[layer_index]
    out = Path(CONTEXT["out"]) / f"layer-{layer}-{name}.json"
    if out.exists():
        return str(out)
    result = {
        "state": CONTEXT["state"],
        "profile": CONTEXT["profile"],
        "layer": layer,
        "readout": name,
        "image_pool": ipool,
        "text_pool": tpool,
    }
    for family in ("coco", "imagenet", "hard"):
        ikey, tkey = f"{family}_images", f"{family}_texts"
        if ikey not in DATA:
            continue
        images = DATA[ikey][:, layer_index, ipool].float()
        texts = DATA[tkey][:, layer_index, tpool].float()
        if family == "coco":
            result[family] = coco_metrics(
                images,
                texts,
                ROWS[ikey],
                ROWS[tkey],
                layer=layer,
                maps=name in ("content_mean", "query_native"),
            )
        elif family == "imagenet":
            result[family] = imagenet_metrics(
                images,
                texts,
                ROWS[ikey],
                ROWS[tkey],
                maps=name in ("content_mean", "query_native"),
            )
        else:
            result[family] = hard_metrics(images, texts, ROWS[ikey], ROWS[tkey])
    if name == "query_native" and DATA["coco_images"].shape[2] > 3:
        result["mask_change"] = {
            modality: mask_change(
                DATA[f"coco_{modality}"], ROWS[f"coco_{modality}"], layer_index
            )
            for modality in ("images", "texts")
        }
    write_json(out, result)
    return str(out)


def load_dataset(directory, dataset, samples, profile, expected_state):
    paths = sorted(directory.glob(f"{dataset}-rank-*-of-*.pt"))
    if not paths:
        return None, None, None
    payloads = [
        torch.load(p, weights_only=True, mmap=True, map_location="cpu") for p in paths
    ]
    checks = payloads[0]["state_checks"]
    expected = [
        i
        for i, row in enumerate(samples)
        if profile["subset"] == "all" or row["robust"]
    ]
    found = torch.cat([p["indices"] for p in payloads]).tolist()
    if sorted(found) != expected or len(paths) != payloads[0]["world_size"]:
        raise ValueError(f"Missing/duplicate rank or samples: {dataset}")
    if checks["state"] != expected_state or any(
        p["state_checks"] != checks or p["profile"] != profile for p in payloads
    ):
        raise ValueError("Mixed model provenance/profile")
    positions = {index: i for i, index in enumerate(expected)}
    output = torch.empty(
        (len(expected), *payloads[0]["features"].shape[1:]), dtype=torch.bfloat16
    )
    for p in payloads:
        output[[positions[int(i)] for i in p["indices"]]] = p["features"]
    return output, [samples[i] for i in expected], checks


def summarize(root):
    rows = [
        json.loads(p.read_text())
        for p in sorted((root / "analysis").glob("*/*/layer-*.json"))
    ]
    write_json(
        root / "results-v2.json",
        {"schema": "unified_semantics_v2_results_1", "rows": rows},
    )
    flat = []
    groups = defaultdict(list)
    for row in rows:
        coco = row["coco"]["test"]
        item = {key: row[key] for key in ("state", "profile", "layer", "readout")}
        item.update(
            coco_test_images=coco["images"],
            **{
                f"coco_{mode}_{key}": value
                for mode in ("raw", "centered")
                for key, value in coco[mode].items()
            },
        )
        item.update(
            cka=coco["geometry"]["linear_cka"],
            shuffled_cka=coco["geometry"]["permutation_cka_mean"],
            rsa=coco["geometry"]["off_diagonal_similarity_rank_correlation"],
        )
        if "imagenet" in row:
            item.update(
                {
                    f"imagenet_{mode}_{key}": value
                    for mode in ("raw", "centered")
                    for key, value in row["imagenet"][mode].items()
                }
            )
        if "hard" in row:
            item.update(
                hard_centered_accuracy=row["hard"]["test"]["centered"]["accuracy"],
                hard_image_shuffle=row["hard"]["test"]["centered"][
                    "within_category_image_shuffle_accuracy"
                ],
            )
        flat.append(item)
        if "dev" in row["coco"]:
            groups[(row["state"], row["profile"], row["readout"])].append(row)
    columns = list(dict.fromkeys(k for row in flat for k in row))
    if flat:
        with (root / "layer-metrics-v2.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat)
    selected = []
    for key, candidates in sorted(groups.items()):
        # Selection depends EXCLUSIVELY on development I2T R@1, with fixed tie break.
        chosen = max(
            candidates,
            key=lambda r: (
                r["coco"]["dev"]["centered"]["i2t_r1"],
                -LAYERS.index(r["layer"]),
            ),
        )
        selected.append(
            {
                "state": key[0],
                "profile": key[1],
                "readout": key[2],
                "selected_layer": chosen["layer"],
                "dev_i2t_r1": chosen["coco"]["dev"]["centered"]["i2t_r1"],
                "test": chosen["coco"]["test"]["centered"],
            }
        )
    write_json(root / "dev-selected-layers.json", selected)
    return rows


def plot(rows, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for state, profile, readout, label in (
        ("final_ema", "bare", "content_mean", "EMA: bare content mean"),
        ("final_ema", "native", "content_mean", "EMA: native content mean"),
        ("final_ema", "neutral", "content_mean", "EMA: neutral content mean"),
        ("final_ema", "native", "query_native", "EMA: native heterogeneous query"),
        ("final_ema", "native", "same_text_mask", "EMA: native slots, both text masks"),
        ("final_ema", "neutral", "query_native", "EMA: neutral shared text query"),
        ("init42", "native", "content_mean", "Init reference 42: native content mean"),
        ("init42", "native", "query_native", "Init reference 42: native query"),
    ):
        chosen = [
            r
            for r in rows
            if (r["state"], r["profile"], r["readout"]) == (state, profile, readout)
        ]
        chosen.sort(key=lambda r: LAYERS.index(r["layer"]))
        if not chosen:
            continue
        x = [LAYERS.index(r["layer"]) for r in chosen]
        for ax, values in (
            (axes[0, 0], [r["coco"]["test"]["centered"]["i2t_r1"] for r in chosen]),
            (axes[0, 1], [r["coco"]["test"]["geometry"]["linear_cka"] for r in chosen]),
            (
                axes[1, 0],
                [r["imagenet"]["centered"]["text_to_image_top1"] for r in chosen],
            ),
            (axes[1, 1], [r["hard"]["test"]["centered"]["accuracy"] for r in chosen]),
        ):
            ax.plot(
                x,
                values,
                label=label,
                linestyle="--" if state.startswith("init") else "-",
            )
    for ax, title in zip(
        axes.flat,
        (
            "COCO centered I2T R@1 (%)",
            "COCO paired linear CKA",
            "ImageNet text-context classifier to images (%)",
            "SugarCrepe centered pair accuracy (%)",
        ),
    ):
        ax.set_title(title)
        ax.set_xlabel("Block output (0 = input; FN = final RMSNorm)")
        ax.set_xticks(
            [0, 4, 8, 12, 16, 20, 24, 28, 29],
            ["0", "4", "8", "12", "16", "20", "24", "28", "FN"],
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.axvspan(28.5, 29.5, color="grey", alpha=0.08)
        ax.grid(alpha=0.2)
    axes[1, 1].axhline(50, color="grey", linestyle=":")
    axes[0, 0].legend(fontsize=6)
    fig.savefig(root / "layer-curves-v2.png", dpi=170)
    fig.savefig(root / "layer-curves-v2.pdf")
    plt.close(fig)


def main():
    global DATA, ROWS, CONTEXT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state", default="final_ema")
    parser.add_argument("--profiles", default="bare,native,neutral")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--robustness-baselines", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    # Fork only from a single-threaded parent. Inheriting a live OpenMP pool
    # can deadlock CPU tensor operations in children; workers enable their
    # own small thread pools after the fork.
    torch.set_num_threads(1)
    if not args.summarize_only:
        protocol = json.loads((args.output_dir / "protocol.json").read_text())
        samples = json.loads((args.output_dir / "samples.json").read_text())
        for profile_name in args.profiles.split(","):
            DATA, ROWS = {}, {}
            checks = None
            for dataset, sample_rows in samples.items():
                data, rows, check = load_dataset(
                    args.output_dir / args.state / profile_name,
                    dataset,
                    sample_rows,
                    protocol["profiles"][profile_name],
                    args.state,
                )
                if data is not None:
                    DATA[dataset], ROWS[dataset], checks = data, rows, check
            required = {
                key
                for key, rows in samples.items()
                if protocol["profiles"][profile_name]["subset"] == "all"
                or any(r["robust"] for r in rows)
            }
            if set(DATA) != required:
                raise ValueError(f"Incomplete datasets for {args.state}/{profile_name}")
            output_profile = profile_name
            if args.robustness_baselines:
                subset_data, subset_rows = {}, {}
                for key in ("coco_images", "coco_texts"):
                    select = torch.tensor([r["robust"] for r in ROWS[key]])
                    subset_data[key] = DATA[key][select]
                    subset_rows[key] = [r for r in ROWS[key] if r["robust"]]
                DATA, ROWS = subset_data, subset_rows
                output_profile += "_robust_subset"
            pairs = [
                ("content_mean", 0, 0),
                ("content_last_sigma", 1, 1),
                ("query_native", 2, 2),
            ]
            if DATA["coco_images"].shape[2] > 3 and not checks["mask_equal"]:
                pairs += [("query_swapped_mask", 3, 3)]
                if protocol["profiles"][profile_name]["prompt"] == "native":
                    pairs += [("same_text_mask", 2, 3), ("same_image_mask", 3, 2)]
                else:
                    pairs += [
                        ("image_textmask_text_imagemask", 2, 3),
                        ("image_imagemask_text_textmask", 3, 2),
                    ]
            out = args.output_dir / "analysis" / args.state / output_profile
            CONTEXT = {
                "state": args.state,
                "profile": output_profile,
                "threads": args.threads,
                "out": str(out),
            }
            tasks = [(i, name, ip, tp) for i in range(30) for name, ip, tp in pairs]
            emit(
                "analysis_start_v2",
                state=args.state,
                profile=profile_name,
                tasks=len(tasks),
            )
            with mp.get_context("fork").Pool(args.workers) as pool:
                for i, _ in enumerate(pool.imap_unordered(analyze_one, tasks), 1):
                    if i % 20 == 0 or i == len(tasks):
                        emit(
                            "analysis_progress_v2",
                            state=args.state,
                            profile=profile_name,
                            done=i,
                            total=len(tasks),
                        )
    if args.skip_summary:
        emit("state_analysis_complete_v2", state=args.state)
        return
    rows = summarize(args.output_dir)
    if args.plot:
        plot(rows, args.output_dir)
    emit("analysis_complete_v2", rows=len(rows))


if __name__ == "__main__":
    main()
