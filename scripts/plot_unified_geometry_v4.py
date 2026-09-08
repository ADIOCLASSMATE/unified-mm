"""Complete, fixed-layer V4 geometry, fit-size, reliability and control plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

READOUTS = ("content_mean", "content_last_sigma", "query_native")
NAMES = ("Content mean", "Last-sigma content", "Native query")
COLORS = ("#4c5562", "#2470af", "#dd8128", "#508b57")
MODES = ("centered_euclidean", "centered_unit_sphere")
DIMS = ("full", "32", "128", "512")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    root = a.output_dir
    rows = json.loads((root / "results-geometry-v4.json").read_text())["rows"]
    layers = json.loads((root / "protocol.json").read_text())["layers"]
    lookup = {(r["state"], r["profile"], r["readout"], r["layer"]): r for r in rows}
    assert len(lookup) == len(rows) == 630, "Do not publish incomplete layer plots"
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    x = np.arange(len(layers), dtype=float)
    x[-1] += 1

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=160)
        fig.savefig(figures / f"{name}.pdf")
        plt.close(fig)

    def base_style(ax):
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    def depth_style(ax):
        base_style(ax)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 30])
        ax.set_xticklabels(["0", "4", "8", "12", "16", "20", "24", "28", "FN"])
        ax.set_xlabel("Backbone depth; FN = final RMSNorm")

    def result(
        readout, family, mode, layer="final_norm", state="final_ema", profile="native"
    ):
        return lookup[state, profile, readout, layer]["families"][family][mode]

    def metric(g, key):
        if not g["valid"]:
            return np.nan
        return (
            g["knn_chance_adjusted"]["10"]
            if key == "knn"
            else g["scores"]["rsa_spearman"]
        )

    for mode in MODES:
        label = "Euclidean" if mode == MODES[0] else "Unit sphere"
        for key, title in (("knn", "Chance-adjusted kNN@10"), ("rsa", "RSA Spearman")):
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
            for row, family in enumerate(("imagenet", "coco")):
                for col, readout in enumerate(READOUTS):
                    ax = axes[row, col]
                    for state, profile, color, style, name in (
                        ("final_ema", "native", "#2470af", "-", "EMA native"),
                        ("final_raw", "native", "#dd8128", "--", "raw native"),
                        ("final_ema", "bare", "#508b57", "-.", "EMA bare"),
                        ("final_ema", "neutral", "#9061a8", ":", "EMA neutral"),
                    ):
                        ys = [
                            metric(
                                result(readout, family, mode, layer, state, profile)[
                                    "geometry"
                                ]["primary"],
                                key,
                            )
                            for layer in layers
                        ]
                        ax.plot(x, ys, color=color, linestyle=style, label=name)
                    initial = np.array(
                        [
                            [
                                metric(
                                    result(readout, family, mode, layer, state)[
                                        "geometry"
                                    ]["primary"],
                                    key,
                                )
                                for layer in layers
                            ]
                            for state in ("init42", "init43", "init44")
                        ]
                    )
                    valid = ~np.isnan(initial).all(0)
                    ax.fill_between(
                        x[valid],
                        np.nanmin(initial[:, valid], 0),
                        np.nanmax(initial[:, valid], 0),
                        color="#6d747b",
                        alpha=0.2,
                        label="3 init-reference range",
                    )
                    ax.plot(
                        x[valid],
                        np.nanmean(initial[:, valid], 0),
                        color="#6d747b",
                        linewidth=1,
                    )
                    ax.axhline(0, color="black", linewidth=0.6)
                    ax.set_title(f"{family.upper()} | {NAMES[col]}")
                    ax.set_ylabel(title)
                    depth_style(ax)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(
                f"B V4 | {label} | fixed held-out semantic units\nNative query is heterogeneous; bare/neutral queries are both text masks. Init ranges are not confidence intervals."
            )
            save(fig, f"layers-{key}-{mode}")

        for split in ("test", "transfer_test"):
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
            for row, family in enumerate(("imagenet", "coco")):
                n = 600 if family == "imagenet" else 8192
                for col, readout in enumerate(READOUTS):
                    ax = axes[row, col]
                    for dim, color in zip(DIMS, COLORS, strict=True):
                        maps = [
                            result(readout, family, mode, layer)["mappings"][
                                f"fit{n}-dim{dim}"
                            ]
                            for layer in layers
                        ]
                        ys = [
                            m[split]["paired"]["r2"] if m["valid"] else np.nan
                            for m in maps
                        ]
                        ax.plot(
                            x,
                            ys,
                            color=color,
                            label="Full 1024-D" if dim == "full" else f"PCA {dim}-D",
                        )
                        if dim == "32":
                            ax.plot(
                                x,
                                [
                                    m[split]["shuffled_fit"]["r2"]
                                    if m["valid"]
                                    else np.nan
                                    for m in maps
                                ],
                                color=color,
                                linestyle=":",
                                alpha=0.7,
                                label="PCA 32-D shuffled fit",
                            )
                    ax.axhline(0, color="black", linewidth=0.6)
                    ax.axhline(1, color="#508b57", linewidth=0.6, linestyle="--")
                    ax.set_yscale("symlog", linthresh=1)
                    domain = (
                        family.upper()
                        if split == "test"
                        else f"{family.upper()} to {'COCO' if family == 'imagenet' else 'ImageNet'}"
                    )
                    ax.set_title(f"{domain} | {NAMES[col]}")
                    ax.set_ylabel("Test R²; symlog outside [-1, 1]")
                    depth_style(ax)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(
                f"B V4 EMA native | {label} | {'strict transfer' if split == 'transfer_test' else 'held-out units'}\nImageNet full map is sample-underidentified (600 classes). COCO fit = 8192 scenes. No target-domain recalibration."
            )
            save(fig, f"rotation-{split}-{mode}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    sizes = (512, 2048, 8192)
    for row, mode in enumerate(MODES):
        for col, readout in enumerate(READOUTS):
            ax = axes[row, col]
            maps = result(readout, "coco", mode)["mappings"]
            for dim, color in zip(DIMS, COLORS, strict=True):
                selected = [maps[f"fit{n}-dim{dim}"] for n in sizes]
                score = np.array(
                    [
                        m["test"]["paired"]["r2"] if m["valid"] else np.nan
                        for m in selected
                    ]
                )
                low = np.array(
                    [
                        m["test"]["bootstrap"]["r2_95_interval"][0]
                        if m["valid"]
                        else np.nan
                        for m in selected
                    ]
                )
                high = np.array(
                    [
                        m["test"]["bootstrap"]["r2_95_interval"][1]
                        if m["valid"]
                        else np.nan
                        for m in selected
                    ]
                )
                ax.plot(sizes, score, "o-", color=color, label=f"{dim} dimensions")
                ax.fill_between(sizes, low, high, color=color, alpha=0.12)
                for n, mapping in zip(sizes, selected, strict=True):
                    for repeat in mapping.get("repeat_fits", []):
                        ax.plot(
                            n,
                            repeat["test"]["paired"]["r2"],
                            "x",
                            color=color,
                            alpha=0.7,
                        )
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, [str(n) for n in sizes])
            ax.set_xlabel("Independent fit scenes")
            ax.set_ylabel("Orthogonal-map test R²")
            ax.set_title(f"{NAMES[col]} | {'Euclidean' if row == 0 else 'Unit sphere'}")
            base_style(ax)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "B V4 EMA native | fixed final RMSNorm | same 2048 held-out scenes\nBands: test-scene conditional 95% bootstrap. Crosses: two fit resamples (not confidence intervals)."
    )
    save(fig, "coco-fit-size-final-norm")

    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        sizes = (1, 3, 5, 10, 15)
        for col, readout in enumerate(READOUTS):
            ax = axes[col]
            g = result(readout, "imagenet", mode)["geometry"]
            for key, color, name in (
                ("crossmodal", "#2470af", "Image-text"),
                ("image_repeat", "#dd8128", "Independent image views"),
            ):
                ax.plot(
                    sizes,
                    [
                        g["prototype_sizes"][str(n)][key]["scores"]["knn_10"]
                        for n in sizes
                    ],
                    "o-",
                    color=color,
                    label=name,
                )
            ax.axhline(
                g["repeated_views"]["y"]["scores"]["knn_10"],
                color="#508b57",
                linestyle=":",
                label="Independent text templates",
            )
            ax.axhline(
                10 / 199,
                color="black",
                linewidth=0.6,
                linestyle="--",
                label="Random neighbor identity",
            )
            ax.set_title(NAMES[col])
            ax.set_xlabel("Images per class prototype")
            ax.set_ylabel("kNN@10 identity overlap")
            ax.set_xticks(sizes)
            base_style(ax)
        axes[0].legend(fontsize=7)
        fig.suptitle(
            f"B V4 EMA native final RMSNorm | ImageNet reliability | {mode}\n200 held-out mapping classes; separate image views are a reference, not a strict upper bound."
        )
        save(fig, f"imagenet-prototype-size-{mode}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, readout in enumerate(READOUTS):
        for mode, color in zip(MODES, COLORS[1:3], strict=True):
            r = result(readout, "coco", mode)
            sizes = (1, 3, 5)
            axes[0, col].plot(
                sizes,
                [
                    r["geometry"]["caption_counts"][str(n)]["scores"]["knn_10"]
                    for n in sizes
                ],
                "o-",
                color=color,
                label=mode.replace("centered_", ""),
            )
            axes[1, col].plot(
                sizes,
                [
                    r["mappings"]["fit8192-dim32"]["caption_test"][str(n)]["paired"][
                        "r2"
                    ]
                    for n in sizes
                ],
                "o-",
                color=color,
            )
        for row in range(2):
            axes[row, col].set_title(NAMES[col])
            axes[row, col].set_xlabel("Captions averaged per test scene")
            axes[row, col].set_xticks((1, 3, 5))
            base_style(axes[row, col])
        axes[0, col].set_ylabel("kNN@10 identity overlap")
        axes[1, col].set_ylabel("Frozen 32-D map test R²")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "B V4 EMA native final RMSNorm | COCO caption-count control\nRaw features averaged before normalization; maps always fitted using all training captions."
    )
    save(fig, "coco-caption-count")

    states = ("final_ema", "final_raw", "init42", "init43", "init44")
    disjoint = json.loads((root / "aro-disjoint-v4.json").read_text())
    aro_lookup = {
        (r["state"], r["profile"], r["readout"], r["mode"], r["dimension"]): r[
            "cohorts"
        ]["verified_disjoint"]
        for r in disjoint["rows"]
    }
    for mode in MODES:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for col, readout in enumerate(READOUTS):
            records = [
                aro_lookup[state, "native", readout, mode, "32"] for state in states
            ]
            for row, task in enumerate(sorted(records[0])):
                ax = axes[row, col]
                positions = np.arange(len(states))
                for offset, key, color in (
                    (-0.18, "paired_fit", "#2470af"),
                    (0.18, "shuffled_image", "#9da4aa"),
                ):
                    ax.bar(
                        positions + offset,
                        [r[task][key]["cosine"] for r in records],
                        width=0.34,
                        color=color,
                        label=key.replace("_", " "),
                    )
                    if key == "paired_fit":
                        intervals = np.array(
                            [
                                r[task]["bootstrap"]["cosine"]["accuracy_95"]
                                for r in records
                            ]
                        )
                        ax.vlines(
                            positions + offset,
                            intervals[:, 0],
                            intervals[:, 1],
                            color="black",
                            linewidth=0.8,
                            label="Test-group 95% interval",
                        )
                ax.plot(
                    positions,
                    [r[task]["original_coordinates"]["cosine"] for r in records],
                    "x",
                    color="#dd8128",
                    label="Original coordinates",
                )
                ax.axhline(0.5, color="black", linestyle="--", linewidth=0.6)
                ax.set_xticks(positions, [s.replace("final_", "") for s in states])
                ax.set_ylim(0, 1)
                ax.set_title(f"{task.replace('aro_vg_', '')} | {NAMES[col]}")
                ax.set_ylabel("Correct-vs-negative cosine accuracy")
                base_style(ax)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"B V4 final RMSNorm | ARO hard negatives | {mode}\nCOCO 8192-fit 32-D maps; 868 original images with verified COCO-pool-disjoint IDs, split across the two tasks."
        )
        save(fig, f"aro-controls-{mode}")

    robustness = json.loads((root / "robustness-geometry-v4.json").read_text())["rows"]
    for family in ("imagenet", "coco"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for row, mode in enumerate(MODES):
            for col, readout in enumerate(READOUTS):
                ax = axes[row, col]
                chosen = {
                    r["profile"]: r
                    for r in robustness
                    if r["layer"] == "final_norm"
                    and r["source_family"] == family
                    and r["target_family"] == family
                    and r["readout"] == readout
                    and r["mode"] == mode
                }
                for offset, dim, color in (
                    (-0.15, "full", COLORS[0]),
                    (0.15, "32", COLORS[1]),
                ):
                    records = [
                        chosen[profile]["maps"][dim]
                        for profile in ("native_sigma1", "native_sigma2", "native_mean")
                    ]
                    scores = np.array(
                        [r["error_reduction_over_native_denominator"] for r in records]
                    )
                    intervals = np.array(
                        [r["error_reduction_95_interval"] for r in records]
                    )
                    positions = np.arange(3) + offset
                    ax.bar(
                        positions,
                        scores,
                        width=0.28,
                        color=color,
                        label=f"{dim} dimensions",
                    )
                    ax.vlines(
                        positions,
                        intervals[:, 0],
                        intervals[:, 1],
                        color="black",
                        linewidth=0.8,
                    )
                ax.axhline(0, color="black", linewidth=0.6)
                ax.set_xticks(
                    np.arange(3), ("Order / slot 1", "Order / slot 2", "Posterior mean")
                )
                ax.set_title(
                    f"{NAMES[col]} | {'Euclidean' if row == 0 else 'Unit sphere'}"
                )
                ax.set_ylabel("Paired error reduction / native variance")
                base_style(ax)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"B V4 EMA native final RMSNorm | {family.upper()} perturbations\nAll analysis parameters frozen on native fit data. Below zero = larger error. Intervals resample paired semantic units."
        )
        save(fig, f"robustness-{family}")
    print(
        json.dumps(
            {"plots": len(list(figures.glob("*.png"))), "directory": str(figures)}
        )
    )


if __name__ == "__main__":
    main()
