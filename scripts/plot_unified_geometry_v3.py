"""Plot complete V3 layer curves without selecting a test-layer maximum."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    protocol = json.loads((root / "protocol.json").read_text())
    rows = json.loads((root / "results-geometry-v3.json").read_text())["rows"]
    layers = protocol["layers"]
    lookup = {(r["state"], r["profile"], r["readout"], r["layer"]): r for r in rows}
    expected = len(protocol["states_and_profiles"]) * len(layers) * 2
    if len(lookup) != expected:
        raise ValueError(
            f"Refuse to plot an incomplete analysis: {len(lookup)}/{expected}"
        )
    x = np.arange(len(layers), dtype=float)
    x[-1] += 1  # Separate final RMSNorm from block 28 in the tick labels.
    colors = {"content_mean": "#2462ad", "query_native": "#df791a"}
    names = {"content_mean": "Content", "query_native": "Query"}

    def values(state, profile, readout, family, mode):
        return [
            lookup[(state, profile, readout, layer)]["families"][family][mode]
            for layer in layers
        ]

    def axis_style(ax):
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 30])
        ax.set_xticklabels(["0", "4", "8", "12", "16", "20", "24", "28", "FN"])
        ax.set_xlabel("Backbone depth (FN = final RMSNorm)")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    def save(fig, filename):
        fig.savefig(root / f"{filename}.png", dpi=170)
        fig.savefig(root / f"{filename}.pdf")
        plt.close(fig)

    for mode in protocol["preprocessing"]:
        mode_name = "Euclidean" if mode == "centered_euclidean" else "Unit sphere"
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for col, family in enumerate(("imagenet", "coco")):
            for row, metric in enumerate(("knn", "rsa")):
                ax = axes[row, col]

                def metric_value(v, metric=metric):
                    g = v["geometry"]
                    if not g["valid"]:
                        return np.nan
                    return (
                        g["knn_chance_adjusted"]["10"]
                        if metric == "knn"
                        else g["scores"]["rsa_spearman"]
                    )

                for readout, readout_name in names.items():
                    ys = list(
                        map(
                            metric_value,
                            values("final_ema", "native", readout, family, mode),
                        )
                    )
                    ax.plot(
                        x,
                        ys,
                        color=colors[readout],
                        label=f"EMA native {readout_name}",
                    )
                    initial = np.array(
                        [
                            list(
                                map(
                                    metric_value,
                                    values(state, "native", readout, family, mode),
                                )
                            )
                            for state in ("init42", "init43", "init44")
                        ]
                    )
                    valid = ~np.isnan(initial).all(axis=0)
                    ax.fill_between(
                        x[valid],
                        np.nanmin(initial[:, valid], axis=0),
                        np.nanmax(initial[:, valid], axis=0),
                        color=colors[readout],
                        alpha=0.17,
                    )
                    ax.plot(
                        x[valid],
                        np.nanmean(initial[:, valid], axis=0),
                        color=colors[readout],
                        linestyle=":",
                        linewidth=1.2,
                        label=f"Init references {readout_name}",
                    )
                for profile, color in (("bare", "#53915b"), ("neutral", "#9563b3")):
                    ys = list(
                        map(
                            metric_value,
                            values("final_ema", profile, "content_mean", family, mode),
                        )
                    )
                    ax.plot(
                        x,
                        ys,
                        color=color,
                        linestyle="--",
                        label=f"EMA {profile} Content",
                    )
                ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
                ax.set_title(
                    f"{family.upper()} — {'neighbor identity' if metric == 'knn' else 'distance ordering'}"
                )
                ax.set_ylabel(
                    "Chance-adjusted kNN@10" if metric == "knn" else "RSA Spearman"
                )
                axis_style(ax)
        axes[0, 0].legend(fontsize=7, loc="upper left", ncol=2)
        fig.suptitle(
            f"B semantic geometry | {mode_name} | fixed held-out units\nInit bands are 3 initialization references, not confidence intervals",
            fontsize=12,
        )
        save(fig, f"geometry-layers-{mode}")

        for split, title in (
            ("test", "Held-out within-dataset"),
            ("transfer_test", "Strict cross-dataset transfer"),
        ):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for row, readout in enumerate(names):
                for col, family in enumerate(("imagenet", "coco")):
                    ax = axes[row, col]
                    records = values("final_ema", "native", readout, family, mode)
                    for dim, color in (
                        ("full", "#7b7f86"),
                        ("32", "#2462ad"),
                        ("128", "#df791a"),
                    ):
                        mappings = [v["mappings"][dim] for v in records]
                        ys = [
                            m[split]["paired"]["r2"] if m["valid"] else np.nan
                            for m in mappings
                        ]
                        ax.plot(
                            x,
                            ys,
                            color=color,
                            label="Full 1024-D (underidentified)"
                            if dim == "full"
                            else f"Independent PCA {dim}-D",
                        )
                        if dim == "32":
                            null = [
                                m[split]["shuffled_fit"]["r2"] if m["valid"] else np.nan
                                for m in mappings
                            ]
                            ax.plot(
                                x,
                                null,
                                color=color,
                                linestyle=":",
                                alpha=0.7,
                                label="PCA 32-D shuffled-fit control",
                            )
                    ax.axhline(
                        0,
                        color="black",
                        linewidth=0.7,
                        label="Predict fit-mean baseline",
                    )
                    ax.axhline(
                        1,
                        color="#53915b",
                        linewidth=0.7,
                        linestyle="--",
                        label="Exact shape match",
                    )
                    target = "COCO" if family == "imagenet" else "ImageNet"
                    domain = (
                        family.upper()
                        if split == "test"
                        else f"{family.upper()} → {target}"
                    )
                    ax.set_title(f"{domain} — {names[readout]}")
                    ax.set_ylabel("Orthogonal-map R² (1 − normalized squared error)")
                    axis_style(ax)
            axes[0, 0].legend(fontsize=7, loc="lower left")
            fig.suptitle(
                f"B native EMA | {title} | {mode_name}\nFit-only centering/PCA/global RMS units/Q; no target-domain adaptation",
                fontsize=12,
            )
            save(fig, f"procrustes-{split}-{mode}")


if __name__ == "__main__":
    main()
