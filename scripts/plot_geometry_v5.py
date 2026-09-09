"""Render traceable V5 layer/readout, held-out, ARO and robustness figures.

Default requires every setting. --settings is for separate, explicitly partial
rendering preflights only; it never writes the formal figures directory.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.bootstrap_geometry_v5 import endpoint_plan
from utils.research.geometry_v5_assets import RUN, emit, write_json

PRIMARY = [
    "b_native",
    "f_native",
    "dinov2_qwen",
    "mae_qwen",
    "siglip",
    "janusflow_understanding",
    "janusflow_generation",
    "showo2_understanding",
    "showo2_generation",
]
LABELS = {
    "b_native": "B",
    "f_native": "F",
    "dinov2_qwen": "DINOv2 + Qwen",
    "mae_qwen": "MAE + Qwen",
    "siglip": "SigLIP content",
    "siglip_native": "SigLIP trained pooler",
    "janusflow_understanding": "JanusFlow understanding",
    "janusflow_generation": "JanusFlow generation",
    "showo2_understanding": "Show-o2 understanding",
    "showo2_generation": "Show-o2 generation",
    "b_bare": "B bare",
    "b_neutral": "B neutral",
    "f_bare": "F bare",
    "f_neutral": "F neutral",
}
METRICS = {
    "linear_cka": "Linear CKA",
    "rsa_spearman": "Distance RSA",
    "knn_10": "kNN@10 overlap",
}
COLORS = {s: plt.get_cmap("tab10")(i % 10) for i, s in enumerate(PRIMARY)}
COLORS["siglip_native"] = "#555555"


def save(fig, directory, name, manifest, scope):
    fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {
            "name": name,
            "png": str(directory / f"{name}.png"),
            "pdf": str(directory / f"{name}.pdf"),
            "scope": scope,
        }
    )
    emit("v5_figure_saved", name=name)


def find_endpoint(
    endpoints,
    setting,
    family,
    mode,
    endpoint="fixed",
    dimension=32,
    readout=None,
    fit=None,
):
    readout = readout or (
        "native_endpoint" if setting == "siglip_native" else "content_mean"
    )
    fit = fit or (600 if family == "imagenet" else 8192)
    found = [
        r
        for r in endpoints[setting]
        if r["readout"] == readout
        and r["family"] == family
        and r["mode"] == mode
        and r["endpoint"] == endpoint
        and r["fit_points"] == fit
        and (endpoint != "fixed" or r["requested_dimension"] == dimension)
    ]
    assert len(found) == 1, (setting, family, mode, endpoint, dimension, readout, fit)
    return found[0]


def layer_figures(curves, directory, manifest):
    for mode in ("centered_euclidean", "unit_sphere"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.3), sharex=True)
        for setting in PRIMARY:
            if setting not in curves:
                continue
            rows = sorted(
                [r for r in curves[setting] if r["readout"] == "content_mean"],
                key=lambda r: r["pair_index"],
            )
            x = [
                r["pair"]["relative_depth"]
                + (0.035 if r["pair"]["kind"] == "final_norm" else 0)
                for r in rows
            ]
            for i, family in enumerate(("imagenet", "coco")):
                for j, (metric, label) in enumerate(METRICS.items()):
                    values = [
                        r["families"][family][mode]["geometry"]["primary"] for r in rows
                    ]
                    axes[i, j].plot(
                        x,
                        [v["scores"][metric] if v["valid"] else np.nan for v in values],
                        color=COLORS[setting],
                        lw=1.7,
                        label=LABELS[setting],
                    )
                    axes[i, j].set_title(f"{family.upper()} | {label}")
                    axes[i, j].grid(alpha=0.18)
                    axes[i, j].set_xticks(
                        [0, 0.25, 0.5, 0.75, 1, 1.035],
                        ["0", ".25", ".5", ".75", "1", "N"],
                    )
                    axes[i, j].set_xlabel("relative block depth; N = final norm")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(
            f"Common content mean | {mode} | fixed identities, no cross-modal map"
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.96))
        save(
            fig,
            directory,
            f"layers-common-{mode}",
            manifest,
            "all actual layers; matched relative depths; primary common content mean",
        )
    for setting, rows in curves.items():
        if setting == "siglip_native":
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
        for readout in sorted({r["readout"] for r in rows}):
            curve = sorted(
                [r for r in rows if r["readout"] == readout],
                key=lambda r: r["pair_index"],
            )
            x = [r["pair_index"] for r in curve]
            for i, family in enumerate(("imagenet", "coco")):
                for j, (metric, label) in enumerate(METRICS.items()):
                    geo = [
                        r["families"][family]["centered_euclidean"]["geometry"][
                            "primary"
                        ]
                        for r in curve
                    ]
                    axes[i, j].plot(
                        x,
                        [g["scores"][metric] if g["valid"] else np.nan for g in geo],
                        label=readout,
                    )
                    axes[i, j].set_title(f"{family.upper()} | {label}")
                    axes[i, j].set_xlabel("declared pair index (last = final norm)")
                    axes[i, j].grid(alpha=0.18)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
        fig.suptitle(f"{LABELS[setting]} | readout sensitivity | centered Euclidean")
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        save(
            fig,
            directory,
            f"readouts-{setting}",
            manifest,
            "all predeclared readouts; heterogeneous native roles, not matched pooling",
        )


def endpoint_figures(endpoints, directory, manifest):
    settings = [s for s in PRIMARY + ["siglip_native"] if s in endpoints]
    for endpoint in ("fixed", "dev_selected"):
        fig, axes = plt.subplots(
            1, 4, figsize=(16, max(4.5, len(settings) * 0.43)), sharey=True
        )
        for i, family in enumerate(("imagenet", "coco")):
            for j, split in enumerate(("test", "transfer_test")):
                ax = axes[2 * i + j]
                for k, setting in enumerate(settings):
                    row = find_endpoint(
                        endpoints, setting, family, "centered_euclidean", endpoint
                    )
                    if not row["valid"]:
                        continue
                    point = row[split]["paired"]["r2"]
                    low, high = row[split]["bootstrap"]["r2_95_interval"]
                    ax.plot([low, high], [k, k], color=COLORS[setting], lw=2)
                    ax.scatter(point, k, color=COLORS[setting], s=22)
                    ax.text(
                        high,
                        k + 0.15,
                        f"d={row['dimension']}" if endpoint == "dev_selected" else "",
                        fontsize=6,
                    )
                ax.axvline(0, color="black", lw=0.7, ls="--")
                ax.grid(axis="x", alpha=0.18)
                ax.set_title(
                    f"{family.upper()} fit → {'held-out' if split == 'test' else 'cross-domain'}"
                )
                ax.set_xlabel("orthogonal-map R² (95% conditional CI)")
                ax.set_yticks(range(len(settings)), [LABELS[s] for s in settings])
        axes[0].invert_yaxis()
        fig.suptitle(
            "Fixed final norm, 32D"
            if endpoint == "fixed"
            else "Source-dev selected layer + dimension (32/128/512)"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save(
            fig,
            directory,
            f"mapping-{endpoint}",
            manifest,
            "source-only fits; full frozen transfer; common content means plus separately labelled trained SigLIP endpoint",
        )
    fig, axes = plt.subplots(2, 2, figsize=(11, max(6.5, len(settings) * 0.63)))
    for i, family in enumerate(("imagenet", "coco")):
        for j, split in enumerate(("test", "transfer_test")):
            values = np.array(
                [
                    [
                        find_endpoint(
                            endpoints, s, family, "centered_euclidean", dimension=d
                        )
                        .get(split, {})
                        .get("paired", {})
                        .get("r2", np.nan)
                        for d in (32, 128, 512)
                    ]
                    for s in settings
                ]
            )
            ax = axes[i, j]
            bound = max(1, np.nanmax(np.abs(values)))
            im = ax.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
            for (r, c), value in np.ndenumerate(values):
                ax.text(
                    c,
                    r,
                    f"{value:.2f}" if np.isfinite(value) else "invalid",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            ax.set_yticks(
                range(len(settings)), [LABELS[s] for s in settings], fontsize=8
            )
            ax.set_xticks(range(3), [32, 128, 512])
            ax.set_title(f"{family.upper()} fit | {split} R²")
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(
        "Fixed final norm: common subspace budget sensitivity (no test selection)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(
        fig,
        directory,
        "mapping-dimensions",
        manifest,
        "all common dimensions, invalid cells explicit, negative values retained",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for setting in settings:
        for ax, split in zip(axes, ("test", "transfer_test"), strict=True):
            values = [
                find_endpoint(endpoints, setting, "coco", "centered_euclidean", fit=n)
                for n in (512, 2048, 8192)
            ]
            ax.plot(
                [512, 2048, 8192],
                [v[split]["paired"]["r2"] if v["valid"] else np.nan for v in values],
                marker="o",
                color=COLORS[setting],
                label=LABELS[setting],
            )
            ax.set_xscale("log", base=2)
            ax.set_xticks([512, 2048, 8192], [512, 2048, 8192])
            ax.set_title(split)
            ax.set_xlabel("COCO fit scene count")
            ax.set_ylabel("R², fixed final norm / 32D")
            ax.axhline(0, color="black", lw=0.7, ls="--")
            ax.grid(alpha=0.18)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    save(
        fig,
        directory,
        "coco-fit-size",
        manifest,
        "fixed test identities and 32D final norm across all three fit sizes",
    )
    for metric in ("cosine", "distance"):
        fig, axes = plt.subplots(
            2, 2, figsize=(12, max(7, len(settings) * 0.74)), sharey=True
        )
        for i, endpoint in enumerate(("fixed", "dev_selected")):
            for j, task in enumerate(("aro_vg_attribution", "aro_vg_relation")):
                ax = axes[i, j]
                for k, setting in enumerate(settings):
                    row = find_endpoint(
                        endpoints, setting, "coco", "centered_euclidean", endpoint
                    )
                    if not row["valid"]:
                        continue
                    data = row["aro"][task]
                    low, high = data["bootstrap"][metric]["accuracy_95"]
                    ax.plot([low, high], [k, k], color=COLORS[setting], lw=2)
                    ax.scatter(
                        data["paired_fit"][metric], k, color=COLORS[setting], s=24
                    )
                    ax.scatter(
                        data["shuffled_image"][metric],
                        k,
                        color=COLORS[setting],
                        marker="x",
                        s=24,
                    )
                ax.set_title(
                    f"{endpoint} | {'attribute (451)' if j == 0 else 'relation (417)'}"
                )
                ax.set_yticks(
                    range(len(settings)), [LABELS[s] for s in settings], fontsize=8
                )
                ax.axvline(0.5, color="black", lw=0.7, ls="--")
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel("accuracy: dot = paired map + 95% CI; × = shuffled image")
                ax.grid(axis="x", alpha=0.18)
                ax.set_ylim(len(settings) - 0.5, -0.5)
        fig.suptitle(
            f"Strict identity-disjoint ARO | {metric} | COCO8192 map, no ARO fitting/selection"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(
            fig,
            directory,
            f"aro-{metric}",
            manifest,
            "both tasks and fixed/dev-selected endpoints, raw paired accuracy and image-shuffle control",
        )


def robustness_figures(root, endpoints, directory, manifest):
    for setting in (
        "b_native",
        "f_native",
        "janusflow_generation",
        "showo2_generation",
    ):
        if setting not in endpoints:
            continue
        audit = json.loads((root / "audits" / f"robustness-{setting}.json").read_text())
        reports = [json.loads(Path(p).read_text()) for p in audit["paths"]]
        rows = [
            r
            for r in reports
            if r["endpoint"]["endpoint"] == "fixed"
            and r["endpoint"]["dimension"] == 32
            and r["endpoint"]["mode"] == "centered_euclidean"
        ]
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        for i, source in enumerate(("imagenet", "coco")):
            for j, target in enumerate(("imagenet", "coco")):
                points = [
                    (r["endpoint"]["readout"], v)
                    for r in rows
                    if r["endpoint"]["family"] == source
                    for v in r["rows"]
                    if v["target_family"] == target
                ]
                points.sort(key=lambda v: (v[0], v[1]["profile"]))
                ax = axes[i, j]
                for k, (_, value) in enumerate(points):
                    lo, hi = value["error_reduction_95"]
                    ax.plot([lo, hi], [k, k], color="#336699", lw=1.5)
                    ax.scatter(
                        value["error_reduction_over_original_denominator"],
                        k,
                        color="#336699",
                        s=14,
                    )
                ax.set_yticks(
                    range(len(points)),
                    [f"{pool} | {v['profile']}" for pool, v in points],
                    fontsize=6.5,
                )
                ax.set_ylim(len(points) - 0.5, -0.5)
                ax.set_title(f"{source.upper()} fit → {target.upper()} robust test")
                ax.axvline(0, color="black", ls="--", lw=0.7)
                extreme = max(
                    abs(v) for _, value in points for v in value["error_reduction_95"]
                )
                label = "error reduction / SAME native denominator (95% CI)"
                if extreme > 20:
                    ax.set_xscale("symlog", linthresh=1)
                    label += "\nsymlog display; linear between -1 and 1"
                ax.set_xlabel(label)
                ax.grid(axis="x", alpha=0.18)
        fig.suptitle(
            f"{LABELS[setting]} | all readouts, fixed final norm / 32D | negative = degraded"
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.96))
        fig.text(
            0.5,
            0.012,
            "All source means/PCA/RMS/Q frozen. B/F sigma also moves native query position; image midpoint reuses unchanged text.",
            ha="center",
            fontsize=7,
        )
        save(
            fig,
            directory,
            f"robustness-{setting}",
            manifest,
            "all readouts at fixed32; 32 ImageNet test classes / 512 COCO test scenes; all three predetermined perturbations; symlog x-axis with linear [-1,1] only when CI magnitudes exceed 20, no value clipping",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--settings")
    args = parser.parse_args()
    root = args.output_dir
    contract = json.loads((root / "comparison-contract.json").read_text())
    settings = args.settings.split(",") if args.settings else list(contract["settings"])
    directory = root / ("figures-preflight" if args.settings else "figures")
    curves, endpoints = {}, {}
    for setting in settings:
        curves[setting] = [
            json.loads(p.read_text())
            for p in sorted((root / "analysis" / setting).glob("*.json"))
        ]
        plans = endpoint_plan(curves[setting], contract["settings"][setting])
        audit = json.loads((root / "audits" / f"endpoints-{setting}.json").read_text())
        assert audit["expected"] == len(plans)
        endpoints[setting] = [json.loads(Path(p).read_text()) for p in audit["paths"]]
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    manifest = []
    layer_figures(curves, directory, manifest)
    endpoint_figures(endpoints, directory, manifest)
    robustness_figures(root, endpoints, directory, manifest)
    write_json(
        directory / "index.json",
        {
            "schema": "geometry_v5_figures_1",
            "partial_preflight": bool(args.settings),
            "settings": settings,
            "figures": manifest,
        },
    )


if __name__ == "__main__":
    main()
