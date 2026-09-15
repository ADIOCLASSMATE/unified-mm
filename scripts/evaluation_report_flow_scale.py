"""Export validated flow-head CFG sweeps without connecting missing evaluations."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from pathlib import Path


def _plot(rows, cfgs, directory, family):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = ("#12655f", "#c47b37", "#5866a6")
    markers = ("o", "s", "^")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#172c35", "text.color": "#172c35",
                         "axes.edgecolor": "#b2c2c5", "grid.color": "#dce5e6",
                         "svg.fonttype": "none", "pdf.fonttype": 42})

    def axis(ax, metric):
        for row, color, marker in zip(rows, colors, markers):
            points = row["results"]
            if not any(point["source"] for point in points):
                continue
            ys = [point[metric] if point["source"] else math.nan for point in points]
            ax.plot(cfgs, ys, color=color, marker=marker, linewidth=2.1, markersize=5,
                    label=f'{family.upper()} depth {row["depth"]} ({row["completed"]}/{len(cfgs)})')
            if metric == "is":
                stds = [point["is_std"] if point["source"] else math.nan for point in points]
                ax.fill_between(cfgs, [y-s for y, s in zip(ys, stds)],
                                [y+s for y, s in zip(ys, stds)], color=color, alpha=0.10, linewidth=0)
        ax.set(xlabel="Classifier-free guidance (CFG)",
               ylabel="FID" if metric == "fid" else "Inception Score")
        ax.set_title("FID ↓  Lower is better" if metric == "fid" else "IS ↑  Higher is better", loc="left", pad=13)
        ax.set_xticks(cfgs)
        ax.set_xlim(min(cfgs)-0.12, max(cfgs)+0.12)
        ax.grid(alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right" if metric == "fid" else "lower right")

    artifacts = {}

    def save(fig, name, formats):
        for fmt in formats:
            destination = directory / f"{family}-{name}.{fmt}"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, suffix="."+fmt, delete=False) as handle:
                    temporary = Path(handle.name)
                fig.savefig(temporary, format=fmt, dpi=220, bbox_inches="tight", facecolor="white")
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            artifacts[f"{name}_{fmt}"] = destination.name

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), layout="constrained")
    for ax, metric in zip(axes, ("fid", "is")):
        axis(ax, metric)
    fig.suptitle(f"{family.upper()} flow head scale · CFG sweep\n"
                 "ImageNet-val 50K | final EMA | Heun 10 | Halton | seed 42", fontsize=12, y=1.07)
    fig.supxlabel("Missing evaluations break the lines. IS shading: ±1 split standard deviation; not a confidence interval.", fontsize=9)
    save(fig, "cfg-sweep", ("png", "pdf", "svg"))
    plt.close(fig)
    for metric in ("fid", "is"):
        fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
        axis(ax, metric)
        save(fig, "cfg-"+metric, ("svg",))
        plt.close(fig)
    return artifacts


def export_flow_head_sweep(root, study, write, *, plots=False):
    relative = "comparisons/flow-head-scale"
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    rows, cfgs = study["rows"], study["cfg_values"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("model", "family", "depth", "width", "head_parameters",
                                                "cfg", "fid", "is", "is_std", "source"))
    writer.writeheader()
    best = []
    for row in rows:
        for point in row["results"]:
            writer.writerow({"model": row["id"], **{key: row[key] for key in
                             ("family", "depth", "width", "head_parameters")}, **point_without_batch(point)})
        present = [point for point in row["results"] if point["source"]]
        best.append({"id": row["id"], "family": row["family"], "depth": row["depth"],
                     "completed": len(present), "total": len(cfgs),
                     "best_fid": min(present, key=lambda p: p["fid"]) if present else None,
                     "best_is": max(present, key=lambda p: p["is"]) if present else None})
    write(directory / "cfg-sweep.csv", stream.getvalue())
    result = {"csv": relative+"/cfg-sweep.csv", "best": best, "plots": {}}
    for family in sorted({row["family"] for row in rows}):
        selected = sorted((row for row in rows if row["family"] == family), key=lambda r: r["depth"])
        if not any(row["completed"] for row in selected):
            continue
        plot_data = {"cfg_values": cfgs, "rows": [{key: row[key] for key in
                     ("id", "depth", "completed", "results")} for row in selected]}
        manifest = directory / f"{family}-cfg-plot-data.json"
        if plots:
            artifacts = _plot(selected, cfgs, directory, family)
            write(manifest, json.dumps({"data": plot_data, "artifacts": artifacts}, ensure_ascii=False, indent=2)+"\n")
        if manifest.exists():
            cached = json.loads(manifest.read_text())
            if cached["data"] == plot_data and all((directory/name).is_file() for name in cached["artifacts"].values()):
                result["plots"][family] = {key: relative+"/"+name for key, name in cached["artifacts"].items()}
    write(directory / "cfg-sweep.json", json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result


def point_without_batch(point):
    return {key: point[key] for key in ("cfg", "fid", "is", "is_std", "source")}
