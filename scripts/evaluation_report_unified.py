"""Export the validated joint / only CFG comparison as tables and figures."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from pathlib import Path


def generation_points(study):
    rows = {(row["cfg"], row["key"]): row for row in study["rows"] if row["task"] == "generation"}
    points = []
    for cfg in study["generation_cfg_values"]:
        if (cfg, "fid") not in rows:
            continue
        fid, score = rows[cfg, "fid"], rows[cfg, "is"]
        point = {"cfg": cfg, "complete": bool(fid["only"] and score["only"])}
        for side in ("baseline", "only"):
            point[side] = None if fid[side] is None else {
                "fid": fid[side]["value"], "is": score[side]["value"],
                "is_std": score[side].get("std"), "source": fid[side]["source"],
            }
        point.update(delta_fid=fid["delta"], delta_is=score["delta"])
        points.append(point)
    return points


def _best(points, side):
    present = [point for point in points if point[side] is not None]
    if not present:
        return None
    fid = min(present, key=lambda point: point[side]["fid"])
    score = max(present, key=lambda point: point[side]["is"])
    return {"best_fid": {"cfg": fid["cfg"], **fid[side]},
            "best_is": {"cfg": score["cfg"], **score[side]},
            "completed": len(present), "total": len(points)}


def _plot(points, directory):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    styles = (("baseline", "B (unified)", "#12655f", "o"),
              ("only", "T2I-only", "#c47b37", "s"))
    cfgs = [point["cfg"] for point in points]
    completed = sum(point["complete"] for point in points)
    suffix = "" if completed == len(points) else f" | T2I-only: {completed}/{len(points)} CFG values complete"
    subtitle = "ImageNet-val 50K | final EMA | Heun 10 | Halton | seed 42" + suffix
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#172c35", "text.color": "#172c35",
                         "axes.edgecolor": "#b2c2c5", "grid.color": "#dce5e6",
                         "svg.fonttype": "none", "pdf.fonttype": 42})

    def values(side, metric):
        return [point[side][metric] if point[side] is not None else math.nan for point in points]

    def axis(ax, metric):
        for side, label, color, marker in styles:
            ys = values(side, metric)
            ax.plot(cfgs, ys, color=color, marker=marker, linewidth=2.1,
                    markersize=5, label=label)
            if metric == "is":
                stds = values(side, "is_std")
                lower = [y - std if std is not None else math.nan for y, std in zip(ys, stds)]
                upper = [y + std if std is not None else math.nan for y, std in zip(ys, stds)]
                ax.fill_between(cfgs, lower, upper, color=color, alpha=0.12, linewidth=0)
        ax.set(xlabel="Classifier-free guidance (CFG)", ylabel="FID" if metric == "fid" else "Inception Score")
        ax.set_title("FID ↓  Lower is better" if metric == "fid" else "IS ↑  Higher is better", loc="left", pad=13)
        ax.set_xticks(cfgs)
        ax.set_xlim(min(cfgs) - 0.12, max(cfgs) + 0.12)
        ax.grid(alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right" if metric == "fid" else "lower right")

    artifacts = {}

    def save(fig, name, formats):
        for fmt in formats:
            destination = directory / f"{name}.{fmt}"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, suffix="." + fmt, delete=False) as handle:
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
    fig.suptitle("B vs T2I-only · CFG sweep\n" + subtitle, fontsize=12, y=1.07)
    fig.supxlabel("IS shading: ±1 standard deviation across 10 stratified splits; not a confidence interval.", fontsize=9)
    save(fig, "cfg-sweep", ("png", "pdf", "svg"))
    plt.close(fig)
    for metric in ("fid", "is"):
        fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
        axis(ax, metric)
        save(fig, "cfg-" + metric, ("svg",))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 5.6), layout="constrained")
    labels = []
    for side, label, color, marker in styles:
        ax.plot(values(side, "is"), values(side, "fid"), color=color,
                marker=marker, linewidth=1.8, markersize=5, label=label)
        for point in points:
            if point[side] is not None:
                labels.append((side, color, point["cfg"], (point[side]["is"], point[side]["fid"])))
    ax.set(xlabel="Inception Score ↑", ylabel="FID ↓",
           title="FID / IS trade-off · point labels show CFG")
    ax.grid(alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.supxlabel(subtitle, fontsize=9)
    # High CFG points cluster tightly: keep every label readable using short
    # callouts chosen in display coordinates, without shifting the data points.
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [ax.get_legend().get_window_extent(renderer)]
    for _, _, _, xy in labels:
        x, y = ax.transData.transform(xy)
        occupied.append(Bbox.from_bounds(x - 5, y - 5, 10, 10))
    for side, color, cfg, xy in reversed(labels):
        preferred = 6 if side == "baseline" else -13
        offsets = [(dx, dy) for dx in (5, -5, 20, -20, 38, -38, 58, -58, 80, -80)
                   for dy in (preferred, 6, -13, 20, -27, 36, -43, 54, -61)]
        offsets.sort(key=lambda offset: abs(offset[0]) + abs(offset[1] - preferred) * 1.1)
        chosen = offsets[-1]
        for dx, dy in offsets:
            annotation = ax.annotate(f"{cfg:.1f}", xy, xytext=(dx, dy), textcoords="offset points",
                                     color=color, fontsize=8, ha="left" if dx > 0 else "right")
            annotation.set_in_layout(False)
            bounds = annotation.get_window_extent(renderer).expanded(1.12, 1.35)
            fits = (ax.bbox.contains(bounds.x0, bounds.y0) and ax.bbox.contains(bounds.x1, bounds.y1)
                    and not any(bounds.overlaps(box) for box in occupied))
            annotation.remove()
            if fits:
                chosen = (dx, dy)
                occupied.append(bounds)
                break
        dx, dy = chosen
        annotation = ax.annotate(f"{cfg:.1f}", xy, xytext=chosen, textcoords="offset points",
                                 color=color, fontsize=8, ha="left" if dx > 0 else "right",
                                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.2},
                                 arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.55,
                                             "linewidth": 0.6, "shrinkB": 4})
        annotation.set_in_layout(False)
    save(fig, "fid-is-tradeoff", ("png", "pdf", "svg"))
    plt.close(fig)
    return artifacts


def export_generation_sweep(root, study, write, *, plots=False):
    points = generation_points(study)
    if not points:
        return None
    relative = "comparisons/unified-training-ablation"
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    fields = ("cfg", "complete", "b_fid", "only_fid", "delta_fid", "b_is", "b_is_std",
              "only_is", "only_is_std", "delta_is", "b_source", "only_source")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for point in points:
        record = {key: point[key] for key in ("cfg", "complete", "delta_fid", "delta_is")}
        for side, prefix in (("baseline", "b"), ("only", "only")):
            record.update({f"{prefix}_{key}": point[side][key] if point[side] else None
                           for key in ("fid", "is", "is_std", "source")})
        writer.writerow(record)
    write(directory / "cfg-sweep.csv", stream.getvalue())
    paired = [point for point in points if point["complete"]]
    result = {"points": points, "completed": len(paired), "total": len(points),
              "complete": len(paired) == len(points), "baseline": _best(points, "baseline"),
              "only": _best(points, "only"), "csv": relative + "/cfg-sweep.csv", "plots": {},
              "paired_wins": {side: {metric: sum((point[side][metric] < point[other][metric]
                                                if metric == "fid" else point[side][metric] > point[other][metric])
                                                for point in paired) for metric in ("fid", "is")}
                              for side, other in (("baseline", "only"), ("only", "baseline"))}}
    manifest = directory / "cfg-plot-data.json"
    if plots:
        artifacts = _plot(points, directory)
        write(manifest, json.dumps({"points": points, "artifacts": artifacts}, ensure_ascii=False, indent=2) + "\n")
    if manifest.exists():
        data = json.loads(manifest.read_text())
        # A normal report refresh must never display plots from older results.
        if data["points"] == points and all((directory / name).is_file() for name in data["artifacts"].values()):
            result["plots"] = {key: relative + "/" + name for key, name in data["artifacts"].items()}
    return result
