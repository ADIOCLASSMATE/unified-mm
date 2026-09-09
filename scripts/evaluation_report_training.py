"""Read loss histories independently of the final-checkpoint evaluation inventory."""
from __future__ import annotations

from datetime import UTC, datetime
import csv
import io
import json
import math
import os
from pathlib import Path

import yaml

from utils.experiment_registry import GROUPS, is_temporary_training_run, experiment_identity, read_run_identity

FIELDS = ["step_loss", "train/loss_t2i", "train/loss_i2t", "train/loss_climbmix",
          "train/weighted_contribution_t2i", "train/weighted_contribution_i2t",
          "train/weighted_contribution_climbmix"]
VALIDATION_FIELDS = ["val/loss", "val/loss_t2i", "val/loss_i2t", "val/loss_climbmix",
                     "val/weighted_contribution_t2i", "val/weighted_contribution_i2t",
                     "val/weighted_contribution_climbmix"]
COLUMNS = ["step", "total", "t2i", "i2t", "climbmix", "weighted_t2i", "weighted_i2t", "weighted_climbmix"]
PALETTE = ["#087e8b", "#d1495b", "#5c4d9e", "#d58a00", "#27844c", "#2563b5",
           "#aa3377", "#71752c", "#ac6031", "#526578", "#26a59a", "#bd7bad",
           "#6b8e23", "#5450c4", "#eb7134", "#1982a5", "#aa6b05", "#6a7a8b",
           "#b54060", "#427dba", "#73559c", "#3d8270", "#b98439", "#b75627",
           "#478333", "#647dc1", "#8f573d", "#96546e"]


def read_history(path: Path):
    """Accept an unfinished final write; reject damaged complete records.

    On resume/rollback, the later segment replaces the old tail. Missing tasks
    and nonfinite samples stay null; a missing task must never become zero loss.
    """
    points, dropped, nonfinite, partial = {}, 0, 0, False
    protocol = None
    if not path.exists():
        return [], {"partial_tail": False, "replaced_points": 0, "nonfinite_values": 0, "loss_protocol": None}
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    partial = True
                    break
                raise ValueError(f"malformed training log: {path}:{line_no}") from None
            step = record.get("global_step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"invalid training step: {path}:{line_no}")
            if points and step <= next(reversed(points)):
                stale = [s for s in points if s >= step]
                dropped += len(stale)
                for s in stale:
                    del points[s]
            values = []
            for field in FIELDS:
                value = record.get("metrics", {}).get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ValueError(f"invalid loss: {path}:{line_no}: {field}")
                if value is not None and not math.isfinite(value):
                    nonfinite += 1
                    value = None
                values.append(value)
            points[step] = [step, *values]
            protocol = record.get("loss_protocol")
    return list(points.values()), {"partial_tail": partial, "replaced_points": dropped,
                                  "nonfinite_values": nonfinite, "loss_protocol": protocol}


def collect_validation(repo: Path, root: Path, directory: Path, config: dict, *, training_protocol=None):
    """Read this run's periodic loss only, excluding standalone final/EMA evals.

    Canonical migrated files override the same step in the old run directory.
    The old validator emits zero for inactive tasks: require nonzero targets
    before treating those zeroes as observations. val/loss_text is caption CE.
    """
    canonical = repo / "output/evaluation/training-validation" / directory.relative_to(repo / "output")
    paths = {}
    for location in (directory, canonical):
        for pattern in ("validation_metrics_step_*.json", "validation_climbmix_metrics_step_*.json",
                        "validation_unified_loss_metrics_step_*.json"):
            paths.update({p.name: p for p in location.glob(pattern)})
    by_step, metadata, incomplete, nonfinite = {}, {}, [], 0
    for name, path in sorted(paths.items(), key=lambda item: (
            int(item[0].rsplit("_", 1)[-1].removesuffix(".json")),
            2 if "unified_loss" in item[0] else 1 if "climbmix" in item[0] else 0)):
        content = path.read_text()
        try:
            record = json.loads(content)
        except json.JSONDecodeError:
            if not content.endswith("\n"):
                incomplete.append(os.path.relpath(path, root))
                continue
            raise ValueError(f"malformed validation record: {path}") from None
        pure_text = name.startswith("validation_climbmix_metrics_step_")
        unified = name.startswith("validation_unified_loss_metrics_step_")
        expected_schema = ("selfless_unified_loss_validation_metrics_v1" if unified else
                           "selfless_climbmix_validation_metrics_v1" if pure_text else "selfless_flow_validation_metrics_v1")
        if record.get("schema") != expected_schema:
            raise ValueError(f"unknown validation loss schema: {path}")
        if (pure_text or unified) and record.get("complete") is not True:
            raise ValueError(f"incomplete loss validation: {path}")
        step = record.get("global_step")
        prefix = ("validation_unified_loss_metrics_step_" if unified else
                  "validation_climbmix_metrics_step_" if pure_text else "validation_metrics_step_")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0 or name != f"{prefix}{step}.json":
            raise ValueError(f"validation step/filename mismatch: {path}")
        metrics = record.get("metrics", {})
        protocol = record.get("loss_protocol", {})
        if unified:
            active = set(protocol.get("schedule", []))
            if (protocol.get("name") != "unified_schedule_microbatch_mean_v1" or not active
                    or active - {"t2i", "i2t", "climbmix"} or record.get("model_weights") != "current"):
                raise ValueError(f"invalid unified validation protocol: {path}")
            required = ["val/loss", *[f"val/{key}_{source}" for source in active
                                      for key in ("loss", "weighted_contribution")]]
            if any(metrics.get(key) is None for key in required):
                raise ValueError(f"unified validation is missing an active source: {path}")
        values = []
        for field in VALIDATION_FIELDS:
            value = metrics.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"invalid validation loss: {path}: {field}")
            if value is not None and not math.isfinite(value):
                nonfinite += 1
                value = None
            values.append(value)
        counts = {}
        for task, field, columns in (("image", "val/image_target_tokens", (1, 4)),
                                     ("caption", "val/text_target_tokens", (2, 5)),
                                     ("climbmix", "val/climbmix_target_tokens", (3, 6))):
            count = metrics.get(field)
            if count is not None and (isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count) or count < 0):
                raise ValueError(f"invalid validation target count: {path}: {field}")
            counts[task] = count
            if count == 0:
                for column in columns:
                    values[column] = None
        if counts["image"] == counts["caption"] == 0 and values[3] is None:
            values[0] = None
        if unified:
            source_columns = {"t2i": (1, 4, "image"), "i2t": (2, 5, "caption"), "climbmix": (3, 6, "climbmix")}
            if values[0] is None or any(values[source_columns[s][0]] is None or
                    values[source_columns[s][1]] is None or not counts[source_columns[s][2]] for s in active):
                raise ValueError(f"unified validation has invalid active task values/targets: {path}")
            total = sum(values[source_columns[s][1]] for s in active)
            if not math.isclose(values[0], total, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f"unified validation total does not match its contributions: {path}")
            for s in set(source_columns) - active:
                raw, weighted, count_name = source_columns[s]
                values[raw] = values[weighted] = None
                counts[count_name] = 0
        if step not in by_step or unified:
            # A complete unified record replaces this step atomically. Never
            # splice a legacy task or EMA observation into its current weights.
            by_step[step] = [step, *values]
            metadata[step] = {"source": os.path.relpath(path, root),
                              "validation_seed": record.get("validation_seed"),
                              "training_seed": record.get("training_seed"), "target_counts": counts,
                              "loss_protocol": protocol if unified else {"name": "legacy_text_only" if pure_text else "legacy_imagenet_pair"},
                              "model_weights": record.get("model_weights"), "model_mode": record.get("model_mode")}
        else:
            for i, value in enumerate(values, 1):
                if value is not None:
                    by_step[step][i] = value
            metadata[step]["target_counts"].update({k: v for k, v in counts.items() if v is not None})
        if pure_text:
            metadata[step].update(pure_text_source=os.path.relpath(path, root),
                                  pure_text_independence=record.get("independence"),
                                  pure_text_protocol=record.get("protocol"),
                                  pure_text_data=record.get("source"), pure_text_weights=record.get("model_weights"))
        if unified:
            metadata[step]["subsets"] = record.get("subsets")
            image_subset = record.get("imagenet_subset")
            # Full IDs remain in the linked raw artifact; retain the sampling
            # contract here so a seed/subset change breaks the plotted line.
            metadata[step]["imagenet_subset"] = ({k: v for k, v in image_subset.items()
                if k not in {"sample_ids", "img_ids", "class_indices"}} if image_subset else None)
            text = record.get("pure_text")
            if text is not None:
                metadata[step].update(pure_text_source=os.path.relpath(path, root),
                    pure_text_independence=text.get("independence"), pure_text_protocol=text.get("protocol"),
                    pure_text_data=text.get("source"), pure_text_weights="current")
    points = [by_step[s] for s in sorted(by_step)]
    details = [metadata[s] for s in sorted(by_step)]
    experiment = config.get("experiment", {})
    interval = experiment.get("val_every", 2000)
    if interval is None:
        interval = 2000
    unified_expected = (isinstance(training_protocol, dict)
                        and training_protocol.get("name") == "unified_schedule_microbatch_mean_v1"
                        and (experiment.get("loss_validation") or {}).get("enabled", True))
    training = config.get("training", {})
    weights = ("EMA（训练配置）" if training.get("use_ema") and training.get("ema_validate") else
               "当前训练权重（训练配置）" if "ema_validate" in training else "记录未注明权重类型")
    if details and all(d.get("pure_text_source") == d["source"] and d.get("pure_text_weights") == "current" for d in details):
        weights = "当前训练权重（验证记录）"
    if details and all(d.get("model_weights") == "current" for d in details):
        weights = "当前训练权重（验证记录）"
    reason = ""
    if not points:
        if unified_expected:
            weights = "当前训练权重（统一 loss 协议）"
        if interval == 0:
            reason = "训练配置未启用周期验证"
        elif unified_expected:
            reason = f"尚未产生与训练同口径的验证 loss；每 {interval:,} 步验证一次"
        elif "downstream_validation" in experiment:
            reason = "尚无验证 loss；配置使用下游任务分数验证"
        elif config.get("dataset", {}).get("params", {}).get("schedule") == ["climbmix"]:
            reason = "尚无 ClimbMix 纯文本验证 loss"
        else:
            reason = "尚无验证 loss 记录"
    return {"points": points, "details": details, "records": len(points), "log_every": interval,
            "unified_records": sum(d.get("loss_protocol", {}).get("name") == "unified_schedule_microbatch_mean_v1" for d in details),
            "first_logged_step": points[0][0] if points else None,
            "last_logged_step": points[-1][0] if points else None,
            "available": {key: sum(row[i] is not None for row in points) for i, key in enumerate(COLUMNS) if i},
            "model_weights": weights, "missing_reason": reason,
            "validation_max_batches": config.get("experiment", {}).get("validation_max_batches"),
            "incomplete_files": incomplete, "nonfinite_values": nonfinite,
            "gaps": sum(b[0] - a[0] > interval * 1.5 for a, b in zip(points, points[1:])) if interval > 0 else 0}


def collect_training(repo: Path, root: Path, models: list[dict]):
    output = repo / "output"
    known = {m["run"]: m for m in models}
    inventory = {}
    # Only the unified training namespace. Evaluation snapshots and smoke/debug
    # replicas are not fresh ablation runs and cannot provide formal histories.
    for directory in sorted(output.glob("unified-*")):
        if is_temporary_training_run(directory.name):
            continue
        if "-lr-sweep-" in directory.name:
            for config in sorted(directory.glob("*/config.yaml")):
                inventory[config.parent] = config
        elif (directory / "config.yaml").exists() or directory.name in known:
            inventory[directory] = directory / "config.yaml"
    # Include configured controls/new arms before their first metric or checkpoint exists.
    configured = [*(repo / "configs/selfless").glob("unified_b_x0_flow_depth*_100b_ascend64.yaml"),
                  *(repo / "configs/selfless").glob("unified_single_*_0p6b_100b_ascend16.yaml")]
    for config in sorted(configured):
        declared = yaml.safe_load(config.read_text())
        directory = output / declared["experiment"]["project"]
        inventory.setdefault(directory, config)
    for run in known:
        inventory.setdefault(output / run, output / run / "config.yaml")

    runs = []
    for directory, config_path in inventory.items():
        name = str(directory.relative_to(output))
        spec = known.get(name, {})
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        params = config.get("dataset", {}).get("params", {})
        schedule = params.get("schedule", [])
        training = config.get("training", {})
        target = training.get("max_train_steps")
        if training.get("stop_after_steps"):
            target = min(target, training["stop_after_steps"]) if target else training["stop_after_steps"]
        identity = (read_run_identity(directory, config, presentation=spec)
                    if (directory / "experiment_identity.json").exists()
                    else experiment_identity(name, config, presentation=spec))
        if identity["purpose"] == "temporary":
            continue
        group, label = identity["group"], identity["label"]
        metrics_path = directory / "training_metrics.jsonl"
        points, diagnostics = read_history(metrics_path)
        runtime_path = directory / "training_runtime_metrics.json"
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        recorded_step = points[-1][0] if points else None
        reached = max(runtime.get("global_step", 0), recorded_step or 0)
        complete = bool(target and reached >= target)
        status = "训练已完成" if complete else "已记录部分训练" if points else "等待训练日志"
        available = {key: sum(row[i] is not None for row in points) for i, key in enumerate(COLUMNS) if i}
        expected_interval = config.get("experiment", {}).get("log_every", 10)
        gaps = sum(b[0] - a[0] > expected_interval * 1.5 for a, b in zip(points, points[1:]))
        relative = lambda p: os.path.relpath(p, root)
        stat = metrics_path.stat() if metrics_path.exists() else None
        runs.append({"id": identity["id"], "experiment_identity": identity, "label": label, "run": name, "group": group,
            "config": relative(config_path) if config_path.exists() else None,
            "source": relative(metrics_path) if stat else None,
            "source_bytes": stat.st_size if stat else 0,
            "source_updated_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds") if stat else None,
            "runtime": relative(runtime_path) if runtime_path.exists() else None,
            "status": status, "complete": complete, "target_steps": target, "reached_step": reached,
            "last_logged_step": recorded_step, "first_logged_step": points[0][0] if points else None,
            "records": len(points), "schedule": schedule,
            "text_weight": config.get("model", {}).get("lambda_text"),
            "image_weight": config.get("model", {}).get("lambda_image"),
            "world_size": runtime.get("world_size"), "log_every": expected_interval,
            "available": available, "gaps": gaps, **diagnostics, "points": points,
            "validation": collect_validation(repo, root, directory, config,
                                             training_protocol=diagnostics["loss_protocol"])})
    group_order = list(GROUPS)
    model_order = {m["id"]: n for n, m in enumerate(models)}
    runs.sort(key=lambda r: (group_order.index(r["group"]), model_order.get(r["id"], 100), r["label"]))
    for i, run in enumerate(runs):
        run["color"] = PALETTE[i % len(PALETTE)]
    return {"schema": "unified_training_loss_report_v2", "columns": COLUMNS, "metric_fields": FIELDS,
        "validation_metric_fields": VALIDATION_FIELDS,
        "groups": GROUPS, "runs": runs, "run_count": len(runs),
        "runs_with_history": sum(bool(r["points"]) for r in runs),
        "records": sum(r["records"] for r in runs),
        "runs_with_validation": sum(bool(r["validation"]["points"]) for r in runs),
        "validation_records": sum(r["validation"]["records"] for r in runs),
        "scope": "output/unified-* formal runs, single-source controls, historical controls and 1.7B LR sweep; smoke/debug/replay excluded",
        "normalization": "T2I: mean flow velocity MSE; I2T / ClimbMix: mean CE over valid targets; step_loss and weighted contributions include model weights and gradient accumulation",
        "validation_normalization": "Unified records use the training schedule, model weights and microbatch-mean total; raw tasks are target-weighted means. Legacy ImageNet records contain only lambda_image*T2I + lambda_text*I2T and are labeled separately. val/loss_text is legacy caption CE.",
        "json": "training-loss/curves.json", "csv": "training-loss/curves.csv",
        "validation_csv": "training-loss/validation.csv"}


def export_training(root: Path, data: dict, write):
    write(root / data["json"], json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["model_id", "model_label", "group", *COLUMNS])
    for run in data["runs"]:
        for point in run["points"]:
            writer.writerow([run["id"], run["label"], run["group"], *point])
    write(root / data["csv"], buffer.getvalue())
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["model_id", "model_label", "group", *COLUMNS, "source", "validation_seed",
                     "training_seed", "image_target_tokens", "caption_target_tokens", "climbmix_target_tokens",
                     "pure_text_source", "pure_text_independence", "loss_protocol", "model_weights", "model_mode", "imagenet_subset"])
    for run in data["runs"]:
        for point, detail in zip(run["validation"]["points"], run["validation"]["details"]):
            writer.writerow([run["id"], run["label"], run["group"], *point, detail["source"],
                             detail["validation_seed"], detail["training_seed"],
                             *[detail["target_counts"][key] for key in ("image", "caption", "climbmix")],
                             detail.get("pure_text_source"), detail.get("pure_text_independence"),
                             json.dumps(detail.get("loss_protocol"), ensure_ascii=False),
                             detail.get("model_weights"), detail.get("model_mode"),
                             json.dumps(detail.get("imagenet_subset"), ensure_ascii=False)])
    write(root / data["validation_csv"], buffer.getvalue())
    return {**data, "runs": [{**{k: v for k, v in r.items() if k not in {"points", "validation"}},
                             "validation": {k: v for k, v in r["validation"].items() if k not in {"points", "details"}}}
                            for r in data["runs"]]}


def plot_training(root: Path, data: dict):
    """Standalone scientific figures; retain every sample in JSON/CSV above."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    font = next((f.name for f in font_manager.fontManager.ttflist if f.name in
                 {"Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"}), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    artifacts = []
    for group, title in {"all": "All unified ablations", **GROUPS}.items():
        runs = [r for r in data["runs"] if (r["points"] or r["validation"]["points"])
                and (group == "all" or r["group"] == group or (group == "ablation" and r["id"] == "b_x0"))]
        if not runs:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        handles, labels = [], []
        for run in runs:
            handles.append(Line2D([], [], color=run["color"], lw=1.5))
            labels.append(run["label"])
            validation = run["validation"]
            for column, ax in enumerate(axes.flat, 1):
                xval, yval, protocols, previous = [], [], [], None
                for row, detail in zip(validation["points"], validation["details"]):
                    if previous and (row[0] - previous[0] > validation["log_every"] * 1.5 or
                                     detail["validation_seed"] != previous[1]["validation_seed"] or
                                     detail["target_counts"] != previous[1]["target_counts"] or
                                     detail.get("loss_protocol") != previous[1].get("loss_protocol") or
                                     detail.get("imagenet_subset") != previous[1].get("imagenet_subset") or
                                     detail.get("model_weights") != previous[1].get("model_weights")):
                        xval.append(float("nan")); yval.append(float("nan")); protocols.append(None)
                    xval.append(row[0]); yval.append(row[column] if row[column] is not None else float("nan"))
                    protocols.append(detail.get("loss_protocol", {}).get("name") == "unified_schedule_microbatch_mean_v1")
                    previous = (row[0], detail)
                if any(math.isfinite(y) for y in yval):
                    for unified, style in ((True, "--"), (False, ":")):
                        selected = [v if p is unified else float("nan") for v, p in zip(yval, protocols)]
                        if any(math.isfinite(v) for v in selected):
                            ax.plot(xval, selected, color=run["color"], ls=style, lw=1.4, marker="o", ms=3,
                                    markerfacecolor="white", zorder=3)
            if not run["points"]:
                continue
            values = np.array([[float("nan") if v is None else v for v in r] for r in run["points"]])
            steps = values[:, 0]
            for column, ax in enumerate(axes.flat, 1):
                y = values[:, column]
                if not np.isfinite(y).any():
                    continue
                # A trailing 20-record mean, reset at missing data or log gaps.
                smooth, window = [], []
                for i, v in enumerate(y):
                    if not math.isfinite(v) or (i and steps[i] - steps[i-1] > run["log_every"] * 1.5):
                        window = []
                    if math.isfinite(v):
                        window.append(v)
                        window = window[-20:]
                        smooth.append(sum(window) / len(window))
                    else:
                        smooth.append(float("nan"))
                # NaN separator prevents connecting missing optimizer steps.
                xplot, yplot = [], []
                for i, (x, yv) in enumerate(zip(steps, smooth)):
                    if i and x - steps[i-1] > run["log_every"] * 1.5:
                        xplot.append(float("nan")); yplot.append(float("nan"))
                    xplot.append(x); yplot.append(yv)
                ax.plot(xplot, yplot, color=run["color"], lw=1.2, label=run["label"])
        text_title = "ClimbMix · text CE" + ("" if any(r["validation"]["available"]["climbmix"] for r in runs) else " (validation not yet recorded)")
        for ax, title_metric in zip(axes.flat, ("Total loss (legacy validation has a different protocol)", "T2I · flow MSE", "I2T · caption CE", text_title)):
            ax.set_title(title_metric, loc="left")
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Loss")
            ax.grid(alpha=.18)
            if not ax.lines:
                ax.text(.5, .5, "No task loss recorded", ha="center", va="center", transform=ax.transAxes)
        handles.extend([Line2D([], [], color="#33444a", lw=1.5),
                        Line2D([], [], color="#33444a", lw=1.5, ls="--", marker="o", ms=4, markerfacecolor="white"),
                        Line2D([], [], color="#33444a", lw=1.5, ls=":", marker="o", ms=4, markerfacecolor="white")])
        labels.extend(["Train · trailing 20 points", "Validation · training loss protocol", "Validation · legacy protocol"])
        text_probe_note = ("\nClimbMix validation includes source rows potentially seen during training"
                           if any(d.get("pure_text_independence") == "may_have_been_seen_in_training"
                                  for r in runs for d in r["validation"]["details"]) else "")
        fig.suptitle(f"{title} · training and validation loss\nSnapshot: {data['updated_at']} · Raw data: curves.csv / validation.csv{text_probe_note}", fontsize=14)
        legend_rows = math.ceil(len(labels) / 3)
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, .035 + legend_rows * .023, 1, .91 if text_probe_note else .94))
        for suffix in ("png", "svg"):
            path = root / "training-loss" / f"{group}.{suffix}"
            temporary = path.with_name(path.stem + ".tmp." + suffix)
            fig.savefig(temporary, dpi=160, facecolor="white")
            temporary.replace(path)
            artifacts.append({"label": title, "group": group, "format": suffix, "path": str(path.relative_to(root))})
        plt.close(fig)
    return artifacts
