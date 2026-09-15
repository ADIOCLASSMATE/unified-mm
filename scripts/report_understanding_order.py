#!/usr/bin/env python3
"""Compare formal B random MC64 with paired random/Halton MC1 evaluations."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.build_evaluation_report import write
from utils.evaluation.aro import ARO_TASKS, conditional_score, DEBIASED_SCORE

LABELS = {"mmbench_dev_en": "MMBench circular", "seed_bench_image": "SEED image",
          "sugarcrepe": "SugarCrepe", "aro_vg_relation": "ARO Relation",
          "aro_vg_attribution": "ARO Attribution"}
ARM_LABELS = {"random_mc64": "Random MC64", "random_mc1": "Random MC1",
              "halton_mc1": "Fixed Halton MC1", "halton_shifted_mc64": "Shifted Halton MC64"}


def arm_roots(plan):
    return plan.get("comparison_arms") or {"random_mc64": plan["baseline"], **{
        arm["id"].replace("-", "_"): str(Path(plan["output"])/arm["id"])
        for arm in plan["arms"] if not arm["limit"]}}


def task_root(spec, task):
    return Path(spec[task] if isinstance(spec, dict) else spec)


def read(path):
    return json.loads(path.read_text())


def primary(summary):
    value = summary["metrics"]
    for key in value["primary_metric"].split("."):
        value = value[key]
    return value


def load_predictions(path):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = {row["item_index"]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"duplicate prediction IDs: {path}")
    return result


def correct(row):
    scores = row["candidate_scores"]
    label = row["label"]
    if row["task"] in ARO_TASKS:
        return int(conditional_score(scores[label]) > conditional_score(scores[1-label]))
    if row["task"] == "sugarcrepe":
        return int(scores[label][DEBIASED_SCORE] > scores[1-label][DEBIASED_SCORE])
    return int(max(range(len(scores)), key=lambda i: scores[i][DEBIASED_SCORE]) == label)


def paired_comparison(baseline, candidate, *, seed=424242):
    import numpy as np
    if baseline.keys() != candidate.keys():
        raise ValueError("paired comparison has different sample IDs")
    units = defaultdict(list)
    for index, old in baseline.items():
        new = candidate[index]
        for field in ("task", "item_id", "image_id", "label", "category", "metadata"):
            if old[field] != new[field]:
                raise ValueError(f"paired sample differs: {index}/{field}")
        if [(c["text"], c["token_count"]) for c in old["candidate_scores"]] != [
                (c["text"], c["token_count"]) for c in new["candidate_scores"]]:
            raise ValueError(f"paired candidates differ: {index}")
        circular = old["task"] == "mmbench_dev_en"
        meta = old["metadata"]
        cluster = (meta["original_index"] if circular else
                   meta.get("source_image", meta.get("data_id", old["image_id"])))
        units[str(cluster)].append((correct(old), correct(new)))
    # ARO rows from the same source image may have different crop IDs.
    clusters = []
    for group in units.values():
        values = [(int(all(x[0] for x in group)), int(all(x[1] for x in group)))] if circular else group
        clusters.append((sum(b-a for a,b in values), len(values),
                         sum(a == 0 and b == 1 for a,b in values),
                         sum(a == 1 and b == 0 for a,b in values),
                         sum(a for a,b in values), sum(b for a,b in values)))
    values = np.asarray(clusters, dtype=float)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(2000):
        picked = values[rng.integers(len(values), size=len(values))]
        deltas.append(picked[:, 0].sum() / picked[:, 1].sum())
    total = int(values[:, 1].sum())
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"units": total, "image_or_question_clusters": len(clusters),
            "baseline": float(values[:,4].sum()/total), "candidate": float(values[:,5].sum()/total),
            "delta_pp": float(100*values[:,0].sum()/total),
            "ci95_pp": [float(100*lo), float(100*hi)],
            "wrong_to_right": int(values[:,2].sum()), "right_to_wrong": int(values[:,3].sum()),
            "interval_method": "2000 paired cluster bootstrap replicates; image or circular question; exploratory per-task 95% CI"}


def collect(plan):
    baseline = Path(plan["baseline"])
    original_manifest = read(baseline/"manifest.json")
    roots = arm_roots(plan)
    rows = []
    for task, label in LABELS.items():
        row = {"task": task, "label": label, "scores": {}, "comparisons": {}}
        loaded = {}
        for arm, spec in roots.items():
            root = task_root(spec, task)
            summary_path = root/"summaries"/f"{task}.json"
            if not summary_path.exists():
                continue
            manifest = read(root/"manifest.json")
            for key in ("checkpoint", "checkpoint_step", "seed", "model_dtype", "world_size",
                        "asset_manifest_readable_identity", "dual_stream_attention_contract",
                        "scoring_contract", "max_length", "cache_shard_dir", "reported_score_variant"):
                if manifest[key] != original_manifest[key]:
                    raise ValueError(f"{arm}: mismatched {key}")
            expected_order = "spatial_halton_shifted" if arm == "halton_shifted_mc64" else "spatial_halton" if arm == "halton_mc1" else "random"
            if manifest["image_sigma_order"] != expected_order or manifest["mc_samples"] != (64 if arm.endswith("mc64") else 1):
                raise ValueError(f"{arm}: order/MC mismatch")
            if arm == "halton_shifted_mc64" and manifest.get("image_order_distribution") != "halton_base2_base3_uniform_torus_shift_v1":
                raise ValueError("shifted Halton distribution was not recorded")
            summary = read(summary_path)
            if summary["metrics"]["records"] != plan["records"][task]:
                raise ValueError(f"{arm}/{task}: incomplete summary")
            row["scores"][arm] = primary(summary)
            loaded[arm] = load_predictions(root/"predictions"/f"{task}.jsonl")
        pairs = [("random_mc64", arm) for arm in roots if arm != "random_mc64"] + [
            ("random_mc1", "halton_mc1"), ("halton_mc1", "halton_shifted_mc64"), ("random_mc1", "halton_shifted_mc64")]
        for reference, candidate in pairs:
            if reference not in loaded or candidate not in loaded:
                continue
            paired = paired_comparison(loaded[reference], loaded[candidate])
            if abs(paired["baseline"] - row["scores"][reference]) > 1e-12 or abs(paired["candidate"] - row["scores"][candidate]) > 1e-12:
                raise ValueError(f"{task}: saved metrics disagree with paired predictions")
            row["comparisons"][f"{candidate}_minus_{reference}"] = paired
        rows.append(row)
    complete = all(len(row["scores"]) == len(roots) for row in rows)
    return {"schema": "understanding_order_comparison_v1", "updated_at": datetime.now(timezone.utc).isoformat(),
            "complete": complete, "rows": rows, "protocol": plan, "arms": list(roots),
            "candidate_arm": plan.get("comparison_candidate", "halton_mc1"),
            "scope": "Same formal B checkpoint and full task datasets; order ablation kept separate from formal evaluation selection. ARO retains text priors. ImageNet and retrieval are not part of this five-task comparison."}


def export(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    destination.mkdir(parents=True, exist_ok=True)
    write(destination/"comparison.json", json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    rows = report["rows"]
    arms = report["arms"]
    candidate = report["candidate_arm"]
    candidate_label = ARM_LABELS[candidate]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    x = np.arange(len(rows))
    width = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = [row["scores"].get(arm, float("nan"))*100 for row in rows]
        ax.bar(x+(i-(len(arms)-1)/2)*width, vals, width, label=ARM_LABELS[arm], color=("#66778a", "#66b1ac", "#d48348", "#7d62a8")[i])
    ax.set_xticks(x, [row["label"] for row in rows]);ax.set_ylabel("Accuracy / strict win rate (%)")
    ax.set_ylim(0,100);ax.legend();ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True)
    ax.set_title("Formal B: image-order comparison" + ("" if report["complete"] else " (in progress)"))
    for suffix in ("png", "pdf", "svg"):
        tmp = destination/f".comparison.tmp.{suffix}"
        fig.savefig(tmp, dpi=160);tmp.replace(destination/f"comparison.{suffix}")
    plt.close(fig)
    lines = ["# 正式 B：理解评测图像顺序对照", "", report["scope"], "",
             "| 任务 | " + " | ".join(ARM_LABELS[arm] for arm in arms) + f" | {candidate_label} − Random MC64 (pp) | 95% CI (pp) |", "|" + "---|"*(len(arms)+3)]
    for row in rows:
        scores = [f"{row['scores'][arm]*100:.3f}%" if arm in row["scores"] else "待完成" for arm in arms]
        c = row["comparisons"].get(f"{candidate}_minus_random_mc64")
        lines.append("| "+" | ".join([row["label"], *scores, f"{c['delta_pp']:+.3f}" if c else "—", str(c["ci95_pp"]) if c else "—"])+" |")
    write(destination/"REPORT.md", "\n".join(lines)+"\n")
    table = []
    for row in rows:
        cells = [html.escape(row["label"])] + [f"{row['scores'][arm]*100:.3f}%" if arm in row["scores"] else "待完成" for arm in arms]
        for reference in ("random_mc64", "random_mc1"):
            c = row["comparisons"].get(f"{candidate}_minus_{reference}")
            cells += [f"{c['delta_pp']:+.3f} pp<br><small>95% CI [{c['ci95_pp'][0]:+.3f}, {c['ci95_pp'][1]:+.3f}]</small>" if c else "待完成"]
        table.append("<tr>"+"".join(f"<td>{cell}</td>" for cell in cells)+"</tr>")
    conclusions = []
    if report["complete"]:
        for row in rows:
            c = row["comparisons"][f"{candidate}_minus_random_mc64"]
            verdict = "提升" if c["delta_pp"] > 0 else "下降" if c["delta_pp"] < 0 else "持平"
            conclusions.append(f"{html.escape(row['label'])}：{candidate_label} 相比 random MC64 {verdict} {abs(c['delta_pp']):.3f} 个百分点。")
    refresh = "" if report["complete"] else '<meta http-equiv="refresh" content="60">'
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}
<title>正式 B：理解评测图像顺序对照</title><style>body{{font:16px/1.7 system-ui,sans-serif;color:#253445;background:#f7f8fa;margin:40px auto;max-width:1200px;padding:0 24px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:14px;text-align:left;border-bottom:1px solid #dde3e8}}small{{color:#657687}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}a{{color:#1e6f82}}</style>
<a href="../../../../index.html#matrix">← 返回评测网页</a><h1>正式 B：理解评测图像顺序对照</h1>
<p>{'全量对照已完成。' if report['complete'] else '评测进行中，页面每分钟刷新；未完成分数保留空白。'}</p>
<p>同一 final FP32 EMA，step 95,415；相同数据、候选文本、posterior 和 seed 424242。Random MC64 为正式基线，Random MC1 与固定 Halton MC1 为单序对照；Shifted Halton MC64 使用 64 个独立随机二维平移，同一图像的所有候选共享顺序，平均 log-likelihood。ARO 使用含文本先验的条件似然。</p>
<img src="comparison.png?v={report['updated_at']}" alt="各图像顺序设置的理解评测分数"><div class="scroll"><table><thead><tr><th>任务</th>{''.join('<th>'+ARM_LABELS[arm]+'</th>' for arm in arms)}<th>{candidate_label} − Random MC64</th><th>{candidate_label} − Random MC1</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p>{'<br>'.join(conclusions)}</p><p>按同一图像聚类进行 2,000 次配对 bootstrap；MMBench 按原始问题合并所有选项轮换。95% 区间为逐任务探索性区间，未校正多重比较。当前对照涵盖上述五项，ImageNet 分类和 COCO / Flickr 检索尚未补测。</p>
<p><a href="comparison.json">逐样本胜负变化汇总 JSON</a> · <a href="comparison.pdf">PDF</a> · <a href="comparison.svg">SVG</a> · <a href="REPORT.md">文字报告</a></p><small>更新时间：{report['updated_at']}</small></html>'''
    write(destination/"index.html", page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    previous = None
    for _ in range(1440):
        paths = [task_root(spec, task)/"summaries"/f"{task}.json"
                 for spec in arm_roots(plan).values() for task in LABELS]
        stamp = tuple(p.stat().st_mtime_ns if p.exists() else None for p in paths)
        if stamp != previous:
            report = collect(plan)
            export(report, Path(plan["output"])/"comparison")
            print(json.dumps({"updated_at": report["updated_at"], "complete": report["complete"], "scores": sum(len(r["scores"]) for r in report["rows"])}), flush=True)
            previous = stamp
            if report["complete"]:
                break
        if not args.watch:
            break
        state = Path(plan["output"])/"status.json"
        if state.exists() and read(state)["status"] == "FAILED":
            break
        groups = plan.get("job_groups", [])
        states = [Path(plan["output"])/group["id"]/"status.json" for group in groups]
        if states and all(p.exists() and read(p)["status"] in ("FAILED", "SUCCEEDED") for p in states):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
