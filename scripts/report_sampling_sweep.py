#!/usr/bin/env python3
"""Audit saved sweep images, plot metrics, and refresh the evaluation homepage."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_unified_t2i_sampling import locked, read, report, require, validate_metrics, write, now
from build_evaluation_report import build
from utils.experiment_registry import current_presentation, presentation_sort_key


def matrix_model_presentations(protocol):
    return {mid: current_presentation({**spec, "id": mid}) for mid, spec in protocol["models"].items()}


def plot(root, state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = next((f.name for f in font_manager.fontManager.ttflist if f.name in
                 {"Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"}), "DejaVu Sans")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42, "font.family": font})
    done = [a for a in state["tasks"] if a["status"] == "done"]
    if state["phase"] == "matrix":
        if not done:
            return
        protocol = read(root / "protocol.json")
        models = matrix_model_presentations(protocol)
        model_ids = sorted(models, key=lambda mid: presentation_sort_key(models[mid]))
        colors = {"spatial_halton": "#286aa5", "confidence_stability": "#137c5b", "random": "#8652aa", "sequential": "#a26b22"}
        labels = {"spatial_halton": "Halton", "confidence_stability": "Velocity stability", "random": "Random", "sequential": "E native sequential"}
        views = [("matrix-metrics", model_ids, "All nine models, including E and its sequential control"),
                 ("matrix-metrics-zoom", [m for m in model_ids if m != "e_on_b"],
                  "Expanded view of eight models | E excluded here; see full matrix for E")]
        for filename, included, subtitle in views:
            names = [models[mid]["label"] for mid in included]
            fig, axes = plt.subplots(1, 2, figsize=(13, 6.3), layout="constrained")
            for strategy, color in colors.items():
                arms = [a for a in done if a["strategy"] == strategy and a["model"] in included]
                if not arms:
                    continue
                y = [included.index(a["model"]) + {"spatial_halton": -.2, "confidence_stability": .2, "random": 0, "sequential": .4}[strategy] for a in arms]
                for axis, field in zip(axes, ["fid", "is"]):
                    axis.errorbar([a["result"][field] for a in arms], y,
                        xerr=[a["result"]["is_std"] for a in arms] if field == "is" else None,
                        fmt="o", color=color, label=labels[strategy], capsize=3)
            for axis, label in zip(axes, ["FID (lower is better)", "IS +/- split SD (higher is better)"]):
                axis.set(yticks=list(range(len(names))), yticklabels=names, xlabel=label)
                axis.invert_yaxis()
                axis.grid(alpha=.2)
            axes[1].legend(frameon=False, loc="best")
            fig.suptitle("Final-EMA ablation matrix | CFG 2.0, Heun 10 | paired ImageNet-val 50K\n" + subtitle)
            for ext in ("png", "pdf"):
                fig.savefig(root / f"{filename}.{ext}", bbox_inches="tight", pad_inches=.15)
            plt.close(fig)
        return
    if state["phase"] == "order":
        if not done:
            return
        done.sort(key=lambda a: a["result"]["fid"])
        labels = {"spatial_halton": "Halton", "sequential": "Sequential", "spatial_uniform": "Center-out",
                  "random": "Random", "confidence_cfg": "CFG agreement", "confidence_cfg_reverse": "CFG reverse",
                  "confidence_stability": "Velocity stability", "confidence_halton": "Probe control"}
        names = [labels[a["strategy"]] for a in done]
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), layout="constrained")
        y = list(range(len(done)))
        axes[0].plot([a["result"]["fid"] for a in done], y, "o", color="#245ca6")
        axes[1].errorbar([a["result"]["is"] for a in done], y,
                         xerr=[a["result"]["is_std"] for a in done], fmt="o", capsize=3, color="#247b65")
        for axis, label in zip(axes, ("FID (lower is better)", "IS +/- split SD (higher is better)")):
            axis.set(yticks=y, yticklabels=names, xlabel=label)
            axis.invert_yaxis()
            axis.grid(alpha=.2)
        fig.suptitle("B reveal order | CFG 2.0, Heun 10 | paired ImageNet-val 50K")
        for ext in ("png", "pdf"):
            fig.savefig(root / f"order-sweep.{ext}", bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(9, 4.7), layout="constrained")
        for arm, name in zip(done, names):
            r = arm["result"]
            axis.scatter(r["generation_seconds"] / 60, r["fid"], s=45, label=name)
        axis.set(xlabel="50K generation minutes on 16 NPUs (includes probes)", ylabel="FID (lower is better)")
        axis.grid(alpha=.2)
        axis.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
        fig.suptitle("B reveal-order quality / generation cost")
        for ext in ("png", "pdf"):
            fig.savefig(root / f"order-cost.{ext}", bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
        return
    groups = {"cfg-sweep": [("Heun 10", sorted([a for a in done if a["phase"] == "cfg"], key=lambda a: a["cfg"]))]}
    if state.get("cfg_selection"):
        groups["heun-sweep"] = [(f"CFG {cfg:.1f}", sorted([a for a in done if a["cfg"] == cfg], key=lambda a: a["steps"]))
                                 for cfg in sorted({a["cfg"] for a in state["cfg_selection"].values()})]
    for filename, series in groups.items():
        if not any(arms for _, arms in series):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), layout="constrained")
        for label, arms in series:
            if not arms:
                continue
            x = [a["cfg" if filename == "cfg-sweep" else "steps"] for a in arms]
            axes[0].plot(x, [a["result"]["fid"] for a in arms], "o-", label=label, linewidth=1.7)
            axes[1].errorbar(x, [a["result"]["is"] for a in arms],
                             yerr=[a["result"]["is_std"] for a in arms], fmt="o-", capsize=3, label=label, linewidth=1.7)
        for axis, ylabel in zip(axes, ("FID (lower is better)", "IS +/- split SD (higher is better)")):
            axis.set(xlabel="CFG" if filename == "cfg-sweep" else "Heun steps", ylabel=ylabel)
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
            if filename == "heun-sweep":
                axis.set_xticks([5, 10, 20, 50, 100])
        fig.suptitle("B final EMA | ImageNet-val 50K | paired noise, seed 42")
        for extension in ("png", "pdf"):
            fig.savefig(root / f"{filename}.{extension}", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


def audit(root, state, protocol):
    from PIL import Image
    require(state["status"] == "complete", "sweep is incomplete")
    reference = None
    image_count = 0
    for arm in state["tasks"]:
        require(arm["status"] == "done", "unfinished arm")
        metric_path = Path(arm["result"]["metrics_path"])
        require(validate_metrics(metric_path, protocol, arm) == arm["result"], "metric changed after selection")
        records = []
        for index in protocol["saved_image_indices"]:
            path = metric_path.parent / arm.get("strategy", "spatial_halton") / f"{index:08d}.png"
            with Image.open(path) as image:
                require(image.size == (256, 256) and image.mode == "RGB", f"invalid image dimensions/mode: {path}")
                image.verify()
            records.append(read(path.with_suffix(".json")))
            if state["phase"] in {"order", "matrix"}:
                trace = read(path.parent / "order_trace" / f"{index:08d}.json")
                ranks = [r for row in trace["generation_order"] for r in row]
                require(len(trace["generation_order"]) == 16 and all(len(row) == 16 for row in trace["generation_order"])
                        and sorted(ranks) == list(range(1, 257)), "invalid reveal-order permutation")
                require(trace["policy"] == protocol["order_policies"][arm["strategy"]], "order policy mismatch")
                if arm["strategy"].startswith("confidence_"):
                    baseline_arm = next(a for a in state["tasks"] if a["strategy"] == "spatial_halton"
                                        and a.get("model") == arm.get("model"))
                    base_path = Path(baseline_arm["result"]["metrics_path"]).parent / "spatial_halton/order_trace" / f"{index:08d}.json"
                    base_ranks = [v for row in read(base_path)["generation_order"] for v in row]
                    require(all((a - 1) // 16 == (b - 1) // 16 for a, b in zip(ranks, base_ranks)),
                            "full evaluation changed Halton candidate blocks")
                    if arm["strategy"] == "confidence_halton":
                        require(ranks == base_ranks, "probe control changed Halton order")
                    scores = [s for row in trace["confidence_proxy"] for s in row]
                    import math
                    require(len(scores) == 256 and all(math.isfinite(s) for s in scores), "invalid confidence scores")
                    positions = sorted(range(256), key=lambda p: ranks[p])
                    if arm["strategy"] != "confidence_halton":
                        for start in range(0, 256, 16):
                            block = [scores[p] for p in positions[start:start + 16]]
                            require(block == sorted(block, reverse=arm["strategy"] == "confidence_cfg_reverse"), "trace contradicts confidence order")
                if state["phase"] == "matrix" and arm["strategy"] == "random":
                    base = next(a for a in state["tasks"] if a["model"] == "b_x0" and a["strategy"] == "random")
                    base_trace = Path(base["result"]["metrics_path"]).parent / "random/order_trace" / f"{index:08d}.json"
                    require(trace["generation_order"] == read(base_trace)["generation_order"],
                            "random orders differ across models under the matched RNG/partition")
            image_count += 1
        if reference is None:
            reference = records
        require(records == reference, "images are not paired across parameters")
    for evidence in read(root / "launch/input-audit.json")["files"]:
        stat = Path(evidence["input"]).stat()
        require((stat.st_size, stat.st_mtime_ns) == (evidence["size"], evidence["mtime_ns"]), "sweep input changed")
    if state["phase"] == "order":
        from PIL import ImageChops
        baseline = next(a for a in state["tasks"] if a["strategy"] == "spatial_halton")
        control = next(a for a in state["tasks"] if a["strategy"] == "confidence_halton")
        old = read(root / "cfg-refinement.json")["best_fid"]["result"]
        require(abs(baseline["result"]["fid"] - old["fid"]) < 1e-5 and abs(baseline["result"]["is"] - old["is"]) < 1e-5,
                "Halton baseline does not reproduce the selected CFG/Heun result")
        identical = 0
        for index in protocol["saved_image_indices"]:
            left = Path(baseline["result"]["metrics_path"]).parent / "spatial_halton" / f"{index:08d}.png"
            right = Path(control["result"]["metrics_path"]).parent / "confidence_halton" / f"{index:08d}.png"
            with Image.open(left) as a, Image.open(right) as b:
                identical += ImageChops.difference(a, b).getbbox() is None
        write(root / "controls.json", {"baseline_reproduced": True, "probe_control_identical_images": identical,
              "images_compared": len(protocol["saved_image_indices"]),
              "probe_control_fid_delta": control["result"]["fid"] - baseline["result"]["fid"],
              "probe_control_is_delta": control["result"]["is"] - baseline["result"]["is"],
              "probe_control_time_ratio": control["result"]["generation_seconds"] / baseline["result"]["generation_seconds"]})
    if state["phase"] == "matrix":
        from PIL import ImageChops
        expected = {(m, s) for m in protocol["models"] for s in protocol.get("matrix_strategies", ["spatial_halton", "confidence_stability"])}
        expected.add(("e_on_b", "sequential"))
        require(len(state["tasks"]) == len(expected) and {(a["model"], a["strategy"]) for a in state["tasks"]} == expected,
                "incomplete full matrix coverage")
        old_state = read(Path(protocol["baseline_reproduction"]) / "state.json")
        reproduction = []
        for strategy in protocol.get("matrix_strategies", ["spatial_halton", "confidence_stability"]):
            old = next(a for a in old_state["tasks"] if a["strategy"] == strategy)
            current = next(a for a in state["tasks"] if a["model"] == "b_x0" and a["strategy"] == strategy)
            delta = {k: current["result"][k] - old["result"][k] for k in ["fid", "is", "is_std"]}
            require(all(abs(v) < 1e-5 for v in delta.values()), "B metrics changed under generalized implementation")
            identical = 0
            for index in protocol["saved_image_indices"]:
                paths = [Path(a["result"]["metrics_path"]).parent / strategy / f"{index:08d}.png" for a in [old, current]]
                with Image.open(paths[0]) as a, Image.open(paths[1]) as b:
                    identical += ImageChops.difference(a, b).getbbox() is None
            require(identical == len(protocol["saved_image_indices"]), "B paired images changed")
            reproduction.append({"strategy": strategy, "deltas": delta, "identical_images": identical})
        write(root / "baseline-reproduction.json", {"complete": True, "at": now(), "checks": reproduction})
    write(root / "audit.json", {"complete": True, "at": now(), "arms": len(state["tasks"]),
          "samples_per_arm": 50000, "images_verified": image_count, "paired_image_identities_verified": True,
          "input_sizes_and_mtimes_verified": True, "runtime_hashing_enabled": False,
          "order_traces_verified": state["phase"] in {"order", "matrix"},
          "random_orders_paired_across_models": state["phase"] == "matrix" and "random" in protocol.get("matrix_strategies", [])})


def matrix_analysis(root, state, protocol):
    """Derive the final Chinese comparison only from all validated 50K arms."""
    require(state["phase"] == "matrix" and state["status"] == "complete", "matrix analysis needs every arm")
    require(read(root / "audit.json")["complete"], "matrix analysis needs the full artifact audit")
    repo = Path(__file__).resolve().parents[1]
    labels = {mid: spec["label"] for mid, spec in matrix_model_presentations(protocol).items()}
    comparisons = []
    for mid, spec in protocol["models"].items():
        rows = {a["strategy"]: a["result"] for a in state["tasks"] if a["model"] == mid}
        previous_spec = spec["previous_evaluation"]
        previous_root = repo / "output/evaluation" / previous_spec["root"]
        previous_path = previous_root / ("t2i-fid-is/metrics.json" if previous_spec.get("layout") == "v9" else "core/t2i-fid-is/metrics.json")
        old_payload = read(previous_path)
        require(old_payload["cfg"] == 3.5 and str(old_payload["sampling_steps"]) == "10"
                and old_payload["project_formal_protocol"] and old_payload["samples_evaluated"] == 50000,
                "historical CFG comparison protocol mismatch")
        old = old_payload["strategies"][spec["native_strategy"]]
        b, s = rows["spatial_halton"], rows["confidence_stability"]
        comparisons.append({"model": mid, "label": labels[mid], "halton": b, "stability": s,
            "stability_minus_halton_fid": s["fid"] - b["fid"],
            "stability_minus_halton_is": s["is"] - b["is"],
            "stability_time_ratio": s["generation_seconds"] / b["generation_seconds"],
            "previous_native_strategy": spec["native_strategy"], "previous_metrics": old,
            "previous_metrics_path": str(previous_path), "cfg2_native": rows[spec["native_strategy"]],
            "cfg2_minus_cfg3p5_native_fid": rows[spec["native_strategy"]]["fid"] - old["fid"]})
        if "random" in rows:
            comparisons[-1].update(random=rows["random"],
                random_minus_halton_fid=rows["random"]["fid"] - b["fid"],
                random_minus_halton_is=rows["random"]["is"] - b["is"],
                random_time_ratio=rows["random"]["generation_seconds"] / b["generation_seconds"])
    result = {"complete": True, "at": now(), "models": comparisons,
        "best_cfg3p5_native_fid": min(comparisons, key=lambda x: x["previous_metrics"]["fid"])["model"],
        "best_cfg2_native_fid": min(comparisons, key=lambda x: x["cfg2_native"]["fid"])["model"],
        "best_halton_fid": min(comparisons, key=lambda x: x["halton"]["fid"])["model"],
        "best_stability_fid": min(comparisons, key=lambda x: x["stability"]["fid"])["model"],
        "best_halton_is": max(comparisons, key=lambda x: x["halton"]["is"])["model"],
        "best_stability_is": max(comparisons, key=lambda x: x["stability"]["is"])["model"],
        "stability_fid_improved": [x["model"] for x in comparisons if x["stability_minus_halton_fid"] < 0],
        "stability_is_improved": [x["model"] for x in comparisons if x["stability_minus_halton_is"] > 0],
        "cfg2_native_fid_improved": [x["model"] for x in comparisons if x["cfg2_minus_cfg3p5_native_fid"] < 0]}
    has_random = "random" in protocol.get("matrix_strategies", [])
    if has_random:
        result.update(best_random_fid=min(comparisons, key=lambda x: x["random"]["fid"])["model"],
            best_random_is=max(comparisons, key=lambda x: x["random"]["is"])["model"],
            random_fid_improved=[x["model"] for x in comparisons if x["random_minus_halton_fid"] < 0],
            random_is_improved=[x["model"] for x in comparisons if x["random_minus_halton_is"] > 0])
    write(root / "analysis.json", result)
    lines = ["# 完整消融矩阵：CFG=2.0、Heun=10", "",
        f"9 个模型、{len(state['tasks'])} 组独立 16 卡评测均完成 ImageNet-val 50K。每组保留相同的 64 张图及实际解码顺序，共 {len(state['tasks'])*64:,} 张配对样图。", "",
        f"- Halton 下最低 FID：**{labels[result['best_halton_fid']]}**。",
        f"- confidence_stability 下最低 FID：**{labels[result['best_stability_fid']]}**。",
        f"- 最高 IS：Halton 为 **{labels[result['best_halton_is']]}**，Stability 为 **{labels[result['best_stability_is']]}**。",
        f"- Stability 相比 Halton：**{len(result['stability_fid_improved'])}/9** 个模型 FID 降低，**{len(result['stability_is_improved'])}/9** 个模型 IS 提高。",
        f"- 保持各模型原生顺序，将 CFG 3.5 改为 2.0：**{len(result['cfg2_native_fid_improved'])}/9** 个模型 FID 降低。E 的这项比较采用额外的 sequential 结果。", "",
        f"- 原生顺序下最低 FID 的模型，从 CFG=3.5 时的 **{labels[result['best_cfg3p5_native_fid']]}** 变为 CFG=2.0 时的 **{labels[result['best_cfg2_native_fid']]}**。因此模型的 FID 排名也取决于推理参数。", "",
        "| 模型 | Halton FID ↓ | Halton IS ↑ | Stability FID ↓ | Stability IS ↑ | ΔFID (S−H) | 生成耗时 S/H |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        b, s = row["halton"], row["stability"]
        lines.append(f"| {row['label']} | {b['fid']:.4f} | {b['is']:.2f} ± {b['is_std']:.2f} | {s['fid']:.4f} | {s['is']:.2f} ± {s['is_std']:.2f} | {row['stability_minus_halton_fid']:+.4f} | {row['stability_time_ratio']:.3f}× |")
    if has_random:
        lines += ["", "## Random 顺序消融", "",
            f"Random 下最低 FID：**{labels[result['best_random_fid']]}**；最高 IS：**{labels[result['best_random_is']]}**。与 Halton 比较，**{len(result['random_fid_improved'])}/9** 个模型 FID 降低，**{len(result['random_is_improved'])}/9** 个模型 IS 提高。", "",
            "| 模型 | Random FID ↓ | Random IS ↑ | ΔFID (R−H) | ΔIS (R−H) | 生成耗时 R/H |",
            "|---|---:|---:|---:|---:|---:|"]
        for row in comparisons:
            r = row["random"]
            lines.append(f"| {row['label']} | {r['fid']:.4f} | {r['is']:.2f} ± {r['is_std']:.2f} | {row['random_minus_halton_fid']:+.4f} | {row['random_minus_halton_is']:+.2f} | {row['random_time_ratio']:.3f}× |")
        lines += ["", "Random 对全部 256 个位置取均匀随机排列；仍使用相同的逐位置初始噪声。固定评测 seed=42 和相同 batch/rank 划分，九个模型保存样本的随机顺序逐一匹配。它不使用 Stability 的候选探测。"]
    lines += ["", "## 与原 CFG=3.5 的原生顺序比较", "",
        "| 模型 | 原生顺序 | CFG 3.5 FID / IS | CFG 2.0 FID / IS | ΔFID |", "|---|---|---:|---:|---:|"]
    for row in comparisons:
        old, current = row["previous_metrics"], row["cfg2_native"]
        lines.append(f"| {row['label']} | {row['previous_native_strategy']} | {old['fid']:.4f} / {old['inception_score_mean']:.2f} | {current['fid']:.4f} / {current['is']:.2f} | {row['cfg2_minus_cfg3p5_native_fid']:+.4f} |")
    e = next(row for row in comparisons if row["model"] == "e_on_b")
    lines += ["", "## E 的顺序敏感性", "",
        f"E 在训练时使用 sequential。本轮 CFG=2.0 下，原生 sequential 的 FID / IS 为 **{e['cfg2_native']['fid']:.4f} / {e['cfg2_native']['is']:.2f}**；Halton 为 **{e['halton']['fid']:.4f} / {e['halton']['is']:.2f}**，Stability 为 **{e['stability']['fid']:.4f} / {e['stability']['is']:.2f}**。",
        "", "这说明降低 CFG 与改变生成顺序需要分开解释。即使 Stability 相比 Halton 的指标改善，也不能据此认定它优于 E 的原生顺序。"]
    e_alternatives = [e[k]["fid"] for k in ["halton", "stability", *(["random"] if has_random else [])]]
    if min(e_alternatives) > e["cfg2_native"]["fid"]:
        lines += ["", "**E 在本协议下应保留 sequential。**"]
    if has_random:
        lines += ["", f"E-Random 的 FID / IS 为 **{e['random']['fid']:.4f} / {e['random']['is']:.2f}**；对照 E-sequential 见上文。"]
    d = next(row for row in comparisons if row["model"] == "d_on_b")
    lines += ["", "## D 的 FID 与 IS 取舍", "",
        f"D 从 Halton 切换到 Stability：FID **{d['halton']['fid']:.4f} → {d['stability']['fid']:.4f}**，IS **{d['halton']['is']:.2f} → {d['stability']['is']:.2f}**。"]
    if d["stability_minus_halton_fid"] > 0 and d["stability_minus_halton_is"] > 0:
        lines += ["", "在这个检查点上，Stability 的 IS 提升伴随 FID 上升，不能将它视为两项指标同时改善。"]
    if has_random:
        lines += ["", f"D-Random 的 FID / IS 为 **{d['random']['fid']:.4f} / {d['random']['is']:.2f}**。"]
    lines += ["", "## 协议与解释边界", "",
        "全部使用各自 step-95415 final EMA。A 的严格注意力、旧 A/B 的 shared-query 条件、D 的动态 XT 条件刷新和 F 的无内容流结构均保留。顺序策略只改变生成顺序及其必要探测计算。",
        "", "Stability 在每组 16 个 Halton 候选中，按一次 dt=0.1 的 Euler 探测前后归一化引导速度变化从小到大排序。两次探测只使用提示词和已生成内容，最终解码从相同逐位置噪声重新开始。该分数不是校准后的概率。",
        "", "CFG 2.0 来自 B 的选优，本实验没有对每个模型分别调参。IS 的 ± 是十个分层 split 的标准差，不是跨随机种子的误差条。小幅差异不代表统计显著。计时来自不同独立节点的实际生成墙钟，包含探测开销。",
        "", "真实权重 smoke 使用所有 16 张开发机 NPU；每个模型都通过完整生成、每卡 256 样本容量检查和相同探测形状的固定顺序复跑。BF16 在不同融合矩阵形状下存在数值差异，原始数值控制保留于 `smoke/audit.json`。",
        "", "B 各组均复现上一轮对应策略的 FID/IS 和全部 64 张配对 PNG，见 `baseline-reproduction.json`。完整输入、指标、图片与顺序核验见 `audit.json`。", "",
        "新旧 CFG 的评测条件与旧 B 控制组的历史更名对应关系，见 `launch/cfg-native-protocol-comparison.json`。", "",
        "[完整结果 CSV](results.csv) · [原始指标](summary.json) · [冻结协议](protocol.json) · [全矩阵 PNG](matrix-metrics.png) · [全矩阵 PDF](matrix-metrics.pdf) · [八模型放大图](matrix-metrics-zoom.png) · [放大图 PDF](matrix-metrics-zoom.pdf)"]
    (root / "RESULTS_ZH.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    previous = None
    while True:
        with locked(root):
            state, protocol = read(root / "state.json"), read(root / "protocol.json")
            signature = (state["status"], state["phase"], tuple((a["id"], a["status"]) for a in state["tasks"]))
            if signature != previous:
                report(root, state, protocol)
        if signature != previous:
            plot(root, state)
            if state["status"] == "complete":
                audit(root, state, protocol)
                if state["phase"] == "matrix":
                    matrix_analysis(root, state, protocol)
            build(repo / "output/evaluation", repo / "configs/protocols/evaluation_report.json")
            print(f"REPORT_UPDATED phase={state['phase']} status={state['status']} done={sum(a['status'] == 'done' for a in state['tasks'])}/{len(state['tasks'])}", flush=True)
            previous = signature
        if not args.watch or state["status"] in {"complete", "failed"}:
            require(state["status"] != "failed", state.get("error", "sweep failed"))
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
