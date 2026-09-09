#!/usr/bin/env python3
"""Build the local evaluation homepage from explicitly selected raw results.

CPU only (PyYAML; optional Matplotlib figures). Missing results stay missing; invalidated runs and
checkpoint/protocol mismatches cannot silently enter the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.evaluation_report_training import collect_training, export_training, plot_training
from scripts.evaluation_report_provenance import collect_provenance, export_provenance
from utils.experiment_registry import current_presentation, presentation_sort_key
TEXT_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "openbookqa", "mmlu")
BENCHMARKS = {"sugarcrepe": 7511, "aro_vg_relation": 23937, "aro_vg_attribution": 28748,
              "mmbench_dev_en": 4329, "seed_bench_image": 14233}
LABELS = {"fid": "FID ↓", "is": "IS ↑", "top1": "ImageNet Top-1", "top5": "ImageNet Top-5",
          "text_macro": "文本八项均分", "mmlu": "MMLU 5-shot", "arc_easy": "ARC-E", "arc_challenge": "ARC-C",
          "hellaswag": "HellaSwag", "piqa": "PIQA", "winogrande": "WinoGrande", "boolq": "BoolQ", "openbookqa": "OpenBookQA",
          "sugarcrepe": "SugarCrepe", "aro_vg_relation": "ARO Relation", "aro_vg_attribution": "ARO Attribution",
          "mmbench_dev_en": "MMBench circular", "seed_bench_image": "SEED image",
          "t2i_loss": "T2I loss ↓", "i2t_loss": "I2T loss ↓", "i2t_ppl": "I2T PPL ↓"}
for _dataset in ("coco", "flickr"):
    for _direction in ("i2t", "t2i"):
        for _k in (1, 5, 10):
            LABELS[f"{_dataset}_{_direction}_r{_k}"] = f"{'COCO' if _dataset == 'coco' else 'Flickr'} {_direction.upper()} R@{_k}"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"result path escapes evaluation directory: {relative}")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_not_invalidated(path: Path, root: Path):
    for parent in [path if path.is_dir() else path.parent, *path.parents]:
        if not parent.is_relative_to(root):
            break
        if (parent / "evaluation_invalidation.json").exists():
            raise ValueError(f"invalidated evaluation cannot be selected: {parent}")


def check_checkpoint(payload, spec, selection, root):
    source = payload.get("checkpoint") or payload.get("evaluation_model_source", {}).get("path")
    require(source, f"missing checkpoint identity for {spec['id']}")
    observed, expected = Path(source).resolve(), Path(spec["checkpoint"]).resolve()
    if observed != expected:
        alias = selection.get("historical_checkpoint_alias")
        evidence_file = selection.get("alias_evidence")
        require(alias and evidence_file, f"checkpoint mismatch for {spec['id']}: {observed} != {expected}")
        evidence = read(within(root, evidence_file))
        require(evidence.get("original_run_project") == alias and
                Path(evidence["current_output_root"]).name == spec["run"] and
                evidence.get("control_name") == "flow_head_no_diagonal" and
                observed.parent.name == alias and observed.name == expected.name and
                observed.parent.parent == expected.parent.parent,
                f"unverified historical checkpoint alias for {spec['id']}")
    step = payload.get("checkpoint_step", payload.get("evaluation_model_source", {}).get("global_step"))
    require(step == spec["source"]["global_step"], f"checkpoint step mismatch for {spec['id']}")


def model_metrics(root: Path, spec: dict, selection: dict) -> dict:
    result = {"metrics": {}, "components": {}, "notes": [], "selected_root": selection.get("root")}
    if not selection:
        result.update(status="未收录正式指标", complete=False)
        return result
    run = within(root, selection["root"])
    require(run.is_dir(), f"selected evaluation does not exist: {run}")
    check_not_invalidated(run, root)
    v9 = selection.get("layout") == "v9"
    core = run if v9 else run / "core"
    native = run if v9 else run / "pretraining-native-understanding"
    if selection.get("note"):
        result["notes"].append(selection["note"])

    def source(path):
        check_not_invalidated(path, root)
        return str(path.relative_to(root))

    def put(key, value, path, *, std=None):
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"invalid {key}: {path}")
        percent = key not in {"fid", "is", "t2i_loss", "i2t_loss", "i2t_ppl"}
        require(not percent or 0 <= value <= 1, f"invalid accuracy unit for {key}: {path}")
        result["metrics"][key] = {"value": value, "source": source(path), "percent": percent}
        if std is not None:
            require(math.isfinite(std) and std >= 0, f"invalid split standard deviation: {path}")
            result["metrics"][key]["std"] = std

    def component(name, path, valid):
        result["components"][name] = {"complete": valid, "source": source(path) if path.exists() else None}

    path = core / "t2i-fid-is/metrics.json"
    done = False
    if path.exists():
        data = read(path)
        check_checkpoint(data, spec, selection, root)
        require(data.get("schema") == "selfless_imagenet_val_t2i_fid_is_v2" and data.get("project_formal_protocol") is True,
                f"non-formal generation result: {path}")
        require(data.get("samples_evaluated") == 50000 and str(data.get("sampling_steps")) == "10" and
                data.get("flow_solver") == "heun" and data.get("cfg") == 3.5 and data.get("seed") == 42,
                f"generation protocol mismatch: {path}")
        strategy = "sequential" if spec["image_order"] == "sequential" else "spatial_halton"
        metrics = data["strategies"][strategy]
        require(metrics["count"] == 50000, f"incomplete generation: {path}")
        put("fid", metrics["fid"], path)
        put("is", metrics["inception_score_mean"], path, std=metrics["inception_score_std"])
        done = True
    component("generation", path, done)

    path = core / "text/summary.json"
    done = False
    if path.exists():
        data = read(path)
        check_checkpoint(data, spec, selection, root)
        require(data.get("protocol", {}).get("protocol_schema") == "selfless_text_benchmark_v3",
                f"obsolete text scoring protocol: {path}")
        if data.get("complete") is True:
            require(set(data["primary_metrics"]) == set(TEXT_TASKS), f"incomplete eight-task text result: {path}")
            for key in TEXT_TASKS:
                put(key, data["primary_metrics"][key], path)
            put("text_macro", data["macro_average_primary"], path)
            done = True
    component("text", path, done)

    path = native / "imagenet-classification/summary.json"
    done = False
    if path.exists():
        data = read(path)
        check_checkpoint(data, spec, selection, root)
        require(data.get("schema") == "selfless_imagenet1k_zeroshot_classification_summary_v2" and
                data.get("project_formal_protocol") is True, f"obsolete classification protocol: {path}")
        if data.get("complete_formal_target") is True and data.get("records") == 50000:
            put("top1", data["top_1_accuracy"], path)
            put("top5", data["top_5_accuracy"], path)
            done = True
    component("classification", path, done)

    bench_root = native / "retained-benchmarks"
    manifest_path = bench_root / "manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else None
    if manifest:
        check_checkpoint(manifest, spec, selection, root)
        require(manifest.get("schema") == "selfless_multimodal_likelihood_evaluation_v5" and
                manifest.get("project_formal_protocol") is True, f"obsolete benchmark protocol: {manifest_path}")
        expected_mc = 1 if spec["image_order"] == "sequential" else 64
        require(manifest.get("mc_samples") == expected_mc, f"wrong image-order MC count: {manifest_path}")
    for task, count in BENCHMARKS.items():
        path = bench_root / "summaries" / f"{task}.json"
        done = False
        if path.exists():
            require(manifest is not None, f"benchmark has no source manifest: {path}")
            data = read(path)
            metrics = data["metrics"]
            require(data.get("task") == task and data.get("mc_samples") == expected_mc,
                    f"benchmark task or sampling mismatch: {path}")
            if metrics.get("records") == count:
                key = "circular_accuracy_language_prior_debiased" if task == "mmbench_dev_en" else (
                    "accuracy_language_prior_debiased" if task == "seed_bench_image" else "language_prior_debiased_pairwise.win_rate")
                require(metrics.get("primary_metric") == key, f"wrong benchmark primary metric: {path}")
                value = metrics
                for part in key.split("."):
                    value = value[part]
                put(task, value, path)
                done = True
        component(task, path, done)

    for dataset, folder, image_count, caption_count in (("coco", "mscoco-5k", 5000, 25010), ("flickr", "flickr30k-1k", 1000, 5000)):
        path = native / "standard-retrieval" / folder / "summary.json"
        done = False
        if path.exists():
            data = read(path)
            check_checkpoint(data, spec, selection, root)
            require(data.get("schema") == "selfless_cross_dataset_retrieval_summary_v3" and
                    data.get("project_formal_protocol") is True, f"obsolete retrieval protocol: {path}")
            if data.get("complete_formal_target") is True and data.get("images") == image_count and data.get("captions") == caption_count:
                for direction, field in (("i2t", "image_to_text"), ("t2i", "text_to_image")):
                    for k in (1, 5, 10):
                        put(f"{dataset}_{direction}_r{k}", data[field][f"recall_at_{k}"], path)
                done = True
        component(dataset, path, done)

    path = core / "validation" / f"validation_metrics_step_{spec['source']['global_step']}.json"
    if path.exists():
        data = read(path)
        for key, field in (("t2i_loss", "val/loss_t2i"), ("i2t_loss", "val/loss_i2t"), ("i2t_ppl", "val/ppl_text")):
            if field in data.get("metrics", {}):
                put(key, data["metrics"][field], path)
    completed = sum(c["complete"] for c in result["components"].values())
    result["complete"] = completed == len(result["components"])
    result["status"] = f"{completed}/{len(result['components'])} 组已完成"
    return result


def gallery_data(root: Path, selection: dict):
    gallery = within(root, selection["qualitative"])
    manifest, summary = read(gallery / "manifest.json"), read(gallery / "summary.json")
    require(summary.get("complete") is True, "qualitative gallery is incomplete")
    prefix = str(gallery.relative_to(root))

    def image_path(name):
        path = within(gallery, name)
        require(path.is_file(), f"missing gallery image: {path}")
        return str(path.relative_to(root))

    samples = json.loads(json.dumps(manifest["samples"]))
    for rows in samples.values():
        for row in rows:
            for field in ("input_image", "original_image", "reference_image"):
                if row.get(field):
                    row[field] = image_path(row[field])
    records, seen, counts = [], set(), Counter()
    with (gallery / "results.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            mode = row.get("seed") if row["task"] == "t2i" else row.get("decoding") if row["task"] == "text" else None
            key = (row["model"], row["task"], row["sample_id"], mode)
            require(key not in seen, f"duplicate qualitative result: {key}")
            seen.add(key)
            counts[row["model"], row["task"]] += 1
            record = {k: row[k] for k in ("model", "task", "sample_id", "seed", "decoding", "text", "generated_tokens", "stop_reason") if k in row}
            if row.get("image"):
                record["image"] = image_path(row["image"])
            records.append(record)
    expected_keys = {(m["id"], task, row["id"], mode) for m in manifest["models"] for task, rows in manifest["samples"].items()
                     for row in rows for mode in ((42, 43) if task == "t2i" else ("greedy", "sample") if task == "text" else (None,))}
    require(seen == expected_keys, "qualitative sample identities/coverage mismatch")
    for model in manifest["models"]:
        for task, expected in manifest["expected_per_model"].items():
            require(counts[model["id"], task] == expected, f"qualitative count mismatch: {model['id']}/{task}")
    return {"manifest": manifest, "samples": samples, "records": records, "root": prefix,
            "images_verified": sum(r["task"] == "t2i" for r in records)}


def sampling_sweep_data(root: Path, selection: dict, models: list):
    if __package__:
        from .sweep_unified_t2i_sampling import validate_metrics, task, winners
    else:
        from sweep_unified_t2i_sampling import validate_metrics, task, winners
    sweeps = []
    for selected in selection.get("sampling_sweeps", []):
        directory = within(root, selected["root"])
        check_not_invalidated(directory, root)
        summary = read(directory / "summary.json")
        protocol = summary["protocol"]
        spec = next(m for m in models if m["id"] == selected["model"])
        require(Path(protocol["model_source"]).resolve() == Path(spec["checkpoint"]).resolve()
                and protocol["checkpoint_step"] == spec["source"]["global_step"], "sweep checkpoint mismatch")
        rows, completed, paired_samples = [], [], None
        for result in summary["results"]:
            arm = task(result["cfg"], result["heun_steps"], result["phase"])
            row = {**result, "id": arm["id"]}
            if result["status"] == "done":
                path = Path(result["metrics_path"]).resolve()
                require(path.is_relative_to(root), "sweep metric escapes evaluation directory")
                check_not_invalidated(path, root)
                metric = validate_metrics(path, protocol, arm)
                require(all(result[key] == metric[key] for key in ("fid", "is", "is_std")), "sweep summary metric mismatch")
                row["metrics_path"] = str(path.relative_to(root))
                arm.update(status="done", result=metric)
                completed.append(arm)
                samples, images = [], []
                for index in protocol.get("saved_image_indices", []):
                    image_path = path.parent / "spatial_halton" / f"{index:08d}.png"
                    record = read(image_path.with_suffix(".json"))
                    samples.append({key: record[key] for key in ("global_sample_index", "image_id", "canonical_noise_seed", "prompt")})
                    images.append(str(image_path.relative_to(root)))
                if paired_samples is None:
                    paired_samples = samples
                require(samples == paired_samples, "sweep images do not share sample identities, prompts and noise")
                row["images"] = images
            rows.append(row)
        sweep = {"label": selected["label"], "root": str(directory.relative_to(root)),
                 "status": summary["status"], "phase": summary["phase"], "error": summary.get("error"),
                 "completed": len(completed), "total": len(rows), "rows": rows,
                 "samples": paired_samples or [], "cfg_selection": summary.get("cfg_selection"),
                 "conclusion": None, "plots": []}
        if summary["status"] == "complete":
            require(len(completed) == len(rows), "completed sweep has unfinished arms")
            cfg_arms = [a for a in completed if a["phase"] == "cfg"]
            selected_cfg = winners(cfg_arms)
            require(all(summary["cfg_selection"][key]["id"] == arm["id"] for key, arm in selected_cfg.items()),
                    "sweep CFG selection mismatch")
            for arm in selected_cfg.values():
                observed = {a["steps"] for a in completed if a["cfg"] == arm["cfg"]}
                require(set(protocol["heun_steps"]) <= observed, "sweep Heun coverage incomplete")
            best = winners(completed)
            baseline = next(a for a in completed if a["cfg"] == 3.5 and a["steps"] == 10)
            sweep["conclusion"] = {**best, "baseline": baseline,
                                    "fid_reduction": baseline["result"]["fid"] - best["best_fid"]["result"]["fid"],
                                    "is_increase": best["best_is"]["result"]["is"] - baseline["result"]["is"]}
        for filename in ("cfg-sweep.png", "heun-sweep.png"):
            path = directory / filename
            if path.is_file():
                sweep["plots"].append(str(path.relative_to(root)))
        sweeps.append(sweep)
    return sweeps


def order_sweep_data(root: Path, selection: dict, models: list):
    if __package__:
        from .sweep_unified_t2i_sampling import validate_metrics, winners
    else:
        from sweep_unified_t2i_sampling import validate_metrics, winners
    studies = []
    for selected in selection.get("order_sweeps", []):
        directory = within(root, selected["root"])
        check_not_invalidated(directory, root)
        summary, state = read(directory / "summary.json"), read(directory / "state.json")
        protocol = summary["protocol"]
        spec = next(m for m in models if m["id"] == selected["model"])
        require(Path(protocol["model_source"]).resolve() == Path(spec["checkpoint"]).resolve()
                and protocol["checkpoint_step"] == spec["source"]["global_step"], "order sweep checkpoint mismatch")
        require(summary["status"] == state["status"], "order summary/state mismatch")
        refine = read(directory / "cfg-refinement.json")
        previous_protocol = read(Path(protocol["previous_sweep"]) / "protocol.json")
        for arm in refine["arms"]:
            require(validate_metrics(arm["result"]["metrics_path"], previous_protocol, arm) == arm["result"],
                    "CFG refinement evidence changed")
        cfgs = {a["cfg"] for a in refine["arms"]}
        best_cfg = winners(refine["arms"])["best_fid"]
        require(cfgs == set(refine["cfg_values"]) and min(cfgs) < best_cfg["cfg"] < max(cfgs)
                and best_cfg["cfg"] == protocol["cfg_fixed"]
                and all(a["steps"] == protocol["heun_fixed"] for a in refine["arms"]), "CFG refinement incomplete")
        rows, complete, samples = [], [], None
        for arm in state["tasks"]:
            row = {"id": arm["id"], "strategy": arm["strategy"], "status": arm["status"],
                   "cfg": arm["cfg"], "steps": arm["steps"]}
            if arm["status"] == "done":
                path = Path(arm["result"]["metrics_path"]).resolve()
                require(path.is_relative_to(root), "order metric escapes evaluation directory")
                check_not_invalidated(path, root)
                metric = validate_metrics(path, protocol, arm)
                require(metric == arm["result"], "order summary metric mismatch")
                row.update({**metric, "metrics_path": str(path.relative_to(root))})
                row["images"], row["orders"] = [], []
                identities = []
                for index in protocol["saved_image_indices"]:
                    image_path = path.parent / arm["strategy"] / f"{index:08d}.png"
                    require(image_path.is_file(), "order sample image missing")
                    record = read(image_path.with_suffix(".json"))
                    identities.append({k: record[k] for k in ("global_sample_index", "image_id", "canonical_noise_seed", "prompt")})
                    row["images"].append(str(image_path.relative_to(root)))
                    order_path = image_path.parent / "order_trace" / f"{index:08d}.json"
                    order = read(order_path)
                    require(order["strategy"] == arm["strategy"] and order["global_sample_index"] == index
                            and order["policy"] == protocol["order_policies"][arm["strategy"]], "order trace identity mismatch")
                    row["orders"].append({**order, "source": str(order_path.relative_to(root))})
                if samples is None:
                    samples = identities
                require(identities == samples, "order images are not paired")
                complete.append(arm)
            rows.append(row)
        conclusion = None
        if state["status"] == "complete":
            require(len(complete) == len(rows) and {a["strategy"] for a in complete} == set(protocol["strategies"]), "order sweep is incomplete")
            selected_arms = winners(complete)
            require(all(state["order_selection"][k]["id"] == a["id"] for k, a in selected_arms.items()), "order selection mismatch")
            baseline = next(a for a in complete if a["strategy"] == "spatial_halton")
            conclusion = {**selected_arms, "baseline": baseline,
                          "fid_reduction": baseline["result"]["fid"] - selected_arms["best_fid"]["result"]["fid"]}
        studies.append({"root": str(directory.relative_to(root)), "status": state["status"],
                        "completed": len(complete), "total": len(rows), "rows": rows,
                        "samples": samples or [], "refinement": refine, "conclusion": conclusion,
                        "controls": read(directory / "controls.json") if (directory / "controls.json").exists() else None,
                        "plots": [str(p.relative_to(root)) for p in [directory / "order-sweep.png", directory / "order-cost.png"] if p.is_file()]})
    return studies


def matrix_sweep_data(root: Path, selection: dict, models: list):
    if __package__:
        from .sweep_unified_t2i_sampling import validate_metrics
    else:
        from sweep_unified_t2i_sampling import validate_metrics
    studies = []
    for selected in selection.get("matrix_sweeps", []):
        directory = within(root, selected["root"])
        check_not_invalidated(directory, root)
        protocol, state = read(directory / "protocol.json"), read(directory / "state.json")
        require(protocol["schema"] == "unified_t2i_ablation_matrix_v1" and state["phase"] == "matrix",
                "unknown ablation matrix protocol")
        require(set(protocol["models"]) == set(selection["models"]), "matrix omits selected formal models")
        matrix_strategies = protocol.get("matrix_strategies", ["spatial_halton", "confidence_stability"])
        require(len(matrix_strategies) == len(set(matrix_strategies))
                and {"spatial_halton", "confidence_stability"} <= set(matrix_strategies)
                <= {"spatial_halton", "confidence_stability", "random"}, "unknown matrix comparison strategies")
        expected = {(m, s) for m in protocol["models"] for s in matrix_strategies}
        expected.add(("e_on_b", "sequential"))
        require(len(state["tasks"]) == len(expected) and {(a["model"], a["strategy"]) for a in state["tasks"]} == expected,
                "matrix arm coverage mismatch")
        specs = {m["id"]: m for m in models}
        for mid, spec in protocol["models"].items():
            current = specs[mid]
            require(Path(spec["model_source"]).resolve() == Path(current["checkpoint"]).resolve()
                    and spec["checkpoint_step"] == current["source"]["global_step"], "matrix checkpoint mismatch")
        rows, samples = [], None
        for arm in state["tasks"]:
            require(arm["cfg"] == protocol["cfg_fixed"] == 2 and arm["steps"] == protocol["heun_fixed"] == 10,
                    "matrix sampling settings changed")
            row = {k: arm[k] for k in ["id", "model", "strategy", "status", "cfg", "steps"]}
            row["label"] = specs[arm["model"]]["label"]
            if arm["status"] == "done":
                path = Path(arm["result"]["metrics_path"]).resolve()
                require(path.is_relative_to(directory), "matrix metric escapes its frozen study")
                check_not_invalidated(path, root)
                result = validate_metrics(path, protocol, arm)
                require(result == arm["result"], "matrix metrics changed")
                row.update({**result, "metrics_path": str(path.relative_to(root))})
                row["images"], row["orders"], identities = [], [], []
                for index in protocol["saved_image_indices"]:
                    image_path = path.parent / arm["strategy"] / f"{index:08d}.png"
                    record = read(image_path.with_suffix(".json"))
                    identities.append({k: record[k] for k in ["global_sample_index", "image_id", "canonical_noise_seed", "prompt"]})
                    row["images"].append(str(image_path.relative_to(root)))
                    trace_path = image_path.parent / "order_trace" / f"{index:08d}.json"
                    trace = read(trace_path)
                    require(trace["strategy"] == arm["strategy"] and trace["global_sample_index"] == index
                            and trace["policy"] == protocol["order_policies"][arm["strategy"]], "matrix trace identity mismatch")
                    ranks = [v for line in trace["generation_order"] for v in line]
                    require(len(trace["generation_order"]) == 16 and all(len(line) == 16 for line in trace["generation_order"])
                            and sorted(ranks) == list(range(1, 257)), "matrix trace permutation invalid")
                    row["orders"].append({**trace, "source": str(trace_path.relative_to(root))})
                if samples is None:
                    samples = identities
                require(identities == samples, "matrix prompts, image IDs or noise differ")
            rows.append(row)
        complete = [r for r in rows if r["status"] == "done"]
        comparison = []
        for mid, spec in protocol["models"].items():
            values = {r["strategy"]: r for r in rows if r["model"] == mid}
            comparison.append({"model": mid, "label": specs[mid]["label"], "native_strategy": spec["native_strategy"],
                "previous": {k: specs[mid]["metrics"].get(k) for k in ["fid", "is"]}, "strategies": values})
        conclusion = None
        if state["status"] == "complete":
            require(len(complete) == len(rows), "complete matrix has unfinished arms")
            audit_path = directory / "audit.json"
            require(audit_path.is_file(), "completed matrix requires its full image/input audit")
            audit = read(audit_path)
            require(audit["complete"] and audit["arms"] == len(rows) and audit["images_verified"] == len(rows) * 64
                    and audit["order_traces_verified"] and audit["paired_image_identities_verified"], "matrix audit incomplete")
            baseline = [r for r in complete if r["strategy"] == "spatial_halton"]
            stability = [r for r in complete if r["strategy"] == "confidence_stability"]
            previous_best = min(comparison, key=lambda m: m["previous"]["fid"]["value"])
            conclusion = {"best_halton": min(baseline, key=lambda r: r["fid"]),
                "best_stability": min(stability, key=lambda r: r["fid"]),
                "best_is_halton": max(baseline, key=lambda r: r["is"]),
                "best_is_stability": max(stability, key=lambda r: r["is"]),
                "best_cfg3p5_native": {"model": previous_best["model"], "label": previous_best["label"],
                    "fid": previous_best["previous"]["fid"]["value"]},
                "best_cfg2_native": min([m["strategies"][m["native_strategy"]] for m in comparison], key=lambda r: r["fid"]),
                "cfg2_native_fid_improved": [m["model"] for m in comparison
                    if m["strategies"][m["native_strategy"]]["fid"] < m["previous"]["fid"]["value"]],
                "cfg2_native_is_improved": [m["model"] for m in comparison
                    if m["strategies"][m["native_strategy"]]["is"] > m["previous"]["is"]["value"]],
                "improved_fid_models": [r["model"] for r in stability if r["fid"] < next(b["fid"] for b in baseline if b["model"] == r["model"])],
                "improved_is_models": [r["model"] for r in stability if r["is"] > next(b["is"] for b in baseline if b["model"] == r["model"])]}
            random = [r for r in complete if r["strategy"] == "random"]
            if random:
                conclusion.update(best_random=min(random, key=lambda r: r["fid"]),
                    best_is_random=max(random, key=lambda r: r["is"]),
                    random_improved_fid_models=[r["model"] for r in random if r["fid"] < next(b["fid"] for b in baseline if b["model"] == r["model"])],
                    random_improved_is_models=[r["model"] for r in random if r["is"] > next(b["is"] for b in baseline if b["model"] == r["model"])])
        studies.append({"root": str(directory.relative_to(root)), "status": state["status"],
            "completed": len(complete), "total": len(rows), "rows": rows, "models": comparison,
            "samples": samples or [], "conclusion": conclusion, "strategies": matrix_strategies,
            "plots": [str(p.relative_to(root)) for p in [directory / "matrix-metrics.png", directory / "matrix-metrics-zoom.png"] if p.exists()]})
    return studies


def build(root: Path, selection_file: Path, *, plots: bool = False):
    root = root.resolve()
    selection = read(selection_file)
    require(selection.get("schema") == "unified_evaluation_report_selection_v1", "unknown report selection schema")
    gallery = gallery_data(root, selection)
    models = []
    known = {m["id"] for m in gallery["manifest"]["models"]}
    require(set(selection["models"]) <= known, "selected metric model absent from qualitative inventory")
    specs = [current_presentation(spec) for spec in gallery["manifest"]["models"]]
    specs.sort(key=presentation_sort_key)
    for spec in specs:
        models.append({**spec, **model_metrics(root, spec, selection["models"].get(spec["id"], {}))})
    sampling_sweeps = sampling_sweep_data(root, selection, models)
    updated = datetime.now(UTC).isoformat(timespec="seconds")
    provenance = collect_provenance(REPO, root)
    export_provenance(root, provenance, write)
    training = collect_training(REPO, root, models)
    training["updated_at"] = updated
    training_summary = export_training(root, training, write)
    plot_manifest = root / "training-loss/plots.json"
    if plots:
        plot_data = {"updated_at": updated, "artifacts": plot_training(root, training)}
        write(plot_manifest, json.dumps(plot_data, ensure_ascii=False, indent=2) + "\n")
    if plot_manifest.exists():
        training["plots"] = training_summary["plots"] = read(plot_manifest)
    artifacts = []
    entries = [("数据集来源与合成协议", provenance["document"]),
               ("全模型逐步训练 loss CSV", training["csv"]),
               ("全模型训练期间验证 loss CSV", training["validation_csv"]),
               ("完整定性长表与 ZIP", gallery["root"] + "/index.html"),
               ("B FID 全量复核", "comparisons/bx0-fid-recheck-20260908/REPORT.md"),
               ("B / D 同噪声复核图", "comparisons/bx0-fid-recheck-20260908/paired_generation.html"),
               ("C / D / E 评测加载审计", "audits/audit-cde-evaluation-20260908-4AcX8p/REPORT.md"),
               ("跨模型 Geometry V5", "research/cross-model-geometry-v5-20260907/RESULTS_ZH.md"),
               ("目录迁移与完整性核验", "migrations/20260908-consolidation/journal.json")]
    for label, path in entries:
        if within(root, path).exists():
            artifacts.append({"label": label, "path": path})
    # The directory inventory includes every retained experiment, not only the
    # current final-EMA selections used in the comparable metric table.
    folders = [{"label": p.name, "path": p.name + "/"} for p in sorted(root.iterdir()) if p.is_dir() and not p.is_symlink()]
    summary = {"schema": "unified_evaluation_report_v1", "updated_at": updated, "models": models,
               "qualitative_root": gallery["root"], "qualitative_models": len(models),
               "qualitative_records": len(gallery["records"]), "images_verified": gallery["images_verified"],
               "formal_models": sum(bool(m["metrics"]) for m in models),
               "formal_complete_models": sum(m["complete"] for m in models),
               "artifacts": artifacts, "folders": folders, "runtime_hashing_enabled": False,
               "data_provenance": provenance, "training": training_summary,
               "sampling_sweeps": sampling_sweeps,
               "order_sweeps": order_sweep_data(root, selection, models),
               "matrix_sweeps": matrix_sweep_data(root, selection, models),
               "selection": selection, "scope": "project-native suite; external official generation scorers have separate result availability"}
    data = {**summary, "training": training, "samples": gallery["samples"], "records": gallery["records"], "labels": LABELS}
    template = (REPO / "scripts/assets/evaluation_report.html").read_text(encoding="utf-8")
    require(template.count("__REPORT_EXTENSIONS__") == 1, "invalid report extension marker")
    extensions = "\n".join((REPO / "scripts/assets" / name).read_text(encoding="utf-8")
                           for name in ("evaluation_dashboard.js", "evaluation_training.js", "evaluation_provenance.js"))
    template = template.replace("__REPORT_EXTENSIONS__", extensions)
    # Generated text can contain HTML/script delimiters. It is data, never code.
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    require(template.count("__REPORT_DATA__") == 1, "invalid report template")
    write(root / "index.html", template.replace("__REPORT_DATA__", embedded))
    write(root / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write(root / "selection.json", json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("updated_at", "qualitative_models", "qualitative_records", "images_verified", "formal_models", "formal_complete_models")}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "output/evaluation")
    parser.add_argument("--selection", type=Path, default=REPO / "configs/protocols/evaluation_report.json")
    parser.add_argument("--plots", action="store_true", help="also refresh standalone PNG/SVG loss figures (requires Matplotlib)")
    args = parser.parse_args()
    build(args.root, args.selection, plots=args.plots)
