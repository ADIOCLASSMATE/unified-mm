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
from scripts.evaluation_report_unified import export_generation_sweep
from scripts.evaluation_report_flow_scale import export_flow_head_sweep
from scripts.evaluation_report_status import render_forward_architecture, render_research_status
from utils.experiment_registry import current_presentation, presentation_sort_key, registered_experiments, experiment_identity, task_training_labels
from utils.evaluation.model_contracts import S2_ATTENTION_CONTRACTS, scoring_contract, validate_formal_image_order_scoring
from utils.evaluation.aro import ARO_TASKS, ARO_PRIMARY
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


def report_model_specs(root: Path, selection: dict, gallery_models: list[dict]) -> list[dict]:
    """Include selected final evaluations even when the paired gallery predates them."""
    specs = {m["id"]: {**current_presentation(m), "qualitative_available": True} for m in gallery_models}
    registered = registered_experiments()
    for model_id, selected in selection["models"].items():
        if model_id in specs:
            continue
        run = selected.get("run")
        require(run in registered and registered[run]["id"] == model_id,
                f"selected metric model requires its registered run: {model_id}")
        directory = within(root, selected["root"])
        check_not_invalidated(directory, root)
        evaluation = read(directory / "native_full_evaluation_summary.json")
        require(evaluation.get("complete") is True, f"new model evaluation is incomplete: {model_id}")
        source = evaluation["model_source"]
        checkpoint = (REPO / "output" / run / "hf_model-final-ema").resolve()
        require(source.get("kind") == "hf_final_ema" and source.get("floating_dtype") == "float32"
                and Path(source["path"]).resolve() == checkpoint,
                f"wrong final EMA source for {model_id}")
        saved = read(checkpoint / "config.json")
        export = read(checkpoint / "ema_export_metadata.json")
        require(source["global_step"] == export["source_global_step"] == selected["checkpoint_step"],
                f"checkpoint step mismatch for {model_id}")
        identity = experiment_identity(run)
        specs[model_id] = {"id": model_id, "label": identity["label"], "group": identity["group"],
            "run": run, "checkpoint": str(checkpoint), "source": source,
            "architecture": saved["architecture_variant"], "backbone_attention": saved["dual_stream_attention_contract"],
            "flow_attention": saved.get("flow_head_attention_contract", "architecture_owned"),
            "flow_condition": saved.get("dynamic_xt_flow_condition_contract", saved.get("flow_condition_contract", "architecture_owned")),
            "image_order": saved.get("training_image_sigma_order", "random"),
            "task_training": task_training_labels(identity), "qualitative_available": False}
    return sorted(specs.values(), key=presentation_sort_key)


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
        if spec.get("backbone_attention") in S2_ATTENTION_CONTRACTS:
            require(data["protocol"].get("scoring_contract") == scoring_contract(spec["backbone_attention"], None),
                    f"wrong S2 text scoring contract: {path}")
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
        is_s2 = spec.get("backbone_attention") in S2_ATTENTION_CONTRACTS
        require(manifest.get("dual_stream_attention_contract") == spec.get("backbone_attention"),
                f"benchmark architecture mismatch: {manifest_path}")
        if is_s2:
            benchmark_summary = read(bench_root / "summary.json")
            expected_mc = validate_formal_image_order_scoring(manifest, benchmark_summary.get("scoring", {}))
        else:
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
                key = ARO_PRIMARY if task in ARO_TASKS else "circular_accuracy_language_prior_debiased" if task == "mmbench_dev_en" else (
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
        require(set(protocol["models"]) <= set(selection["models"]), "matrix references unselected formal models")
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


def flow_head_scale_data(root: Path, selection: dict):
    """Compare final EMA head capacities; unfinished training never supplies scores."""
    selected = selection.get("flow_head_scale")
    if not selected:
        return None
    root = root.resolve()
    cfgs = selected["cfg_values"]
    require(isinstance(cfgs, list) and cfgs
            and all(isinstance(cfg, (int, float)) and not isinstance(cfg, bool)
                    and math.isfinite(cfg) and cfg >= 1 for cfg in cfgs)
            and cfgs == sorted(set(cfgs)), "flow head scale CFG coverage mismatch")
    rows, reference = [], None
    for model in selected["models"]:
        spec = current_presentation(model)
        checkpoint = REPO / "output" / spec["run"] / "hf_model-final-ema"
        sources = dict(spec.get("metrics", {}))
        pending = spec.get("pending_metrics", {})
        require(set(pending) <= {str(cfg) for cfg in cfgs}, "unexpected pending scale CFG source")
        for cfg, relative in pending.items():
            if within(root, relative).is_file():
                require(cfg not in sources, "duplicate scale CFG source")
                sources[cfg] = relative
        require(spec["training_status"] in ("complete", "running"), "unknown scale training status")
        require(spec["training_status"] == "complete" or not sources,
                "unfinished scale training cannot supply final scores")
        require(set(sources) <= {str(cfg) for cfg in cfgs}, "unexpected scale CFG source")
        row = {**spec, "checkpoint": str(checkpoint), "results": []}
        row.pop("metrics", None)
        row.pop("pending_metrics", None)
        for cfg in cfgs:
            result = {"cfg": cfg, "fid": None, "is": None, "is_std": None, "source": None}
            relative = sources.get(str(cfg))
            if relative:
                path = within(root, relative)
                check_not_invalidated(path, root)
                data = read(path)
                expected = {"schema": "selfless_imagenet_val_t2i_fid_is_v2",
                            "project_formal_protocol": True, "runtime_hashing_enabled": False,
                            "samples_requested": 50000, "samples_evaluated": 50000,
                            "split": "val", "seed": 42, "cfg": cfg, "cfg_schedule": "constant",
                            "flow_solver": "heun", "temperature": 1.0, "parallel_rate": 1,
                            "backbone_kv_cache": True, "weight_source": "hf_final_ema"}
                for key, value in expected.items():
                    require(data.get(key) == value, f"{path}: scale {key} mismatch")
                require(str(data.get("sampling_steps")) == "10", f"{path}: scale steps mismatch")
                check_checkpoint(data, {**spec, "checkpoint": str(checkpoint),
                                        "source": {"global_step": 95415}}, {}, root)
                require(data["evaluation_model_source"]["kind"] == "hf_final_ema",
                        f"{path}: scale requires final EMA")
                head = data["architecture"]["flow_head"]
                require(all(head[key] == spec[key] for key in ("depth", "width", "architecture")),
                        f"{path}: scale head architecture mismatch")
                require(data["parameters"]["flow_head"] == spec["head_parameters"]
                        and data["parameters"]["total"] == spec["total_parameters"],
                        f"{path}: scale parameter count mismatch")
                contracts = data["implementation_contracts"]
                require(contracts["canonical_initial_noise_enabled"] is True
                        and contracts["paired_sample_count"] == 50000
                        and contracts["ordered_sample_count"] == 50000, f"{path}: scale pairing incomplete")
                precision = data["precision_protocol"]
                require(precision["model_dtype"] == "bf16" and precision["vae_dtype"] == "fp32"
                        and precision["flow_integrator_dtype"] == "fp32", f"{path}: scale precision mismatch")
                metric = data["metric_protocol"]
                require(metric["protocol_name"] == "imagenet_val_fid50k_torch_fidelity_stratified_is"
                        and metric["reference_distribution"] == "imagenet_val_50000"
                        and metric["is_split_assignment"] == "stratified_by_synset"
                        and metric["is_splits"] == 10, f"{path}: scale metric protocol mismatch")
                matched = {key: data[key] for key in ("metric_protocol", "precision_protocol",
                                                     "real_stats_path", "real_stats_metadata")}
                if reference is None:
                    reference = matched
                require(matched == reference, f"{path}: scale scoring/reference protocols differ")
                require(set(data["strategies"]) == {"spatial_halton"}, f"{path}: scale order mismatch")
                score = data["strategies"]["spatial_halton"]
                require(score["count"] == 50000
                        and data["mechanism_diagnostics"]["generated_latent_finite_rate"] == 1.0,
                        f"{path}: scale generation incomplete")
                values = {"fid": score["fid"], "is": score["inception_score_mean"],
                          "is_std": score["inception_score_std"]}
                require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and value >= 0 for value in values.values()),
                        f"{path}: scale score is not finite/nonnegative")
                result.update(**values, source=str(path.relative_to(root)), global_batch=data["batch_size"])
            row["results"].append(result)
        row["completed"] = sum(r["source"] is not None for r in row["results"])
        rows.append(row)
    return {"cfg_values": cfgs, "rows": rows, "completed": sum(row["completed"] for row in rows),
            "total": len(rows) * len(cfgs), "protocol": {
                "checkpoint_step": 95415, "weight_source": "hf_final_ema", "samples": 50000,
                "seed": 42, "sampling_steps": 10, "flow_solver": "heun", "strategy": "spatial_halton",
                "canonical_initial_noise": True}}


def unified_training_ablation_data(root: Path, selection: dict, models: list):
    """Compare the joint recipe with each task's exposure-matched only control."""
    selected = selection.get("unified_training_ablation")
    if not selected:
        return None
    root = root.resolve()
    baseline = next(model for model in models if model["id"] == selected["baseline"])
    baseline_root = selection["models"][baseline["id"]]["root"]
    baseline_paths = {
        "understanding": "pretraining-native-understanding/pretraining_native_understanding_summary.json",
        "generation": "core/t2i-fid-is/metrics.json",
        "text": "core/text/summary.json",
    }
    rows, controls = [], []

    def load(relative, spec):
        path = within(root, relative)
        check_not_invalidated(path, root)
        if not path.exists():
            return None
        data = read(path)
        identity = {**data, "checkpoint_step": data.get("checkpoint_step", data.get("global_step",
                    data.get("evaluation_model_source", {}).get("global_step")))}
        check_checkpoint(identity, spec, {}, root)
        require(data.get("weight_source") == "hf_final_ema", f"only comparison requires final EMA: {path}")
        return data

    def equal(b, o, keys, label):
        require(all(key in b and key in o and b[key] == o[key] for key in keys),
                f"only comparison {label} protocol mismatch")

    def add(task, key, bvalue, ovalue, bpath, opath, *, group=None, aggregate=False, label=None, cfg=None):
        percent = key not in ("fid", "is")
        for value in (bvalue, ovalue):
            if value is None:
                continue
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0 and (not percent or value <= 1),
                    f"invalid only comparison score: {key}")
        def metric(value, path):
            return None if value is None else {"value": value, "percent": percent, "source": path}
        delta = None if ovalue is None else (bvalue - ovalue) * (100 if percent else 1)
        winner = None if delta is None else "tie" if delta == 0 else (
            "baseline" if (delta < 0 if key == "fid" else delta > 0) else "only")
        rows.append({"task": task, "group": group or task, "key": key, "label": label or LABELS[key],
                     "aggregate": aggregate, "lower_is_better": key == "fid",
                     "baseline": metric(bvalue, bpath), "only": metric(ovalue, opath),
                     "delta": delta, "winner": winner})
        if cfg is not None:
            rows[-1]["cfg"] = cfg

    tasks = [control["task"] for control in selected["controls"]]
    require(len(tasks) == len(set(tasks)) and set(tasks) <= set(baseline_paths), "duplicate/unknown only task")
    for control in selected["controls"]:
        task = control["task"]
        bpath = baseline_root + "/" + baseline_paths[task]
        opath = control["source"]
        spec = {**control, "checkpoint": str(REPO / "output" / control["run"] / "hf_model-final-ema"),
                "source": {"global_step": control["checkpoint_step"]}}
        b, o = load(bpath, baseline), load(opath, spec)
        require(b is not None, f"missing B reference for only comparison: {task}")
        complete = o is not None and (o.get("complete") is True if task != "generation" else
                                     o.get("samples_evaluated") == 50000)
        if task != "generation":
            require(b.get("complete") is True, f"incomplete B reference: {task}")
        if not complete:
            o = None
        controls.append({**control, "checkpoint": spec["checkpoint"], "complete": complete,
                         "completed_at": o.get("completed_at") if o else None})

        if task == "text":
            for data in (b, o):
                if data is None:
                    continue
                require(set(data["primary_metrics"]) == set(TEXT_TASKS)
                        and all(data["tasks"][key]["complete"] is True for key in TEXT_TASKS),
                        "incomplete only text task coverage")
                require(math.isclose(data["macro_average_primary"],
                                     sum(data["primary_metrics"].values()) / len(TEXT_TASKS)),
                        "only text macro differs from task mean")
            if o:
                equal(b, o, ("protocol",), "text")
                for key in TEXT_TASKS:
                    equal(b["tasks"][key], o["tasks"][key], ("samples",), key)
            add(task, "text_macro", b["macro_average_primary"], o["macro_average_primary"] if o else None,
                bpath, opath, aggregate=True)
            for key in TEXT_TASKS:
                add(task, key, b["primary_metrics"][key], o["primary_metrics"][key] if o else None, bpath, opath)
        elif task == "understanding":
            if o:
                equal(b, o, ("selection_contract", "dataset_contract", "adaptation"), "understanding")
            for data in (b, o):
                if data is not None:
                    require(not data["coverage"]["missing_required_tasks"], "incomplete only understanding coverage")
            bc = b["imagenet1k_zeroshot_classification"]
            oc = o["imagenet1k_zeroshot_classification"] if o else None
            for data in (bc, oc):
                if data is not None:
                    require(data["complete_formal_target"] is True and data["records"] == 50000,
                            "incomplete only ImageNet coverage")
            if oc:
                equal(bc, oc, ("scoring", "class_text_template", "classes"), "ImageNet")
            for key, source_key in (("top1", "top_1_accuracy"), ("top5", "top_5_accuracy")):
                add(task, key, bc[source_key], oc[source_key] if oc else None, bpath, opath)
            for group in ("paper_compositional_benchmarks", "internal_ablation_diagnostics"):
                for key, bm in b[group].items():
                    om = o[group].get(key) if o else None
                    if om:
                        equal(bm, om, ("kind", "mc_samples", "language_prior_null_images"), key)
                        equal(bm["metrics"], om["metrics"], ("records", "primary_metric"), key)
                    def primary(item):
                        if item is None:
                            return None
                        value = item["metrics"]
                        for part in value["primary_metric"].split("."):
                            value = value[part]
                        return value
                    add(task, key, primary(bm), primary(om), bpath, opath,
                        group="diagnostics" if group == "internal_ablation_diagnostics" else task)
            for dataset, prefix, count in (("mscoco_karpathy_test_5k", "coco", 5000),
                                            ("flickr30k_karpathy_test_1k", "flickr", 1000)):
                br = b["standard_cross_dataset_retrieval"][dataset]
                ori = o["standard_cross_dataset_retrieval"][dataset] if o else None
                for data in (br, ori):
                    if data is not None:
                        require(data["complete_formal_target"] is True and data["images"] == count,
                                f"incomplete only retrieval: {dataset}")
                if ori:
                    equal(br, ori, ("scoring", "images", "captions", "split", "primary_metric"), dataset)
                add(task, prefix + "_mr", br["mean_recall_at_1_5_10"],
                    ori["mean_recall_at_1_5_10"] if ori else None, bpath, opath,
                    label=("COCO-5K" if prefix == "coco" else "Flickr30K-1K") + " 检索 mR")
                for direction, short in (("image_to_text", "i2t"), ("text_to_image", "t2i")):
                    for k in (1, 5, 10):
                        add(task, f"{prefix}_{short}_r{k}", br[direction][f"recall_at_{k}"],
                            ori[direction][f"recall_at_{k}"] if ori else None, bpath, opath, group="retrieval")
        else:
            cfgs = selected.get("generation_cfg_values", [3.5])
            require(isinstance(cfgs, list) and cfgs
                    and all(isinstance(cfg, (int, float)) and not isinstance(cfg, bool)
                            and math.isfinite(cfg) and cfg >= 1 for cfg in cfgs)
                    and cfgs == sorted(set(cfgs)), "invalid only generation CFG coverage")
            done_cfgs, batches = [], {}
            for cfg in cfgs:
                bp = selected.get("generation_baseline_sources", {}).get(str(cfg), bpath if cfg == 3.5 else None)
                op = control.get("cfg_sources", {}).get(str(cfg), opath if cfg == 3.5 else None)
                require(bp and op, f"missing explicit only generation CFG {cfg} source")
                cb, co = load(bp, baseline), load(op, spec)
                require(cb is not None, f"missing B generation CFG {cfg} reference")
                equal(b, cb, ("schema", "metric_protocol", "precision_protocol", "split", "real_source",
                              "real_stats_path", "real_stats_metadata", "cfg_schedule", "temperature",
                              "parallel_rate", "backbone_kv_cache"), "generation across CFG")
                if co and co.get("samples_evaluated") != 50000:
                    co = None
                for data in (cb, co):
                    if data is None:
                        continue
                    require(data["samples_evaluated"] == data["samples_requested"] == 50000
                            and data["cfg"] == cfg and str(data["sampling_steps"]) == "10"
                            and data["flow_solver"] == "heun" and data["seed"] == 42
                            and data["project_formal_protocol"] is True, "only generation protocol mismatch")
                    require(data["implementation_contracts"]["canonical_initial_noise_enabled"] is True
                            and data["strategies"]["spatial_halton"]["count"] == 50000,
                            "only generation pairing/coverage mismatch")
                if co:
                    equal(cb, co, ("schema", "metric_protocol", "precision_protocol", "split", "real_source",
                                  "real_stats_path", "real_stats_metadata", "cfg_schedule", "temperature",
                                  "parallel_rate", "backbone_kv_cache"), "generation")
                    done_cfgs.append(cfg)
                bg = cb["strategies"]["spatial_halton"]
                og = co["strategies"]["spatial_halton"] if co else None
                for key, source_key in (("fid", "fid"), ("is", "inception_score_mean")):
                    add(task, key, bg[source_key], og[source_key] if og else None, bp, op, cfg=cfg)
                    if key == "is":
                        for side, metrics in (("baseline", bg), ("only", og)):
                            if metrics is not None:
                                std = metrics.get("inception_score_std")
                                require(isinstance(std, (int, float)) and not isinstance(std, bool)
                                        and math.isfinite(std) and std >= 0, "invalid only generation IS std")
                                rows[-1][side]["std"] = std
                batches[str(cfg)] = {"baseline": cb["batch_size"], "only": co["batch_size"] if co else None}
            controls[-1].update(complete=len(done_cfgs) == len(cfgs), completed_cfgs=done_cfgs,
                                cfg_total=len(cfgs), generation_batch_sizes=batches)
    completed = sum(control["complete"] for control in controls)
    return {"schema": "unified_training_ablation_v1", "baseline": {
                "id": baseline["id"], "label": "联合 B", "checkpoint": baseline["checkpoint"],
                "checkpoint_step": baseline["source"]["global_step"],
                "physical_positions": selected["baseline_physical_positions"]},
            "controls": controls, "rows": rows, "completed": completed, "total": len(controls),
            "complete": completed == len(controls), "delta_convention": "baseline_minus_only",
            "generation_cfg_values": selected.get("generation_cfg_values", [3.5]),
            "source": "comparisons/unified-training-ablation/comparison.json"}


def build(root: Path, selection_file: Path, *, plots: bool = False, unified_plots: bool = False,
          flow_scale_plots: bool = False):
    root = root.resolve()
    selection = read(selection_file)
    require(selection.get("schema") == "unified_evaluation_report_selection_v1", "unknown report selection schema")
    gallery = gallery_data(root, selection)
    style_instruction = None
    if selection.get("style_instruction"):
        study_root = within(root, selection["style_instruction"])
        style_instruction = read(study_root / "summary.json")
        require(style_instruction.get("complete") is True and read(study_root / "COMPLETED.json").get("complete") is True,
                "style instruction study is incomplete")
        style_instruction["root"] = selection["style_instruction"]
    models = []
    specs = report_model_specs(root, selection, gallery["manifest"]["models"])
    for spec in specs:
        models.append({**spec, **model_metrics(root, spec, selection["models"].get(spec["id"], {}))})
    sampling_sweeps = sampling_sweep_data(root, selection, models)
    flow_head_scale = flow_head_scale_data(root, selection)
    unified_training_ablation = unified_training_ablation_data(root, selection, models)
    updated = datetime.now(UTC).isoformat(timespec="seconds")
    if flow_head_scale:
        flow_head_scale["updated_at"] = updated
        flow_head_scale["cfg_sweep"] = export_flow_head_sweep(
            root, flow_head_scale, write, plots=plots or flow_scale_plots)
    if unified_training_ablation:
        unified_training_ablation["updated_at"] = updated
        unified_training_ablation["generation_sweep"] = export_generation_sweep(
            root, unified_training_ablation, write, plots=plots or unified_plots)
        write(within(root, unified_training_ablation["source"]),
              json.dumps(unified_training_ablation, ensure_ascii=False, indent=2) + "\n")
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
               "qualitative_root": gallery["root"], "qualitative_models": len(gallery["manifest"]["models"]),
               "qualitative_records": len(gallery["records"]), "images_verified": gallery["images_verified"],
               "qualitative_contract": gallery["manifest"].get("contract", {}),
               "qualitative_speed": selection.get("qualitative_speed"),
               "style_instruction": style_instruction,
               "generation_capacity": selection.get("generation_capacity"),
               "formal_models": sum(bool(m["metrics"]) for m in models),
               "formal_complete_models": sum(m["complete"] for m in models),
               "artifacts": artifacts, "folders": folders, "runtime_hashing_enabled": False,
               "data_provenance": provenance, "training": training_summary,
               "sampling_sweeps": sampling_sweeps,
               "order_sweeps": order_sweep_data(root, selection, models),
               "matrix_sweeps": matrix_sweep_data(root, selection, models),
               "flow_head_scale": flow_head_scale,
               "unified_training_ablation": unified_training_ablation,
               "d_cfg_sweep": (read(within(root, selection["d_cfg_sweep"]["output"] + "/comparison.json"))
                               if selection.get("d_cfg_sweep") and within(root, selection["d_cfg_sweep"]["output"] + "/comparison.json").exists() else None),
               "s2_cfg_sweep": (read(within(root, selection["s2_cfg_sweep"]["output"] + "/comparison.json"))
                                if selection.get("s2_cfg_sweep") and within(root, selection["s2_cfg_sweep"]["output"] + "/comparison.json").exists() else None),
               "selection": selection, "scope": "project-native suite; external official generation scorers have separate result availability"}
    data = {**summary, "training": training, "samples": gallery["samples"], "records": gallery["records"], "labels": LABELS}
    template = (REPO / "scripts/assets/evaluation_report.html").read_text(encoding="utf-8")
    require(template.count("__FORWARD_ARCHITECTURE__") == 1, "invalid forward architecture marker")
    template = template.replace("__FORWARD_ARCHITECTURE__", render_forward_architecture())
    require(template.count("__RESEARCH_STATUS__") == 1, "invalid research status marker")
    template = template.replace("__RESEARCH_STATUS__", render_research_status(root, models))
    require(template.count("__REPORT_EXTENSIONS__") == 1, "invalid report extension marker")
    extensions = "\n".join((REPO / "scripts/assets" / name).read_text(encoding="utf-8")
                           for name in ("evaluation_dashboard.js", "evaluation_flow_head_scale.js",
                                        "evaluation_unified_training.js",
                                        "evaluation_training.js", "evaluation_provenance.js", "evaluation_style_instruction.js"))
    template = template.replace("__REPORT_EXTENSIONS__", extensions)
    template = template.replace("</style>", (REPO / "scripts/assets/evaluation_style_instruction.css").read_text() + "\n</style>", 1)
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
    parser.add_argument("--unified-plots", action="store_true", help="refresh B / T2I-only CFG figures (requires Matplotlib)")
    parser.add_argument("--flow-scale-plots", action="store_true", help="refresh flow-head scale CFG figures (requires Matplotlib)")
    args = parser.parse_args()
    build(args.root, args.selection, plots=args.plots, unified_plots=args.unified_plots,
          flow_scale_plots=args.flow_scale_plots)
