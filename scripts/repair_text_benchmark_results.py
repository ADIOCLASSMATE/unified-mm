#!/usr/bin/env python3
"""Repair legacy v2 text scores in place, retaining a pre-P1 backup.

Seven tasks reuse unchanged summed likelihoods. WinoGrande needs fresh suffix
inference: optionally pass a separate v3 WinoGrande-only evaluation directory.
Until then the suite and its eight-task macro are explicitly incomplete.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_selfless_text_benchmarks import (
    DEFAULT_DATA_ROOT,
    DEFAULT_TASKS,
    LM_EVAL_REFERENCE,
    TEXT_NORMALIZATION,
    TEXT_PROTOCOL_SCHEMA,
    WINOGRANDE_SCORING,
    aggregate_mc_rows,
    argmax,
    atomic_write_text,
    load_multiple_choice_task,
    primary_metric,
    utc_now,
)
from utils.evaluation_model_source import resolve_evaluation_model_source


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def renormalize_rows(task, rows, examples):
    """Validate source alignment before changing any persisted result."""
    if task == "winogrande":
        raise ValueError("WinoGrande requires fresh shared-suffix inference")
    rows = sorted(deepcopy(rows), key=lambda row: int(row["item_index"]))
    if [int(row["item_index"]) for row in rows] != list(range(len(examples))):
        raise ValueError(f"{task}: incomplete or duplicate sample coverage")
    for row, example in zip(rows, examples):
        if row.get("schema") != "selfless_text_multiple_choice_sample_v1":
            raise ValueError(f"{task}: expected legacy v1 samples")
        if (row["task"], row["item_id"], int(row["label"])) != (task, example.item_id, example.label):
            raise ValueError(f"{task}: sample does not match the frozen dataset")
        lengths = [len(choice) for choice in example.choices]
        raw = row["choice_loglikelihoods"]
        if len(raw) != len(lengths) or any(length <= 0 for length in lengths):
            raise ValueError(f"{task}: invalid choice lengths")
        if any(not math.isfinite(float(value)) for value in raw):
            raise ValueError(f"{task}: non-finite likelihood")
        normalized = [float(value) / length for value, length in zip(raw, lengths)]
        prediction = argmax(raw)
        prediction_normalized = argmax(normalized)
        row.update(
            schema="selfless_text_multiple_choice_sample_v2",
            protocol_schema=TEXT_PROTOCOL_SCHEMA,
            normalization=TEXT_NORMALIZATION,
            choice_char_counts=lengths,
            choice_normalized_loglikelihoods=normalized,
            prediction=prediction,
            prediction_normalized=prediction_normalized,
            correct=prediction == example.label,
            correct_normalized=prediction_normalized == example.label,
        )
    return rows


def resolve_repair_checkpoint(original, run, relocated_checkpoint=None):
    """Accept only an explicitly declared relocation of a missing HF export.

The declaration supplies identity; readable provenance checks catch accidental
cross-step/architecture merges without violating the project's no-hash rule.
"""
    old_path = Path(original["checkpoint"]).resolve()
    if relocated_checkpoint is None:
        return old_path
    new_path = relocated_checkpoint.resolve()
    if old_path.exists() and old_path != new_path:
        raise ValueError("relocation cannot replace an existing checkpoint")
    source = resolve_evaluation_model_source(new_path)
    if not source.is_hf_final_ema or source.global_step != original["checkpoint_step"]:
        raise ValueError("relocated checkpoint has different weight provenance")
    recorded_ema = run.get("ema", {})
    for field in ("floating_dtype", "state_key_count"):
        if source.metadata.get(field) != recorded_ema.get(field):
            raise ValueError(f"relocated checkpoint provenance differs: {field}")
    config = read_json(new_path / "config.json")
    if config.get("dual_stream_attention_contract") != run.get("dual_stream_attention_contract"):
        raise ValueError("relocated checkpoint attention contract differs")
    return new_path


def repair_results(text_dir, data_root, winogrande_text_dir=None, relocated_checkpoint=None):
    text_dir = text_dir.resolve()
    backup = text_dir / "legacy-before-p1"
    source = backup if backup.is_dir() else text_dir
    original = read_json(source / "summary.json")
    run = read_json(source / "evaluation_run.json")
    checkpoint = resolve_repair_checkpoint(original, run, relocated_checkpoint)
    if original.get("protocol", {}).get("protocol_schema") != "selfless_text_benchmark_v2":
        raise ValueError("repair requires the original v2 summary (or its pre-P1 backup)")
    if original["protocol"].get("lm_eval_reference", {}).get("commit") != LM_EVAL_REFERENCE["commit"]:
        raise ValueError("source does not use the frozen dataset/prompt reference")
    if run.get("limit") != 0 or set(original["tasks"]) != set(DEFAULT_TASKS):
        raise ValueError("repair requires a complete, unlimited eight-task source")
    prepared = {}
    changes = {}
    for task in DEFAULT_TASKS:
        if task == "winogrande":
            continue
        examples = load_multiple_choice_task(task, data_root)
        old_rows = read_rows(source / "tasks" / task / "samples.jsonl")
        new_rows = renormalize_rows(task, old_rows, examples)
        prepared[task] = new_rows
        old_predictions = {int(row["item_index"]): row["prediction_normalized"] for row in old_rows}
        changes[task] = {
            "old_primary": primary_metric(task, original["tasks"][task]),
            "normalized_prediction_changes": sum(
                row["prediction_normalized"] != old_predictions[row["item_index"]] for row in new_rows
            ),
            "samples": len(new_rows),
        }
    if winogrande_text_dir is not None:
        fresh = read_json(winogrande_text_dir / "summary.json")
        fresh_run = read_json(winogrande_text_dir / "evaluation_run.json")
        protocol = fresh.get("protocol", {})
        if fresh.get("complete") is not True or protocol.get("protocol_schema") != TEXT_PROTOCOL_SCHEMA or (
            protocol.get("winogrande_scoring") != WINOGRANDE_SCORING
        ):
            raise ValueError("WinoGrande input is not a completed suffix-scoring run")
        if Path(fresh["checkpoint"]).resolve() != checkpoint:
            raise ValueError("WinoGrande checkpoint differs from the source")
        for field in ("checkpoint_step", "model_dtype", "max_length", "limit", "dual_stream_attention_contract"):
            if fresh_run.get(field) != run.get(field):
                raise ValueError(f"WinoGrande inference contract differs: {field}")
        prepared["winogrande"] = read_rows(winogrande_text_dir / "tasks" / "winogrande" / "samples.jsonl")
        examples = load_multiple_choice_task("winogrande", data_root)
        rows = sorted(prepared["winogrande"], key=lambda row: int(row["item_index"]))
        if len(rows) != len(examples) or any(
            (row["item_index"], row["item_id"], row["label"]) != (ex.item_index, ex.item_id, ex.label)
            or row.get("protocol_schema") != TEXT_PROTOCOL_SCHEMA
            or row.get("schema") != "selfless_text_multiple_choice_sample_v2"
            or row.get("normalization") != TEXT_NORMALIZATION
            for row, ex in zip(rows, examples)
        ):
            raise ValueError("WinoGrande sample coverage or protocol is invalid")
    # Validate everything above before the first write. The original is never
    # overwritten in the backup, including on retries or a later WinoGrande merge.
    if not backup.exists():
        backup.mkdir()
        for name in ("summary.json", "evaluation_run.json"):
            shutil.copy2(text_dir / name, backup / name)
        shutil.copytree(text_dir / "tasks", backup / "tasks")
    metrics = {}
    for task, rows in prepared.items():
        metrics[task] = aggregate_mc_rows(text_dir, task, rows, len(rows))
        if task in changes:
            changes[task]["new_primary"] = primary_metric(task, metrics[task])
    pending = [] if "winogrande" in metrics else ["winogrande"]
    if pending:
        metrics["winogrande"] = {
            "schema": "selfless_text_multiple_choice_invalidated_v1",
            "task": "winogrande", "complete": False,
            "runtime_hashing_enabled": False,
            "reason": "legacy_option_and_suffix_scores_require_new_suffix_inference",
            "legacy_results": str(backup / "tasks" / "winogrande"),
        }
        write_json(text_dir / "tasks" / "winogrande" / "metrics.json", metrics["winogrande"])
    primary = {task: None if task in pending else primary_metric(task, metrics[task]) for task in DEFAULT_TASKS}
    report = {
        "schema": "selfless_text_p1_correction_v1",
        "runtime_hashing_enabled": False,
        "corrected_at": utc_now(), "backup": str(backup),
        "checkpoint": original["checkpoint"], "checkpoint_step": original["checkpoint_step"],
        "resolved_checkpoint": str(checkpoint),
        "checkpoint_relocation_explicit": relocated_checkpoint is not None,
        "normalization": TEXT_NORMALIZATION, "changes": changes,
        "pending_tasks": pending,
        "winogrande_source": str(winogrande_text_dir.resolve()) if winogrande_text_dir else None,
        "legacy_winogrande_accuracy": original["tasks"]["winogrande"]["accuracy"],
        "legacy_macro_average_primary": original["macro_average_primary"],
    }
    summary = deepcopy(original)
    summary.update(
        checkpoint=str(checkpoint),
        schema="selfless_text_benchmark_summary_v4", complete=not pending,
        tasks=metrics, primary_metrics=primary, pending_tasks=pending,
        macro_average_primary=None if pending else sum(primary.values()) / len(primary),
        completed_at=None if pending else utc_now(), p1_correction=report,
    )
    summary["protocol"].update(protocol_schema=TEXT_PROTOCOL_SCHEMA,
        normalization=TEXT_NORMALIZATION, winogrande_scoring=WINOGRANDE_SCORING)
    run.update(schema="selfless_text_benchmark_run_v4", protocol_schema=TEXT_PROTOCOL_SCHEMA,
        checkpoint=str(checkpoint),
        complete=not pending, completed_at=summary["completed_at"],
        pending_tasks=pending, p1_correction=report)
    if relocated_checkpoint is not None:
        run["model_source"] = str(checkpoint)
        run["ema"]["path"] = str(checkpoint)
    write_json(text_dir / "summary.json", summary)
    write_json(text_dir / "evaluation_run.json", run)
    write_json(text_dir / "p1_correction.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text_dir", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--winogrande_text_dir", type=Path)
    parser.add_argument("--relocated_checkpoint", type=Path,
                        help="Explicit current path of the same, relocated original HF export")
    args = parser.parse_args()
    print(json.dumps(repair_results(args.text_dir, args.data_root, args.winogrande_text_dir,
                                   args.relocated_checkpoint), indent=2))


if __name__ == "__main__":
    main()
