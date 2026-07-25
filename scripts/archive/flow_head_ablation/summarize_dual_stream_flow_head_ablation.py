#!/usr/bin/env python3
"""Validate the archived 3x3 DF/FH screen with the amended quality selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.archive.flow_head_ablation.dual_stream_flow_head_ablation import (
    EXPECTED_FLOW_HEAD_PARAMETERS,
    POSITION_VARIANTS,
    SUMMARY_SCHEMA,
    TRAIN_VARIANTS,
    canonical_sha256,
    cell_id,
    file_sha256,
    parse_cell_id,
)

ARCHITECTURES = ("DF0", *TRAIN_VARIANTS)
CELLS = tuple(
    cell_id(architecture, position, allow_baseline=True)
    for architecture in ARCHITECTURES
    for position in POSITION_VARIANTS
)
BASELINE_METRICS = {
    "DF0-FH0": {"fid": 25.0669, "inception_score_mean": 61.5120},
    "DF0-FH1": {"fid": 25.2985, "inception_score_mean": 60.3427},
    "DF0-FH4": {"fid": 26.1535, "inception_score_mean": 60.1577},
}
FID_THRESHOLD = 24.5669
IS_THRESHOLD = 61.0120


def _load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read metrics {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"metrics must be an object: {path}")
    return dict(payload)


def _metric_row(path: str | Path, expected_cell: str) -> dict[str, Any]:
    architecture, position = parse_cell_id(
        expected_cell, allow_baseline=True
    )
    payload = _load(path)
    strategy = payload.get("strategies", {}).get("spatial_halton")
    if not isinstance(strategy, Mapping):
        raise ValueError(f"{path}: missing spatial_halton metrics")
    flow_head = payload.get("architecture", {}).get("flow_head", {})
    actual_architecture = str(flow_head.get("variant", "DF0")).upper()
    position_contract = flow_head.get("position_contract", {})
    actual_position = str(position_contract.get("variant", "")).upper()
    if (actual_architecture, actual_position) != (architecture, position):
        raise ValueError(
            f"{path}: expected {expected_cell}, found "
            f"{actual_architecture}-{actual_position or 'UNKNOWN'}"
        )
    count = int(payload.get("parameters", {}).get("flow_head", -1))
    if count != EXPECTED_FLOW_HEAD_PARAMETERS:
        raise ValueError(
            f"{path}: flow-head parameters={count}, "
            f"expected={EXPECTED_FLOW_HEAD_PARAMETERS}"
        )
    provenance = None
    training_runtime = None
    if architecture != "DF0":
        training_evidence = (
            payload.get("training_protocol", {})
            .get("dual_stream_flow_head", {})
        )
        provenance = training_evidence.get("provenance")
        training_runtime = training_evidence.get("training_runtime")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{path}: missing DF training provenance")
        if not isinstance(training_runtime, Mapping):
            raise ValueError(f"{path}: missing DF training runtime metrics")
        if str(provenance.get("ablation_id")) != expected_cell:
            raise ValueError(f"{path}: provenance cell ID mismatch")
        provenance_architecture = provenance.get("architecture", {})
        if (
            provenance_architecture.get("variant") != architecture
            or provenance_architecture.get("position_variant") != position
        ):
            raise ValueError(f"{path}: provenance architecture drifted")
    return {
        "cell_id": expected_cell,
        "architecture": architecture,
        "position": position,
        "metrics_path": str(Path(path).resolve()),
        "metrics_sha256": file_sha256(path),
        "fid": float(strategy["fid"]),
        "inception_score_mean": float(strategy["inception_score_mean"]),
        "inception_score_std": float(strategy["inception_score_std"]),
        "generation_wall_seconds": float(strategy["generation_wall_seconds"]),
        "generation_samples_per_second": float(
            strategy["generation_samples_per_second"]
        ),
        "flow_content_cache_peak_bytes_per_sample": int(
            strategy.get("flow_content_cache_peak_bytes_per_sample", 0)
        ),
        "flow_cfg_content_cache_divergence_by_layer": strategy.get(
            "flow_cfg_content_cache_divergence_by_layer"
        ),
        "flow_head_parameters": count,
        "provenance": dict(provenance) if provenance is not None else None,
        "training_runtime": (
            dict(training_runtime)
            if training_runtime is not None
            else None
        ),
    }


def _delta(
    lookup: Mapping[str, Mapping[str, Any]],
    left: str,
    right: str,
) -> dict[str, float]:
    return {
        "fid": float(lookup[left]["fid"] - lookup[right]["fid"]),
        "inception_score_mean": float(
            lookup[left]["inception_score_mean"]
            - lookup[right]["inception_score_mean"]
        ),
    }


def _select(lookup: Mapping[str, Mapping[str, Any]]) -> tuple[str, list[str], str]:
    passing = [
        cell
        for cell in CELLS
        if cell.startswith(("DF1-", "DF2-")) and lookup[cell]["passes_screen"]
    ]
    if not passing:
        return (
            "DF0-FH0",
            [],
            "No dynamic architecture-position cell passed the quality gate.",
        )
    best_fid = min(float(lookup[cell]["fid"]) for cell in passing)
    fid_pool = [
        cell
        for cell in passing
        if float(lookup[cell]["fid"]) <= best_fid + 0.25
    ]
    if len(fid_pool) == 1:
        selected = fid_pool[0]
        return selected, passing, f"{selected} is the only cell within 0.25 FID of the best."
    best_is = max(
        float(lookup[cell]["inception_score_mean"]) for cell in fid_pool
    )
    is_pool = [
        cell
        for cell in fid_pool
        if float(lookup[cell]["inception_score_mean"]) >= best_is - 0.5
    ]
    if len(is_pool) == 1:
        selected = is_pool[0]
        return selected, passing, f"{selected} has the preferred IS within the near-best FID set."
    best_throughput = max(
        float(lookup[cell]["generation_samples_per_second"])
        for cell in is_pool
    )
    throughput_pool = [
        cell
        for cell in is_pool
        if abs(
            float(lookup[cell]["generation_samples_per_second"])
            - best_throughput
        )
        <= 1e-12
    ]
    architecture_order = {"DF2": 0, "DF1": 1}
    position_order = {"FH0": 0, "FH1": 1, "FH4": 2}
    selected = min(
        throughput_pool,
        key=lambda cell: (
            architecture_order[lookup[cell]["architecture"]],
            position_order[lookup[cell]["position"]],
        ),
    )
    return (
        selected,
        passing,
        (
            f"{selected} wins the throughput and deterministic "
            "architecture/position tie-break."
        ),
    )


def summarize(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    missing = [cell for cell in CELLS if cell not in paths]
    if missing:
        raise ValueError(f"missing metrics paths for cells: {missing}")
    rows = [_metric_row(paths[cell], cell) for cell in CELLS]
    lookup = {row["cell_id"]: row for row in rows}
    for baseline_cell, expected in BASELINE_METRICS.items():
        for metric, anchor in expected.items():
            if abs(float(lookup[baseline_cell][metric]) - anchor) > 5e-4:
                raise ValueError(
                    f"{baseline_cell} {metric} no longer matches its anchor"
                )

    dynamic_rows = [
        lookup[cell]
        for cell in CELLS
        if cell.startswith(("DF1-", "DF2-"))
    ]
    pairing_keys = (
        "flow_head_parameter_count",
        "flow_head_parameter_schema_sha256",
        "flow_head_initial_state_sha256",
        "train_order_sha256",
        "augmentation_sha256",
        "runtime_source_manifest_sha256",
    )
    reference_provenance = dynamic_rows[0]["provenance"]
    for row in dynamic_rows[1:]:
        for key in pairing_keys:
            if row["provenance"].get(key) != reference_provenance.get(key):
                raise ValueError(
                    f"dynamic-cell pairing evidence differs for {key}"
                )

    canonical = lookup["DF0-FH0"]
    for row in dynamic_rows:
        matched_baseline = lookup[f"DF0-{row['position']}"]
        row["delta_fid_vs_canonical_df0_fh0"] = (
            row["fid"] - canonical["fid"]
        )
        row["delta_is_vs_canonical_df0_fh0"] = (
            row["inception_score_mean"]
            - canonical["inception_score_mean"]
        )
        row["delta_fid_vs_matched_df0"] = (
            row["fid"] - matched_baseline["fid"]
        )
        row["delta_is_vs_matched_df0"] = (
            row["inception_score_mean"]
            - matched_baseline["inception_score_mean"]
        )
        row["sampling_wall_ratio_vs_df0_fh0"] = (
            row["generation_wall_seconds"]
            / canonical["generation_wall_seconds"]
        )
        row["passes_quality_gate"] = (
            row["fid"] <= FID_THRESHOLD
            and row["inception_score_mean"] >= IS_THRESHOLD
        )
        # Sampling wall time is retained as a diagnostic, not a screen gate.
        row["passes_screen"] = row["passes_quality_gate"]

    architecture_estimands = {}
    for position in POSITION_VARIANTS:
        architecture_estimands[position] = {
            "dual_df1_minus_df0": _delta(
                lookup, f"DF1-{position}", f"DF0-{position}"
            ),
            "attention_only_df2_minus_df0": _delta(
                lookup, f"DF2-{position}", f"DF0-{position}"
            ),
            "content_mlp_df1_minus_df2": _delta(
                lookup, f"DF1-{position}", f"DF2-{position}"
            ),
        }
    position_estimands = {}
    for architecture in ARCHITECTURES:
        position_estimands[architecture] = {
            "rope_given_add_fh1_minus_fh0": _delta(
                lookup, f"{architecture}-FH1", f"{architecture}-FH0"
            ),
            "remove_add_given_rope_fh4_minus_fh1": _delta(
                lookup, f"{architecture}-FH4", f"{architecture}-FH1"
            ),
            "pure_rope_fh4_minus_fh0": _delta(
                lookup, f"{architecture}-FH4", f"{architecture}-FH0"
            ),
        }
    interaction_estimands = {}
    for architecture in TRAIN_VARIANTS:
        interaction_estimands[architecture] = {}
        for label in (
            "rope_given_add_fh1_minus_fh0",
            "remove_add_given_rope_fh4_minus_fh1",
        ):
            interaction_estimands[architecture][label] = {
                metric: (
                    position_estimands[architecture][label][metric]
                    - position_estimands["DF0"][label][metric]
                )
                for metric in ("fid", "inception_score_mean")
            }

    selected, passing, rationale = _select(lookup)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "phase": "architecture_position_screen",
        "baselines": {
            cell: {
                **metrics,
                "retrained": False,
            }
            for cell, metrics in BASELINE_METRICS.items()
        },
        "thresholds": {
            "fid_max": FID_THRESHOLD,
            "inception_score_min": IS_THRESHOLD,
        },
        "sampling_efficiency": {
            "role": "reported_diagnostic_not_gate",
            "metric": "sampling_wall_ratio_vs_df0_fh0",
        },
        "pairing": {
            key: reference_provenance[key] for key in pairing_keys
        },
        "rows": rows,
        "estimands": {
            "architecture": architecture_estimands,
            "position": position_estimands,
            "architecture_position_interaction": interaction_estimands,
        },
        "decision": {
            "selected": selected,
            "passing_candidates": passing,
            "rationale": rationale,
            "trigger_confirmation": selected != "DF0-FH0",
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for cell in CELLS:
        parser.add_argument(
            f"--{cell.lower()}",
            dest=cell.lower().replace("-", "_"),
            required=True,
        )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        cell: getattr(args, cell.lower().replace("-", "_"))
        for cell in CELLS
    }
    payload = summarize(paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
