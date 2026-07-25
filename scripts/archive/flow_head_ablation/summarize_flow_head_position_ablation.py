#!/usr/bin/env python3
"""Validate, select, and summarize formal FH0--FH4 FID/IS evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.modeling_model.image_flow_position import (  # noqa: E402
    FLOW_HEAD_POSITION_SPECS,
    SUPPORTED_FLOW_HEAD_POSITION_VARIANTS,
)
from scripts.flow_head_position_ablation import (  # noqa: E402
    CONFIRMATION_SEEDS,
    SCREEN_SEED,
    SELECTOR_SCHEMA,
    SUMMARY_SCHEMA,
    canonical_sha256,
)


class SummaryError(ValueError):
    pass


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise SummaryError(f"{label} must be finite, got {result!r}")
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"failed to read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SummaryError(f"{path} must contain a JSON object")
    return payload


def _row(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    metrics = _load_json(path)
    if not bool(metrics.get("official_protocol")):
        raise SummaryError(f"{path}: evaluation is not official 10K protocol")
    architecture = metrics.get("architecture")
    if not isinstance(architecture, Mapping):
        raise SummaryError(f"{path}: missing architecture")
    variant_id = str(architecture.get("ablation_id", ""))
    if variant_id not in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS:
        raise SummaryError(f"{path}: invalid FH ID {variant_id!r}")
    actual_contract = architecture.get("flow_head", {}).get("position_contract")
    expected_contract = FLOW_HEAD_POSITION_SPECS[variant_id].as_contract((80, 80))
    if actual_contract != expected_contract:
        raise SummaryError(
            f"{path}: {variant_id} position contract mismatch: "
            f"expected={expected_contract}, actual={actual_contract}"
        )
    strategy = metrics.get("strategies", {}).get("spatial_halton")
    if not isinstance(strategy, Mapping):
        raise SummaryError(f"{path}: missing spatial_halton metrics")
    protocol = metrics.get("training_protocol", {}).get("flow_head_position")
    if not isinstance(protocol, Mapping):
        raise SummaryError(f"{path}: missing FH training provenance")
    provenance = protocol.get("provenance")
    if not isinstance(provenance, Mapping):
        raise SummaryError(f"{path}: missing compact FH provenance")
    seed = int(provenance.get("training_seed", -1))
    if provenance.get("ablation_id") != variant_id:
        raise SummaryError(f"{path}: provenance ID mismatch")
    if provenance.get("architecture", {}).get("flow_head_position") != expected_contract:
        raise SummaryError(f"{path}: provenance architecture mismatch")
    validation_path = protocol.get("validation_metrics_path")
    validation = _load_json(validation_path)
    if (
        validation.get("ablation_id") != variant_id
        or int(validation.get("training_seed", -1)) != seed
    ):
        raise SummaryError(f"{path}: final validation identity mismatch")
    validation_metrics = validation.get("metrics")
    if not isinstance(validation_metrics, Mapping):
        raise SummaryError(f"{path}: final validation metrics are missing")
    early_keys = (
        "val/flow/context_0_v_mse",
        "val/flow/context_1_v_mse",
    )
    early_values = [
        _finite(validation_metrics.get(key), f"{path}:{key}") for key in early_keys
    ]
    return {
        "id": variant_id,
        "seed": seed,
        "phase": str(provenance.get("phase")),
        "fid": _finite(strategy.get("fid"), f"{path}:FID"),
        "is": _finite(
            strategy.get("inception_score_mean"),
            f"{path}:IS",
        ),
        "is_std": _finite(
            strategy.get("inception_score_std"),
            f"{path}:IS std",
        ),
        "early_flow_v_mse": statistics.fmean(early_values),
        "val_flow_v_mse": _finite(
            validation_metrics.get("val/flow/v_mse"),
            f"{path}:validation flow MSE",
        ),
        "generation_wall_seconds": _finite(
            strategy.get("generation_wall_seconds"),
            f"{path}:generation wall seconds",
        ),
        "generation_samples_per_second": _finite(
            strategy.get("generation_samples_per_second"),
            f"{path}:generation samples/s",
        ),
        "peak_cuda_allocated_mib": _finite(
            metrics.get("distributed", {}).get("peak_cuda_allocated_mib"),
            f"{path}:peak CUDA memory",
        ),
        "flow_head_parameters": int(metrics.get("parameters", {}).get("flow_head", -1)),
        "parameter_schema_sha256": provenance.get(
            "initial_parameter_schema_sha256"
        ),
        "initial_state_sha256": provenance.get("initial_parameter_state_sha256"),
        "runtime_source_manifest_sha256": provenance.get(
            "runtime_source_manifest_sha256"
        ),
        "config_fingerprint": provenance.get("config_fingerprint"),
        "provenance_sha256": provenance.get("provenance_sha256"),
        "metrics_path": str(path.resolve()),
        "validation_metrics_path": str(Path(validation_path).resolve()),
        "position_contract": expected_contract,
    }


def _pareto(rows: list[dict[str, Any]]) -> list[str]:
    frontier = []
    for candidate in rows:
        dominated = any(
            other["id"] != candidate["id"]
            and other["fid"] <= candidate["fid"]
            and other["is"] >= candidate["is"]
            and (
                other["fid"] < candidate["fid"]
                or other["is"] > candidate["is"]
            )
            for other in rows
        )
        if not dominated:
            frontier.append(candidate["id"])
    return [value for value in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS if value in frontier]


def _selector(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["id"]: row for row in rows}
    best_fid = min(row["fid"] for row in rows)
    near_best = {
        row["id"] for row in rows if row["fid"] <= best_fid + 1.0
    }
    pareto = set(_pareto(rows))
    baseline = lookup["FH0"]
    early_loss = {
        row["id"]
        for row in rows
        if row["early_flow_v_mse"] < baseline["early_flow_v_mse"]
        and row["fid"] <= baseline["fid"] + 0.5
    }
    selected = {"FH0"} | near_best | pareto | early_loss
    ordered = [
        value for value in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS if value in selected
    ]
    return {
        "schema": SELECTOR_SCHEMA,
        "screen_seed": SCREEN_SEED,
        "mandatory_ids": ["FH0"],
        "near_best_fid_threshold": 1.0,
        "near_best_fid_ids": [
            value for value in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS if value in near_best
        ],
        "fid_is_pareto_ids": [
            value for value in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS if value in pareto
        ],
        "early_loss_fid_guardrail": 0.5,
        "early_loss_guardrail_ids": [
            value for value in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS if value in early_loss
        ],
        "selected_ids": ordered,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant_id in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS:
        group = [row for row in rows if row["id"] == variant_id]
        if not group:
            continue
        fids = [row["fid"] for row in group]
        scores = [row["is"] for row in group]
        losses = [row["early_flow_v_mse"] for row in group]
        output.append(
            {
                "id": variant_id,
                "seeds": sorted(row["seed"] for row in group),
                "fid_mean": statistics.fmean(fids),
                "fid_std": statistics.pstdev(fids),
                "is_mean": statistics.fmean(scores),
                "is_std_across_seeds": statistics.pstdev(scores),
                "early_flow_v_mse_mean": statistics.fmean(losses),
                "early_flow_v_mse_std": statistics.pstdev(losses),
            }
        )
    return output


def _pairing_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    schema_values = {row["parameter_schema_sha256"] for row in rows}
    counts = {row["flow_head_parameters"] for row in rows}
    if len(schema_values) != 1 or None in schema_values or len(counts) != 1:
        raise SummaryError("FH learned parameter schemas/counts differ")
    for seed in sorted({row["seed"] for row in rows}):
        state_values = {
            row["initial_state_sha256"] for row in rows if row["seed"] == seed
        }
        if len(state_values) != 1 or None in state_values:
            raise SummaryError(
                f"FH initial parameter bytes are not paired for seed {seed}"
            )
    source_values = {row["runtime_source_manifest_sha256"] for row in rows}
    if len(source_values) != 1 or None in source_values:
        raise SummaryError("FH runs do not share one runtime source manifest")
    return {
        "schema": "selfless_flow_head_position_pairing_gate_v1",
        "validated_runs": len(rows),
        "flow_head_parameter_count": next(iter(counts)),
        "parameter_schema_sha256": next(iter(schema_values)),
        "runtime_source_manifest_sha256": next(iter(source_values)),
        "paired_initial_state_by_seed": {
            str(seed): next(
                row["initial_state_sha256"]
                for row in rows
                if row["seed"] == seed
            )
            for seed in sorted({row["seed"] for row in rows})
        },
    }


def _effects(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {row["id"]: row for row in aggregates}
    required = set(SUPPORTED_FLOW_HEAD_POSITION_VARIANTS)
    if set(lookup) != required:
        return {}
    fid = {key: row["fid_mean"] for key, row in lookup.items()}
    return {
        "delta_Rf_Ac1_FH1_minus_FH0": fid["FH1"] - fid["FH0"],
        "delta_Rf_Ac0_FH3_minus_FH2": fid["FH3"] - fid["FH2"],
        "delta_Ac_Rf0_FH2_minus_FH0": fid["FH2"] - fid["FH0"],
        "delta_Ac_Rf1_FH3_minus_FH1": fid["FH3"] - fid["FH1"],
        "delta_Aq_FH4_minus_FH3": fid["FH4"] - fid["FH3"],
        "Ac_x_Rf_interaction": (
            (fid["FH3"] - fid["FH2"]) - (fid["FH1"] - fid["FH0"])
        ),
    }


def summarize(
    metric_paths: list[str | Path],
    *,
    phase: str,
    screen_summary: str | Path | None = None,
) -> dict[str, Any]:
    rows = [_row(path) for path in metric_paths]
    pairs = [(row["id"], row["seed"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise SummaryError("duplicate FH ID/seed metrics")
    if phase == "screen":
        expected = {
            (variant_id, SCREEN_SEED)
            for variant_id in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS
        }
        if set(pairs) != expected:
            raise SummaryError(f"screen metrics mismatch: expected={expected}, got={set(pairs)}")
        if {row["phase"] for row in rows} != {"screen"}:
            raise SummaryError("screen evidence contains non-screen runs")
        selector = _selector(rows)
        source_screen_sha = None
    elif phase == "confirmation":
        if screen_summary is None:
            raise SummaryError("confirmation summary requires --screen-summary")
        screen = _load_json(screen_summary)
        if screen.get("schema") != SUMMARY_SCHEMA or screen.get("phase") != "screen":
            raise SummaryError("confirmation source is not a formal FH screen summary")
        selector = screen.get("selector")
        if not isinstance(selector, Mapping) or selector.get("schema") != SELECTOR_SCHEMA:
            raise SummaryError("confirmation source selector is invalid")
        selected = list(selector["selected_ids"])
        expected = {
            (variant_id, seed)
            for variant_id in selected
            for seed in CONFIRMATION_SEEDS
        }
        if set(pairs) != expected:
            raise SummaryError(
                f"confirmation metrics mismatch: expected={expected}, got={set(pairs)}"
            )
        if {row["phase"] for row in rows} != {"confirmation"}:
            raise SummaryError("confirmation evidence contains non-confirmation runs")
        source_screen_sha = hashlib_sha256_file(screen_summary)
    else:
        raise SummaryError(f"unknown phase {phase!r}")
    aggregates = _aggregate(rows)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "phase": phase,
        "screen_summary_sha256": source_screen_sha,
        "runs": sorted(rows, key=lambda row: (row["id"], row["seed"])),
        "aggregates": aggregates,
        "pairing_gate": _pairing_gate(rows),
        "selector": dict(selector),
        "fid_is_pareto_frontier": _pareto(
            [
                {
                    "id": row["id"],
                    "fid": row["fid_mean"],
                    "is": row["is_mean"],
                }
                for row in aggregates
            ]
        ),
        "effects": _effects(aggregates),
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    return payload


def hashlib_sha256_file(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# Flow-head 2D RoPE ablation: {payload['phase']}",
        "",
        "| ID | Seeds | FID ↓ | IS ↑ | Early flow MSE ↓ |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["aggregates"]:
        lines.append(
            f"| {row['id']} | {','.join(map(str, row['seeds']))} | "
            f"{row['fid_mean']:.4f} ± {row['fid_std']:.4f} | "
            f"{row['is_mean']:.4f} ± {row['is_std_across_seeds']:.4f} | "
            f"{row['early_flow_v_mse_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Selector: " + ", ".join(payload["selector"]["selected_ids"]),
            "",
            "FID contrasts:",
            "",
        ]
    )
    for key, value in payload["effects"].items():
        lines.append(f"- `{key}`: {value:+.4f}")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "confirmation"), required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--screen-summary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = summarize(
        args.metrics,
        phase=args.phase,
        screen_summary=args.screen_summary,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(payload), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
