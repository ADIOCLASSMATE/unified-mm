#!/usr/bin/env python3
"""Bind historical flow-head paths to the consolidated archive layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = REPO_ROOT / "output" / "flow_head_ablation"
STATIC_ROOT = ARCHIVE_ROOT / "static_position_screen"
DYNAMIC_ROOT = ARCHIVE_ROOT / "dynamic_dual_stream_screen"
FINAL_DECISION_PATH = DYNAMIC_ROOT / "evidence" / "final_decision.json"
AMENDED_SUMMARY_PATH = (
    DYNAMIC_ROOT / "evidence" / "amended_quality_only_summary.json"
)
OUTPUT_PATH = ARCHIVE_ROOT / "relocation_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def bind_file(path: Path) -> dict[str, object]:
    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def bind_size(path: Path) -> dict[str, object]:
    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
    }


def directory_inventory(path: Path) -> dict[str, int]:
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return {
        "file_count": len(files),
        "size_bytes": sum(candidate.stat().st_size for candidate in files),
    }


def validate_active_runtime_isolation() -> None:
    active_runtime_files = (
        REPO_ROOT / "pretrain" / "train_selfless_flow.py",
        REPO_ROOT / "scripts" / "evaluate_single_stream_fid_is.py",
        REPO_ROOT / "utils" / "utils.py",
    )
    forbidden = (
        "scripts.archive.flow_head_ablation",
        "require_flow_head_position_ablation_protocol",
        "require_dual_stream_flow_head_ablation_protocol",
    )
    for path in active_runtime_files:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                raise RuntimeError(
                    f"Active runtime {relative(path)} still exposes {token!r}"
                )


def bind_run(run_dir: Path) -> dict[str, object]:
    candidates = (
        run_dir / "config.yaml",
        run_dir / "training_runtime_metrics.json",
        run_dir / "dual_stream_flow_head_training_provenance.json",
        run_dir / "flow_head_position_training_provenance.json",
        run_dir / "fid_is_cfg3p5_10k_ema" / "metrics.json",
        run_dir / "fid_is_selected_cfg3p5_ema" / "metrics.json",
        run_dir / "hf_model-final-ema" / "config.json",
    )
    key_artifact_patterns = (
        "checkpoint-*/metadata.json",
        "checkpoint-*/ema_state.pt",
        "ema_state-final.pt",
        "hf_model-final/model.safetensors",
        "hf_model-final-ema/model.safetensors",
        "image_flow_adapter-final.pt",
    )
    key_artifacts = sorted(
        {
            path
            for pattern in key_artifact_patterns
            for path in run_dir.glob(pattern)
            if path.is_file()
        }
    )
    return {
        "run": run_dir.name,
        "archive_path": relative(run_dir),
        **directory_inventory(run_dir),
        "bound_files": [
            bind_file(candidate) for candidate in candidates if candidate.is_file()
        ],
        "key_artifacts": [bind_size(path) for path in key_artifacts],
    }


def write_final_decision() -> None:
    raw_path = DYNAMIC_ROOT / "evidence" / "raw_preregistered_summary.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    amended = json.loads(AMENDED_SUMMARY_PATH.read_text(encoding="utf-8"))
    roles = {
        "DF0-FH0": "historical_static_anchor",
        "DF0-FH1": "archive_only",
        "DF0-FH4": "archive_only",
        "DF1-FH0": "default_baseline",
        "DF1-FH1": "archive_only_hybrid_empirical_upper_bound",
        "DF1-FH4": "pure_rope_baseline",
        "DF2-FH0": "archive_only",
        "DF2-FH1": "archive_only",
        "DF2-FH4": "archive_only_failure",
    }
    rows = []
    for row in raw["rows"]:
        rows.append(
            {
                "cell_id": row["cell_id"],
                "fid": row["fid"],
                "inception_score_mean": row["inception_score_mean"],
                "inception_score_std": row["inception_score_std"],
                "generation_samples_per_second": row[
                    "generation_samples_per_second"
                ],
                "role": roles[row["cell_id"]],
            }
        )
    payload = {
        "schema": "selfless_flow_head_final_baseline_decision_v1",
        "status": "completed_and_archived",
        "date": "2026-07-25",
        "scope": "matched ImageNet-100 seed-42 10K screen",
        "active_interface": {
            "architecture": "DF1",
            "default_cell": "DF1-FH0",
            "supported_cells": ["DF1-FH0", "DF1-FH4"],
            "supported_position_variants": ["FH0", "FH4"],
        },
        "decision_basis": (
            "user-directed interface convergence: retain the shared-attention/"
            "shared-MLP dynamic content architecture and the two atomic position "
            "endpoints"
        ),
        "selection_policy": {
            "quality_thresholds_retained_as_descriptive_screen": True,
            "sampling_efficiency_reported": True,
            "sampling_efficiency_is_gate": False,
            "post_result_amendment": True,
        },
        "quality_only_selector_outcome": amended["decision"],
        "rows": rows,
        "raw_preregistered_summary": bind_file(raw_path),
        "amended_quality_only_summary": bind_file(AMENDED_SUMMARY_PATH),
    }
    FINAL_DECISION_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    validate_active_runtime_isolation()
    write_final_decision()

    static_runs = sorted(
        path for path in (STATIC_ROOT / "runs").iterdir() if path.is_dir()
    )
    dynamic_runs = sorted(
        path for path in (DYNAMIC_ROOT / "runs").iterdir() if path.is_dir()
    )
    if len(static_runs) != 11:
        raise RuntimeError(
            f"Expected 11 archived static-position runs, found {len(static_runs)}"
        )
    if len(dynamic_runs) != 6:
        raise RuntimeError(
            f"Expected 6 archived dynamic runs, found {len(dynamic_runs)}"
        )

    relocations = []
    for run_dir in (*static_runs, *dynamic_runs):
        relocations.append(
            {
                "original": f"output/{run_dir.name}",
                "archived": relative(run_dir),
                "kind": "run_directory",
            }
        )
    relocations.extend(
        [
            {
                "original": "output/flow_head_position_ablation",
                "archived": relative(
                    STATIC_ROOT / "evidence" / "flow_head_position_ablation"
                ),
                "kind": "evidence_directory",
            },
            {
                "original": "output/aborted-validation-image-probe-0724",
                "archived": relative(
                    DYNAMIC_ROOT
                    / "incidents"
                    / "aborted-validation-image-probe-0724"
                ),
                "kind": "incident_directory",
            },
            {
                "original": "output/selfless-flow-dual-df1-fh1-train-probe-0724",
                "archived": relative(
                    DYNAMIC_ROOT
                    / "incidents"
                    / "selfless-flow-dual-df1-fh1-train-probe-0724"
                ),
                "kind": "incident_directory",
            },
            {
                "original": "output/selfless-flow-dual-df1-train-probe",
                "archived": relative(
                    DYNAMIC_ROOT
                    / "incidents"
                    / "selfless-flow-dual-df1-train-probe"
                ),
                "kind": "incident_directory",
            },
            {
                "original": "output/selfless-flow-dual-df2-s42",
                "archived": relative(
                    DYNAMIC_ROOT / "incidents" / "selfless-flow-dual-df2-s42"
                ),
                "kind": "incident_directory",
            },
            {
                "original": "output/dual_stream_flow_head_position_ablation_summary.json",
                "archived": relative(
                    DYNAMIC_ROOT / "evidence" / "raw_preregistered_summary.json"
                ),
                "kind": "evidence_file",
            },
            {
                "original": "output/dual_stream_flow_head_cuda_smoke.json",
                "archived": relative(
                    DYNAMIC_ROOT / "evidence" / "dual_stream_flow_head_cuda_smoke.json"
                ),
                "kind": "evidence_file",
            },
            {
                "original": "output/dual_stream_flow_head_position_cuda_smoke.json",
                "archived": relative(
                    DYNAMIC_ROOT
                    / "evidence"
                    / "dual_stream_flow_head_position_cuda_smoke.json"
                ),
                "kind": "evidence_file",
            },
        ]
    )

    evidence_files = [
        STATIC_ROOT / "evidence" / "flow_head_position_ablation" / "screen_summary.json",
        STATIC_ROOT / "evidence" / "flow_head_position_ablation" / "screen_summary.md",
        DYNAMIC_ROOT / "evidence" / "raw_preregistered_summary.json",
        AMENDED_SUMMARY_PATH,
        DYNAMIC_ROOT / "evidence" / "dual_stream_flow_head_cuda_smoke.json",
        DYNAMIC_ROOT
        / "evidence"
        / "dual_stream_flow_head_position_cuda_smoke.json",
        FINAL_DECISION_PATH,
    ]
    for path in evidence_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    code_roots = (
        REPO_ROOT / "scripts" / "archive" / "flow_head_ablation",
        REPO_ROOT / "script" / "ablation" / "archive" / "flow_head_ablation",
        REPO_ROOT / "configs" / "ablation" / "archive" / "flow_head_ablation",
    )
    code_files = sorted(
        path
        for root in code_roots
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    code_files.extend(
        [
            REPO_ROOT
            / "docs"
            / "archive"
            / "SELFLESS_FLOW_FLOW_HEAD_2D_ROPE_ABLATION_PROPOSAL_HISTORICAL.md",
            REPO_ROOT
            / "docs"
            / "archive"
            / "SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL_HISTORICAL.md",
        ]
    )
    static_inventory = directory_inventory(STATIC_ROOT)
    dynamic_inventory = directory_inventory(DYNAMIC_ROOT)
    data_inventory = {
        "file_count": (
            static_inventory["file_count"] + dynamic_inventory["file_count"]
        ),
        "size_bytes": (
            static_inventory["size_bytes"] + dynamic_inventory["size_bytes"]
        ),
    }

    payload = {
        "schema": "selfless_flow_head_ablation_archive_relocation_v1",
        "archive_root": relative(ARCHIVE_ROOT),
        "policy": {
            "raw_evidence_mutated": False,
            "recorded_legacy_paths_remain_inside_raw_evidence": True,
            "path_resolution": "apply relocations by longest matching original prefix",
            "all_checkpoints_preserved": True,
            "active_baseline_runs_are_archived_with_the_full_matrix": True,
            "sampling_efficiency_is_gate": False,
            "active_runtime_imports_archive": False,
        },
        "counts": {
            "static_run_directories": len(static_runs),
            "dynamic_run_directories": len(dynamic_runs),
            "relocations": len(relocations),
            "archived_code_files": len(code_files),
            "archived_data_files": data_inventory["file_count"],
            "archived_data_size_bytes": data_inventory["size_bytes"],
        },
        "relocations": relocations,
        "evidence_files": [bind_file(path) for path in evidence_files],
        "code_files": [bind_file(path) for path in sorted(set(code_files))],
        "run_bindings": [bind_run(path) for path in (*static_runs, *dynamic_runs)],
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(relative(OUTPUT_PATH))


if __name__ == "__main__":
    main()
