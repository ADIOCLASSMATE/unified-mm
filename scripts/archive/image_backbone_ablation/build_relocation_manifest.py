#!/usr/bin/env python3
"""Bind the original ablation paths to the consolidated archive layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = REPO_ROOT / "output" / "image_backbone_ablation"
RUNS_ROOT = ARCHIVE_ROOT / "runs"
EVIDENCE_ROOT = ARCHIVE_ROOT / "evidence"
OUTPUT_PATH = ARCHIVE_ROOT / "relocation_manifest.json"
PRUNED_EVIDENCE_PATH = (
    EVIDENCE_ROOT
    / "pruned_partial_runs"
    / "pruning_manifest_20260726.json"
)
PRUNED_RUN_NAMES = (
    "selfless-flow-image-embedder-e6-seed44",
    "selfless-flow-image-embedder-qf-e2-q1-seed43",
    "selfless-flow-image-embedder-qf-e2-q1-seed44",
    "selfless-flow-image-embedder-qf-e2-q1-seed45",
)


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


def main() -> None:
    run_dirs = sorted(
        path for path in RUNS_ROOT.glob("selfless-flow-image-embedder-*") if path.is_dir()
    )
    if len(run_dirs) != 43:
        raise RuntimeError(f"Expected 43 retained run directories, found {len(run_dirs)}")

    relocations = [
        {
            "original": f"output/{path.name}",
            "archived": relative(path),
            "kind": "run_directory",
        }
        for path in run_dirs
    ]
    relocations.extend(
        [
            {
                "original": "output/image_embedder_ablation",
                "archived": relative(EVIDENCE_ROOT / "screening_and_confirmation"),
                "kind": "evidence_directory",
            },
            {
                "original": "output/image_mask_position_ablation",
                "archived": relative(EVIDENCE_ROOT / "mask_position_q_factor"),
                "kind": "evidence_directory",
            },
        ]
    )
    relocations.extend(
        {
            "original": f"output/{run_name}",
            "archived": relative(
                EVIDENCE_ROOT / "pruned_partial_runs" / run_name
            ),
            "kind": "pruned_run_evidence",
        }
        for run_name in PRUNED_RUN_NAMES
    )

    evidence_files = [
        EVIDENCE_ROOT
        / "screening_and_confirmation"
        / "expanded_seed42_summary.json",
        EVIDENCE_ROOT
        / "screening_and_confirmation"
        / "confirmation_d1_summary.json",
        EVIDENCE_ROOT
        / "screening_and_confirmation"
        / "evaluation_retry_incidents.json",
        EVIDENCE_ROOT / "mask_position_q_factor" / "legacy_bridge_summary.json",
        EVIDENCE_ROOT / "mask_position_q_factor" / "legacy_bridge_runs.csv",
        EVIDENCE_ROOT
        / "mask_position_q_factor"
        / "evaluation_retry_incidents.json",
        REPO_ROOT
        / "configs"
        / "ablation"
        / "image_mask_position_legacy_q1_equivalence_v1.json",
        REPO_ROOT
        / "configs"
        / "ablation"
        / "image_mask_position_legacy_q1_reuse_manifest_v1.json",
        REPO_ROOT
        / "configs"
        / "ablation"
        / "image_mask_position_q0_metrics_attestation_v1.json",
        PRUNED_EVIDENCE_PATH,
    ]
    for path in evidence_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    run_bindings = []
    for run_dir in run_dirs:
        bound_files = []
        for candidate in (
            run_dir / "config.yaml",
            run_dir / "fid_is_selected_cfg3p5_ema" / "metrics.json",
            run_dir / "hf_model-final-ema" / "config.json",
        ):
            if candidate.is_file():
                bound_files.append(bind_file(candidate))
        run_bindings.append(
            {
                "run": run_dir.name,
                "archive_path": relative(run_dir),
                "bound_files": bound_files,
            }
        )

    payload = {
        "schema": "selfless_flow_image_backbone_archive_relocation_v1",
        "archive_root": relative(ARCHIVE_ROOT),
        "policy": {
            "raw_evidence_mutated": False,
            "recorded_legacy_paths_remain_inside_raw_evidence": True,
            "path_resolution": "apply relocations by longest matching original prefix",
            "old_q1_training_reused_not_rerun": True,
            "completed_run_artifacts_preserved": True,
            "stopped_partial_runs_pruned": True,
        },
        "counts": {
            "run_directories": len(run_dirs),
            "pruned_partial_run_directories": len(PRUNED_RUN_NAMES),
            "relocations": len(relocations),
        },
        "relocations": relocations,
        "evidence_files": [bind_file(path) for path in evidence_files],
        "run_bindings": run_bindings,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(relative(OUTPUT_PATH))


if __name__ == "__main__":
    main()
