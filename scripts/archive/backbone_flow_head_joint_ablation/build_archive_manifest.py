#!/usr/bin/env python3
"""Build the relocation manifest for the completed joint ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "output/backbone_flow_head_joint_ablation"
RUN_ROOT = OUTPUT_ROOT / "runs"
MANIFEST_PATH = OUTPUT_ROOT / "archive_manifest.json"

RUN_NAMES = (
    "selfless-flow-bfh-e2q1-df1-fh0-s42",
    "selfless-flow-bfh-e2q1-df1-fh4-s42",
    "selfless-flow-bfh-e2q0-df1-fh0-s42",
    "selfless-flow-bfh-e2q0-df1-fh4-s42",
    "selfless-flow-bfh-e2bq0-df1-fh0-s42",
    "selfless-flow-bfh-e2bq0-df1-fh4-s42",
)

RELOCATIONS = [
    {
        "historical_path": "configs/ablation/backbone_flow_head_joint",
        "archive_path": "configs/ablation/archive/backbone_flow_head_joint_ablation",
    },
    {
        "historical_path": "script/ablation/train_eval_backbone_flow_head_joint_ablation.sh",
        "archive_path": "script/ablation/archive/backbone_flow_head_joint_ablation/train_eval_backbone_flow_head_joint_ablation.sh",
    },
    {
        "historical_path": "scripts/backbone_flow_head_joint_ablation.py",
        "archive_path": "scripts/archive/backbone_flow_head_joint_ablation/backbone_flow_head_joint_ablation.py",
    },
    {
        "historical_path": "scripts/smoke_backbone_flow_head_joint_ablation.py",
        "archive_path": "scripts/archive/backbone_flow_head_joint_ablation/smoke_backbone_flow_head_joint_ablation.py",
    },
    {
        "historical_path": "tests/test_backbone_flow_head_joint_ablation.py",
        "archive_path": "scripts/archive/backbone_flow_head_joint_ablation/tests/test_backbone_flow_head_joint_ablation.py",
    },
    {
        "historical_path": "docs/SELFLESS_FLOW_BACKBONE_FLOW_HEAD_JOINT_ABLATION_PROPOSAL.md",
        "archive_path": "docs/archive/SELFLESS_FLOW_BACKBONE_FLOW_HEAD_JOINT_ABLATION_PROPOSAL_HISTORICAL.md",
    },
    *[
        {
            "historical_path": f"output/{run}",
            "archive_path": f"output/backbone_flow_head_joint_ablation/runs/{run}",
        }
        for run in RUN_NAMES
    ],
]


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix != ".pyc"
    )


def hashed_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in sorted(set(paths)):
        records.append(
            {
                "path": relative(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def run_binding(run_name: str) -> dict:
    root = RUN_ROOT / run_name
    files = files_under(root)
    bound_relpaths = (
        "config.yaml",
        "training_runtime_metrics.json",
        "joint_ablation_preflight.json",
        "joint_ablation_validated_result.json",
        "fid_is_cfg3p5_10k_ema/metrics.json",
        "hf_model-final-ema/config.json",
    )
    key_relpaths = (
        "ema_state-final.pt",
        "image_flow_adapter-final.pt",
        "hf_model-final/model.safetensors",
        "hf_model-final-ema/model.safetensors",
    )
    bound_files = [
        root / item for item in bound_relpaths if (root / item).is_file()
    ]
    key_artifacts = [
        {
            "path": relative(root / item),
            "size_bytes": (root / item).stat().st_size,
        }
        for item in key_relpaths
        if (root / item).is_file()
    ]
    return {
        "run": run_name,
        "archive_path": relative(root),
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "bound_files": hashed_records(bound_files),
        "key_artifacts": key_artifacts,
    }


def main() -> None:
    code_roots = [
        REPO_ROOT / "configs/ablation/archive/backbone_flow_head_joint_ablation",
        REPO_ROOT / "script/ablation/archive/backbone_flow_head_joint_ablation",
        REPO_ROOT / "scripts/archive/backbone_flow_head_joint_ablation",
        REPO_ROOT
        / "docs/archive/SELFLESS_FLOW_BACKBONE_FLOW_HEAD_JOINT_ABLATION_PROPOSAL_HISTORICAL.md",
        REPO_ROOT / "docs/SELFLESS_FLOW_BACKBONE_FLOW_HEAD_JOINT_ABLATION.md",
        REPO_ROOT / "docs/assets/backbone_flow_head_joint_ablation",
    ]
    evidence_roots = [
        OUTPUT_ROOT / "evidence",
        OUTPUT_ROOT / "incidents",
        OUTPUT_ROOT / "smoke",
    ]
    code_files = hashed_records(
        [file for root in code_roots for file in files_under(root)]
    )
    evidence_files = hashed_records(
        [file for root in evidence_roots for file in files_under(root)]
    )
    run_bindings = [run_binding(run_name) for run_name in RUN_NAMES]
    archive = {
        "schema": "selfless_backbone_flow_head_joint_archive_v1",
        "archive_date": "2026-07-26",
        "archive_root": "output/backbone_flow_head_joint_ablation",
        "policy": {
            "data_loss": "none",
            "run_artifacts": "moved intact; no checkpoint or evaluation file removed",
            "immutable_summary_paths": "resolved through relocations; original summary bytes unchanged",
            "active_default": "E2-Q0__DF1-FH4",
        },
        "relocations": RELOCATIONS,
        "code_files": code_files,
        "evidence_files": evidence_files,
        "run_bindings": run_bindings,
        "counts": {
            "archived_code_files": len(code_files),
            "archived_evidence_files": len(evidence_files),
            "run_directories": len(run_bindings),
            "run_files": sum(item["file_count"] for item in run_bindings),
            "run_size_bytes": sum(item["size_bytes"] for item in run_bindings),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(archive, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
