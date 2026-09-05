#!/usr/bin/env python3
"""Aggregate comparable formal checkpoint evaluations into a trend report."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


GENERATION_PROTOCOL = "imagenet_val_fid50k_torch_fidelity_stratified_is"
GENERATION_REFERENCE = "imagenet_val_50000"
GENERATION_SCOPE = "same_protocol_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation_root",
        type=Path,
        action="append",
        required=True,
        help="Completed full-evaluation root; repeat once per checkpoint.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def read_summary(root: Path) -> dict[str, Any]:
    path = root / "full_evaluation_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    if payload.get("complete") is not True or payload.get("profile") != "formal":
        raise ValueError(f"trend inputs must be complete formal evaluations: {path}")
    if payload.get("schema") != "unified_full_checkpoint_evaluation_summary_v4":
        raise ValueError(f"trend input uses an obsolete protocol: {path}")
    if payload.get("runtime_hashing_enabled", True) is not False:
        raise ValueError(f"trend input violates the no-hash contract: {path}")
    contract = payload.get("dataset_contract", {})
    if contract.get("training_split") != "imagenet_train":
        raise ValueError(f"training split is not ImageNet train: {path}")
    if contract.get("evaluation_split") != "imagenet_val":
        raise ValueError(f"evaluation split is not ImageNet val: {path}")
    return payload


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite trend metric: {label}={number}")
    return number


def trend_row(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    t2i = summary["generation"]["imagenet_val_t2i"]
    validation = summary["understanding"]["heldout_validation"]
    expected_generation = {
        "project_formal_protocol": True,
        "leaderboard_comparable_to_adm_dit": False,
        "protocol_name": GENERATION_PROTOCOL,
        "reference_distribution": GENERATION_REFERENCE,
        "comparison_scope": GENERATION_SCOPE,
        "samples": 50_000,
        "is_split_assignment": "stratified_by_synset",
    }
    mismatches = {
        key: {"expected": value, "actual": t2i.get(key)}
        for key, value in expected_generation.items()
        if t2i.get(key) != value
    }
    if mismatches:
        raise ValueError(f"T2I trend protocol is invalid at {root}: {mismatches}")
    if not str(t2i.get("strategy", "")).strip():
        raise ValueError(f"T2I trend strategy is missing at {root}")
    fid = finite(t2i["fid"], "fid")
    is_mean = finite(t2i["inception_score_mean"], "inception_score_mean")
    is_std = finite(t2i["inception_score_std"], "inception_score_std")
    if fid < 0.0 or is_mean < 1.0 - 1.0e-9 or is_std < 0.0:
        raise ValueError(f"T2I trend metric range is invalid: {root}")
    return {
        "global_step": int(summary["global_step"]),
        "checkpoint": str(summary["checkpoint"]),
        "evaluation_root": str(root.resolve()),
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "validation_loss": finite(validation["val/loss"], "validation_loss"),
        "validation_loss_i2t": finite(
            validation["val/loss_i2t"], "validation_loss_i2t"
        ),
        "validation_loss_t2i": finite(
            validation["val/loss_t2i"], "validation_loss_t2i"
        ),
        "validation_text_ppl": finite(
            validation["val/ppl_text"], "validation_text_ppl"
        ),
        "pure_text_macro": finite(
            summary["pure_text"]["macro_average_primary"], "pure_text_macro"
        ),
        "pure_text_primary_metrics": summary["pure_text"]["primary_metrics"],
    }


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Unified 0.6B checkpoint trend",
        "",
        "All rows use ImageNet train for training, ImageNet val for evaluation, "
        "project-defined ImageNet-val FID-50K and synset-stratified 10-split IS.",
        "",
        "| step | FID ↓ | IS ↑ | val loss ↓ | I2T loss ↓ | T2I loss ↓ | text PPL ↓ | text macro ↑ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {global_step} | {fid:.6f} | {inception_score_mean:.6f} ± "
            "{inception_score_std:.6f} | {validation_loss:.6f} | "
            "{validation_loss_i2t:.6f} | "
            "{validation_loss_t2i:.6f} | {validation_text_ppl:.6f} | "
            "{pure_text_macro:.6f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = [trend_row(root, read_summary(root)) for root in args.evaluation_root]
    rows.sort(key=lambda row: row["global_step"])
    steps = [row["global_step"] for row in rows]
    if len(set(steps)) != len(steps):
        raise ValueError(f"duplicate checkpoint steps in trend inputs: {steps}")
    payload = {
        "schema": "unified_checkpoint_trend_v2",
        "complete": True,
        "runtime_hashing_enabled": False,
        "dataset_contract": {
            "training_split": "imagenet_train",
            "evaluation_split": "imagenet_val",
        },
        "steps": steps,
        "rows": rows,
    }
    atomic_write(
        args.output_dir / "trend.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(args.output_dir / "trend.md", markdown(rows))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
