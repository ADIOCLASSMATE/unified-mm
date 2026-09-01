#!/usr/bin/env python3
"""Combine image and pure-text metrics into one unified evaluation report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from utils.evaluation_model_source import resolve_evaluation_model_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    return parser.parse_args()


def atomic_write_text(path: Path, payload: str) -> None:
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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def forbidden_audit_fields(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if "sha256" in lowered or "digest" in lowered:
                found.append(path)
            found.extend(forbidden_audit_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(forbidden_audit_fields(item, f"{prefix}[{index}]"))
    return found


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = resolve_evaluation_model_source(checkpoint)
    image = read_json(args.output_root / "evaluation_summary.json")
    text = read_json(args.output_root / "text" / "summary.json")
    if not bool(image.get("complete", False)) or not bool(text.get("complete", False)):
        raise ValueError("image or text evaluation is incomplete")
    if Path(image["checkpoint"]).resolve() != checkpoint:
        raise ValueError("image evaluation checkpoint differs from requested checkpoint")
    if Path(text["checkpoint"]).resolve() != checkpoint:
        raise ValueError("text evaluation checkpoint differs from requested checkpoint")
    for name, payload in (("image", image), ("text", text)):
        weight_source = payload.get("weight_source")
        if source.is_hf_final_ema and weight_source != source.kind:
            raise ValueError(f"{name} evaluation lacks final-HF source identity")
        if weight_source is not None and weight_source != source.kind:
            raise ValueError(f"{name} evaluation weight source differs")
    if int(image["global_step"]) != int(text["checkpoint_step"]):
        raise ValueError("image and text evaluation checkpoint steps differ")
    dataset_contract = image.get("dataset_contract", {})
    if dataset_contract.get("training_split") != "imagenet_train":
        raise ValueError("image training split is not ImageNet train")
    if dataset_contract.get("evaluation_split") != "imagenet_val":
        raise ValueError("image evaluation split is not ImageNet val")
    report = {
        "schema": "unified_full_checkpoint_evaluation_summary_v2",
        "complete": True,
        "profile": args.profile,
        "runtime_hashing_enabled": False,
        "checkpoint": str(checkpoint),
        "global_step": int(image["global_step"]),
        "weight_source": source.kind,
        "model_source": source.report(),
        "dataset_contract": image["dataset_contract"],
        "generation": {
            "imagenet_val_t2i": image["t2i_fid_is"],
            "qualitative_images": image["qualitative_artifacts"]["flow_images"],
        },
        "understanding": {
            "heldout_validation": image["heldout_validation"],
            "qualitative_captions": image["qualitative_artifacts"][
                "caption_jsonl"
            ],
        },
        "pure_text": {
            "macro_average_primary": text["macro_average_primary"],
            "primary_metrics": text["primary_metrics"],
            "tasks": text["tasks"],
            "protocol": text["protocol"],
        },
        "components": {
            "image_summary": str(
                (args.output_root / "evaluation_summary.json").resolve()
            ),
            "text_summary": str(
                (args.output_root / "text" / "summary.json").resolve()
            ),
        },
        "completed_at": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    forbidden = forbidden_audit_fields(report)
    if forbidden:
        raise ValueError(f"no-hash report contains forbidden audit fields: {forbidden}")
    destination = args.output_root / "full_evaluation_summary.json"
    atomic_write_text(
        destination,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
