#!/usr/bin/env python3
"""Run the pinned clean-fid API for MJHQ-30K overall and category FID."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from cleanfid import fid

EXPECTED_CLEANFID_VERSION = "0.1.35"
MJHQ_CATEGORIES = (
    "animals",
    "art",
    "fashion",
    "food",
    "indoor",
    "landscape",
    "logo",
    "people",
    "plants",
    "vehicles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--generated_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=12)
    return parser.parse_args()


def _compute(reference: Path, generated: Path, args: argparse.Namespace) -> float:
    return float(
        fid.compute_fid(
            str(reference),
            str(generated),
            mode="clean",
            model_name="inception_v3",
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )


def main() -> None:
    args = parse_args()
    installed_version = version("clean-fid")
    if installed_version != EXPECTED_CLEANFID_VERSION:
        raise RuntimeError(
            "clean-fid version mismatch: "
            f"expected {EXPECTED_CLEANFID_VERSION}, got {installed_version}"
        )
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid clean-fid batch/worker setting")

    category_fid = {
        category: _compute(
            args.reference_dir / category,
            args.generated_dir / category,
            args,
        )
        for category in MJHQ_CATEGORIES
    }
    result = {
        "cleanfid_version": installed_version,
        "mode": "clean",
        "model_name": "inception_v3",
        "fid": _compute(args.reference_dir, args.generated_dir, args),
        "category_fid": category_fid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
