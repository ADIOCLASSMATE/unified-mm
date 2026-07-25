#!/usr/bin/env python3
"""Compatibility entry point for the frozen FH screen source manifest.

The frozen evaluator references ``Mapping`` while validating FH checkpoint
provenance, but the corresponding import was omitted.  Keep the evaluator
source byte-for-byte unchanged while screen jobs are active, inject the
missing symbol into its module namespace, and delegate to its normal entry
point.  The permanent import belongs in the evaluator after the frozen runs
finish.
"""

import sys
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts import evaluate_single_stream_fid_is as evaluator


def main() -> None:
    evaluator.Mapping = Mapping
    evaluator.main()


if __name__ == "__main__":
    main()
