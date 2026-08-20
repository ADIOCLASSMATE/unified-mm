"""Read-only training-log health checks shared by staged sweep selectors."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


ERROR_PATTERN = re.compile(
    r"Traceback|RuntimeError|OutOfMemory|\bOOM\b|non-finite|\bNaN\b",
    re.IGNORECASE,
)
GRAD_PATTERN = re.compile(
    r"GradientNorm: Step: (\d+) \| PreClip: ([0-9.eE+-]+)"
)


def training_health(
    run_root: Path,
    *,
    abnormal_pre_clip_norm: float = 100.0,
) -> dict[str, Any]:
    errors: list[str] = []
    grad_norms: list[dict[str, float | int]] = []
    paths = sorted((run_root / "prelaunch_audit").glob("**/training.log"))
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if ERROR_PATTERN.search(line):
                errors.append(line[-500:])
            match = GRAD_PATTERN.search(line)
            if match:
                grad_norms.append(
                    {
                        "step": int(match.group(1)),
                        "pre_clip": float(match.group(2)),
                    }
                )
    abnormal = [
        row
        for row in grad_norms
        if float(row["pre_clip"]) > float(abnormal_pre_clip_norm)
    ]
    return {
        "training_logs": [str(path) for path in paths],
        "error_lines": errors[:20],
        "pre_clip_grad_norms": grad_norms,
        "gradient_log_status": (
            "available" if grad_norms else "unavailable_or_legacy_cadence_bug"
        ),
        "abnormal_pre_clip_norm_threshold": float(abnormal_pre_clip_norm),
        "abnormal_grad_norms": abnormal,
        "passed": not errors and not abnormal,
    }
