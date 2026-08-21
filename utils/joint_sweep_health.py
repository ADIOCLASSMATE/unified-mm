"""Read-only training-log health checks shared by staged sweep selectors."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ERROR_PATTERN = re.compile(
    r"Traceback|RuntimeError|OutOfMemory|\bOOM\b|non-finite|\bNaN\b",
    re.IGNORECASE,
)
GRAD_PATTERN = re.compile(
    r"GradientNorm: Step: (\d+) \| PreClip: ([0-9.eE+-]+)"
)
STEP_PATTERN = re.compile(r"\[RANK 0\] Step: (\d+)")


def _expected_gradient_cadence(run_root: Path) -> int | None:
    config_path = run_root / "config.yaml"
    if not config_path.is_file():
        return None
    try:
        value = int(OmegaConf.load(config_path).experiment.log_grad_norm_every)
    except (AttributeError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def training_health(
    run_root: Path,
    *,
    abnormal_pre_clip_norm: float = 100.0,
    max_step: int | None = None,
) -> dict[str, Any]:
    if max_step is not None and max_step < 0:
        raise ValueError("max_step must be non-negative")
    errors: list[str] = []
    grad_norms: list[dict[str, float | int]] = []
    latest_step = 0
    paths = sorted((run_root / "prelaunch_audit").glob("**/training.log"))
    scanned_paths: list[Path] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        path_steps = [
            int(match.group(1))
            for line in lines
            for pattern in (STEP_PATTERN, GRAD_PATTERN)
            if (match := pattern.search(line)) is not None
        ]
        # A later continuation log is outside a shorter candidate's sample
        # budget. This matters when a completed long run is reused at an
        # earlier sweep gate.
        if (
            max_step is not None
            and path_steps
            and min(path_steps) > max_step
        ):
            continue
        scanned_paths.append(path)
        current_step: int | None = None
        for line in lines:
            step_match = STEP_PATTERN.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                if max_step is None or current_step <= max_step:
                    latest_step = max(latest_step, current_step)
            if ERROR_PATTERN.search(line) and (
                max_step is None
                or current_step is None
                or current_step <= max_step
            ):
                errors.append(line[-500:])
            match = GRAD_PATTERN.search(line)
            if match:
                gradient_step = int(match.group(1))
                current_step = gradient_step
                if max_step is not None and gradient_step > max_step:
                    continue
                latest_step = max(latest_step, gradient_step)
                grad_norms.append(
                    {
                        "step": gradient_step,
                        "pre_clip": float(match.group(2)),
                    }
                )
    abnormal = [
        row
        for row in grad_norms
        if float(row["pre_clip"]) > float(abnormal_pre_clip_norm)
    ]
    expected_cadence = _expected_gradient_cadence(run_root)
    if grad_norms:
        gradient_log_status = "available"
    elif expected_cadence is not None and latest_step < expected_cadence:
        gradient_log_status = "pending_first_sample"
    else:
        gradient_log_status = "unavailable_or_legacy_cadence_bug"
    return {
        "training_logs": [str(path) for path in scanned_paths],
        "max_step": max_step,
        "error_lines": errors[:20],
        "pre_clip_grad_norms": grad_norms,
        "latest_logged_step": latest_step,
        "expected_gradient_cadence": expected_cadence,
        "gradient_log_status": gradient_log_status,
        "abnormal_pre_clip_norm_threshold": float(abnormal_pre_clip_norm),
        "abnormal_grad_norms": abnormal,
        "passed": not errors and not abnormal,
    }
