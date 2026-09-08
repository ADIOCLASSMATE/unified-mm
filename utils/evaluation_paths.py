"""Canonical result paths; training weights and state stay in their run directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = REPO_ROOT / "output/evaluation"


def training_validation_root(run_root: str | Path) -> Path:
    run_root = Path(run_root).resolve()
    try:
        identity = run_root.relative_to(REPO_ROOT / "output")
    except ValueError:
        identity = Path(run_root.name)
    return EVALUATION_ROOT / "training-validation" / identity


def validation_output_dir(config) -> Path:
    """Standalone evaluators keep their explicitly selected output directory."""
    return Path(config.experiment.get("validation_output_dir", config.experiment.output_dir))


def training_validation_file(run_root: str | Path, filename: str) -> Path:
    """Read relocated results, with fallback for older, unmigrated experiments."""
    current = training_validation_root(run_root) / filename
    return current if current.exists() else Path(run_root) / filename
