import json
from pathlib import Path

import pytest

from scripts.select_imagenet1k_caption_joint_stage import collect_stage


def _write_candidate(root: Path, run_id: str, *, text: float, image: float, ratio: float):
    run_root = root / f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}"
    run_root.mkdir(parents=True)
    (run_root / "validation_metrics_step_2404.json").write_text(
        json.dumps(
            {
                "schema": "selfless_flow_validation_metrics_v1",
                "global_step": 2404,
                "metrics": {"val/loss_text": text, "val/loss_image_flow": image},
            }
        )
    )
    probe = run_root / "gradient_probe" / "checkpoint-2404" / "probe.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        json.dumps(
            {
                "schema": "selfless_caption_t2i_gradient_probe_v1",
                "status": "complete",
                "summary": {
                    "ratio_g_image_over_g_text": {"median": ratio},
                    "cosine": {"median": 0.2},
                    "task_conflict": {"persistent_negative": False},
                },
            }
        )
    )


def test_stage_selector_uses_both_losses_and_gradient_balance(tmp_path: Path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "staged_training": {
                    "stage_boundaries": [2404],
                    "top_k_to_continue": {"2404": 2},
                },
                "candidates": [
                    {"id": "a", "backbone_lr": 1e-5, "flow_lr": 2e-5},
                    {"id": "b", "backbone_lr": 2e-5, "flow_lr": 4e-5},
                    {"id": "c", "backbone_lr": 5e-6, "flow_lr": 1e-5},
                ],
            }
        )
    )
    _write_candidate(tmp_path, "a", text=2.0, image=0.4, ratio=0.2)
    _write_candidate(tmp_path, "b", text=1.0, image=0.5, ratio=0.2)
    _write_candidate(tmp_path, "c", text=3.0, image=0.3, ratio=0.01)
    result = collect_stage(
        sweep, tmp_path, step=2404, lambda_text=0.2, require_complete=True
    )
    assert result["status"] == "complete"
    assert result["selected"] == ["a", "b"]


def test_stage_selector_reports_missing_probe(tmp_path: Path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "staged_training": {
                    "stage_boundaries": [2404],
                    "top_k_to_continue": {"2404": 1},
                },
                "candidates": [
                    {"id": "a", "backbone_lr": 1e-5, "flow_lr": 2e-5}
                ],
            }
        )
    )
    result = collect_stage(
        sweep, tmp_path, step=2404, lambda_text=0.2, require_complete=False
    )
    assert result["status"] == "incomplete"
    with pytest.raises(FileNotFoundError):
        collect_stage(
            sweep, tmp_path, step=2404, lambda_text=0.2, require_complete=True
        )
