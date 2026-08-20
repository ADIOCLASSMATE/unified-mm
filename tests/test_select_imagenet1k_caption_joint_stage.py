import json
from pathlib import Path

import pytest

from scripts.select_imagenet1k_caption_joint_stage import collect_stage


def _write_candidate(
    root: Path,
    run_id: str,
    *,
    text: float,
    image: float,
    ratio: float,
    cosine: float = 0.2,
    negative_fraction: float = 0.0,
    step: int = 2404,
):
    run_root = root / f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}"
    run_root.mkdir(parents=True)
    (run_root / f"validation_metrics_step_{step}.json").write_text(
        json.dumps(
            {
                "schema": "selfless_flow_validation_metrics_v1",
                "global_step": step,
                "metrics": {"val/loss_text": text, "val/loss_image_flow": image},
            }
        )
    )
    probe = run_root / "gradient_probe" / f"checkpoint-{step}" / "probe.json"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        json.dumps(
            {
                "schema": "selfless_caption_t2i_gradient_probe_v1",
                "status": "complete",
                "summary": {
                    "ratio_g_image_over_g_text": {"median": ratio},
                    "cosine": {"median": cosine},
                    "task_conflict": {
                        "persistent_negative": negative_fraction >= 0.5,
                        "negative_fraction": negative_fraction,
                    },
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


def test_stage_selector_penalizes_persistent_negative_cosine(tmp_path: Path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "staged_training": {
                    "stage_boundaries": [2404],
                    "top_k_to_continue": {"2404": 1},
                },
                "candidates": [
                    {"id": "aligned", "backbone_lr": 1e-5, "flow_lr": 2e-5},
                    {"id": "conflict", "backbone_lr": 2e-5, "flow_lr": 4e-5},
                ],
            }
        )
    )
    _write_candidate(
        tmp_path,
        "aligned",
        text=1.0,
        image=0.5,
        ratio=0.2,
        cosine=0.05,
        negative_fraction=0.1,
    )
    _write_candidate(
        tmp_path,
        "conflict",
        text=1.0,
        image=0.5,
        ratio=0.2,
        cosine=-0.2,
        negative_fraction=0.8,
    )

    result = collect_stage(
        sweep, tmp_path, step=2404, lambda_text=0.2, require_complete=True
    )

    assert result["selected"] == ["aligned"]
    assert result["ranking"][1]["task_conflict"]["persistent_negative"] is True


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


def test_later_stage_only_requires_prior_stage_selection(tmp_path: Path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "staged_training": {
                    "stage_boundaries": [2404, 4808],
                    "top_k_to_continue": {"2404": 2, "4808": 1},
                },
                "candidates": [
                    {"id": "a", "backbone_lr": 1e-5, "flow_lr": 2e-5},
                    {"id": "b", "backbone_lr": 2e-5, "flow_lr": 4e-5},
                    {"id": "c", "backbone_lr": 5e-6, "flow_lr": 1e-5},
                ],
            }
        )
    )
    _write_candidate(
        tmp_path, "a", text=2.0, image=0.4, ratio=0.2, step=4808
    )
    _write_candidate(
        tmp_path, "b", text=1.0, image=0.5, ratio=0.2, step=4808
    )
    prior = tmp_path / "stage-2404.json"
    prior.write_text(
        json.dumps(
            {
                "schema": "selfless_caption_t2i_stage_selection_v1",
                "status": "complete",
                "stage_step": 2404,
                "selected": ["a", "b"],
            }
        )
    )

    result = collect_stage(
        sweep,
        tmp_path,
        step=4808,
        lambda_text=0.2,
        require_complete=True,
        candidate_selection_path=prior,
    )

    assert result["status"] == "complete"
    assert result["expected_candidates"] == 2
    assert result["missing"] == []
    assert result["candidate_selection"] == {
        "path": str(prior),
        "stage_step": 2404,
        "selected": ["a", "b"],
    }
    assert {row["id"] for row in result["ranking"]} == {"a", "b"}


def test_later_stage_rejects_unknown_prior_selection_id(tmp_path: Path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "staged_training": {
                    "stage_boundaries": [2404, 4808],
                    "top_k_to_continue": {"2404": 1, "4808": 1},
                },
                "candidates": [
                    {"id": "a", "backbone_lr": 1e-5, "flow_lr": 2e-5}
                ],
            }
        )
    )
    prior = tmp_path / "stage-2404.json"
    prior.write_text(
        json.dumps(
            {
                "schema": "selfless_caption_t2i_stage_selection_v1",
                "status": "complete",
                "stage_step": 2404,
                "selected": ["missing"],
            }
        )
    )

    with pytest.raises(ValueError, match="absent from sweep"):
        collect_stage(
            sweep,
            tmp_path,
            step=4808,
            lambda_text=0.2,
            require_complete=True,
            candidate_selection_path=prior,
        )
