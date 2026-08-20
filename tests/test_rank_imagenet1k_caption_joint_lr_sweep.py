import json

from scripts.rank_imagenet1k_caption_joint_lr_sweep import collect


def _write_metrics(root, run_id, step, text_loss, image_loss):
    run_root = root / f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / f"validation_metrics_step_{step}.json").write_text(
        json.dumps(
            {
                "schema": "selfless_flow_validation_metrics_v1",
                "global_step": step,
                "metrics": {
                    "val/loss_text": text_loss,
                    "val/loss_image_flow": image_loss,
                },
            }
        ),
        encoding="utf-8",
    )


def test_joint_sweep_ranker_balances_final_text_and_image_ranks(tmp_path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {
                    "validation_steps": [10, 20],
                    "rule": "mean final rank with regression tie-break",
                    "top_k_for_generation_evaluation": 2,
                },
                "candidates": [
                    {"id": "text", "backbone_lr": 1e-5, "flow_lr": 1e-5},
                    {"id": "balanced", "backbone_lr": 2e-5, "flow_lr": 2e-5},
                    {"id": "image", "backbone_lr": 3e-5, "flow_lr": 3e-5},
                ],
            }
        ),
        encoding="utf-8",
    )
    values = {
        "text": [(1.2, 0.45), (1.0, 0.5)],
        "balanced": [(1.1, 0.4), (1.1, 0.4)],
        "image": [(1.0, 0.3), (1.2, 0.3)],
    }
    for run_id, checkpoints in values.items():
        for step, (text_loss, image_loss) in zip((10, 20), checkpoints):
            _write_metrics(tmp_path, run_id, step, text_loss, image_loss)

    report = collect(sweep, tmp_path, require_complete=True)

    assert report["status"] == "complete"
    assert report["winner"] == "balanced"
    assert report["top_k"] == ["balanced", "text"]
    balanced = report["ranking"][0]
    assert balanced["text_rank"] == 2.0
    assert balanced["image_rank"] == 2.0
    assert balanced["mean_rank"] == 2.0


def test_joint_sweep_ranker_reports_missing_candidates(tmp_path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {
                    "validation_steps": [10],
                    "rule": "test",
                    "top_k_for_generation_evaluation": 1,
                },
                "candidates": [
                    {"id": "missing", "backbone_lr": 1e-5, "flow_lr": 1e-5}
                ],
            }
        ),
        encoding="utf-8",
    )

    report = collect(sweep, tmp_path, require_complete=False)

    assert report["status"] == "incomplete"
    assert report["winner"] is None
    assert report["missing"][0]["missing_validation_steps"] == [10]
