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


def test_joint_sweep_ranker_balances_final_and_best_text_image_ranks(tmp_path):
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
    assert report["winner"] is None
    assert report["validation_leader"] == "image"
    assert report["top_k"] == ["image", "text"]
    image = report["ranking"][0]
    assert image["final_text_rank"] == 3.0
    assert image["best_text_rank"] == 1.5
    assert image["final_image_rank"] == 1.0
    assert image["best_image_rank"] == 1.0
    assert image["mean_rank"] == 1.625


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


def test_joint_sweep_ranker_only_requires_staged_selected_candidates(tmp_path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {
                    "validation_steps": [10, 20],
                    "rule": "test",
                    "top_k_for_generation_evaluation": 3,
                },
                "candidates": [
                    {"id": "keep", "backbone_lr": 1e-5, "flow_lr": 2e-5},
                    {"id": "stopped", "backbone_lr": 2e-5, "flow_lr": 4e-5},
                ],
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "stage-4808.json"
    selection.write_text(json.dumps({"selected": ["keep"]}), encoding="utf-8")
    _write_metrics(tmp_path, "keep", 10, 1.2, 0.5)
    _write_metrics(tmp_path, "keep", 20, 1.0, 0.4)
    _write_metrics(tmp_path, "stopped", 10, 0.5, 0.2)

    report = collect(
        sweep,
        tmp_path,
        require_complete=True,
        candidate_selection_path=selection,
    )

    assert report["status"] == "complete"
    assert report["expected_candidates"] == 1
    assert report["winner"] is None
    assert report["validation_leader"] == "keep"


def test_joint_sweep_ranker_supports_lambda_project_names(tmp_path):
    project = "selfless-flow-imagenet1k-caption-joint-lambda-b1e5-f2e5-lt0p1"
    sweep = tmp_path / "lambda.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {
                    "validation_steps": [10, 20],
                    "rule": "test",
                    "top_k_for_generation_evaluation": 1,
                },
                "candidates": [
                    {
                        "id": "b1e5-f2e5-lt0p1",
                        "run_project": project,
                        "backbone_lr": 1e-5,
                        "flow_lr": 2e-5,
                        "lambda_text": 0.1,
                        "evaluation_model_subdir": "hf_model-20-ema-eval",
                        "generation_evaluation_subdir": (
                            "generation-evaluation/lambda-step-20"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / project
    run_root.mkdir()
    for step, text, image in ((10, 2.0, 0.6), (20, 1.5, 0.5)):
        (run_root / f"validation_metrics_step_{step}.json").write_text(
            json.dumps(
                {
                    "schema": "selfless_flow_validation_metrics_v1",
                    "global_step": step,
                    "metrics": {
                        "val/loss_text": text,
                        "val/loss_image_flow": image,
                    },
                }
            ),
            encoding="utf-8",
        )

    report = collect(sweep, tmp_path, require_complete=True)

    assert report["ranking"][0]["run_project"] == project
    assert report["ranking"][0]["lambda_text"] == 0.1
    assert report["ranking"][0]["evaluation_model_subdir"] == (
        "hf_model-20-ema-eval"
    )
    assert report["ranking"][0]["generation_evaluation_subdir"] == (
        "generation-evaluation/lambda-step-20"
    )


def test_joint_sweep_ranker_requires_final_gradient_probe_when_configured(tmp_path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "selection": {
                    "validation_steps": [10],
                    "rule": "test",
                    "require_gradient_probe": True,
                    "top_k_for_generation_evaluation": 1,
                },
                "candidates": [
                    {"id": "probe", "backbone_lr": 1e-5, "flow_lr": 2e-5}
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_metrics(tmp_path, "probe", 10, 1.0, 0.5)

    incomplete = collect(sweep, tmp_path, require_complete=False)
    assert "missing_gradient_probe" in incomplete["missing"][0]

    probe_path = (
        tmp_path
        / "selfless-flow-imagenet1k-caption-joint-sweep-probe"
        / "gradient_probe"
        / "checkpoint-10"
        / "probe.json"
    )
    probe_path.parent.mkdir(parents=True)
    probe_path.write_text(
        json.dumps(
            {
                "schema": "selfless_caption_t2i_gradient_probe_v1",
                "status": "complete",
                "summary": {
                    "ratio_g_image_over_g_text": {"median": 0.1},
                    "cosine": {"median": -0.2},
                    "task_conflict": {"persistent_negative": True},
                },
            }
        ),
        encoding="utf-8",
    )

    complete = collect(sweep, tmp_path, require_complete=True)
    assert complete["status"] == "complete"
    assert complete["ranking"][0]["gradient_probe"]["cosine_median"] == -0.2
