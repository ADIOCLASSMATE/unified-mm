import json

from scripts.build_imagenet1k_caption_joint_lambda_sweep import build_manifest


def test_lambda_manifest_uses_probe_candidates_and_one_top_lr(tmp_path):
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(
        json.dumps(
            {
                "schema": "selfless_caption_t2i_gradient_probe_v1",
                "status": "complete",
                "checkpoint": {"path": "fixed", "model_sha256": "abc"},
                "batches": [{}] * 16,
                "summary": {
                    "ratio_g_image_over_g_text": {"median": 0.1},
                    "cosine": {"median": -0.1},
                    "task_conflict": {"persistent_negative": True},
                    "lambda_text": {
                        "raw_center": 0.1,
                        "center": 0.1,
                        "bounds": [0.025, 0.4],
                        "scaled": {"0.5x": 0.05, "1.0x": 0.1, "2.0x": 0.2},
                        "reference_points": [0.05, 0.1, 0.2],
                        "candidates": [0.05, 0.1, 0.2],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    ranking_path = tmp_path / "ranking.json"
    ranking_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "validation_leader": "b1e5-f2e5",
                "ranking": [
                    {
                        "id": "b1e5-f2e5",
                        "checkpoints": [{"step": 2404}, {"step": 4808}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    lr_sweep_path = tmp_path / "lr.json"
    lr_sweep_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "b1e5-f2e5",
                        "backbone_lr": 1e-5,
                        "flow_lr": 2e-5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(probe_path, ranking_path, lr_sweep_path)

    assert [row["lambda_text"] for row in manifest["candidates"]] == [
        0.05,
        0.1,
        0.2,
    ]
    assert all(
        row["top_lr_id"] == "b1e5-f2e5" for row in manifest["candidates"]
    )
    by_lambda = {row["lambda_text"]: row for row in manifest["candidates"]}
    assert by_lambda[0.2]["run_project"] == (
        "selfless-flow-imagenet1k-caption-joint-sweep-b1e5-f2e5"
    )
    assert by_lambda[0.2]["source"] == "lr_sweep_baseline"
    assert by_lambda[0.2]["reuse_existing_run"] is True
    assert by_lambda[0.1]["run_project"] == (
        "selfless-flow-imagenet1k-caption-joint-lambda-b1e5-f2e5-lt0p1"
    )
    assert by_lambda[0.1]["source"] == "lambda_sweep"
    assert by_lambda[0.1]["reuse_existing_run"] is False
    assert all(
        row["evaluation_model_subdir"] == "hf_model-4808-ema-eval"
        for row in manifest["candidates"]
    )
    assert all(
        row["generation_evaluation_subdir"]
        == "generation-evaluation/lambda-step-4808"
        for row in manifest["candidates"]
    )
    assert manifest["lambda_policy"]["cartesian_product_with_lr"] is False
    assert manifest["source_probe"]["task_conflict"]["persistent_negative"] is True
