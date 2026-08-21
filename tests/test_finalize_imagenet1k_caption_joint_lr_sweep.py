import json

import pytest

from scripts.finalize_imagenet1k_caption_joint_lr_sweep import collect_final


def _write_generation_metrics(root, run_id, *, clip, fid, inception):
    run_root = root / f"selfless-flow-imagenet1k-caption-joint-sweep-{run_id}"
    caption = run_root / "generation-evaluation/i2t-clip/metrics.json"
    image = run_root / "generation-evaluation/t2i-fid-is/metrics.json"
    caption.parent.mkdir(parents=True, exist_ok=True)
    image.parent.mkdir(parents=True, exist_ok=True)
    caption.write_text(
        json.dumps(
            {
                "schema": "selfless_imagenet1k_i2t_clip_metrics_v1",
                "samples": 1000,
                "class_balance": {
                    "class_count": 1000,
                    "min_samples_per_class": 1,
                    "max_samples_per_class": 1,
                },
                "generation": {"seed": 424242},
                "clip": {"caption_clip_score": clip},
            }
        ),
        encoding="utf-8",
    )
    image.write_text(
        json.dumps(
            {
                "official_protocol": True,
                "samples_evaluated": 50000,
                "seed": 42,
                "sampling_steps": "100",
                "strategies": {
                    "spatial_halton": {
                        "count": 50000,
                        "fid": fid,
                        "inception_score_mean": inception,
                        "inception_score_std": 3.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_initialization_metrics(root):
    path = root / "initialization-metrics.json"
    path.write_text(
        json.dumps(
            {
                "official_protocol": True,
                "samples_evaluated": 50000,
                "seed": 42,
                "sampling_steps": 100,
                "strategies": {
                    "spatial_halton": {
                        "count": 50000,
                        "fid": 25.0,
                        "inception_score_mean": 150.0,
                        "inception_score_std": 4.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_finalizer_balances_caption_fid_and_inception_ranks(tmp_path):
    validation_path = tmp_path / "validation-ranking.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema": "selfless_imagenet1k_caption_joint_lr_ranking_v1",
                "status": "complete",
                "winner": "caption",
                "top_k": ["caption", "balanced", "image"],
                "ranking": [
                    {
                        "id": "caption",
                        "backbone_lr": 1e-5,
                        "flow_lr": 1e-5,
                        "overall_rank": 1,
                    },
                    {
                        "id": "balanced",
                        "backbone_lr": 2e-5,
                        "flow_lr": 2e-5,
                        "overall_rank": 2,
                    },
                    {
                        "id": "image",
                        "backbone_lr": 3e-5,
                        "flow_lr": 3e-5,
                        "overall_rank": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_generation_metrics(
        tmp_path, "caption", clip=0.40, fid=30.0, inception=100.0
    )
    _write_generation_metrics(
        tmp_path, "balanced", clip=0.35, fid=20.0, inception=200.0
    )
    _write_generation_metrics(
        tmp_path, "image", clip=0.30, fid=10.0, inception=150.0
    )

    report = collect_final(
        validation_path,
        tmp_path,
        require_complete=True,
        initialization_metrics_path=_write_initialization_metrics(tmp_path),
    )

    assert report["status"] == "complete"
    assert report["winner"] == "balanced"
    assert report["ranking"][0]["generation_mean_rank"] == 4.0 / 3.0
    assert report["excluded"][0]["id"] == "caption"
    assert report["ranking"][0]["t2i_vs_initialization"] == {
        "fid_delta": -5.0,
        "fid_status": "improved",
        "inception_score_delta": 50.0,
        "inception_score_status": "improved",
        "joint_status": "improved_both",
    }


def test_finalizer_reports_missing_generation_metrics(tmp_path):
    validation_path = tmp_path / "validation-ranking.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema": "selfless_imagenet1k_caption_joint_lr_ranking_v1",
                "status": "complete",
                "winner": "candidate",
                "top_k": ["candidate"],
                "ranking": [
                    {
                        "id": "candidate",
                        "backbone_lr": 1e-5,
                        "flow_lr": 2e-5,
                        "overall_rank": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = collect_final(
        validation_path,
        tmp_path,
        require_complete=False,
        initialization_metrics_path=_write_initialization_metrics(tmp_path),
    )

    assert report["status"] == "incomplete"
    assert report["winner"] is None
    assert report["missing"][0]["id"] == "candidate"


def test_finalizer_uses_candidate_generation_evaluation_subdir(tmp_path):
    validation_path = tmp_path / "validation-ranking.json"
    project = "selfless-flow-imagenet1k-caption-joint-sweep-b1e5-f2e5"
    evaluation_subdir = "generation-evaluation/lambda-step-4808"
    validation_path.write_text(
        json.dumps(
            {
                "schema": "selfless_imagenet1k_caption_joint_lr_ranking_v1",
                "status": "complete",
                "validation_leader": "b1e5-f2e5-lt0p2",
                "top_k": ["b1e5-f2e5-lt0p2"],
                "ranking": [
                    {
                        "id": "b1e5-f2e5-lt0p2",
                        "run_project": project,
                        "evaluation_model_subdir": "hf_model-4808-ema-eval",
                        "generation_evaluation_subdir": evaluation_subdir,
                        "lambda_text": 0.2,
                        "backbone_lr": 1e-5,
                        "flow_lr": 2e-5,
                        "overall_rank": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / project / evaluation_subdir
    expected_model_path = str(
        tmp_path / project / "hf_model-4808-ema-eval"
    )
    caption_path = run_root / "i2t-clip/metrics.json"
    image_path = run_root / "t2i-fid-is/metrics.json"
    caption_path.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    caption_path.write_text(
        json.dumps(
            {
                "schema": "selfless_imagenet1k_i2t_clip_metrics_v1",
                "samples": 1000,
                "model": {
                    "path": expected_model_path,
                    "weights_sha256": "model-sha",
                },
                "class_balance": {
                    "class_count": 1000,
                    "min_samples_per_class": 1,
                    "max_samples_per_class": 1,
                },
                "generation": {"seed": 424242},
                "split": {
                    "strategy": "stratified",
                    "seed": 42,
                    "val_samples_per_class": 50,
                    "validation_overlap_train": False,
                },
                "clip": {"caption_clip_score": 0.35},
            }
        ),
        encoding="utf-8",
    )
    image_path.write_text(
        json.dumps(
            {
                "official_protocol": True,
                "samples_evaluated": 50000,
                "seed": 42,
                "sampling_steps": "100",
                "model_path": expected_model_path,
                "strategies": {
                    "spatial_halton": {
                        "count": 50000,
                        "fid": 20.0,
                        "inception_score_mean": 200.0,
                        "inception_score_std": 3.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assets_path = run_root / "prelaunch_audit/assets.json"
    assets_path.parent.mkdir(parents=True)
    assets_path.write_text(
        json.dumps(
            {
                "model": {
                    "path": expected_model_path,
                    "weights_sha256": "model-sha",
                }
            }
        ),
        encoding="utf-8",
    )

    report = collect_final(
        validation_path,
        tmp_path,
        require_complete=True,
        initialization_metrics_path=_write_initialization_metrics(tmp_path),
    )

    assert report["status"] == "complete"
    assert report["winner"] == "b1e5-f2e5-lt0p2"
    assert report["ranking"][0]["generation_evaluation_subdir"] == evaluation_subdir
    assert report["ranking"][0]["evaluation_model_subdir"] == (
        "hf_model-4808-ema-eval"
    )
    assert report["ranking"][0]["lambda_text"] == 0.2


def test_finalizer_rejects_t2i_protocol_drift_from_initialization(tmp_path):
    validation_path = tmp_path / "validation-ranking.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema": "selfless_imagenet1k_caption_joint_lr_ranking_v1",
                "status": "complete",
                "top_k": ["candidate"],
                "ranking": [
                    {
                        "id": "candidate",
                        "backbone_lr": 1e-5,
                        "flow_lr": 2e-5,
                        "overall_rank": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_generation_metrics(
        tmp_path, "candidate", clip=0.3, fid=20.0, inception=175.0
    )
    image_path = (
        tmp_path
        / "selfless-flow-imagenet1k-caption-joint-sweep-candidate"
        / "generation-evaluation/t2i-fid-is/metrics.json"
    )
    image = json.loads(image_path.read_text(encoding="utf-8"))
    image["seed"] = 43
    image_path.write_text(json.dumps(image), encoding="utf-8")

    with pytest.raises(ValueError, match="protocol mismatch.*seed"):
        collect_final(
            validation_path,
            tmp_path,
            require_complete=True,
            initialization_metrics_path=_write_initialization_metrics(tmp_path),
        )
