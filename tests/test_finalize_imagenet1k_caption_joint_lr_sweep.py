import json

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
                "strategies": {
                    "spatial_halton": {
                        "fid": fid,
                        "inception_score_mean": inception,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


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
    )

    assert report["status"] == "complete"
    assert report["winner"] == "balanced"
    assert report["ranking"][0]["generation_mean_rank"] == 5.0 / 3.0


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
    )

    assert report["status"] == "incomplete"
    assert report["winner"] is None
    assert report["missing"][0]["id"] == "candidate"
