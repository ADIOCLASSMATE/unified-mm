from omegaconf import OmegaConf

from utils.joint_sweep_health import training_health


def _run(tmp_path, *, step: int, grad_line: str = ""):
    root = tmp_path / f"run-{step}"
    log = root / "prelaunch_audit" / "stage-2404" / "training.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        f"[RANK 0] Step: {step} | Loss: 1.0\n{grad_line}",
        encoding="utf-8",
    )
    OmegaConf.save(
        OmegaConf.create({"experiment": {"log_grad_norm_every": 1202}}),
        root / "config.yaml",
    )
    return training_health(root)


def test_health_reports_gradient_sample_pending_before_cadence(tmp_path):
    health = _run(tmp_path, step=1200)
    assert health["gradient_log_status"] == "pending_first_sample"
    assert health["latest_logged_step"] == 1200


def test_health_reports_missing_sample_at_or_after_cadence(tmp_path):
    health = _run(tmp_path, step=1250)
    assert health["gradient_log_status"] == "unavailable_or_legacy_cadence_bug"


def test_health_reports_available_preclip_norm(tmp_path):
    health = _run(
        tmp_path,
        step=1250,
        grad_line="GradientNorm: Step: 1202 | PreClip: 2.5 | MaxNorm: 1.0",
    )
    assert health["gradient_log_status"] == "available"
    assert health["pre_clip_grad_norms"] == [{"step": 1202, "pre_clip": 2.5}]
    assert health["latest_logged_step"] == 1250


def test_gradient_event_advances_latest_step_between_loss_logs(tmp_path):
    health = _run(
        tmp_path,
        step=1200,
        grad_line="GradientNorm: Step: 1202 | PreClip: 0.9 | MaxNorm: 1.0",
    )
    assert health["latest_logged_step"] == 1202


def test_health_max_step_excludes_later_continuation_events(tmp_path):
    root = tmp_path / "reused-long-run"
    early_log = root / "prelaunch_audit" / "stage-2404" / "training.log"
    early_log.parent.mkdir(parents=True)
    early_log.write_text(
        "\n".join(
            [
                "[RANK 0] Step: 1202 | Loss: 1.0",
                "GradientNorm: Step: 1202 | PreClip: 0.5 | MaxNorm: 1.0",
                "[RANK 0] Step: 2404 | Loss: 0.9",
                "GradientNorm: Step: 2404 | PreClip: 0.4 | MaxNorm: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    later_log = root / "prelaunch_audit" / "stage-12020" / "training.log"
    later_log.parent.mkdir(parents=True)
    later_log.write_text(
        "\n".join(
            [
                "[RANK 0] Step: 2450 | Loss: 0.8",
                "GradientNorm: Step: 3606 | PreClip: 250.0 | MaxNorm: 1.0",
                "RuntimeError: failure after the selected sample budget",
            ]
        ),
        encoding="utf-8",
    )
    OmegaConf.save(
        OmegaConf.create({"experiment": {"log_grad_norm_every": 1202}}),
        root / "config.yaml",
    )

    health = training_health(root, max_step=2404)

    assert health["passed"] is True
    assert health["max_step"] == 2404
    assert health["latest_logged_step"] == 2404
    assert health["pre_clip_grad_norms"] == [
        {"step": 1202, "pre_clip": 0.5},
        {"step": 2404, "pre_clip": 0.4},
    ]
    assert health["error_lines"] == []
    assert health["training_logs"] == [str(early_log)]
