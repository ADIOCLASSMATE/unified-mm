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
