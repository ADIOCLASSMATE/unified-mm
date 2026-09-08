import json

import pytest

from scripts.build_evaluation_report import (
    check_checkpoint,
    gallery_data,
    model_metrics,
    within,
)


def test_invalidated_result_is_never_used_for_a_score(tmp_path):
    run = tmp_path / "d/checkpoints/r1"
    run.mkdir(parents=True)
    (run / "evaluation_invalidation.json").write_text('{"invalidated": true}')
    with pytest.raises(ValueError, match="invalidated"):
        model_metrics(tmp_path, {"id": "d_on_b"}, {"root": "d/checkpoints/r1"})


def test_absent_formal_scores_stay_empty_instead_of_zero(tmp_path):
    result = model_metrics(tmp_path, {"id": "text_only"}, {})
    assert result["metrics"] == {}
    assert result["complete"] is False


def test_checkpoint_identity_requires_explicit_evidence_for_a_rename(tmp_path):
    spec = {"id": "b_no_flowdiag", "checkpoint": "/training/b-no-diagonal/hf_model-final-ema", "run": "b-no-diagonal", "source": {"global_step": 95415}}
    source = {"checkpoint": "/training/b/hf_model-final-ema", "checkpoint_step": 95415}
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        check_checkpoint(source, spec, {}, tmp_path)
    evidence = {"original_run_project": "b", "current_output_root": "output/b-no-diagonal", "control_name": "flow_head_no_diagonal"}
    (tmp_path / "identity.json").write_text(json.dumps(evidence))
    selection = {"historical_checkpoint_alias": "b", "alias_evidence": "identity.json"}
    check_checkpoint(source, spec, selection, tmp_path)
    with pytest.raises(ValueError, match="step mismatch"):
        check_checkpoint({**source, "checkpoint_step": 60000}, spec, selection, tmp_path)


def test_gallery_rejects_missing_image_and_wrong_sample_coverage(tmp_path):
    gallery = tmp_path / "qualitative/test"
    gallery.mkdir(parents=True)
    manifest = {"models": [{"id": "b"}], "samples": {"t2i": [{"id": "s1"}], "i2t": [], "text": []}, "expected_per_model": {"t2i": 2, "i2t": 0, "text": 0}}
    (gallery / "manifest.json").write_text(json.dumps(manifest))
    (gallery / "summary.json").write_text('{"complete":true}')
    rows = [{"task": "t2i", "model": "b", "sample_id": "s1", "seed": seed, "image": "sample.png"} for seed in (42, 43)]
    result = gallery / "results.jsonl"
    result.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="missing gallery image"):
        gallery_data(tmp_path, {"qualitative": "qualitative/test"})
    (gallery / "sample.png").write_bytes(b"image presence checked here; decoding is audited separately")
    gallery_data(tmp_path, {"qualitative": "qualitative/test"})
    result.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="coverage mismatch"):
        gallery_data(tmp_path, {"qualitative": "qualitative/test"})


def test_result_paths_cannot_escape_the_evaluation_directory(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        within(tmp_path, "../training")
