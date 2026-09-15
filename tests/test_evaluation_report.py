import json

import pytest

from scripts.build_evaluation_report import (
    check_checkpoint,
    flow_head_scale_data,
    gallery_data,
    model_metrics,
    sampling_sweep_data,
    unified_training_ablation_data,
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


def test_selected_model_can_precede_paired_gallery(tmp_path, monkeypatch):
    from scripts import build_evaluation_report as report
    monkeypatch.setattr(report, "REPO", tmp_path)
    run = "unified-s2-single-0p6b-100b-imagenet-split-s42-r1"
    checkpoint = tmp_path / "output" / run / "hf_model-final-ema"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({
        "architecture_variant": "showo2_unified", "dual_stream_attention_contract": "showo2_omni_attention"}))
    (checkpoint / "ema_export_metadata.json").write_text('{"source_global_step":95415}')
    root = tmp_path / "output/evaluation"
    directory = root / "s2/final"
    directory.mkdir(parents=True)
    (directory / "native_full_evaluation_summary.json").write_text(json.dumps({
        "complete": True, "model_source": {"kind": "hf_final_ema", "floating_dtype": "float32",
        "path": str(checkpoint), "global_step": 95415}}))
    selected = {"run": run, "root": "s2/final", "checkpoint_step": 95415}
    selection = {"models": {"s2_single": selected}}
    specs = report.report_model_specs(root, selection, [])
    assert [(m["id"], m["qualitative_available"]) for m in specs] == [("s2_single", False)]
    assert specs[0]["architecture"] == "showo2_unified"
    selected["checkpoint_step"] = 60000
    with pytest.raises(ValueError, match="step mismatch"):
        report.report_model_specs(root, selection, [])


@pytest.mark.parametrize("s2,mc,accepted", [(True, 1, True), (True, 64, False), (False, 1, False), (False, 64, True)])
def test_report_enforces_model_specific_benchmark_scoring(tmp_path, s2, mc, accepted):
    from utils.evaluation.model_contracts import S2_ATTENTION, S2_SCORING
    directory = tmp_path / "run/pretraining-native-understanding/retained-benchmarks"
    directory.mkdir(parents=True)
    attention = S2_ATTENTION if s2 else "xlnet_content_diagonal"
    manifest = {"checkpoint": "/training/model/hf_model-final-ema", "checkpoint_step": 95415,
        "schema": "selfless_multimodal_likelihood_evaluation_v5", "project_formal_protocol": True,
        "dual_stream_attention_contract": attention, "mc_samples": mc,
        "scoring_contract": S2_SCORING if s2 else "selfless",
        "image_order_mc_contract": "not_applicable_full_image_ar" if s2 else "random"}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    scoring = {**manifest, "contract": manifest["scoring_contract"]}
    (directory / "summary.json").write_text(json.dumps({"scoring": scoring}))
    spec = {"id": "test_model", "checkpoint": manifest["checkpoint"], "source": {"global_step": 95415},
            "backbone_attention": attention, "image_order": "random"}
    if accepted:
        assert model_metrics(tmp_path, spec, {"root": "run"})["metrics"] == {}
    else:
        with pytest.raises(ValueError, match="mc_samples|MC count"):
            model_metrics(tmp_path, spec, {"root": "run"})


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


def test_unfinished_sampling_sweep_never_announces_an_optimum(tmp_path):
    sweep = tmp_path / "sampling"
    sweep.mkdir()
    summary = {"status": "running", "phase": "cfg", "results": [],
               "protocol": {"model_source": "/training/b/hf_model-final-ema", "checkpoint_step": 95415}}
    (sweep / "summary.json").write_text(json.dumps(summary))
    models = [{"id": "b_x0", "checkpoint": "/training/b/hf_model-final-ema", "source": {"global_step": 95415}}]
    selection = {"sampling_sweeps": [{"model": "b_x0", "label": "B sweep", "root": "sampling"}]}
    result = sampling_sweep_data(tmp_path, selection, models)
    assert result[0]["conclusion"] is None
    assert result[0]["completed"] == 0
    summary["protocol"]["checkpoint_step"] = 60000
    (sweep / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="sweep checkpoint mismatch"):
        sampling_sweep_data(tmp_path, selection, models)


@pytest.fixture
def scale_case(tmp_path):
    from scripts.build_evaluation_report import REPO

    b = {"id": "scale_test_b", "run": "scale-test-b", "label": "B", "family": "b",
         "depth": 16, "width": 1280, "architecture": "contextual_dual_stream",
         "head_parameters": 321543696, "total_parameters": 918660624,
         "training_status": "complete", "metrics": {"2.0": "b/metrics.json"}}
    f = {**b, "id": "scale_test_f", "run": "scale-test-f", "label": "F", "family": "f",
         "architecture": "positionwise_adaln_mlp", "width": 1960,
         "head_parameters": 321655616, "total_parameters": 918772544,
         "training_status": "running", "metrics": {}}
    data = {"schema": "selfless_imagenet_val_t2i_fid_is_v2", "project_formal_protocol": True,
            "runtime_hashing_enabled": False, "samples_requested": 50000, "samples_evaluated": 50000,
            "split": "val", "seed": 42, "cfg": 2.0, "cfg_schedule": "constant", "sampling_steps": "10",
            "flow_solver": "heun", "temperature": 1.0, "parallel_rate": 1, "backbone_kv_cache": True,
            "weight_source": "hf_final_ema", "batch_size": 2048,
            "evaluation_model_source": {"kind": "hf_final_ema", "global_step": 95415,
                                        "path": str(REPO / "output/scale-test-b/hf_model-final-ema")},
            "architecture": {"flow_head": {key: b[key] for key in ("depth", "width", "architecture")}},
            "parameters": {"flow_head": b["head_parameters"], "total": b["total_parameters"]},
            "implementation_contracts": {"canonical_initial_noise_enabled": True,
                                         "paired_sample_count": 50000, "ordered_sample_count": 50000},
            "precision_protocol": {"model_dtype": "bf16", "vae_dtype": "fp32", "flow_integrator_dtype": "fp32"},
            "metric_protocol": {"protocol_name": "imagenet_val_fid50k_torch_fidelity_stratified_is",
                                "reference_distribution": "imagenet_val_50000", "is_splits": 10,
                                "is_split_assignment": "stratified_by_synset"},
            "real_stats_path": "/data/imagenet_val50000.pt", "real_stats_metadata": {"split": "val"},
            "mechanism_diagnostics": {"generated_latent_finite_rate": 1.0},
            "strategies": {"spatial_halton": {"count": 50000, "fid": 4.1581,
                                               "inception_score_mean": 222.88, "inception_score_std": 2.45}}}
    path = tmp_path / "b/metrics.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    return {"flow_head_scale": {"cfg_values": [1.0, 2.0, 3.5], "models": [b, f]}}, path, data


def test_scale_reads_raw_scores_and_leaves_pending_f_empty(tmp_path, scale_case):
    selection, _, _ = scale_case
    result = flow_head_scale_data(tmp_path, selection)
    assert (result["completed"], result["total"]) == (1, 6)
    assert result["rows"][0]["results"][1] == {
        "cfg": 2.0, "fid": 4.1581, "is": 222.88, "is_std": 2.45,
        "source": "b/metrics.json", "global_batch": 2048}
    for row in result["rows"][1]["results"]:
        assert all(row[key] is None for key in ("fid", "is", "is_std", "source"))
    selection["flow_head_scale"]["models"][1]["metrics"] = {"2.0": "b/metrics.json"}
    with pytest.raises(ValueError, match="unfinished scale training"):
        flow_head_scale_data(tmp_path, selection)


@pytest.mark.parametrize(("keys", "value", "error"), [
    (("cfg",), 3.5, "scale cfg mismatch"),
    (("samples_evaluated",), 10000, "scale samples_evaluated mismatch"),
    (("evaluation_model_source", "global_step"), 60000, "checkpoint step mismatch"),
    (("architecture", "flow_head", "depth"), 8, "head architecture mismatch"),
    (("parameters", "flow_head"), 164072976, "parameter count mismatch"),
    (("strategies", "spatial_halton", "fid"), float("nan"), "score is not finite"),
    (("implementation_contracts", "canonical_initial_noise_enabled"), False, "pairing incomplete"),
])
def test_scale_rejects_wrong_identity_protocol_or_invalid_score(tmp_path, scale_case, keys, value, error):
    selection, path, data = scale_case
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        flow_head_scale_data(tmp_path, selection)


def test_scale_allows_generation_batch_difference_but_requires_same_scoring(tmp_path, scale_case):
    selection, path, data = scale_case
    second = path.with_name("cfg1.json")
    selection["flow_head_scale"]["models"][0]["metrics"]["1.0"] = "b/cfg1.json"
    data.update(cfg=1.0, batch_size=4096)
    second.write_text(json.dumps(data))
    assert flow_head_scale_data(tmp_path, selection)["completed"] == 2
    data["real_stats_path"] = "/data/different_reference.pt"
    second.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="scoring/reference protocols differ"):
        flow_head_scale_data(tmp_path, selection)


def test_scale_rejects_invalidated_evaluation(tmp_path, scale_case):
    selection, path, _ = scale_case
    (path.parent / "evaluation_invalidation.json").write_text('{"invalidated": true}')
    with pytest.raises(ValueError, match="invalidated"):
        flow_head_scale_data(tmp_path, selection)


@pytest.fixture
def unified_text_case(tmp_path):
    from scripts.build_evaluation_report import REPO, TEXT_TASKS

    model = {"id": "b_x0", "run": "joint-test-b", "source": {"global_step": 95415},
             "checkpoint": str(REPO / "output/joint-test-b/hf_model-final-ema")}
    control = {"id": "text_only", "label": "text-only", "run": "joint-test-text-only",
               "checkpoint_step": 95368, "task": "text", "source": "only/summary.json",
               "physical_positions": {"text": 100000595968, "i2t": 0, "t2i": 0}}
    selection = {"models": {"b_x0": {"root": "baseline"}}, "unified_training_ablation": {
        "baseline": "b_x0", "baseline_physical_positions": {"text": 100049879040, "i2t": 50024939520, "t2i": 50024939520},
        "controls": [control]}}
    data = {"checkpoint": model["checkpoint"], "checkpoint_step": 95415, "weight_source": "hf_final_ema",
            "complete": True, "protocol": {"normalization": "original_choice_characters", "mmlu_fewshot": 5},
            "primary_metrics": {key: 0.4 for key in TEXT_TASKS}, "macro_average_primary": 0.4,
            "tasks": {key: {"samples": 100, "complete": True} for key in TEXT_TASKS}}
    baseline_path = tmp_path / "baseline/core/text/summary.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps(data))
    data = {**data, "checkpoint": str(REPO / "output/joint-test-text-only/hf_model-final-ema"),
            "checkpoint_step": 95368, "primary_metrics": {key: 0.5 for key in TEXT_TASKS}, "macro_average_primary": 0.5}
    only_path = tmp_path / control["source"]
    only_path.parent.mkdir()
    only_path.write_text(json.dumps(data))
    return selection, [model], only_path, data


def test_unified_text_comparison_preserves_direction_sources_and_missing_results(tmp_path, unified_text_case):
    selection, models, path, data = unified_text_case
    result = unified_training_ablation_data(tmp_path, selection, models)
    assert result["complete"] is True
    assert len(result["rows"]) == 9
    for row in result["rows"]:
        assert row["delta"] == pytest.approx(-10)
        assert row["winner"] == "only"
        assert row["baseline"]["source"] == "baseline/core/text/summary.json"
        assert row["only"]["source"] == "only/summary.json"
    data["complete"] = False
    path.write_text(json.dumps(data))
    partial = unified_training_ablation_data(tmp_path, selection, models)
    assert partial["complete"] is False
    assert all(row["only"] is None and row["delta"] is None and row["winner"] is None for row in partial["rows"])
    path.unlink()
    assert unified_training_ablation_data(tmp_path, selection, models)["completed"] == 0


@pytest.mark.parametrize(("keys", "value", "error"), [
    (("checkpoint_step",), 60000, "checkpoint step mismatch"),
    (("checkpoint",), "/another/model/hf_model-final-ema", "checkpoint mismatch"),
    (("protocol", "mmlu_fewshot"), 0, "text protocol mismatch"),
    (("tasks", "mmlu", "samples"), 99, "mmlu protocol mismatch"),
    (("tasks", "mmlu", "complete"), False, "incomplete only text task coverage"),
    (("macro_average_primary",), 0.7, "macro differs"),
])
def test_unified_comparison_rejects_noncomparable_text_results(tmp_path, unified_text_case, keys, value, error):
    selection, models, path, data = unified_text_case
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        unified_training_ablation_data(tmp_path, selection, models)


def test_unified_comparison_rejects_invalidated_only_result(tmp_path, unified_text_case):
    selection, models, path, _ = unified_text_case
    (path.parent / "evaluation_invalidation.json").write_text('{}')
    with pytest.raises(ValueError, match="invalidated"):
        unified_training_ablation_data(tmp_path, selection, models)


def test_unified_generation_requires_matching_cfg_and_reference_and_marks_lower_fid_better(tmp_path, unified_text_case, scale_case):
    selection, models, _, _ = unified_text_case
    _, _, generation = scale_case
    control = selection["unified_training_ablation"]["controls"][0]
    control.update(task="generation", checkpoint_step=95415)
    generation.update(cfg=3.5, real_source="cached_original_imagenet_val")
    generation["evaluation_model_source"]["path"] = models[0]["checkpoint"]
    baseline_path = tmp_path / "baseline/core/t2i-fid-is/metrics.json"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(json.dumps(generation))
    from scripts.build_evaluation_report import REPO
    generation["evaluation_model_source"]["path"] = str(REPO / "output" / control["run"] / "hf_model-final-ema")
    generation["strategies"]["spatial_halton"].update(fid=6.0, inception_score_mean=300.0)
    only_path = tmp_path / control["source"]
    only_path.write_text(json.dumps(generation))
    rows = unified_training_ablation_data(tmp_path, selection, models)["rows"]
    assert rows[0]["key"] == "fid" and rows[0]["winner"] == "baseline" and rows[0]["delta"] < 0
    assert rows[1]["key"] == "is" and rows[1]["winner"] == "only" and rows[1]["delta"] < 0
    generation["cfg"] = 2.0
    only_path.write_text(json.dumps(generation))
    with pytest.raises(ValueError, match="generation protocol mismatch"):
        unified_training_ablation_data(tmp_path, selection, models)
    generation.update(cfg=3.5, real_stats_path="/another/reference.pt")
    only_path.write_text(json.dumps(generation))
    with pytest.raises(ValueError, match="generation protocol mismatch"):
        unified_training_ablation_data(tmp_path, selection, models)


@pytest.mark.parametrize("cfgs", [[1.0, 2.0, 3.5], [1.0 + i / 2 for i in range(11)]])
def test_unified_missing_cfg_never_uses_another_cfg_score(tmp_path, unified_text_case, scale_case, cfgs):
    from scripts.build_evaluation_report import REPO

    selection, models, _, _ = unified_text_case
    _, _, data = scale_case
    study = selection["unified_training_ablation"]
    control = study["controls"][0]
    control.update(task="generation", checkpoint_step=95415, cfg_sources={})
    study.update(generation_cfg_values=cfgs, generation_baseline_sources={})
    data.update(real_source="cached_original_imagenet_val")
    data["evaluation_model_source"]["path"] = models[0]["checkpoint"]
    for cfg in study["generation_cfg_values"]:
        data["cfg"] = cfg
        path = tmp_path / ("baseline/core/t2i-fid-is/metrics.json" if cfg == 3.5 else f"baseline/cfg{cfg}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        study["generation_baseline_sources"][str(cfg)] = str(path.relative_to(tmp_path))
        control["cfg_sources"][str(cfg)] = control["source"] if cfg == 3.5 else f"only/cfg{cfg}.json"
    data["evaluation_model_source"]["path"] = str(REPO / "output" / control["run"] / "hf_model-final-ema")
    data["cfg"] = 3.5
    (tmp_path / control["source"]).write_text(json.dumps(data))
    result = unified_training_ablation_data(tmp_path, selection, models)
    assert not result["complete"] and result["controls"][0]["completed_cfgs"] == [3.5]
    assert len(result["rows"]) == 2 * len(cfgs)
    for row in result["rows"]:
        assert (row["only"] is None) == (row["cfg"] != 3.5)
        if row["cfg"] != 3.5:
            assert row["delta"] is None and row["winner"] is None
    (tmp_path / control["cfg_sources"]["1.0"]).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="generation protocol mismatch"):
        unified_training_ablation_data(tmp_path, selection, models)
