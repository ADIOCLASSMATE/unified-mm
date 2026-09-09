import json

import pytest
import yaml

from scripts.evaluation_report_training import collect_training, collect_validation, export_training, read_history


def record(step, **metrics):
    return json.dumps({"global_step": step, "metrics": metrics}) + "\n"


def test_task_losses_do_not_use_mixed_text_loss_or_missing_zeroes(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    path.write_text(record(10, **{"step_loss": 0.25, "train/loss_text": 5.0, "train/loss_i2t": 5.0,
                                  "train/loss_image_flow": 0.0, "train/weighted_contribution_i2t": 0.25}))
    rows, _ = read_history(path)
    assert rows == [[10, 0.25, None, 5.0, None, None, 0.25, None]]


def test_rollback_replaces_stale_tail_and_unfinished_write_waits_for_next_build(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    path.write_text(record(10, step_loss=1) + record(20, step_loss=2) + record(30, step_loss=3)
                    + record(20, step_loss=0.2) + '{"global_step":30')
    rows, info = read_history(path)
    assert [r[:2] for r in rows] == [[10, 1], [20, 0.2]]
    assert info["partial_tail"] is True
    assert info["replaced_points"] == 2


def test_nonfinite_data_stays_missing_but_complete_corrupt_lines_are_rejected(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    path.write_text(record(10, step_loss=float("nan")))
    rows, info = read_history(path)
    assert rows[0][1] is None
    assert info["nonfinite_values"] == 1
    path.write_text('{"global_step":10\n')
    with pytest.raises(ValueError, match="malformed training log"):
        read_history(path)


def test_discovery_includes_no_ema_runs_and_uses_actual_sweep_stop(tmp_path):
    root = tmp_path / "output/evaluation"
    root.mkdir(parents=True)
    for name in ("unified-a-0p6b-t2i-only-run", "unified-b-smoke", "unified-a-1p7b-lr-sweep-1b-s42/arm"):
        run = tmp_path / "output" / name
        run.mkdir(parents=True)
        config = {"training": {"max_train_steps": 95415, "stop_after_steps": 955},
                  "dataset": {"params": {"schedule": ["t2i"]}}}
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        (run / "training_metrics.jsonl").write_text(record(950, **{"step_loss": 0.6, "train/loss_t2i": 0.6}))
        (run / "training_runtime_metrics.json").write_text('{"global_step":955}')
    declared = tmp_path / "configs/selfless/unified_single_t2i_0p6b_100b_ascend16.yaml"
    declared.parent.mkdir(parents=True)
    declared.write_text(yaml.safe_dump({"experiment": {"project": "unified-b-t2i-only-planned"}}))
    data = collect_training(tmp_path, root, [{"id": "future", "run": "unified-future", "label": "Future arm"}])
    assert len(data["runs"]) == 4
    completed = [r for r in data["runs"] if r["points"]]
    assert all(r["complete"] and r["target_steps"] == 955 and r["last_logged_step"] == 950 for r in completed)
    pending = next(r for r in data["runs"] if r["id"] == "future")
    assert pending["points"] == [] and pending["status"] == "等待训练日志"
    assert any(r["label"] == "历史 B · T2I-only" and r["source"] is None for r in data["runs"])
    assert not any("smoke" in r["run"] for r in data["runs"])


def validation_record(path, step, **metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "selfless_flow_validation_metrics_v1", "global_step": step,
                                "validation_seed": 424242, "training_seed": 42, "metrics": metrics}) + "\n")


@pytest.mark.parametrize("new_protocol,enabled,interval,expected_current", [
    (True, True, 10000, True),
    (True, False, 10000, False),
    (False, True, 10000, False),
    (True, True, 0, True),
])
def test_first_validation_status_uses_actual_training_protocol(tmp_path, new_protocol, enabled, interval, expected_current):
    run = tmp_path / "output/unified-new-run"
    run.mkdir(parents=True)
    config = {"experiment": {"val_every": interval, "loss_validation": {"enabled": enabled},
                             "downstream_validation": {}},
              "training": {"use_ema": True, "ema_validate": True}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    row = {"global_step": 10, "metrics": {"step_loss": .7}}
    if new_protocol:
        row["loss_protocol"] = {"name": "unified_schedule_microbatch_mean_v1"}
    (run / "training_metrics.jsonl").write_text(json.dumps(row) + "\n")
    result = collect_training(tmp_path, tmp_path / "output/evaluation", [])['runs'][0]['validation']
    assert result['records'] == 0 and result['points'] == []
    assert ('当前训练权重' in result['model_weights']) == expected_current
    if interval == 0:
        assert result['log_every'] == 0 and '未启用周期验证' in result['missing_reason']
    elif expected_current:
        assert '10,000' in result['missing_reason'] and '同口径' in result['missing_reason']


def test_validation_prefers_canonical_periodic_results_and_masks_inactive_tasks(tmp_path):
    run = tmp_path / "output/unified-a-caption-only"
    root = tmp_path / "output/evaluation"
    name = "validation_metrics_step_2000.json"
    validation_record(run / name, 2000, **{"val/loss_i2t": 99})
    canonical = root / "training-validation" / run.name / name
    validation_record(canonical, 2000, **{"val/loss": .1, "val/loss_t2i": 0, "val/loss_i2t": 2,
        "val/loss_text": 2, "val/image_target_tokens": 0, "val/text_target_tokens": 500,
        "val/weighted_contribution_t2i": 0, "val/weighted_contribution_i2t": .1})
    # Final EMA evaluation has a different protocol and must not join this history.
    validation_record(root / "formal" / run.name / "validation_metrics_step_95415.json", 95415, **{"val/loss": 77})
    result = collect_validation(tmp_path, root, run, {})
    assert result["points"] == [[2000, .1, None, 2, None, None, .1, None]]
    assert result["records"] == 1 and result["available"]["climbmix"] == 0
    assert result["details"][0]["source"] == str(canonical.relative_to(root))


def test_validation_t2i_only_caption_zero_is_missing_and_partial_files_wait(tmp_path):
    run = tmp_path / "output/unified-a-t2i-only"
    root = tmp_path / "output/evaluation"
    validation_record(run / "validation_metrics_step_2000.json", 2000, **{
        "val/loss": .7, "val/loss_t2i": .7, "val/loss_i2t": 0, "val/loss_text": 0,
        "val/text_target_tokens": 0, "val/image_target_tokens": 256})
    (run / "validation_metrics_step_4000.json").write_text('{"global_step":4000')
    result = collect_validation(tmp_path, root, run, {})
    assert result["points"][0][2:5] == [.7, None, None]
    assert len(result["incomplete_files"]) == 1
    validation_record(run / "validation_metrics_step_4000.json", 6000)
    with pytest.raises(ValueError, match="step/filename mismatch"):
        collect_validation(tmp_path, root, run, {})


def test_pure_text_validation_joins_matching_step_without_changing_joint_total(tmp_path):
    run = tmp_path / "output/unified-a"
    root = tmp_path / "output/evaluation"
    directory = root / "training-validation" / run.name
    validation_record(directory / "validation_metrics_step_2000.json", 2000, **{
        "val/loss": .8, "val/loss_t2i": .7, "val/loss_i2t": 2,
        "val/image_target_tokens": 256, "val/text_target_tokens": 99})
    for step in (2000, 4000):
        path = directory / f"validation_climbmix_metrics_step_{step}.json"
        path.write_text(json.dumps({"schema": "selfless_climbmix_validation_metrics_v1", "global_step": step,
            "metrics": {"val/loss_climbmix": 3, "val/climbmix_target_tokens": 1000,
                        "val/weighted_contribution_climbmix": .15}, "complete": True,
            "model_weights": "current", "validation_seed": 424242, "training_seed": 42,
            "protocol": "climbmix_fixed_document_ce_v1", "independence": "may_have_been_seen_in_training",
            "source": {"manifest": "fixed-manifest.json"}}) + "\n")
    result = collect_validation(tmp_path, root, run, {})
    assert result["points"][0] == [2000, .8, .7, 2, 3, None, None, .15]
    assert result["points"][1] == [4000, None, None, None, 3, None, None, .15]
    assert result["details"][0]["target_counts"] == {"image": 256, "caption": 99, "climbmix": 1000}
    assert result["details"][0]["pure_text_independence"] == "may_have_been_seen_in_training"
    assert result["details"][0]["source"] != result["details"][0]["pure_text_source"]


def unified_record(directory, step, **updates):
    directory.mkdir(parents=True, exist_ok=True)
    record = {"schema": "selfless_unified_loss_validation_metrics_v1", "global_step": step,
        "complete": True, "model_weights": "current", "model_mode": "train_no_grad",
        "loss_protocol": {"name": "unified_schedule_microbatch_mean_v1", "schedule": ["t2i"] * 8},
        "metrics": {"val/loss": .7, "val/loss_t2i": .7, "val/image_target_tokens": 256,
                    "val/text_target_tokens": 0, "val/weighted_contribution_t2i": .7}}
    record.update(updates)
    path = directory / f"validation_unified_loss_metrics_step_{step}.json"
    path.write_text(json.dumps(record) + "\n")
    return path


def test_unified_history_keeps_shared_sampling_contract_without_embedding_full_ids(tmp_path):
    run = tmp_path / "output/unified-a"
    root = tmp_path / "output/evaluation"
    directory = root / "training-validation" / run.name
    subset = {"schema": "training_imagenet_subset_v1", "seed": 424242, "per_class": 2,
              "samples": 2000, "classes": 1000, "sample_ids": ["val/image_1"],
              "img_ids": [1], "class_indices": [0]}
    unified_record(directory, 100, imagenet_subset=subset)
    unified_record(directory, 200, imagenet_subset={**subset, "seed": 99})
    details = collect_validation(tmp_path, root, run, {})["details"]
    assert details[0]["imagenet_subset"]["samples"] == 2000
    assert details[0]["imagenet_subset"]["seed"] == 424242
    assert details[1]["imagenet_subset"]["seed"] == 99
    assert not {"sample_ids", "img_ids", "class_indices"} & details[0]["imagenet_subset"].keys()


def test_unified_result_replaces_step_whole_without_splicing_legacy_tasks(tmp_path):
    run = tmp_path / "output/unified-a"
    root = tmp_path / "output/evaluation"
    directory = root / "training-validation" / run.name
    for step in (2000, 4000):
        validation_record(directory / f"validation_metrics_step_{step}.json", step,
                          **{"val/loss": .9, "val/loss_t2i": .7, "val/loss_i2t": 4,
                             "val/image_target_tokens": 256, "val/text_target_tokens": 99})
    path = unified_record(directory, 4000)
    result = collect_validation(tmp_path, root, run, {})
    assert result["points"][0] == [2000, .9, .7, 4, None, None, None, None]
    assert result["points"][1] == [4000, .7, .7, None, None, .7, None, None]
    assert result["unified_records"] == 1
    assert result["details"][0]["loss_protocol"]["name"] == "legacy_imagenet_pair"
    assert result["details"][1]["source"] == str(path.relative_to(root))
    assert result["details"][1]["model_weights"] == "current"


def test_unified_pure_text_provenance_is_preserved_and_incomplete_total_rejected(tmp_path):
    run = tmp_path / "output/unified-a"
    root = tmp_path / "output/evaluation"
    directory = root / "training-validation" / run.name
    path = unified_record(directory, 2000,
        loss_protocol={"name": "unified_schedule_microbatch_mean_v1", "schedule": ["climbmix"] * 8},
        metrics={"val/loss": .2, "val/loss_climbmix": 4., "val/weighted_contribution_climbmix": .2,
                 "val/climbmix_target_tokens": 200},
        pure_text={"source": {"manifest": "fixed.json"}, "protocol": "climbmix_fixed_document_ce_v1",
                   "independence": "may_have_been_seen_in_training"})
    result = collect_validation(tmp_path, root, run, {})
    assert result["points"] == [[2000, .2, None, None, 4, None, None, .2]]
    assert result["details"][0]["pure_text_source"] == str(path.relative_to(root))
    assert result["details"][0]["pure_text_independence"] == "may_have_been_seen_in_training"
    record = json.loads(path.read_text())
    record["loss_protocol"]["schedule"].append("t2i")
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="missing an active source"):
        collect_validation(tmp_path, root, run, {})
