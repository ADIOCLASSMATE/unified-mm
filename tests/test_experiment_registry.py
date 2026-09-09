import copy
import json

import pytest

from utils.experiment_registry import current_presentation, experiment_identity, read_run_identity, task_training_labels, write_run_identity


def test_new_model_has_one_config_owned_identity_without_registry_edit(tmp_path):
    directory = tmp_path / "unified-new-research-arm"
    config = {"experiment": {"identity": {"id": "new_arm", "label": "New arm", "group": "main", "purpose": "formal"}},
              "dataset": {"params": {"schedule": ["climbmix", "t2i", "climbmix", "i2t"]}}}
    write_run_identity(directory, config)
    actual = read_run_identity(directory, config)
    assert actual["id"] == "new_arm" and actual["group"] == "main"
    assert actual["active_sources"] == ["climbmix", "i2t", "t2i"]
    assert set(task_training_labels(actual).values()) == {"已训练"}
    changed = copy.deepcopy(config)
    changed["dataset"]["params"]["schedule"] = ["climbmix"]
    with pytest.raises(ValueError, match="identity changed"):
        write_run_identity(directory, changed)
    with pytest.raises(ValueError, match="task identity disagrees"):
        read_run_identity(directory, changed)


def test_unknown_model_does_not_invent_training_tasks():
    identity = experiment_identity("unified-unknown-model")
    assert identity["purpose"] == "unclassified" and identity["active_sources"] is None
    assert all("未记录" in value for value in task_training_labels(identity).values())


def test_temporary_name_cannot_inherit_formal_purpose_from_training_config():
    identity = experiment_identity("unified-new-smoke", {"experiment": {"identity": {"purpose": "formal"}}})
    assert identity["purpose"] == "temporary"


def test_declared_temporary_run_needs_no_special_name():
    identity = experiment_identity("unified-local-check", {"experiment": {"identity": {"purpose": "temporary"}}})
    assert identity["purpose"] == "temporary"


def test_invalid_saved_identity_is_rejected(tmp_path):
    write_run_identity(tmp_path, {})
    path = tmp_path / "experiment_identity.json"
    data = json.loads(path.read_text())
    data["purpose"] = "unknown-purpose"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="group/purpose"):
        read_run_identity(tmp_path)


@pytest.mark.parametrize("run,old_label,old_group,expected_label,expected_group", [
    ("unified-b-x0content-0p6b-100b-imagenet-split-s42-r1", "B · X0-content（新版）", "main", "B", "main"),
    ("unified-c-on-b-x0content-0p6b-100b-imagenet-split-s42-r1", "C on B · 文本 AR", "main", "C · B + 文本 AR", "ablation"),
])
def test_display_rename_preserves_saved_identity_and_resume_task_checks(
        tmp_path, run, old_label, old_group, expected_label, expected_group):
    directory = tmp_path / run
    config = {"dataset": {"params": {"schedule": ["climbmix", "t2i", "climbmix", "i2t"]}}}
    write_run_identity(directory, config)
    path = directory / "experiment_identity.json"
    saved = json.loads(path.read_text())
    saved.update(label=old_label, group=old_group)
    path.write_text(json.dumps(saved))
    before = path.read_bytes()

    actual = read_run_identity(directory, config)
    assert (actual["label"], actual["group"]) == (expected_label, expected_group)
    assert actual["id"] == saved["id"] and actual["active_sources"] == saved["active_sources"]
    write_run_identity(directory, config)
    assert path.read_bytes() == before

    changed = copy.deepcopy(config)
    changed["dataset"]["params"]["schedule"] = ["climbmix"]
    with pytest.raises(ValueError, match="task identity disagrees"):
        read_run_identity(directory, changed)
    with pytest.raises(ValueError, match="identity changed"):
        write_run_identity(directory, changed)


def test_historical_run_cannot_be_relabelled_as_formal_b_by_a_stale_manifest():
    spec = {"run": "unified-b-0p6b-100b-imagenet-split-s42-r1", "id": "b_flowdiag",
            "label": "B", "group": "main", "checkpoint": "/training/legacy-b/hf_model-final-ema"}
    current = current_presentation(spec)
    assert current["group"] == "legacy" and current["label"].startswith("历史 B")
    assert current["checkpoint"] == spec["checkpoint"] and current["id"] == "b_flowdiag"
    assert spec["label"] == "B"  # Never mutate the frozen source manifest.
    with pytest.raises(ValueError, match="id disagrees"):
        current_presentation({**spec, "id": "b_x0"})
