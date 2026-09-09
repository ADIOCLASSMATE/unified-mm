import json
from types import SimpleNamespace

import pytest

from utils.evaluation.text_benchmarks import DEFAULT_TASKS, LM_EVAL_REFERENCE, MultipleChoiceExample
from scripts.repair_text_benchmark_results import (
    renormalize_rows,
    repair_results,
    resolve_repair_checkpoint,
)


def example(task="piqa"):
    return MultipleChoiceExample(0, "id", task, "Q:", ("long", "é猫"), 0, category="subject")


def sample(task="piqa"):
    return dict(schema="selfless_text_multiple_choice_sample_v1", task=task,
                item_index=0, item_id="id", label=0, category="subject",
                choice_loglikelihoods=[-6.0, -8.0], choice_token_counts=[1, 2],
                choice_normalized_loglikelihoods=[-6.0, -4.0], prediction=0,
                prediction_normalized=1, correct=True, correct_normalized=False,
                boundary_adjusted=False, truncated_context_tokens=0)


def test_repair_uses_character_not_token_or_utf8_byte_lengths():
    old = sample()
    row = renormalize_rows("piqa", [old], [example()])[0]
    assert old["prediction_normalized"] == 1  # source remains unchanged
    assert row["prediction_normalized"] == 0
    assert row["choice_char_counts"] == [4, 2]
    assert row["choice_normalized_loglikelihoods"] == [-1.5, -4.0]
    assert row["choice_loglikelihoods"] == old["choice_loglikelihoods"]


@pytest.mark.parametrize("change", [{"item_id": "wrong"}, {"label": 1},
                                  {"choice_loglikelihoods": [float("nan"), -8]}])
def test_repair_rejects_misalignment_and_nonfinite_likelihood(change):
    row = sample()
    row.update(change)
    with pytest.raises(ValueError):
        renormalize_rows("piqa", [row], [example()])


def test_repair_refuses_to_renormalize_legacy_winogrande():
    with pytest.raises(ValueError, match="fresh"):
        renormalize_rows("winogrande", [sample("winogrande")], [example("winogrande")])


def test_in_place_repair_preserves_backup_invalidates_macro_and_is_repeatable(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.repair_text_benchmark_results.load_multiple_choice_task",
                        lambda task, root: [example(task)])
    tasks = {}
    for task in DEFAULT_TASKS:
        directory = tmp_path / "tasks" / task
        directory.mkdir(parents=True)
        (directory / "samples.jsonl").write_text(json.dumps(sample(task)) + "\n")
        tasks[task] = dict(accuracy=1.0, accuracy_normalized=0.0, accuracy_macro=1.0)
        (directory / "metrics.json").write_text(json.dumps(tasks[task]))
    original = dict(tasks=tasks, protocol=dict(protocol_schema="selfless_text_benchmark_v2",
                    lm_eval_reference=LM_EVAL_REFERENCE), checkpoint="/model", checkpoint_step=1,
                    macro_average_primary=0.5)
    (tmp_path / "summary.json").write_text(json.dumps(original))
    (tmp_path / "evaluation_run.json").write_text(json.dumps(dict(limit=0)))
    for _ in range(2):
        report = repair_results(tmp_path, tmp_path)
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["complete"] is False
        assert summary["macro_average_primary"] is None
        assert summary["primary_metrics"]["winogrande"] is None
        assert summary["primary_metrics"]["piqa"] == 1.0
        assert summary["tasks"]["winogrande"]["complete"] is False
        assert report["pending_tasks"] == ["winogrande"]
        assert json.loads((tmp_path / "legacy-before-p1" / "summary.json").read_text()) == original

    fresh = tmp_path / "fresh-winogrande"
    (fresh / "tasks" / "winogrande").mkdir(parents=True)
    row = sample("winogrande")
    row.update(schema="selfless_text_multiple_choice_sample_v2",
               protocol_schema="selfless_text_benchmark_v3",
               normalization="original_choice_characters")
    (fresh / "tasks" / "winogrande" / "samples.jsonl").write_text(json.dumps(row) + "\n")
    (fresh / "summary.json").write_text(json.dumps(dict(
        complete=True, checkpoint="/model", protocol=dict(
            protocol_schema="selfless_text_benchmark_v3",
            winogrande_scoring="shared_suffix_given_prefix_and_option"))))
    (fresh / "evaluation_run.json").write_text(json.dumps(dict(limit=0)))
    report = repair_results(tmp_path, tmp_path, fresh)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert report["pending_tasks"] == []
    assert summary["complete"] is True
    assert summary["primary_metrics"]["winogrande"] == 1.0
    assert summary["macro_average_primary"] == 1.0
    assert json.loads((tmp_path / "legacy-before-p1" / "summary.json").read_text()) == original


def test_checkpoint_relocation_requires_explicit_missing_path_and_matching_provenance(tmp_path, monkeypatch):
    old_path = tmp_path / "old"
    new_path = tmp_path / "renamed"
    new_path.mkdir()
    (new_path / "config.json").write_text(json.dumps(dict(dual_stream_attention_contract="selfless_strict")))
    source = SimpleNamespace(is_hf_final_ema=True, global_step=42,
                             metadata=dict(floating_dtype="float32", state_key_count=10))
    monkeypatch.setattr("scripts.repair_text_benchmark_results.resolve_evaluation_model_source",
                        lambda path: source)
    original = dict(checkpoint=str(old_path), checkpoint_step=42)
    run = dict(ema=source.metadata, dual_stream_attention_contract="selfless_strict")
    assert resolve_repair_checkpoint(original, run) == old_path
    assert resolve_repair_checkpoint(original, run, new_path) == new_path
    source.global_step = 43
    with pytest.raises(ValueError, match="provenance"):
        resolve_repair_checkpoint(original, run, new_path)
    source.global_step = 42
    old_path.mkdir()
    with pytest.raises(ValueError, match="existing"):
        resolve_repair_checkpoint(original, run, new_path)
