import pytest

from utils.evaluation.aro import (
    ARO_PRIMARY, CONDITIONAL_SCORE, DEBIASED_SCORE,
    conditional_score, update_aro_metrics,
)


def prediction(positive_prior=-5.0):
    return {
        "task": "aro_vg_relation", "kind": "pairwise_caption_ranking",
        "label": 1, "category": "on",
        "candidate_scores": [
            {DEBIASED_SCORE: 2.0, "estimated_language_prior_log_score": -8.0,
             "language_prior_alpha": 1.0},
            {DEBIASED_SCORE: 1.0, "estimated_language_prior_log_score": positive_prior,
             "language_prior_alpha": 1.0},
        ],
    }


def test_restoring_prior_changes_ranking_and_keeps_old_diagnostics():
    old = {"language_prior_debiased_pairwise": {"win_rate": 0.0}}
    metrics = update_aro_metrics(old, [prediction()])
    assert metrics["primary_metric"] == ARO_PRIMARY
    assert metrics["conditional_pairwise"]["win_rate"] == 1.0
    assert metrics["language_prior_debiased_pairwise"]["win_rate"] == 0.0
    assert metrics["categories"]["on"]["conditional_pairwise"]["win_rate"] == 1.0
    assert "conditional_pairwise" not in old


def test_ties_do_not_count_as_correct():
    metrics = update_aro_metrics({}, [prediction(), prediction(-7.0), prediction(-9.0)])
    assert metrics["conditional_pairwise"]["win_rate"] == pytest.approx(1 / 3)
    assert metrics["conditional_pairwise"]["tie_rate"] == pytest.approx(1 / 3)
    assert metrics["conditional_pairwise"]["loss_rate"] == pytest.approx(1 / 3)


def test_saved_and_legacy_conditional_scores_are_supported():
    assert conditional_score({CONDITIONAL_SCORE: -2.0}) == -2.0
    assert conditional_score({"normalized_loglikelihood": -3.0}) == -3.0
    with pytest.raises(ValueError, match="alpha"):
        conditional_score({DEBIASED_SCORE: 2.0})
    with pytest.raises(ValueError, match="finite"):
        conditional_score({CONDITIONAL_SCORE: float("nan")})


def test_migration_updates_nested_primary_but_preserves_sugarcrepe():
    from scripts.update_aro_conditional_results import propagate

    metrics = update_aro_metrics({}, [prediction()])
    changes = {"aro_vg_relation": {"metrics": metrics, "aro_metric_revision": {"schema": "test"}}}
    old = {"primary_metrics": {"aro_vg_relation": 0.0, "sugarcrepe": 0.6},
           "paper_compositional_benchmarks": {
               "aro_vg_relation": {"task": "aro_vg_relation", "metrics": {"records": 1}},
               "sugarcrepe": {"metrics": {"primary_metric": "language_prior_debiased_pairwise.win_rate"}}}}
    new = propagate(old, changes)
    assert new["primary_metrics"] == {"aro_vg_relation": 1.0, "sugarcrepe": 0.6}
    assert new["paper_compositional_benchmarks"]["sugarcrepe"] == old["paper_compositional_benchmarks"]["sugarcrepe"]
    assert "categories" not in new["paper_compositional_benchmarks"]["aro_vg_relation"]["metrics"]
