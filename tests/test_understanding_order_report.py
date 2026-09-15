import pytest

from scripts.report_understanding_order import paired_comparison


def row(index, task, label, winner, *, original=1, source_image="image-a"):
    return {"task": task, "item_id": str(index), "item_index": index,
            "image_id": index, "label": label, "category": "test",
            "metadata": {"original_index": original, "source_image": source_image},
            "candidate_scores": [{"text": str(i), "token_count": 1,
                "conditional_mean_token_loglikelihood": float(i == winner),
                "language_prior_debiased_mean_token_loglikelihood": float(i == winner)} for i in (0,1)]}


def test_circular_accuracy_requires_all_rotations_and_counts_questions():
    base = {i: row(i, "mmbench_dev_en", 1, i) for i in (0,1)}
    new = {i: row(i, "mmbench_dev_en", 1, 1) for i in (0,1)}
    result = paired_comparison(base, new)
    assert result["units"] == 1
    assert result["delta_pp"] == 100
    assert result["wrong_to_right"] == 1
    assert result["ci95_pp"] == [100,100]


def test_aro_crops_share_source_image_cluster_and_pairing_is_strict():
    base = {i: row(i, "aro_vg_relation", 1, 0) for i in (0,1)}
    new = {i: row(i, "aro_vg_relation", 1, i) for i in (0,1)}
    result = paired_comparison(base, new)
    assert result["units"] == 2
    assert result["image_or_question_clusters"] == 1
    assert result["delta_pp"] == 50
    assert result["ci95_pp"] == [50,50]
    new[0]["candidate_scores"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="candidates"):
        paired_comparison(base, new)
