import json
import math
from pathlib import Path

from PIL import Image
import pytest
import torch

from scripts.evaluate_cross_dataset_retrieval import (
    LIKELIHOOD_SCORING_CONTRACT,
    retrieval_metrics,
)
from scripts.evaluate_imagenet_pretraining_native import (
    classification_metrics,
    load_openai_clip_class_names,
)
from scripts.language_prior_calibration import language_prior_debiased_scores
from scripts.merge_cross_dataset_retrieval_partitions import merge_score_shards
from scripts.prepare_cross_dataset_retrieval_assets import (
    load_karpathy_test_records,
)


def test_karpathy_normalizer_preserves_five_captions_and_split(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index in range(2):
        filename = f"image-{index}.jpg"
        Image.new("RGB", (4, 4), color=(index, 2, 3)).save(image_root / filename)
        rows.append(
            {
                "split": "test",
                "filename": filename,
                "imgid": index,
                "sentences": [{"raw": f"caption {index}-{j}"} for j in range(5)],
            }
        )
    rows.append(
        {
            "split": "train",
            "filename": "unused.jpg",
            "sentences": [{"raw": f"unused {j}"} for j in range(5)],
        }
    )
    karpathy = tmp_path / "dataset_flickr30k.json"
    karpathy.write_text(json.dumps({"images": rows}), encoding="utf-8")

    records = load_karpathy_test_records(
        dataset="flickr30k",
        karpathy_json=karpathy,
        image_root=image_root,
        expected_images=2,
    )

    assert len(records) == 2
    assert records[0].image_index == 0
    assert records[1].img_id == 8_100_000_001
    assert records[0].captions == tuple(f"caption 0-{j}" for j in range(5))


def test_cross_dataset_retrieval_uses_five_positive_captions_per_image():
    scores = torch.tensor(
        [
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [4.0, 3.0, 2.0, 1.0, 0.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        ]
    )

    metrics = retrieval_metrics(scores)

    assert metrics["images"] == 2
    assert metrics["captions"] == 10
    assert metrics["image_to_text"]["recall_at_1"] == 1.0
    assert metrics["text_to_image"]["recall_at_1"] == 1.0


def test_cross_dataset_retrieval_supports_canonical_ragged_caption_counts():
    scores = torch.tensor(
        [
            [9.0, 8.0, 1.0, 0.0, -1.0],
            [1.0, 0.0, 9.0, 8.0, 7.0],
        ]
    )

    metrics = retrieval_metrics(scores, caption_counts=[2, 3])

    assert metrics["caption_count_distribution"] == {"2": 1, "3": 1}
    assert metrics["image_to_text"]["recall_at_1"] == 1.0
    assert metrics["text_to_image"]["recall_at_1"] == 1.0


def test_language_prior_uses_probability_space_mean_and_fixed_alpha_one():
    conditional = torch.tensor(
        [[math.log(0.8), math.log(0.2)], [math.log(0.4), math.log(0.6)]]
    )
    calibrated, prior = language_prior_debiased_scores(conditional)

    assert torch.allclose(prior.exp(), torch.tensor([0.6, 0.4]), atol=1.0e-6)
    assert torch.allclose(
        calibrated,
        conditional - torch.log(torch.tensor([0.6, 0.4])),
        atol=1.0e-6,
    )


def test_pinned_openai_clip_class_names_are_unique_and_disambiguated():
    root = Path(__file__).resolve().parents[1]
    names, provenance = load_openai_clip_class_names(
        root / "scripts/assets/imagenet1k_openai_clip_classnames.json"
    )
    assert len(names) == len(set(names)) == 1_000
    assert names[744] == "projectile"
    assert names[836] == "sunglass"
    assert provenance["source_commit"] == (
        "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
    )


def test_imagenet_classification_reports_only_top1_and_top5():
    scores = torch.zeros(2, 1_000)
    scores[0, 7] = 2.0
    scores[1, :6] = torch.arange(1.0, 7.0)
    result = classification_metrics(scores, torch.tensor([7, 999]))
    assert result["top_1_accuracy"] == 0.5
    assert result["top_5_accuracy"] == 0.5
    assert set(result) == {
        "records",
        "classes",
        "primary_metric",
        "top_1_accuracy",
        "top_5_accuracy",
    }


def test_columnwise_prior_correction_leaves_t2i_ranks_invariant():
    conditional = torch.tensor(
        [
            [-0.4, -0.1, -1.0, -0.9],
            [-0.6, -0.3, -0.2, -0.5],
        ]
    )
    calibrated, _ = language_prior_debiased_scores(conditional)

    before = retrieval_metrics(conditional, caption_counts=[2, 2])
    after = retrieval_metrics(calibrated, caption_counts=[2, 2])
    assert before["text_to_image"] == after["text_to_image"]


def test_independent_query_partitions_merge_with_exact_coverage(tmp_path: Path):
    expected = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    roots = [tmp_path / "partition-0", tmp_path / "partition-1"]
    for partition_index, root in enumerate(roots):
        (root / "shards").mkdir(parents=True)
        query_indices = [partition_index, partition_index + 2]
        for rank, query_index in enumerate(query_indices):
            torch.save(
                {
                    "query_indices": torch.tensor([query_index]),
                    "conditional_mean_token_loglikelihood": expected[
                        query_index : query_index + 1
                    ],
                    "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
                    "runtime_hashing_enabled": False,
                },
                root / "shards" / f"rank-{rank:05d}-of-00002.pt",
            )

    actual = merge_score_shards(
        [(roots[0], 0, 2, 2), (roots[1], 1, 2, 2)],
        images=4,
        captions=6,
    )

    assert torch.equal(actual, expected)


def test_independent_query_partition_merge_rejects_missing_rows(tmp_path: Path):
    root = tmp_path / "partition-0"
    (root / "shards").mkdir(parents=True)
    torch.save(
        {
            "query_indices": torch.tensor([0]),
            "conditional_mean_token_loglikelihood": torch.zeros((1, 3)),
            "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
            "runtime_hashing_enabled": False,
        },
        root / "shards" / "rank-00000-of-00001.pt",
    )

    with pytest.raises(ValueError, match="incomplete query partition"):
        merge_score_shards(
            [(root, 0, 2, 1)],
            images=4,
            captions=3,
        )


def test_independent_query_partition_merge_rejects_duplicate_rows(tmp_path: Path):
    root = tmp_path / "partition-0"
    (root / "shards").mkdir(parents=True)
    torch.save(
        {
            "query_indices": torch.tensor([0, 0]),
            "conditional_mean_token_loglikelihood": torch.zeros((2, 3)),
            "scoring_contract": LIKELIHOOD_SCORING_CONTRACT,
            "runtime_hashing_enabled": False,
        },
        root / "shards" / "rank-00000-of-00001.pt",
    )

    with pytest.raises(ValueError, match="duplicate rows within"):
        merge_score_shards(
            [(root, 0, 2, 1)],
            images=4,
            captions=3,
        )


def test_paper_protocol_requires_full_imagenet_classification_and_removes_custom_retrieval():
    root = Path(__file__).resolve().parents[1]
    protocol = (
        root / "configs/protocols/pretraining_native_understanding_evaluation_ascend16.yaml"
    ).read_text(encoding="utf-8")
    evaluator = (root / "scripts/evaluate_imagenet_pretraining_native.py").read_text(
        encoding="utf-8"
    )
    assert "imagenet_zero_shot_classification:" in protocol
    assert "images: 50000" in protocol
    assert "classes: 1000" in protocol
    assert "language_prior_debiased_only" in protocol
    assert "real_labels" not in evaluator
    assert "retrieval_1k" not in evaluator
    assert "retrieval_5k" not in evaluator
