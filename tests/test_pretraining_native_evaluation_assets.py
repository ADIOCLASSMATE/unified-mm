import csv
import json
import sys

from PIL import Image
import pytest
from safetensors.torch import save_file
import torch

from scripts.prepare_multimodal_likelihood_assets import (
    ImageRegistry,
    prepare_svo_probes,
    prepare_whatsup_controlled,
)
from scripts import summarize_pretraining_native_understanding as native_summary


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_paper_metric_gate_rejects_nonfinite_or_out_of_range_values(value):
    with pytest.raises(ValueError, match="finite and in"):
        native_summary.require_unit_interval(value, "test metric")


def write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path)


def test_svo_normalizer_retains_official_pair_and_negative_type(tmp_path):
    image_root = tmp_path / "svo-images"
    write_image(image_root / "0.jpg")
    write_image(image_root / "1.jpg")
    csv_path = tmp_path / "svo.csv"
    fields = (
        "sentence",
        "pos_triplet",
        "neg_triplet",
        "pos_url",
        "neg_url",
        "pos_image_id",
        "neg_image_id",
        "subj_neg",
        "verb_neg",
        "obj_neg",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "sentence": "A girl stands in grass.",
                "pos_triplet": "girl,stand,grass",
                "neg_triplet": "dog,stand,grass",
                "pos_url": "https://example.test/positive",
                "neg_url": "https://example.test/negative",
                "pos_image_id": 0,
                "neg_image_id": 1,
                "subj_neg": "True",
                "verb_neg": "False",
                "obj_neg": "False",
            }
        )
    rows, details = prepare_svo_probes(csv_path, image_root, ImageRegistry())
    assert len(rows) == 2
    assert {row["metadata"]["image_role"] for row in rows} == {
        "positive",
        "negative",
    }
    assert {row["metadata"]["negative_type"] for row in rows} == {"subject"}
    assert rows[0]["candidates"] == rows[1]["candidates"]
    assert details["negative_type_counts"] == {"subject": 1}


def test_whatsup_normalizer_preserves_four_image_sets(tmp_path):
    annotations = {}
    roots = {}
    relations = {
        "A": ("left_of", "right_of", "on", "under"),
        "B": ("left_of", "right_of", "in-front_of", "behind"),
    }
    for subset, subset_relations in relations.items():
        root = tmp_path / f"images-{subset}"
        rows = []
        for relation in subset_relations:
            filename = f"cup_{relation}_plate_0.jpeg"
            write_image(root / filename)
            rows.append(
                {
                    "image_path": f"obsolete/root/{filename}",
                    "caption_options": [
                        f"correct {relation}",
                        "wrong one",
                        "wrong two",
                        "wrong three",
                    ],
                }
            )
        annotation = tmp_path / f"controlled-{subset}.json"
        annotation.write_text(json.dumps(rows), encoding="utf-8")
        annotations[subset] = annotation
        roots[subset] = root

    rows, details = prepare_whatsup_controlled(
        annotations["A"],
        roots["A"],
        annotations["B"],
        roots["B"],
        ImageRegistry(),
    )
    assert len(rows) == 8
    assert details["sets"] == 2
    assert {row["metadata"]["subset"] for row in rows} == {"A", "B"}
    assert all(row["label"] == 0 and len(row["candidates"]) == 4 for row in rows)


@pytest.mark.parametrize("s2", [False, True])
def test_pretraining_native_summary_merges_required_components(tmp_path, monkeypatch, s2):
    checkpoint = tmp_path / "checkpoint"
    native_root = tmp_path / "native"
    benchmark_root = tmp_path / "benchmarks"
    coco_root = tmp_path / "coco"
    flickr_root = tmp_path / "flickr"
    output = tmp_path / "summary"
    for path in (checkpoint, native_root, benchmark_root, coco_root, flickr_root):
        path.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "model.image_token_embedder.weight": torch.zeros(1),
            "image_flow_condition_proj.weight": torch.zeros(1),
            "image_flow_head.weight": torch.zeros(1),
        },
        checkpoint / "model.safetensors",
    )
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "ema_export_metadata.json").write_text(
        json.dumps(
            {
                "schema": "selfless_ema_hf_export_v1",
                "floating_dtype": "float32",
                "source_global_step": 58_000,
                "source_world_size": 64,
                "state_key_count": 3,
                "export_kind": "training",
            }
        ),
        encoding="utf-8",
    )
    common_manifest = {
        "complete": True,
        "runtime_hashing_enabled": False,
        "checkpoint": str(checkpoint.resolve()),
        "weight_source": "hf_final_ema",
    }
    (native_root / "manifest.json").write_text(
        json.dumps(
            {
                **common_manifest,
                "schema": (
                    "selfless_imagenet1k_zeroshot_classification_evaluation_v2"
                ),
                "project_formal_protocol": True,
            }
        ),
        encoding="utf-8",
    )
    benchmark_scoring = {
        "primary_candidate_score": (
            "language_prior_debiased_mean_token_loglikelihood"
        ),
        "reported_score_variant": "conditional_aro_debiased_other_tasks",
        "task_primary_candidate_scores": {task: "conditional_mean_token_loglikelihood" for task in ("aro_vg_relation", "aro_vg_attribution")},
        "language_prior_alpha": 1.0,
        "language_prior_estimator": "content_free_gaussian_image_logmeanexp",
        "language_prior_null_image_count": 3,
        "language_prior_uses_labels": False,
        "mc_samples": 64,
    }
    if s2:
        benchmark_scoring.update({
            "dual_stream_attention_contract": "showo2_omni_attention",
            "scoring_contract": "showo2_next_token_ar_target_aligned_v1",
            "contract": "showo2_next_token_ar_target_aligned_v1",
            "image_order_mc_contract": "not_applicable_full_image_ar",
            "mc_samples": 1,
        })
    (benchmark_root / "manifest.json").write_text(
        json.dumps(
            {
                **common_manifest,
                "schema": "selfless_multimodal_likelihood_evaluation_v5",
                "project_formal_protocol": True,
                "records": {
                    "mmbench_dev_en": 4_329,
                    "seed_bench_image": 14_233,
                    "sugarcrepe": 7_511,
                    "aro_vg_relation": 23_937,
                    "aro_vg_attribution": 28_748,
                },
                **benchmark_scoring,
            }
        ),
        encoding="utf-8",
    )
    (native_root / "summary.json").write_text(
        json.dumps(
            {
                "schema": (
                    "selfless_imagenet1k_zeroshot_classification_summary_v2"
                ),
                "task": "imagenet1k_zeroshot_classification",
                "checkpoint_step": 58_000,
                "runtime_hashing_enabled": False,
                "project_formal_protocol": True,
                "complete_formal_target": True,
                "records": 50_000,
                "classes": 1_000,
                "formal_target_records": 50_000,
                "language_prior_image_count": 50_000,
                "class_text_template": "a photo of a {class_name}.",
                "accuracy_unit": "unit_interval",
                "top_1_accuracy": 0.42,
                "top_5_accuracy": 0.68,
                "primary_metric": "top_1_accuracy",
                "scoring": {
                    "primary_candidate_score": (
                        "language_prior_debiased_mean_token_loglikelihood"
                    ),
                    "language_prior_alpha": 1.0,
                    "language_prior_estimator": "candidate_image_logmeanexp",
                },
            }
        ),
        encoding="utf-8",
    )
    benchmark_tasks = {}
    for task, value in (
        ("mmbench_dev_en", 0.38),
        ("seed_bench_image", 0.38),
        ("sugarcrepe", 0.63),
        ("aro_vg_relation", 0.71),
        ("aro_vg_attribution", 0.88),
    ):
        records = {
            "mmbench_dev_en": 4_329,
            "seed_bench_image": 14_233,
            "sugarcrepe": 7_511,
            "aro_vg_relation": 23_937,
            "aro_vg_attribution": 28_748,
        }[task]
        if task == "mmbench_dev_en":
            metrics = {
                "records": records,
                "primary_metric": "circular_accuracy_language_prior_debiased",
                "circular_accuracy_language_prior_debiased": value,
            }
        elif task == "seed_bench_image":
            metrics = {
                "records": records,
                "primary_metric": "accuracy_language_prior_debiased",
                "accuracy_language_prior_debiased": value,
            }
        elif task.startswith("aro_vg_"):
            metrics = {"records": records, "primary_metric": "conditional_pairwise.win_rate",
                       "conditional_pairwise": {"win_rate": value}}
        else:
            metrics = {
                "records": records,
                "primary_metric": "language_prior_debiased_pairwise.win_rate",
                "language_prior_debiased_pairwise": {"win_rate": value},
            }
        benchmark_tasks[task] = {"task": task, "metrics": metrics}
    (benchmark_root / "summary.json").write_text(
        json.dumps(
            {
                "schema": "selfless_multimodal_likelihood_summary_v5",
                "project_formal_protocol": True,
                "checkpoint_step": 58_000,
                "runtime_hashing_enabled": False,
                "accuracy_and_rate_unit": "unit_interval",
                "scoring": benchmark_scoring,
                "tasks": benchmark_tasks,
            }
        ),
        encoding="utf-8",
    )
    for root, task, value in (
        (coco_root, "mscoco_karpathy_test_5k", 0.21),
        (flickr_root, "flickr30k_karpathy_test_1k", 0.31),
    ):
        images, captions, caption_distribution = {
            "mscoco_karpathy_test_5k": (
                5_000,
                25_010,
                {"5": 4_990, "6": 10},
            ),
            "flickr30k_karpathy_test_1k": (1_000, 5_000, {"5": 1_000}),
        }[task]

        def direction(queries, candidates):
            return {
                "queries": queries,
                "candidates": candidates,
                "mean_rank": 1.0,
                "median_rank": 1.0,
                "recall_at_1": value,
                "recall_at_5": value,
                "recall_at_10": value,
                "mean_recall_at_1_5_10": value,
            }

        (root / "manifest.json").write_text(
            json.dumps(
                {
                    **common_manifest,
                    "schema": "selfless_cross_dataset_retrieval_evaluation_v3",
                    "task": task,
                    "project_formal_protocol": True,
                }
            ),
            encoding="utf-8",
        )
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "selfless_cross_dataset_retrieval_summary_v3",
                    "task": task,
                    "checkpoint_step": 58_000,
                    "runtime_hashing_enabled": False,
                    "project_formal_protocol": True,
                    "complete_formal_target": True,
                    "images": images,
                    "captions": captions,
                    "caption_count_distribution": caption_distribution,
                    "recall_unit": "unit_interval",
                    "rank_unit": "one_based_candidate_rank",
                    "coco_five_fold_1k_average": False,
                    "primary_metric": "mean_recall_at_1_5_10",
                    "mean_recall_at_1_5_10": value,
                    "image_to_text": direction(images, captions),
                    "text_to_image": direction(captions, images),
                    "scoring": {
                        "primary_candidate_score": (
                            "language_prior_debiased_mean_token_loglikelihood"
                        ),
                        "language_prior_alpha": 1.0,
                        "language_prior_estimator": "candidate_image_logmeanexp",
                    },
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_pretraining_native_understanding.py",
            "--checkpoint",
            str(checkpoint),
            "--imagenet_classification_root",
            str(native_root),
            "--coco_retrieval_root",
            str(coco_root),
            "--flickr30k_retrieval_root",
            str(flickr_root),
            "--benchmark_root",
            str(benchmark_root),
            "--output_dir",
            str(output),
        ],
    )
    native_summary.main()
    report = json.loads(
        (output / "pretraining_native_understanding_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["complete"] is True
    assert report["global_step"] == 58_000
    assert report["primary_metrics"]["imagenet1k_zeroshot_top_1_accuracy"] == 0.42
    assert report["primary_metrics"]["mscoco_karpathy_test_5k"] == 0.21
    assert report["primary_metrics"]["sugarcrepe"] == 0.63
    assert "caption_hard_negatives" not in report
    assert report["selection_contract"]["language_prior_alpha"] == 1.0
    assert report["selection_contract"]["hard_negative_language_prior_null_images"] == 3
    assert set(report["internal_ablation_diagnostics"]) == {
        "mmbench_dev_en",
        "seed_bench_image",
    }
