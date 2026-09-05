import json
import sys

from safetensors.torch import save_file
import torch

from scripts import summarize_unified_native_checkpoint_trend as native_trend
from scripts import summarize_unified_native_full_evaluation as native_full


def _benchmark(task, records, value):
    metrics = {
        "records": records,
        "primary_metric": "accuracy_language_prior_debiased",
        "accuracy_language_prior_debiased": value,
    }
    if task == "mmbench_dev_en":
        metrics.update(
            {
                "primary_metric": "circular_accuracy_language_prior_debiased",
                "circular_accuracy_language_prior_debiased": value,
                "vanilla_accuracy_language_prior_debiased": value + 0.01,
            }
        )
    if task in {"sugarcrepe", "aro_vg_relation", "aro_vg_attribution"}:
        metrics.update(
            {
                "primary_metric": "language_prior_debiased_pairwise.win_rate",
                "language_prior_debiased_pairwise": {"win_rate": value},
            }
        )
    if task == "sugarcrepe":
        metrics["categories"] = {
            "add_att": {
                "language_prior_debiased_pairwise": {
                    "win_rate": value + 0.02,
                },
            }
        }
    return {"task": task, "metrics": metrics}


def _standard_retrieval(images, i2t, t2i):
    def direction(r1):
        return {
            "queries": images,
            "candidates": captions,
            "recall_at_1": r1,
            "recall_at_5": min(1.0, r1 + 0.1),
            "recall_at_10": min(1.0, r1 + 0.2),
            "mean_recall_at_1_5_10": min(1.0, r1 + 0.1),
            "mean_rank": 2.0,
            "median_rank": 1.0,
        }

    captions = 25_010 if images == 5_000 else 5_000
    image_to_text = direction(i2t)
    text_to_image = direction(t2i)
    text_to_image["queries"] = captions
    text_to_image["candidates"] = images
    return {
        "complete_formal_target": True,
        "images": images,
        "captions": captions,
        "caption_count_distribution": (
            {"5": 4_990, "6": 10}
            if images == 5_000
            else {"5": 1_000}
        ),
        "primary_metric": "mean_recall_at_1_5_10",
        "recall_unit": "unit_interval",
        "rank_unit": "one_based_candidate_rank",
        "coco_five_fold_1k_average": False,
        "scoring": {
            "primary_candidate_score": (
                "language_prior_debiased_mean_token_loglikelihood"
            ),
            "language_prior_alpha": 1.0,
            "language_prior_estimator": "candidate_image_logmeanexp",
        },
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "mean_recall_at_1_5_10": (
            image_to_text["mean_recall_at_1_5_10"]
            + text_to_image["mean_recall_at_1_5_10"]
        )
        / 2,
    }


def _native_summary(checkpoint, step):
    benchmarks = {
        "mmbench_dev_en": _benchmark("mmbench_dev_en", 4_329, 0.38),
        "seed_bench_image": _benchmark("seed_bench_image", 14_233, 0.38),
        "sugarcrepe": _benchmark("sugarcrepe", 7_511, 0.63),
        "aro_vg_relation": _benchmark("aro_vg_relation", 23_937, 0.71),
        "aro_vg_attribution": _benchmark(
            "aro_vg_attribution", 28_748, 0.88
        ),
    }
    return {
        "schema": "pretraining_native_understanding_summary_v5",
        "complete": True,
        "runtime_hashing_enabled": False,
        "checkpoint": str(checkpoint.resolve()),
        "global_step": step,
        "weight_source": "hf_final_ema",
        "dataset_contract": {
            "image_training_split": "imagenet_train",
            "image_evaluation_split": "imagenet_val",
            "train_validation_overlap_allowed": False,
        },
        "primary_metrics": {
            "imagenet1k_zeroshot_top_1_accuracy": 0.25,
            "imagenet1k_zeroshot_top_5_accuracy": 0.5,
            "mscoco_karpathy_test_5k": 0.45,
            "flickr30k_karpathy_test_1k": 0.65,
            "sugarcrepe": 0.63,
            "aro_vg_relation": 0.71,
            "aro_vg_attribution": 0.88,
        },
        "imagenet1k_zeroshot_classification": {
            "complete_formal_target": True,
            "records": 50_000,
            "classes": 1_000,
            "formal_target_records": 50_000,
            "language_prior_image_count": 50_000,
            "primary_metric": "top_1_accuracy",
            "accuracy_unit": "unit_interval",
            "class_text_template": "a photo of a {class_name}.",
            "scoring": {
                "primary_candidate_score": (
                    "language_prior_debiased_mean_token_loglikelihood"
                ),
                "language_prior_alpha": 1.0,
                "language_prior_estimator": "candidate_image_logmeanexp",
            },
            "top_1_accuracy": 0.25,
            "top_5_accuracy": 0.5,
        },
        "standard_cross_dataset_retrieval": {
            "mscoco_karpathy_test_5k": _standard_retrieval(5_000, 0.2, 0.5),
            "flickr30k_karpathy_test_1k": _standard_retrieval(1_000, 0.4, 0.7),
        },
        "paper_compositional_benchmarks": {
            task: benchmarks[task]
            for task in ("sugarcrepe", "aro_vg_relation", "aro_vg_attribution")
        },
        "internal_ablation_diagnostics": {
            task: benchmarks[task]
            for task in ("mmbench_dev_en", "seed_bench_image")
        },
        "selection_contract": {
            "imagenet_validation_images": 50_000,
            "image_text_matching_score_variant": (
                "language_prior_debiased_mean_token_loglikelihood_only"
            ),
            "language_prior_alpha": 1.0,
            "dense_retrieval_language_prior_estimator": (
                "candidate_image_logmeanexp"
            ),
            "hard_negative_language_prior_estimator": (
                "content_free_gaussian_image_logmeanexp"
            ),
            "hard_negative_language_prior_null_images": 3,
        },
    }


def test_selected_native_full_summary_and_trend(tmp_path, monkeypatch):
    step = 58_000
    checkpoint = tmp_path / "hf_model-final-ema"
    core = tmp_path / "core"
    native = tmp_path / "native"
    output = tmp_path / "output"
    checkpoint.mkdir()
    core.mkdir()
    native.mkdir()
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
                "source_global_step": step,
                "source_world_size": 64,
                "state_key_count": 3,
                "export_kind": "training",
            }
        ),
        encoding="utf-8",
    )
    (core / "full_evaluation_summary.json").write_text(
        json.dumps(
            {
                "schema": "unified_full_checkpoint_evaluation_summary_v4",
                "complete": True,
                "profile": "formal",
                "runtime_hashing_enabled": False,
                "checkpoint": str(checkpoint.resolve()),
                "global_step": step,
                "weight_source": "hf_final_ema",
                "dataset_contract": {
                    "training_split": "imagenet_train",
                    "evaluation_split": "imagenet_val",
                },
                "generation": {
                    "imagenet_val_t2i": {
                        "project_formal_protocol": True,
                        "leaderboard_comparable_to_adm_dit": False,
                        "protocol_name": (
                            "imagenet_val_fid50k_torch_fidelity_stratified_is"
                        ),
                        "reference_distribution": "imagenet_val_50000",
                        "comparison_scope": "same_protocol_only",
                        "not_adm_dit_reason": (
                            "validation_reference_and_pytorch_torch_fidelity_extractor"
                        ),
                        "fid_reducer": "symmetric_eigendecomposition",
                        "strategy": "spatial_halton",
                        "samples": 50_000,
                        "is_split_assignment": "stratified_by_synset",
                        "fid": 5.5,
                        "inception_score_mean": 240.0,
                        "inception_score_std": 3.0,
                        "inception_score_splits": [237.0, 243.0] * 5,
                    }
                },
                "understanding": {
                    "heldout_validation": {"val/loss": 0.8},
                    "qualitative_captions": "captions.jsonl",
                },
                "pure_text": {
                    "macro_average_primary": 0.46,
                    "macro_average_role": "internal_cross_task_summary_only",
                    "primary_metrics": {"arc_easy": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )
    (native / "pretraining_native_understanding_summary.json").write_text(
        json.dumps(_native_summary(checkpoint, step)), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_unified_native_full_evaluation.py",
            "--checkpoint",
            str(checkpoint),
            "--core_output_root",
            str(core),
            "--native_output_root",
            str(native),
            "--output_root",
            str(output),
            "--profile",
            "formal",
        ],
    )
    native_full.main()
    summary = json.loads(
        (output / "native_full_evaluation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["complete"] is True
    assert summary["understanding"]["domains"]["zero_shot_classification"] == (
        "imagenet_val_50k"
    )
    assert (
        summary["understanding"]
        ["out_of_domain_general_vlm_benchmarks_in_paper_summary"]
        is False
    )
    assert "pope_coco" in summary["coverage"]["removed_from_protocol"]

    row = native_trend.trend_row(output)
    assert row["global_step"] == step
    assert row["benchmarks"]["sugarcrepe"] == 0.63
    assert row["imagenet1k_zeroshot"]["top_1_accuracy"] == 0.25
    assert row["standard_retrieval"]["mscoco_karpathy_test_5k"]["t2i_r1"] == 0.5
