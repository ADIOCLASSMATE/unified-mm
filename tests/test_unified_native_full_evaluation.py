import json
import sys

from safetensors.torch import save_file
import torch

from scripts import summarize_unified_native_checkpoint_trend as native_trend
from scripts import summarize_unified_native_full_evaluation as native_full


def _retrieval(records, i2t, t2i):
    def direction(r1):
        return {
            "instance_recall_at_1": r1,
            "instance_recall_at_5": min(1.0, r1 + 0.1),
            "instance_recall_at_10": min(1.0, r1 + 0.2),
        }

    return {
        "complete_formal_target": True,
        "records": records,
        "normalized_loglikelihood": {
            "image_to_text": direction(i2t),
            "text_to_image": direction(t2i),
            "mean_bidirectional_instance_recall_at_1": (i2t + t2i) / 2,
        },
    }


def _benchmark(task, records, value):
    metrics = {
        "records": records,
        "primary_metric": "accuracy_normalized_loglikelihood",
        "accuracy_normalized_loglikelihood": value,
    }
    if task == "mmbench_dev_en":
        metrics.update(
            {
                "primary_metric": "circular_accuracy_normalized_loglikelihood",
                "circular_accuracy_normalized_loglikelihood": value,
                "vanilla_accuracy_normalized_loglikelihood": value + 0.01,
            }
        )
    if task == "sugarcrepe":
        metrics["categories"] = {
            "add_att": {
                "accuracy_normalized_loglikelihood": value + 0.02,
            }
        }
    return {"task": task, "metrics": metrics}


def _standard_retrieval(images, i2t, t2i):
    def direction(r1):
        return {
            "recall_at_1": r1,
            "recall_at_5": min(1.0, r1 + 0.1),
            "recall_at_10": min(1.0, r1 + 0.2),
        }

    return {
        "complete_formal_target": True,
        "images": images,
        "captions": images * 5,
        "normalized_loglikelihood": {
            "image_to_text": direction(i2t),
            "text_to_image": direction(t2i),
            "mean_bidirectional_recall_at_1": (i2t + t2i) / 2,
        },
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
            "retrieval_1k": 0.6,
            "retrieval_5k": 0.5,
            "mscoco_karpathy_test_5k": 0.35,
            "flickr30k_karpathy_test_1k": 0.55,
            "sugarcrepe": 0.63,
            "aro_vg_relation": 0.71,
            "aro_vg_attribution": 0.88,
        },
        "retrieval_1k": _retrieval(1_000, 0.3, 0.9),
        "retrieval_5k": _retrieval(5_000, 0.2, 0.8),
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
                        "official_protocol": True,
                        "samples": 50_000,
                        "is_split_assignment": "stratified_by_synset",
                        "fid": 5.5,
                        "inception_score_mean": 240.0,
                        "inception_score_std": 3.0,
                    }
                },
                "understanding": {
                    "heldout_validation": {"val/loss": 0.8},
                    "qualitative_captions": "captions.jsonl",
                },
                "pure_text": {
                    "macro_average_primary": 0.46,
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
    assert summary["understanding"]["domains"]["custom_in_domain"] == "imagenet_val"
    assert (
        summary["understanding"]
        ["out_of_domain_general_vlm_benchmarks_in_paper_summary"]
        is False
    )
    assert "pope_coco" in summary["coverage"]["removed_from_protocol"]

    row = native_trend.trend_row(output)
    assert row["global_step"] == step
    assert row["benchmarks"]["sugarcrepe"] == 0.63
    assert row["retrieval_5k"]["t2i_r1"] == 0.8
    assert row["standard_retrieval"]["mscoco_karpathy_test_5k"]["t2i_r1"] == 0.5
