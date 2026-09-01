import csv
import json
import sys

from PIL import Image
from safetensors.torch import save_file
import torch

from scripts.prepare_multimodal_likelihood_assets import (
    ImageRegistry,
    prepare_svo_probes,
    prepare_whatsup_controlled,
)
from scripts import summarize_pretraining_native_understanding as native_summary


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


def test_pretraining_native_summary_merges_required_components(tmp_path, monkeypatch):
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
        json.dumps(common_manifest), encoding="utf-8"
    )
    (benchmark_root / "manifest.json").write_text(
        json.dumps(common_manifest), encoding="utf-8"
    )
    native_tasks = {
        "retrieval_1k": {
            "primary_metric": "normalized_loglikelihood.mean_bidirectional_instance_recall_at_1",
            "normalized_loglikelihood": {
                "mean_bidirectional_instance_recall_at_1": 0.3
            },
        },
        "retrieval_5k": {
            "primary_metric": "normalized_loglikelihood.mean_bidirectional_instance_recall_at_1",
            "normalized_loglikelihood": {
                "mean_bidirectional_instance_recall_at_1": 0.1
            },
        },
    }
    (native_root / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_step": 58_000,
                "runtime_hashing_enabled": False,
                "tasks": native_tasks,
            }
        ),
        encoding="utf-8",
    )
    benchmark_tasks = {
        task: {
            "task": task,
            "metrics": {
                "primary_metric": "accuracy_normalized_loglikelihood",
                "accuracy_normalized_loglikelihood": value,
            }
        }
        for task, value in (
            ("mmbench_dev_en", 0.38),
            ("seed_bench_image", 0.38),
            ("sugarcrepe", 0.63),
            ("aro_vg_relation", 0.71),
            ("aro_vg_attribution", 0.88),
        )
    }
    (benchmark_root / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_step": 58_000,
                "runtime_hashing_enabled": False,
                "tasks": benchmark_tasks,
            }
        ),
        encoding="utf-8",
    )
    for root, task, value in (
        (coco_root, "mscoco_karpathy_test_5k", 0.21),
        (flickr_root, "flickr30k_karpathy_test_1k", 0.31),
    ):
        (root / "manifest.json").write_text(
            json.dumps({**common_manifest, "task": task}), encoding="utf-8"
        )
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "checkpoint_step": 58_000,
                    "runtime_hashing_enabled": False,
                    "primary_metric": (
                        "normalized_loglikelihood.mean_bidirectional_recall_at_1"
                    ),
                    "normalized_loglikelihood": {
                        "mean_bidirectional_recall_at_1": value
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
            "--imagenet_retrieval_root",
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
    assert "classification" not in report["primary_metrics"]
    assert report["primary_metrics"]["mscoco_karpathy_test_5k"] == 0.21
    assert report["primary_metrics"]["sugarcrepe"] == 0.63
    assert "caption_hard_negatives" not in report
    assert report["selection_contract"]["visual_calibration_enabled"] is False
    assert set(report["internal_ablation_diagnostics"]) == {
        "mmbench_dev_en",
        "seed_bench_image",
    }
