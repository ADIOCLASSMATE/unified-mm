import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_qwen_showo_fid_is import (
    PROTOCOL_NAME,
    build_expected_real_metadata,
    build_fixed_val_records,
    deterministic_sample_seed,
    frechet_distance,
    load_fixed_val_records,
    metric_transform_metadata,
    resolve_original_image_path,
    selection_fingerprint,
    validate_protocol_settings,
    validate_real_stats_metadata,
)


class QwenShowOFIDHelperTests(unittest.TestCase):
    def _records(self):
        records = []
        for class_index in range(100):
            synset = f"n{class_index:08d}"
            for image_index in range(4):
                records.append(
                    {
                        "manifest_index": len(records),
                        "img_id": class_index * 10 + image_index,
                        "synset": synset,
                        "source_path": f"/old/train/{synset}/{synset}_{image_index}.JPEG",
                    }
                )
        names = {f"n{index:08d}": f"class {index}" for index in range(100)}
        return records, names

    def test_fixed_split_is_balanced_and_deterministic(self):
        records, names = self._records()
        first = build_fixed_val_records(
            records, names, val_samples_per_class=2, split_seed=42
        )
        second = build_fixed_val_records(
            records, names, val_samples_per_class=2, split_seed=42
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 200)
        counts = {}
        for index, record in enumerate(first):
            counts[record["synset"]] = counts.get(record["synset"], 0) + 1
            self.assertEqual(record["evaluation_index"], index)
            self.assertEqual(record["prompt"], names[record["synset"]])
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(selection_fingerprint(first), selection_fingerprint(second))

    def test_per_sample_seed_is_stable_and_record_specific(self):
        records, names = self._records()
        selected = build_fixed_val_records(
            records, names, val_samples_per_class=1, split_seed=7
        )
        seeds = [deterministic_sample_seed(123, row) for row in selected]
        self.assertEqual(
            seeds, [deterministic_sample_seed(123, row) for row in selected]
        )
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertNotEqual(
            seeds, [deterministic_sample_seed(124, row) for row in selected]
        )

    def test_authoritative_split_manifest_uses_split_index_order(self):
        records, names = self._records()
        split_rows = []
        validation_index = 0
        train_index = 0
        for record in records:
            image_number = int(record["img_id"]) % 10
            if image_number < 2:
                split = "validation"
                split_index = validation_index
                validation_index += 1
            else:
                split = "train"
                split_index = train_index
                train_index += 1
            split_rows.append(
                {
                    "img_id": record["img_id"],
                    "synset": record["synset"],
                    "split": split,
                    "split_index": split_index,
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                # Deliberately reverse file order: split_index is authoritative.
                for row in reversed(split_rows):
                    handle.write(json.dumps(row) + "\n")
            selected = load_fixed_val_records(
                records,
                path,
                names,
                expected_samples_per_class=2,
            )
        self.assertEqual(len(selected), 200)
        self.assertEqual(
            [row["evaluation_index"] for row in selected], list(range(200))
        )
        self.assertEqual(selected[0]["img_id"], 0)
        self.assertEqual(selected[-1]["img_id"], 991)

    def test_real_metadata_validation_detects_manifest_and_transform_changes(self):
        records, names = self._records()
        selected = build_fixed_val_records(
            records, names, val_samples_per_class=1, split_seed=42
        )
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.jsonl"
            with manifest.open("w", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row) + "\n")
            feature = {
                "backend": "torchmetrics.NoTrainInceptionV3/torch-fidelity",
                "extractor_antialias": True,
                "feature": 2048,
                "feature_name": "2048",
                "logits_name": "logits_unbiased",
                "weights_sha256": "a" * 64,
                "weights_filename": "inception.pth",
                "weights_path": "/tmp/inception.pth",
                "software": {
                    "torch": "2.8.0",
                    "torchmetrics": "1.9.0",
                    "torch-fidelity": "0.4.0",
                },
            }
            expected = build_expected_real_metadata(
                manifest_path=manifest,
                split_manifest_path=manifest,
                selected_records=selected,
                transform=metric_transform_metadata(256),
                feature=feature,
                val_samples_per_class=1,
                split_seed=42,
            )
            self.assertEqual(expected["protocol"], PROTOCOL_NAME)
            validate_real_stats_metadata(expected, expected)

            wrong_transform = json.loads(json.dumps(expected))
            wrong_transform["transform"]["resize"]["size"] = 299
            with self.assertRaisesRegex(ValueError, "transform"):
                validate_real_stats_metadata(wrong_transform, expected)

            wrong_manifest = json.loads(json.dumps(expected))
            wrong_manifest["manifest_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                validate_real_stats_metadata(wrong_manifest, expected)

            wrong_split = json.loads(json.dumps(expected))
            wrong_split["split_manifest_sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "split_manifest_sha256"):
                validate_real_stats_metadata(wrong_split, expected)

            wrong_weights = json.loads(json.dumps(expected))
            wrong_weights["feature"]["weights_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "weights_sha256"):
                validate_real_stats_metadata(wrong_weights, expected)

    def test_original_image_path_rebases_stale_absolute_manifest_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            synset = "n00000001"
            target = root / synset / "sample.JPEG"
            target.parent.mkdir()
            target.write_bytes(b"image")
            record = {
                "img_id": 1,
                "synset": synset,
                "source_path": f"/missing/old/train/{synset}/sample.JPEG",
            }
            self.assertEqual(resolve_original_image_path(record, root), target)

    def test_frechet_distance_is_zero_for_identical_gaussians(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in the lightweight test interpreter")
        generator = torch.Generator().manual_seed(7)
        matrix = torch.randn(8, 8, generator=generator, dtype=torch.float64)
        covariance = matrix @ matrix.T + torch.eye(8, dtype=torch.float64)
        mean = torch.randn(8, generator=generator, dtype=torch.float64)
        self.assertLess(
            abs(frechet_distance(mean, covariance, mean, covariance)), 1.0e-8
        )

    def test_fixed_protocol_rejects_metric_drift(self):
        validate_protocol_settings(
            split_seed=42,
            val_samples_per_class=100,
            image_size=256,
            fid_feature=2048,
        )
        with self.assertRaisesRegex(ValueError, "image_size=299"):
            validate_protocol_settings(
                split_seed=42,
                val_samples_per_class=100,
                image_size=299,
                fid_feature=2048,
            )


if __name__ == "__main__":
    unittest.main()
