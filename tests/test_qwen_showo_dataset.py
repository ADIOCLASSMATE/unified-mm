import json
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scripts.prepare_imagenet100_showo_vq_tokens import (
    read_manifest,
    resolve_source_path,
)
from utils.dataset_qwen_showo_imagenet import (
    QwenShowOImageNetDataset,
    _build_split_indices,
    build_qwen_showo_generation_batch,
    build_qwen_showo_imagenet_dataloaders,
    collate_qwen_showo_imagenet,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 4

    def __init__(self):
        self._ids = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = 10 + len(self._ids)
            ids.append(self._ids[word])
        return ids


def write_fixture(root: Path, classes=3, samples_per_class=8, image_tokens=4):
    image_ids = []
    tokens = []
    manifest_rows = []
    mapping_rows = []
    image_id = 0
    for class_index in range(classes):
        synset = f"n{class_index:08d}"
        mapping_rows.append(f"{synset} class {class_index}, alternate {class_index}\n")
        for sample_index in range(samples_per_class):
            image_ids.append(image_id)
            tokens.append(
                [
                    (class_index + sample_index + token_index) % 16
                    for token_index in range(image_tokens)
                ]
            )
            manifest_rows.append(
                {
                    "img_id": image_id,
                    "synset": synset,
                    "source_path": f"/old/train/{synset}/{image_id}.JPEG",
                }
            )
            image_id += 1
    torch.save(
        {
            "image_ids": torch.tensor(image_ids, dtype=torch.long),
            "tokens": torch.tensor(tokens, dtype=torch.int16),
        },
        root / "tokens.pt",
    )
    with (root / "manifest.jsonl").open("w") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")
    (root / "mapping.txt").write_text("".join(mapping_rows))
    return root / "tokens.pt", root / "manifest.jsonl", root / "mapping.txt"


def make_dataset(root: Path):
    tokens, manifest, mapping = write_fixture(root)
    tokenizer = FakeTokenizer()
    dataset = QwenShowOImageNetDataset(
        tokens_path=str(tokens),
        manifest_jsonl=str(manifest),
        synset_mapping_path=str(mapping),
        tokenizer=tokenizer,
        t2i_token_id=1,
        boi_token_id=2,
        eoi_token_id=3,
        eos_token_id=tokenizer.eos_token_id,
        image_offset=100,
        image_vocab_size=16,
        image_tokens_per_img=4,
        mmap=False,
    )
    return dataset, tokenizer


class QwenShowOImageNetDatasetTest(unittest.TestCase):
    def test_t2i_only_sequence_uses_unified_image_offset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, tokenizer = make_dataset(Path(tmpdir))
            item = dataset[0]
            class_ids = tokenizer.encode("class 0", add_special_tokens=False)

            self.assertEqual(len(dataset), 24)
            self.assertEqual(
                item["input_ids"].tolist(),
                [1, *class_ids, 2, 100, 101, 102, 103, 3, 4],
            )
            self.assertEqual(int(item["image_start"]), 2 + len(class_ids))
            self.assertEqual(item["image_token_ids"].tolist(), [0, 1, 2, 3])
            self.assertEqual(
                item["token_types"].tolist(),
                [2, 0, 0, 2, 1, 1, 1, 1, 2, 2],
            )

    def test_collator_drops_only_class_text_and_labels_only_masked_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset, _ = make_dataset(Path(tmpdir))
            kwargs = dict(
                pad_token_id=0,
                image_mask_token_id=5,
                cond_dropout_prob=1.0,
                fixed_mask_ratio=0.5,
                pad_to_multiple_of=None,
                mask_seed=42,
            )
            batch = collate_qwen_showo_imagenet(
                [dataset[0], dataset[1]], **kwargs
            )
            repeated = collate_qwen_showo_imagenet(
                [dataset[0], dataset[1]], **kwargs
            )

            self.assertTrue(batch["condition_dropped"].all())
            self.assertTrue(torch.equal(batch["image_starts"], torch.tensor([2, 2])))
            self.assertTrue(torch.equal(batch["input_ids"][:, :2], torch.tensor([[1, 2], [1, 2]])))
            self.assertTrue(
                torch.equal(
                    batch["masked_image_positions"],
                    repeated["masked_image_positions"],
                )
            )
            self.assertTrue(
                torch.equal(
                    batch["labels"] != -100,
                    batch["masked_image_positions"],
                )
            )
            self.assertTrue(
                torch.equal(
                    batch["masked_image_positions"].sum(dim=1),
                    torch.tensor([2, 2]),
                )
            )
            self.assertTrue((batch["labels"][batch["labels"] != -100] >= 100).all())

            attention = batch["attention_mask"]
            minimum = torch.finfo(attention.dtype).min
            # Text/task is causal.
            self.assertEqual(float(attention[0, 0, 0, 1]), minimum)
            # BOI and image queries can see the complete BOI/image/EOI span.
            self.assertEqual(float(attention[0, 0, 2, 6]), 0.0)
            self.assertEqual(float(attention[0, 0, 1, 6]), 0.0)

    def test_stratified_split_matches_115k_train_and_10k_val_contract(self):
        class LargeDataset:
            def __init__(self):
                self.img_ids = torch.arange(125_000)
                self.synsets = {
                    image_id: f"n{image_id // 1250:08d}"
                    for image_id in range(125_000)
                }

            def __len__(self):
                return 125_000

        dataset = LargeDataset()
        train_indices, val_indices = _build_split_indices(
            dataset=dataset,
            val_ratio=0.08,
            seed=42,
            strategy="stratified",
            val_samples_per_class=100,
        )
        self.assertEqual(len(train_indices), 115_000)
        self.assertEqual(len(val_indices), 10_000)
        val_counts = {}
        for index in val_indices:
            synset = dataset.synsets[int(dataset.img_ids[index])]
            val_counts[synset] = val_counts.get(synset, 0) + 1
        self.assertEqual(set(val_counts.values()), {100})
        self.assertEqual(len(val_counts), 100)

    def test_loader_interface_and_deterministic_validation_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tokens, manifest, mapping = write_fixture(root)
            split_manifest = root / "split.jsonl"
            explicit_val_ids = [23, 22, 7, 6, 15, 14]
            split_rows = []
            for split_index, image_id in enumerate(explicit_val_ids):
                split_rows.append(
                    {
                        "split_index": split_index,
                        "img_id": image_id,
                        "synset": f"n{image_id // 8:08d}",
                        "split": "validation",
                    }
                )
            with split_manifest.open("w") as handle:
                for row in reversed(split_rows):
                    handle.write(json.dumps(row) + "\n")
            config = OmegaConf.create(
                {
                    "model": {
                        "t2i_token_id": 1,
                        "boi_token_id": 2,
                        "eoi_token_id": 3,
                        "image_mask_token_id": 5,
                        "image_offset": 100,
                        "image_vocab_size": 16,
                        "image_tokens_per_img": 4,
                    },
                    "dataset": {
                        "params": {
                            "tokens_path": str(tokens),
                            "manifest_jsonl": str(manifest),
                            "synset_mapping_path": str(mapping),
                            "split_manifest_jsonl": str(split_manifest),
                            "image_tokens_per_img": 4,
                            "class_prompt_template": "{}",
                            "val_samples_per_class": 2,
                            "split_strategy": "stratified",
                            "split_seed": 42,
                            "cond_dropout_prob": 1.0,
                            "pad_to_multiple_of": None,
                            "mmap": False,
                        },
                        "preprocessing": {"max_seq_length": 32},
                    },
                    "evaluation": {"image_mask_ratio": 0.25},
                    "training": {
                        "batch_size": 4,
                        "dataloader_workers": 0,
                        "min_masking_rate": 0.0,
                    },
                }
            )
            train_loader, val_loader = build_qwen_showo_imagenet_dataloaders(
                config, FakeTokenizer()
            )
            self.assertEqual(len(train_loader.dataset), 18)
            self.assertEqual(len(val_loader.dataset), 6)
            self.assertEqual(
                list(val_loader.dataset.indices), explicit_val_ids
            )

            first = next(iter(val_loader))
            repeated = next(iter(val_loader))
            self.assertTrue(
                torch.equal(
                    first["masked_image_positions"],
                    repeated["masked_image_positions"],
                )
            )
            self.assertTrue(
                torch.equal(
                    first["masked_image_positions"].sum(dim=1),
                    torch.ones(first["input_ids"].shape[0], dtype=torch.long),
                )
            )
            expected_keys = {
                "input_ids",
                "token_types",
                "labels",
                "attention_mask",
                "image_token_mask",
                "class_ids",
                "sample_ids",
            }
            self.assertTrue(expected_keys.issubset(first))

    def test_generation_batch_has_paired_conditional_and_unconditional_inputs(self):
        tokenizer = FakeTokenizer()
        batch = build_qwen_showo_generation_batch(
            ["class zero", "class one with words"],
            tokenizer,
            t2i_token_id=1,
            boi_token_id=2,
            eoi_token_id=3,
            image_mask_token_id=5,
            image_tokens_per_img=4,
            pad_to_multiple_of=None,
        )
        self.assertEqual(batch["input_ids"].shape, batch["uncond_input_ids"].shape)
        self.assertTrue(torch.equal(batch["uncond_image_starts"], torch.tensor([2, 2])))
        self.assertTrue(
            (
                batch["input_ids"][batch["image_token_mask"]]
                == 5
            ).all()
        )
        self.assertTrue(
            (
                batch["uncond_input_ids"][batch["uncond_image_token_mask"]]
                == 5
            ).all()
        )
        self.assertEqual(
            batch["attention_mask"].shape,
            (
                2,
                1,
                batch["input_ids"].shape[1],
                batch["input_ids"].shape[1],
            ),
        )

    def test_token_prep_relocates_manifest_source_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "Data" / "CLS-LOC" / "train" / "n00000001" / "x.JPEG"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"not decoded in this test")
            resolved = resolve_source_path(
                "/stale/ILSVRC/Data/CLS-LOC/train/n00000001/x.JPEG",
                "n00000001",
                root,
            )
            self.assertEqual(resolved, image_path)

            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "img_id": 7,
                        "synset": "n00000001",
                        "source_path": (
                            "/stale/ILSVRC/Data/CLS-LOC/train/"
                            "n00000001/x.JPEG"
                        ),
                    }
                )
                + "\n"
            )
            rows = read_manifest(manifest, root)
            self.assertEqual(rows[0]["resolved_path"], image_path)
            self.assertEqual(rows[0]["manifest_index"], 0)


if __name__ == "__main__":
    unittest.main()
