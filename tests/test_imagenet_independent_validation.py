import json

import pytest
import torch
from omegaconf import OmegaConf

from utils.imagenet_flow_dataloaders import build_imagenet_flow_cache_dataloaders
from utils.dataset_imagenet_flow_cache import (
    POSTERIOR_CACHE_FORMAT,
    POSTERIOR_STATS_LAYOUT,
)
from utils.combined_dataloaders import (
    build_unified_image_validation_dataloader,
)


class Tokenizer:
    eos_token_id = 14

    def encode(self, text, add_special_tokens=False):
        return [100 + index for index, _ in enumerate(text.split())]


def _write_dataset(tmp_path, split, names):
    root = tmp_path / split
    root.mkdir()
    cache_path = root / "cache.pt"
    manifest_path = root / "manifest.jsonl"
    captions_path = root / "captions.jsonl"
    means = torch.zeros((len(names), 4, 1), dtype=torch.float16)
    stds = torch.ones_like(means)
    torch.save(
        {
            "posterior_stats": torch.cat((means, stds), dim=-1),
            "img_ids": torch.arange(1, len(names) + 1, dtype=torch.int64),
            "metadata": {
                "format": POSTERIOR_CACHE_FORMAT,
                "stats_layout": POSTERIOR_STATS_LAYOUT,
            },
        },
        cache_path,
    )
    manifest_rows = []
    caption_rows = []
    for index, name in enumerate(names, start=1):
        relative = f"n00000001/{name}.JPEG"
        manifest_rows.append(
            {
                "img_id": index,
                "source_path": f"/dataset/{split}/{relative}",
                "synset": "n00000001",
                "split": split,
            }
        )
        caption_rows.append(
            {
                "img_id": index,
                "path": relative,
                "recaption_short": "test caption",
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    captions_path.write_text(
        "".join(json.dumps(row) + "\n" for row in caption_rows),
        encoding="utf-8",
    )
    return cache_path, manifest_path, captions_path


def _config(tmp_path, *, val_names=("val_1",)):
    train = _write_dataset(tmp_path, "train", ("train_1", "train_2"))
    val = _write_dataset(tmp_path, "val", val_names)
    return OmegaConf.create(
        {
            "model": {
                "boi_token_id": 11,
                "eoi_token_id": 12,
                "mask_token_id": 13,
                "image_tokens_per_img": 4,
                "image_latent_dim": 1,
            },
            "dataset": {
                "class_name": "UnifiedMixedDataset",
                "preprocessing": {"max_seq_length": 16},
                "params": {
                    "cache_path": str(train[0]),
                    "manifest_jsonl": str(train[1]),
                    "caption_jsonl": str(train[2]),
                    "conditioning_mode": "caption",
                    "expected_split": "train",
                    "expected_records": 2,
                    "caption_sequence_modes": ["t2i"],
                    "image_tokens_per_img": 4,
                    "image_latent_dim": 1,
                    "max_seq_length": 16,
                    "validation": {
                        "cache_path": str(val[0]),
                        "manifest_jsonl": str(val[1]),
                        "caption_jsonl": str(val[2]),
                        "expected_split": "val",
                        "expected_records": len(val_names),
                    },
                },
            },
            "training": {
                "seed": 42,
                "batch_size": 1,
                "total_batch_size": 1,
                "dataloader_workers": 0,
                "runtime_hashing_enabled": False,
            },
        }
    )


def test_independent_validation_uses_all_train_and_all_val_rows(tmp_path):
    config = _config(tmp_path)
    train_loader, val_loader = build_imagenet_flow_cache_dataloaders(
        config, Tokenizer()
    )

    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1
    assert train_loader.dataset.dataset.dataset_split == "train"
    assert val_loader.dataset.dataset.dataset_split == "val"
    assert train_loader.dataset.dataset._is_training_index(0) is True
    assert val_loader.dataset.dataset._is_training_index(0) is False
    assert set(train_loader.dataset.indices) == {0, 1}
    assert list(val_loader.dataset.indices) == [0]


def test_independent_validation_rejects_image_identity_overlap(tmp_path):
    config = _config(tmp_path, val_names=("train_1",))
    with pytest.raises(ValueError, match="identity overlap"):
        build_imagenet_flow_cache_dataloaders(config, Tokenizer())


def test_unified_offline_validation_opens_only_independent_val(tmp_path):
    config = _config(tmp_path)
    image_params = config.dataset.params
    config.dataset.params = OmegaConf.create({"image": image_params})

    loader = build_unified_image_validation_dataloader(
        config,
        Tokenizer(),
        batch_size=1,
        num_workers=0,
    )

    assert len(loader.dataset) == 2
    assert set(loader.dataset.datasets) == {"t2i", "i2t"}
    assert all(
        dataset.dataset_split == "val"
        for dataset in loader.dataset.datasets.values()
    )
