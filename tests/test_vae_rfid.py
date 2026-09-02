from argparse import Namespace
from pathlib import Path

import pytest
import torch

from scripts.evaluate_vae_rfid import (
    CACHE_FORMAT,
    CACHE_LAYOUT,
    REAL_STATS_SCHEMA,
    canonical_cache_metadata,
    load_real_moments,
    stable_posterior_sample,
    validate_full_cache_metadata,
    validate_full_protocol_args,
)


def _cache_metadata() -> dict:
    return {
        "format": CACHE_FORMAT,
        "stats_layout": CACHE_LAYOUT,
        "stats_are_scaled": True,
        "source_mode": "manifest_jsonl",
        "source_manifest_jsonl": "public/datasets/imagenet_full/manifest_val.jsonl",
        "vae": "mar-kl16",
        "vae_dtype": "float16",
        "storage_dtype": "float16",
        "scaling_factor": 0.2325,
        "image_size": 256,
        "posterior_shape": [16, 16, 32],
        "token_shape": [256, 32],
        "runtime_hashing_enabled": False,
        "vae_checkpoint_sha256": "must-not-be-reported",
    }


def test_full_rfid_protocol_is_frozen_to_sample_primary_contract():
    args = Namespace(
        samples=50_000,
        cache_shards=16,
        feature=2048,
        seed=42,
        vae_dtype="fp32",
    )
    validate_full_protocol_args(args, ["sample", "mean"])
    with pytest.raises(ValueError, match="posterior_modes"):
        validate_full_protocol_args(args, ["mean", "sample"])


def test_full_rfid_cache_contract_and_reportable_metadata_are_strict():
    metadata = _cache_metadata()
    validate_full_cache_metadata(metadata, Path("shard.pt"))
    report = canonical_cache_metadata(metadata)
    assert report["runtime_hashing_enabled"] is False
    assert "sha256" not in str(report).lower()

    metadata["source_manifest_jsonl"] = "manifest_train.jsonl"
    with pytest.raises(ValueError, match="validation manifest"):
        validate_full_cache_metadata(metadata, Path("shard.pt"))


def test_real_moments_require_canonical_imagenet_val_metadata(tmp_path):
    path = tmp_path / "moments.pt"
    torch.save(
        {
            "schema": REAL_STATS_SCHEMA,
            "stats": {
                "count": 50_000,
                "sum": torch.zeros(2),
                "outer_sum": torch.eye(2),
            },
            "metadata": {
                "source": {
                    "split": "validation",
                    "classes": 1000,
                    "samples_per_class": 50,
                    "selected_records_sha256": "must-not-be-reported",
                },
                "feature": {
                    "extractor": "torch-fidelity-inception-v3-compat",
                    "feature": 2,
                    "accumulation_dtype": "torch.float32",
                    "weights_sha256": "must-not-be-reported",
                },
                "image_transform": {
                    "resize": 256,
                    "interpolation": "bicubic",
                    "center_crop": 256,
                    "color_mode": "RGB",
                },
            },
        },
        path,
    )
    moments, report = load_real_moments(path, feature=2)
    assert int(moments.count.item()) == 50_000
    assert report["source"] == {
        "split": "validation",
        "classes": 1000,
        "samples_per_class": 50,
    }
    assert "sha256" not in str(report).lower()


def test_posterior_sampling_is_reproducible_by_global_image_id():
    mean = torch.zeros(2, 256, 16)
    std = torch.ones_like(mean)
    image_ids = torch.tensor([1, 17])
    first = stable_posterior_sample(
        mean, std, image_ids, seed=42, storage_dtype=torch.float16
    )
    second = stable_posterior_sample(
        mean, std, image_ids, seed=42, storage_dtype=torch.float16
    )
    changed = stable_posterior_sample(
        mean, std, image_ids, seed=43, storage_dtype=torch.float16
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)

