import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from scripts.evaluate_qwen_showo_fid_is import (
    build_expected_real_metadata,
    feature_metadata,
    load_fixed_val_records,
    load_manifest,
    load_synset_names,
    metric_transform_metadata,
)
from scripts.evaluate_single_stream_fid_is import (
    is_official_flow_protocol,
    load_shared_original_real_stats,
    parse_args,
    shared_feature_moments,
    validate_strategies,
)


def test_formal_flow_evaluator_defaults_to_single_token_decoding():
    with patch.object(sys, "argv", ["evaluate_single_stream_fid_is.py"]):
        assert parse_args().parallel_rate == 1


def test_official_flow_protocol_requires_single_token_decoding():
    settings = {
        "shared_real_count": 10_000,
        "samples": 10_000,
        "is_splits": 10,
    }
    assert is_official_flow_protocol(**settings, parallel_rate=1)
    assert not is_official_flow_protocol(**settings, parallel_rate=4)


def test_formal_flow_evaluator_rejects_oracle_sigma_orders():
    validate_strategies(["spatial_halton"], allow_sigma_strategies=False)
    for strategy in ("sigma", "sigma_replay", "causal_sigma"):
        try:
            validate_strategies([strategy], allow_sigma_strategies=False)
        except ValueError as error:
            assert strategy in str(error)
        else:
            raise AssertionError(f"oracle strategy {strategy!r} was accepted")


def test_shared_feature_moments_loads_cached_protocol_stats():
    payload = {
        "stats": {
            "count": 3,
            "sum": torch.tensor([1.0, 2.0], dtype=torch.float64),
            "outer_sum": torch.tensor(
                [[4.0, 5.0], [5.0, 6.0]], dtype=torch.float64
            ),
        }
    }
    moments = shared_feature_moments(payload, feature=2, device="cpu")
    assert moments.count.item() == 3
    assert torch.equal(moments.sum, payload["stats"]["sum"])
    assert torch.equal(moments.outer_sum, payload["stats"]["outer_sum"])


def test_flow_evaluator_strictly_loads_shared_protocol_stats(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    split_manifest = tmp_path / "split.jsonl"
    mapping = tmp_path / "mapping.txt"
    weights = tmp_path / "inception.pth"
    stats_path = tmp_path / "stats.pt"
    weights.write_bytes(b"fixed-inception-weights")
    mapping.write_text("n00000001 class one\nn00000002 class two\n")

    manifest_rows = [
        {
            "img_id": 1,
            "synset": "n00000001",
            "source_path": "/missing/n00000001/a.JPEG",
        },
        {
            "img_id": 2,
            "synset": "n00000002",
            "source_path": "/missing/n00000002/b.JPEG",
        },
    ]
    split_rows = [
        {
            "img_id": 1,
            "synset": "n00000001",
            "split": "validation",
            "split_index": 0,
        },
        {
            "img_id": 2,
            "synset": "n00000002",
            "split": "validation",
            "split_index": 1,
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows)
    )
    split_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in split_rows)
    )
    selected = load_fixed_val_records(
        load_manifest(manifest),
        split_manifest,
        load_synset_names(mapping),
        expected_classes=2,
        expected_samples_per_class=1,
    )
    metadata = build_expected_real_metadata(
        manifest_path=manifest,
        split_manifest_path=split_manifest,
        selected_records=selected,
        transform=metric_transform_metadata(256),
        feature=feature_metadata(2, weights),
        val_samples_per_class=1,
        split_seed=42,
    )
    torch.save(
        {
            "metadata": metadata,
            "stats": {
                "count": 2,
                "sum": torch.zeros(2, dtype=torch.float64),
                "outer_sum": torch.eye(2, dtype=torch.float64),
            },
        },
        stats_path,
    )
    config = OmegaConf.create(
        {
            "dataset": {
                "params": {
                    "manifest_jsonl": str(manifest),
                    "split_manifest_jsonl": str(split_manifest),
                    "synset_mapping_path": str(mapping),
                    "num_classes": 2,
                    "val_samples_per_class": 1,
                    "split_seed": 42,
                }
            }
        }
    )
    loaded = load_shared_original_real_stats(
        str(stats_path),
        config=config,
        fid_feature=2,
        real_image_size=256,
        inception_weights_path=str(weights),
    )
    assert loaded["metadata"]["selection_sha256"] == metadata["selection_sha256"]

    config.dataset.params.split_seed = 7
    try:
        load_shared_original_real_stats(
            str(stats_path),
            config=config,
            fid_feature=2,
            real_image_size=256,
            inception_weights_path=str(weights),
        )
    except ValueError as error:
        assert "split" in str(error)
    else:
        raise AssertionError("mismatched split metadata was accepted")
