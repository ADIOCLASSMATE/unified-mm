import json
import sys
from argparse import Namespace
from collections.abc import Mapping as CollectionsMapping
from datetime import timedelta
from unittest.mock import patch

import pytest
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
    EVALUATOR_RNG_CONTRACT,
    EVALUATOR_RNG_CONTRACT_SHA256,
    DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS,
    Mapping as EvaluatorMapping,
    build_canonical_initial_noise_bank,
    canonical_json_sha256,
    evaluation_pairing_manifest_hashes,
    evaluation_process_group_timeout_seconds,
    image_embedder_training_artifacts,
    is_official_flow_protocol,
    load_shared_original_real_stats,
    parse_args,
    require_finite_generated_latents,
    require_finite_metric_scalar,
    require_finite_metric_tensor,
    shared_feature_moments,
    validate_image_embedder_ablation_protocol,
    validate_strategies,
)
from scripts import evaluate_single_stream_fid_is as evaluator
from scripts.archive.image_backbone_ablation.image_embedder_ablation_matrix import build_ablation_config
from scripts.archive.image_backbone_ablation.image_embedder_confirmation_protocol import (
    CONFIRMATION_PROVENANCE_SCHEMA,
    canonical_sha256,
)


def test_formal_flow_evaluator_imports_mapping_for_final_aggregation():
    assert EvaluatorMapping is CollectionsMapping


def test_formal_flow_evaluator_has_permanent_process_group_timeout(monkeypatch):
    monkeypatch.delenv("EVAL_PROCESS_GROUP_TIMEOUT_SECONDS", raising=False)
    assert (
        evaluation_process_group_timeout_seconds()
        == DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS
        == 7200
    )
    monkeypatch.setenv("EVAL_PROCESS_GROUP_TIMEOUT_SECONDS", "9000")
    assert evaluation_process_group_timeout_seconds() == 9000
    monkeypatch.setenv("EVAL_PROCESS_GROUP_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        evaluation_process_group_timeout_seconds()


def test_distributed_evaluator_installs_the_permanent_timeout(monkeypatch):
    for key, value in {
        "RANK": "0",
        "WORLD_SIZE": "2",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "2",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("EVAL_PROCESS_GROUP_TIMEOUT_SECONDS", raising=False)
    with (
        patch.object(evaluator.dist, "is_initialized", return_value=False),
        patch.object(evaluator.dist, "init_process_group") as init_process_group,
    ):
        evaluator.init_distributed("cpu")
    assert init_process_group.call_args.kwargs == {
        "backend": "gloo",
        "timeout": timedelta(seconds=7200),
    }


def test_formal_flow_evaluator_defaults_to_single_token_decoding():
    with patch.object(sys, "argv", ["evaluate_single_stream_fid_is.py"]):
        assert parse_args().parallel_rate == 1


def test_canonical_initial_noise_is_batch_and_rank_invariant():
    indices = [7, 19]
    torch.manual_seed(123)
    expected_next_global_random = torch.randn(3)
    torch.manual_seed(123)
    combined, combined_records = build_canonical_initial_noise_bank(
        indices,
        evaluation_seed=42,
    )
    actual_next_global_random = torch.randn(3)
    assert combined.shape == (2, 256, 16)
    assert len(combined_records) == 2
    assert torch.equal(actual_next_global_random, expected_next_global_random)

    separately_built = torch.cat(
        [
            build_canonical_initial_noise_bank(
                [index],
                evaluation_seed=42,
            )[0]
            for index in indices
        ],
        dim=0,
    )
    assert torch.equal(separately_built, combined)
    assert EVALUATOR_RNG_CONTRACT_SHA256 == canonical_json_sha256(
        EVALUATOR_RNG_CONTRACT
    )


def test_pairing_manifest_hashes_are_order_independent_but_identity_sensitive():
    noise = [
        {"global_sample_index": 1, "canonical_noise_sha256": "b" * 64},
        {"global_sample_index": 0, "canonical_noise_sha256": "a" * 64},
    ]
    samples = [
        {"global_sample_index": 1, "image_id": 101},
        {"global_sample_index": 0, "image_id": 100},
    ]
    first = evaluation_pairing_manifest_hashes(
        local_noise_records=noise,
        local_sample_records=samples,
        evaluation_seed=42,
        expected_samples=2,
    )
    second = evaluation_pairing_manifest_hashes(
        local_noise_records=list(reversed(noise)),
        local_sample_records=list(reversed(samples)),
        evaluation_seed=42,
        expected_samples=2,
    )
    assert first == second

    changed_samples = [dict(record) for record in samples]
    changed_samples[0]["image_id"] = 999
    changed = evaluation_pairing_manifest_hashes(
        local_noise_records=noise,
        local_sample_records=changed_samples,
        evaluation_seed=42,
        expected_samples=2,
    )
    assert (
        changed["ordered_eval_sample_manifest_sha256"]
        != first["ordered_eval_sample_manifest_sha256"]
    )


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


def test_formal_flow_evaluator_rejects_nonfinite_generated_latents():
    require_finite_generated_latents(
        torch.zeros(2, 16, 16, 16),
        strategy="spatial_halton",
        rank=0,
        batch_idx=3,
        global_indices=[0, 1],
    )
    bad = torch.zeros(2, 16, 16, 16)
    bad[1, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite generated image latents"):
        require_finite_generated_latents(
            bad,
            strategy="spatial_halton",
            rank=0,
            batch_idx=3,
            global_indices=[0, 1],
        )


def test_formal_flow_evaluator_rejects_nonfinite_metric_tensors_and_scalars():
    require_finite_metric_tensor(torch.zeros(2, 3), label="finite")
    assert require_finite_metric_scalar(1.25, label="finite") == 1.25

    with pytest.raises(FloatingPointError, match="non-finite tensor"):
        require_finite_metric_tensor(
            torch.tensor([[0.0, float("inf")]]),
            label="decoded_generated_images",
        )
    with pytest.raises(FloatingPointError, match="non-finite formal evaluation scalar"):
        require_finite_metric_scalar(float("nan"), label="fid")


def _image_embedder_protocol_args(**overrides):
    values = {
        "require_official_protocol": True,
        "device": "cuda",
        "model_dtype": "bf16",
        "seed": 42,
        "batch_size": 512,
        "samples": 10_000,
        "split": "val",
        "sampling_steps": "100",
        "temperature": 1.0,
        "cfg": 3.5,
        "cfg_schedule": "constant",
        "flow_solver": "heun",
        "parallel_rate": 1,
        "vae_dtype": "fp32",
        "fid_feature": 2048,
        "is_splits": 10,
        "adapter": "none",
        "model_state": "",
        "ema_state": "",
        "allow_sigma_strategies": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_image_embedder_protocol_requires_exact_controls_and_single_strategy():
    args = _image_embedder_protocol_args()
    validate_image_embedder_ablation_protocol(
        args,
        ["spatial_halton"],
        world_size=8,
    )

    with pytest.raises(ValueError, match="strategies"):
        validate_image_embedder_ablation_protocol(
            args,
            ["spatial_uniform", "spatial_halton"],
            world_size=8,
        )
    with pytest.raises(ValueError, match="cfg"):
        validate_image_embedder_ablation_protocol(
            _image_embedder_protocol_args(cfg=4.0),
            ["spatial_halton"],
            world_size=8,
        )


def test_image_embedder_training_artifacts_require_final_step_and_ema(tmp_path):
    config = build_ablation_config("E0", 42)
    checkpoint = tmp_path / "checkpoint-35920"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"global_step": 35_920}),
        encoding="utf-8",
    )
    (checkpoint / "ema_state.pt").write_bytes(b"ema")
    model = tmp_path / "hf_model-final-ema"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")

    protocol = image_embedder_training_artifacts(
        config,
        config_path=str(tmp_path / "config.yaml"),
        model_path=str(model),
    )
    assert protocol["training_seed"] == 42
    assert protocol["final_global_step"] == 35_920
    assert protocol["artifacts"]["ema_state_size_bytes"] == 3

    (checkpoint / "metadata.json").write_text(
        json.dumps({"global_step": 35_919}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="global_step"):
        image_embedder_training_artifacts(
            config,
            config_path=str(tmp_path / "config.yaml"),
            model_path=str(model),
        )


def test_confirmation_training_artifacts_bind_checkpoint_and_hf_provenance(tmp_path):
    config = build_ablation_config(
        "E0",
        43,
        confirmation_screen_json=(
            "output/image_backbone_ablation/evidence/"
            "screening_and_confirmation/expanded_seed42_summary.json"
        ),
    )
    provenance_path = tmp_path / "confirmation_training_provenance.json"
    provenance = {
        "schema": CONFIRMATION_PROVENANCE_SCHEMA,
        "ablation_id": "E0",
        "training_seed": 43,
        "space_to_depth_factor": 1,
        "confirmation_declaration": OmegaConf.to_container(
            config.experiment.confirmation_protocol,
            resolve=True,
        ),
        "confirmation_declaration_sha256": str(
            config.experiment.confirmation_protocol.declaration_sha256
        ),
        "initial_state": {
            "contract": {"schema": "test_init_v1"},
            "image_modules": {
                "parameter_count": 3,
                "parameter_schema_sha256": "a" * 64,
                "state_sha256": "b" * 64,
                "parameters": [],
            },
            "special_token_names_and_ids": [["image_mask", 8]],
            "special_token_rows_sha256": "c" * 64,
        },
        "train_data": {
            "contract": {"schema": "test_order_v1"},
            "dataloader_shuffle_seed": 43,
            "initial_generator_state_sha256": "d" * 64,
            "dataloader_base_seed": 123,
            "dataset_length": 90_000,
            "epoch0_ordered_sample_identity_sha256": "e" * 64,
            "augmentation_contract": {"schema": "test_aug_v1"},
            "epoch0_augmentation_decisions_sha256": "f" * 64,
            "augmentation_seed": 43,
            "latent_hflip_probability": 0.5,
            "batch_size_per_rank": 32,
            "total_batch_size": 256,
            "drop_last": True,
            "num_workers": 8,
            "persistent_workers": True,
            "input_files": {},
        },
        "base_model": {"files": [], "manifest_sha256": "1" * 64},
        "runtime_source": {"files": [], "manifest_sha256": "2" * 64},
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    config.experiment.confirmation_provenance_path = str(provenance_path)
    config.experiment.confirmation_provenance_sha256 = provenance[
        "provenance_sha256"
    ]

    checkpoint = tmp_path / "checkpoint-35920"
    checkpoint.mkdir()
    checkpoint_binding = {
        "path": str(provenance_path),
        "sha256": provenance["provenance_sha256"],
        "declaration_sha256": str(
            config.experiment.confirmation_protocol.declaration_sha256
        ),
    }
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "global_step": 35_920,
                "confirmation_provenance": checkpoint_binding,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "ema_state.pt").write_bytes(b"ema")
    model = tmp_path / "hf_model-final-ema"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    hf_provenance_path = model / provenance_path.name
    hf_provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    protocol = image_embedder_training_artifacts(
        config,
        config_path=str(tmp_path / "config.yaml"),
        model_path=str(model),
    )
    compact = protocol["confirmation"]["provenance"]
    assert compact["ablation_id"] == "E0"
    assert compact["training_seed"] == 43
    assert compact["provenance_sha256"] == provenance["provenance_sha256"]
    assert (
        compact["initial_state"]["image_modules"]["state_sha256"]
        == "b" * 64
    )
    assert (
        compact["train_data"]["epoch0_ordered_sample_identity_sha256"]
        == "e" * 64
    )
    assert compact["runtime_source_manifest_sha256"] == "2" * 64

    tampered = dict(provenance)
    tampered["training_seed"] = 44
    hf_provenance_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        image_embedder_training_artifacts(
            config,
            config_path=str(tmp_path / "config.yaml"),
            model_path=str(model),
        )


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
