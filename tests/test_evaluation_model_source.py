import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from safetensors.torch import save_file
import torch

from scripts.summarize_unified_evaluation import validate_t2i_model_source
from utils.evaluation_model_source import (
    EvaluationModelSource,
    configure_model_source,
    load_model_source_weights,
    resolve_evaluation_model_source,
)


def _dynamic_post_load_fixture(tmp_path, dtype):
    (tmp_path / "config.json").write_text(
        json.dumps({"architecture_variant": "dynamic_xt"}), encoding="utf-8"
    )
    stored = {
        f"model.backbone_flow_time_embedder.mlp.{layer}.{kind}":
        (torch.arange(4).reshape(2, 2).float() + 0.123 + layer)
        if kind == "weight" else torch.tensor([0.123, 0.456])
        for layer in (0, 2) for kind in ("weight", "bias")
    }
    save_file(stored, tmp_path / "model.safetensors")
    loaded = {name: value.to(dtype).clone() for name, value in stored.items()}
    model = SimpleNamespace(
        config=SimpleNamespace(architecture_variant="dynamic_xt"),
        get_parameter=lambda name: loaded[name],
    )
    source = EvaluationModelSource(tmp_path, "hf_final_ema", 95415, 64, {})
    return model, source, loaded, stored


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dynamic_post_load_guard_checks_values_without_overwriting(tmp_path, dtype):
    model, source, loaded, _ = _dynamic_post_load_fixture(tmp_path, dtype)
    expected = {name: value.clone() for name, value in loaded.items()}
    report = load_model_source_weights(model, source)
    validation = report["post_load_validation"]
    assert validation["complete"] is True
    assert validation["comparison"] == "exact_after_model_dtype_cast"
    assert validation["runtime_hashing_enabled"] is False
    assert set(validation["checked_parameters"]) == set(loaded)
    for name in loaded:
        torch.testing.assert_close(loaded[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["changed", "missing_model", "missing_export", "nonfinite", "wrong_architecture"])
def test_dynamic_post_load_guard_fails_closed(tmp_path, failure):
    model, source, loaded, stored = _dynamic_post_load_fixture(tmp_path, torch.bfloat16)
    name = "model.backbone_flow_time_embedder.mlp.0.weight"
    if failure == "changed":
        loaded[name].add_(1)
    elif failure == "missing_model":
        loaded.pop(name)
    elif failure == "missing_export":
        stored.pop(name)
        save_file(stored, tmp_path / "model.safetensors")
    elif failure == "nonfinite":
        loaded[name].fill_(float("nan"))
    else:
        model.config.architecture_variant = "selfless_contextual"
    with pytest.raises(RuntimeError, match="Dynamic-XT"):
        load_model_source_weights(model, source)


def test_final_hf_ema_source_uses_export_provenance(tmp_path):
    model = tmp_path / "hf_model-final-ema"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architecture_variant": "selfless_contextual",
                "training_objective": "selfless_dual_stream",
                "dual_stream_attention_contract": "selfless_strict",
                "image_tokens_per_img": 256,
                "image_latent_dim": 16,
                "image_flow_width": 1280,
                "image_flow_depth": 8,
                "image_flow_batch_mul": 1,
                "mask_token_id": 1,
                "boi_token_id": 2,
                "eoi_token_id": 3,
                "image_mask_token_id": 4,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.image_token_embedder.weight": torch.zeros(1),
            "image_flow_condition_proj.weight": torch.zeros(1),
            "image_flow_head.weight": torch.zeros(1),
        },
        model / "model.safetensors",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "ema_export_metadata.json").write_text(
        json.dumps(
            {
                "schema": "selfless_ema_hf_export_v1",
                "floating_dtype": "float32",
                "source_global_step": 95_415,
                "source_world_size": 64,
                # The source state contains a tied alias while safetensors
                # stores only one physical tensor for that alias group.
                "state_key_count": 4,
                "stored_weight_key_count": 3,
                "export_kind": "training",
            }
        ),
        encoding="utf-8",
    )

    source = resolve_evaluation_model_source(model)
    config = OmegaConf.create(
        {
            "model": {
                "model_path": "base",
                "architecture_variant": "selfless_contextual",
                "training_objective": "selfless_dual_stream",
                "dual_stream_attention_contract": "selfless_strict",
                "image_tokens_per_img": 256,
                "image_latent_dim": 16,
                "image_flow_width": 1280,
                "image_flow_depth": 8,
                "image_flow_batch_mul": 1,
            },
            "training": {
                "from_scratch": True,
                "use_gradient_checkpointing": True,
            },
        }
    )
    configure_model_source(config, source)

    assert source.kind == "hf_final_ema"
    assert source.global_step == 95_415
    assert source.report()["state_key_count"] == 4
    assert source.report()["stored_weight_key_count"] == 3
    assert config.model.model_path == str(model.resolve())
    assert config.training.from_scratch is False
    assert config.training.use_gradient_checkpointing is False
    assert config.model.flow_head_attention_contract == "selfless_strict"
    assert config.model.flow_condition_contract == (
        "backbone_xt_shared_query_content"
    )


def test_final_hf_ema_source_rejects_stored_key_count_mismatch(tmp_path):
    model = tmp_path / "hf_model-final-ema"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    save_file({"weight": torch.zeros(1)}, model / "model.safetensors")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "ema_export_metadata.json").write_text(
        json.dumps(
            {
                "schema": "selfless_ema_hf_export_v1",
                "floating_dtype": "float32",
                "source_global_step": 1,
                "source_world_size": 1,
                "state_key_count": 2,
                "stored_weight_key_count": 2,
                "export_kind": "training",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stored key count disagrees"):
        resolve_evaluation_model_source(model)


def test_raw_hf_export_without_ema_provenance_is_rejected(tmp_path):
    model = tmp_path / "hf_model-final"
    model.mkdir()
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        (model / filename).write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="checkpoint_complete"):
        resolve_evaluation_model_source(model)


def test_dynamic_hf_export_requires_backbone_time_embedder_weights(tmp_path):
    model = tmp_path / "hf_model-final-ema"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architecture_variant": "dynamic_xt",
                "training_objective": "selfless_dual_stream",
                "dual_stream_attention_contract": "xlnet_content_diagonal",
                "image_tokens_per_img": 256,
                "image_latent_dim": 16,
                "image_flow_width": 1280,
                "image_flow_depth": 8,
                "image_flow_batch_mul": 4,
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            "model.image_token_embedder.weight": torch.zeros(1),
            "image_flow_condition_proj.weight": torch.zeros(1),
            "image_flow_head.weight": torch.zeros(1),
        },
        model / "model.safetensors",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "ema_export_metadata.json").write_text(
        json.dumps(
            {
                "schema": "selfless_ema_hf_export_v1",
                "floating_dtype": "float32",
                "source_global_step": 95_415,
                "source_world_size": 64,
                "state_key_count": 3,
                "export_kind": "training",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="backbone_flow_time_embedder"):
        resolve_evaluation_model_source(model)


def test_hf_checkpoint_architecture_is_authoritative(tmp_path):
    model = tmp_path / "hf_model-final-ema"
    model.mkdir()
    hf_config = {
        "architecture_variant": "single_stream_text_ar",
        "training_objective": "selfless_dual_stream",
        "dual_stream_attention_contract": "selfless_strict",
        "image_tokens_per_img": 256,
        "image_latent_dim": 16,
        "image_flow_width": 1280,
        "image_flow_depth": 8,
        "image_flow_batch_mul": 1,
        "mask_token_id": 1,
        "boi_token_id": 2,
        "eoi_token_id": 3,
        "image_mask_token_id": 4,
    }
    (model / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
    save_file(
        {
            "model.image_token_embedder.weight": torch.zeros(1),
            "image_flow_condition_proj.weight": torch.zeros(1),
            "image_flow_head.weight": torch.zeros(1),
        },
        model / "model.safetensors",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "ema_export_metadata.json").write_text(
        json.dumps(
            {
                "schema": "selfless_ema_hf_export_v1",
                "floating_dtype": "float32",
                "source_global_step": 95_415,
                "source_world_size": 64,
                "state_key_count": 3,
                "export_kind": "training",
            }
        ),
        encoding="utf-8",
    )
    config = _evaluation_config()

    configure_model_source(config, resolve_evaluation_model_source(model))

    assert config.model.architecture_variant == "single_stream_text_ar"


def _evaluation_config():
    return OmegaConf.create(
        {
            "model": {
                "model_path": "base",
                "architecture_variant": "selfless_contextual",
                "training_objective": "selfless_dual_stream",
                "dual_stream_attention_contract": "selfless_strict",
                "image_tokens_per_img": 256,
                "image_latent_dim": 16,
                "image_flow_width": 1280,
                "image_flow_depth": 8,
                "image_flow_batch_mul": 1,
                "flow_condition_contract": (
                    "backbone_xt_query_backbone_x0_content"
                ),
            },
            "training": {
                "from_scratch": True,
                "use_gradient_checkpointing": True,
            },
        }
    )


def _write_sharded_source(path, *, model_contract=None):
    path.mkdir()
    (path / "checkpoint_complete.json").write_text(
        json.dumps({"global_step": 10}), encoding="utf-8"
    )
    metadata = {"global_step": 10, "world_size": 64}
    if model_contract is not None:
        metadata["config_contract"] = {"model": model_contract}
    (path / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (path / "ema_manifest.json").write_text(
        json.dumps(
            {
                "schema": "selfless_rank_sharded_fp32_ema_v1",
                "world_size": 64,
                "runtime": {"global_step": 10},
            }
        ),
        encoding="utf-8",
    )


def _text_ar_model_contract():
    return {
        "architecture_variant": "single_stream_text_ar",
        "training_objective": "selfless_dual_stream",
        "dual_stream_attention_contract": "selfless_strict",
        "image_tokens_per_img": 256,
        "image_latent_dim": 16,
        "image_flow_width": 1280,
        "image_flow_depth": 8,
        "image_flow_batch_mul": 1,
    }


def _ordered_evaluation_config():
    config = _evaluation_config()
    config.dataset = OmegaConf.create(
        {
            "class_name": "UnifiedMixedDataset",
            "params": {"image": {"image_sigma_order": "random"}},
        }
    )
    config.experiment = OmegaConf.create(
        {"validation_single_stream_order_strategies": ["spatial_halton"]}
    )
    config.evaluation = OmegaConf.create({"strategies": "spatial_halton"})
    return config


def test_sharded_checkpoint_architecture_is_authoritative(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    _write_sharded_source(
        checkpoint,
        model_contract=_text_ar_model_contract(),
    )
    config = _evaluation_config()

    source = resolve_evaluation_model_source(checkpoint)
    configure_model_source(config, source)

    assert source.kind == "rank_sharded_ema"
    assert config.model.architecture_variant == "single_stream_text_ar"
    assert config.model.model_path == "base"


def test_checkpoint_attention_contract_is_authoritative(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    model_contract = _text_ar_model_contract()
    model_contract["architecture_variant"] = "selfless_contextual"
    model_contract["dual_stream_attention_contract"] = (
        "xlnet_content_diagonal"
    )
    _write_sharded_source(checkpoint, model_contract=model_contract)
    config = _evaluation_config()

    configure_model_source(
        config,
        resolve_evaluation_model_source(checkpoint),
    )

    assert config.model.architecture_variant == "selfless_contextual"
    assert (
        config.model.dual_stream_attention_contract
        == "xlnet_content_diagonal"
    )
    # Missing is deliberately not inferred from the backbone: every legacy
    # A/B flow head used the shared strict mask.
    assert config.model.flow_head_attention_contract == "selfless_strict"
    assert config.model.flow_condition_contract == (
        "backbone_xt_shared_query_content"
    )


def test_checkpoint_flow_head_attention_contract_is_authoritative(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    model_contract = _text_ar_model_contract()
    model_contract.update(
        {
            "architecture_variant": "selfless_contextual",
            "dual_stream_attention_contract": "xlnet_content_diagonal",
            "flow_head_attention_contract": "xlnet_content_diagonal",
        }
    )
    _write_sharded_source(checkpoint, model_contract=model_contract)
    config = _evaluation_config()

    configure_model_source(
        config,
        resolve_evaluation_model_source(checkpoint),
    )

    assert config.model.flow_head_attention_contract == (
        "xlnet_content_diagonal"
    )


def test_checkpoint_flow_condition_contract_is_authoritative(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    model_contract = _text_ar_model_contract()
    model_contract.update(
        {
            "architecture_variant": "selfless_contextual",
            "dual_stream_attention_contract": "xlnet_content_diagonal",
            "flow_head_attention_contract": "xlnet_content_diagonal",
            "flow_condition_contract": (
                "backbone_xt_query_backbone_x0_content"
            ),
        }
    )
    _write_sharded_source(checkpoint, model_contract=model_contract)
    config = _evaluation_config()

    configure_model_source(
        config,
        resolve_evaluation_model_source(checkpoint),
    )

    assert config.model.flow_condition_contract == (
        "backbone_xt_query_backbone_x0_content"
    )


@pytest.mark.parametrize("depth,width", [(8, 1936), (16, 1960), (30, 1968)])
@pytest.mark.parametrize("kind", ["rank_sharded_ema", "hf_final_ema"])
def test_f_checkpoint_owns_parameter_matched_width_and_reference_contract(
    tmp_path, depth, width, kind,
):
    checkpoint = tmp_path / "checkpoint-10"
    model_contract = _text_ar_model_contract()
    model_contract.update(
        {
            "architecture_variant": "positionwise_flow_head_on_b",
            "dual_stream_attention_contract": "xlnet_content_diagonal",
            "flow_head_attention_contract": "not_applicable",
            "image_flow_width": width,
            "image_flow_depth": depth,
            "image_flow_batch_mul": 4,
            "positionwise_reference_flow_width": 1280,
            "positionwise_reference_flow_depth": depth,
            "positionwise_max_parameter_relative_error": 0.005,
            "training_image_sigma_order": "random",
        }
    )
    if kind == "rank_sharded_ema":
        _write_sharded_source(checkpoint, model_contract=model_contract)
        source = resolve_evaluation_model_source(checkpoint)
    else:
        checkpoint.mkdir()
        model_contract.update(mask_token_id=1, boi_token_id=2, eoi_token_id=3, image_mask_token_id=4)
        (checkpoint / "config.json").write_text(json.dumps(model_contract))
        source = EvaluationModelSource(checkpoint, kind, 10, 64, {})
    config = _ordered_evaluation_config()
    config.model.image_flow_batch_mul = 4

    configure_model_source(config, source)

    assert config.model.architecture_variant == "positionwise_flow_head_on_b"
    assert config.model.flow_head_attention_contract == "not_applicable"
    assert config.model.flow_condition_contract == "not_applicable"
    assert config.model.image_flow_width == width
    assert config.model.image_flow_depth == depth
    assert config.model.positionwise_reference_flow_width == 1280
    assert config.model.positionwise_reference_flow_depth == depth
    assert config.model.positionwise_max_parameter_relative_error == 0.005


def test_legacy_f_checkpoint_without_attention_field_is_not_applicable(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint-10"
    model_contract = _text_ar_model_contract()
    model_contract.update(
        {
            "architecture_variant": "positionwise_flow_head_on_b",
            "dual_stream_attention_contract": "xlnet_content_diagonal",
            "image_flow_width": 1936,
            "image_flow_batch_mul": 4,
            "positionwise_reference_flow_width": 1280,
            "positionwise_reference_flow_depth": 8,
            "positionwise_max_parameter_relative_error": 0.005,
        }
    )
    _write_sharded_source(checkpoint, model_contract=model_contract)
    config = _ordered_evaluation_config()
    config.model.image_flow_batch_mul = 4

    configure_model_source(config, resolve_evaluation_model_source(checkpoint))

    assert config.model.flow_head_attention_contract == "not_applicable"
    assert config.model.flow_condition_contract == "not_applicable"


def test_checkpoint_image_contract_mismatch_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    model_contract = _text_ar_model_contract()
    model_contract["image_latent_dim"] = 32
    _write_sharded_source(checkpoint, model_contract=model_contract)

    with pytest.raises(ValueError, match="image_latent_dim"):
        configure_model_source(
            _evaluation_config(),
            resolve_evaluation_model_source(checkpoint),
        )


@pytest.mark.parametrize("depth", [16, 30])
def test_neutral_evaluation_config_uses_checkpoint_flow_capacity(tmp_path, depth):
    checkpoint = tmp_path / "checkpoint-10"
    model_contract = _text_ar_model_contract()
    model_contract.update(image_flow_depth=depth, image_flow_width=1536)
    _write_sharded_source(checkpoint, model_contract=model_contract)
    config = _evaluation_config()
    configure_model_source(config, resolve_evaluation_model_source(checkpoint))
    assert config.model.image_flow_depth == depth
    assert config.model.image_flow_width == 1536
    assert config.model.image_latent_dim == 16


@pytest.mark.parametrize("depth", [0, -1, True, 16.5])
def test_evaluation_rejects_invalid_checkpoint_flow_capacity(tmp_path, depth):
    checkpoint = tmp_path / "checkpoint-10"
    model_contract = _text_ar_model_contract()
    model_contract['image_flow_depth'] = depth
    _write_sharded_source(checkpoint, model_contract=model_contract)
    with pytest.raises(ValueError, match='invalid image_flow_depth'):
        configure_model_source(_evaluation_config(), resolve_evaluation_model_source(checkpoint))


def test_sharded_checkpoint_without_model_contract_is_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    _write_sharded_source(checkpoint)

    with pytest.raises(ValueError, match="config_contract"):
        configure_model_source(
            _evaluation_config(),
            resolve_evaluation_model_source(checkpoint),
        )


def test_t2i_summary_accepts_unified_sharded_source_report(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    _write_sharded_source(
        checkpoint,
        model_contract=_text_ar_model_contract(),
    )
    source = resolve_evaluation_model_source(checkpoint)
    identity = source.report()
    load_report = {
        **identity,
        "ema_checkpoint": str(source.path),
        "keys": 3,
        "missing": [],
        "unexpected": [],
    }

    validate_t2i_model_source(
        {
            "weight_source": source.kind,
            "evaluation_model_source": identity,
            "model_source_load": load_report,
        },
        source,
    )


def test_t2i_summary_rejects_inconsistent_unified_load_report(tmp_path):
    checkpoint = tmp_path / "checkpoint-10"
    _write_sharded_source(
        checkpoint,
        model_contract=_text_ar_model_contract(),
    )
    source = resolve_evaluation_model_source(checkpoint)
    identity = source.report()
    load_report = {
        **identity,
        "ema_checkpoint": str(tmp_path / "checkpoint-other"),
    }

    with pytest.raises(ValueError, match="sharded-EMA load report"):
        validate_t2i_model_source(
            {
                "weight_source": source.kind,
                "evaluation_model_source": identity,
                "model_source_load": load_report,
            },
            source,
        )
