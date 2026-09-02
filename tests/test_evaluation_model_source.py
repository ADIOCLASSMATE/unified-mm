import json

import pytest
from omegaconf import OmegaConf
from safetensors.torch import save_file
import torch

from scripts.summarize_unified_evaluation import validate_t2i_model_source
from utils.evaluation_model_source import (
    configure_model_source,
    resolve_evaluation_model_source,
)


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
                "state_key_count": 3,
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
    assert config.model.model_path == str(model.resolve())
    assert config.training.from_scratch is False
    assert config.training.use_gradient_checkpointing is False


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
