"""Resolve and load canonical evaluation weight sources.

The unified evaluators support two explicit source formats:

* a final Hugging Face EMA export (the standard final-model path), or
* a legacy rank-sharded EMA checkpoint used by retained checkpoint trends.

The HF path is loaded by ``load_model_tokenizer(...from_pretrained...)`` after
``configure_model_source`` rewrites ``config.model.model_path``.  It must not
be overlaid with rank-sharded checkpoint weights afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


HF_EMA_EXPORT_SCHEMA = "selfless_ema_hf_export_v1"
SHARDED_EMA_SCHEMA = "selfless_rank_sharded_fp32_ema_v1"

_REQUIRED_MODEL_CONTRACT_FIELDS = (
    "architecture_variant",
    "training_objective",
    "dual_stream_attention_contract",
    "image_tokens_per_img",
    "image_latent_dim",
    "image_flow_width",
    "image_flow_depth",
    "image_flow_batch_mul",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


@dataclass(frozen=True)
class EvaluationModelSource:
    path: Path
    kind: str
    global_step: int
    world_size: int
    metadata: dict[str, Any]

    @property
    def is_hf_final_ema(self) -> bool:
        return self.kind == "hf_final_ema"

    def report(self) -> dict[str, Any]:
        report = {
            "kind": self.kind,
            "path": str(self.path),
            "global_step": self.global_step,
            "world_size": self.world_size,
        }
        if self.is_hf_final_ema:
            report.update(
                {
                    "loaded_via": "from_pretrained",
                    "floating_dtype": self.metadata.get("floating_dtype"),
                    "state_key_count": self.metadata.get("state_key_count"),
                    "stored_weight_key_count": self.metadata.get(
                        "stored_weight_key_count",
                        self.metadata.get("state_key_count"),
                    ),
                }
            )
        else:
            report["loaded_via"] = "rank_sharded_ema_merge"
        return report


def resolve_evaluation_model_source(path: str | Path) -> EvaluationModelSource:
    source = Path(path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)

    export_metadata_path = source / "ema_export_metadata.json"
    if export_metadata_path.is_file():
        for required in (
            source / "config.json",
            source / "model.safetensors",
            source / "tokenizer.json",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)
        metadata = _read_json(export_metadata_path)
        if metadata.get("schema") != HF_EMA_EXPORT_SCHEMA:
            raise ValueError(
                "unsupported final EMA export schema: "
                f"{metadata.get('schema')!r}"
            )
        step = int(metadata.get("source_global_step", -1))
        world_size = int(metadata.get("source_world_size", -1))
        if step < 0 or world_size <= 0:
            raise ValueError("final EMA export has invalid source provenance")
        if str(metadata.get("floating_dtype")) != "float32":
            raise ValueError("formal final evaluation requires the FP32 EMA export")
        if int(metadata.get("state_key_count", 0)) <= 0:
            raise ValueError("final EMA export has no recorded state keys")
        # ``state_key_count`` describes the merged PyTorch state dict.  A
        # safetensors export stores only one member of each tied-weight alias
        # group (for this model, lm_head.weight is tied to embed_tokens), so
        # its physical key count can legitimately be smaller.  New exports
        # record both counts; legacy exports recorded only the physical count.
        source_state_key_count = int(metadata["state_key_count"])
        stored_weight_key_count = int(
            metadata.get("stored_weight_key_count", source_state_key_count)
        )
        if stored_weight_key_count <= 0:
            raise ValueError("final EMA export has no recorded stored weight keys")
        if stored_weight_key_count > source_state_key_count:
            raise ValueError(
                "final EMA stored weight key count exceeds source state key count"
            )
        if str(metadata.get("export_kind")) != "training":
            raise ValueError("formal final evaluation requires a training EMA export")
        from safetensors import safe_open

        with safe_open(
            str(source / "model.safetensors"), framework="pt", device="cpu"
        ) as handle:
            keys = list(handle.keys())
            if len(keys) != stored_weight_key_count:
                raise ValueError(
                    "final EMA safetensors stored key count disagrees with metadata"
                )
            dtypes = {handle.get_slice(key).get_dtype() for key in keys}
        if dtypes != {"F32"}:
            raise ValueError(f"final EMA safetensors must be FP32; got {sorted(dtypes)}")
        required_prefixes = (
            "model.image_token_embedder.",
            "image_flow_condition_proj.",
            "image_flow_head.",
        )
        if str(metadata.get("architecture_variant", "")).strip().lower() == "dynamic_xt":
            # Export metadata from older checkpoints may not carry the model
            # contract, so the authoritative HF config is checked below too.
            required_prefixes += ("model.backbone_flow_time_embedder.",)
        hf_config = _read_json(source / "config.json")
        if hf_config.get("architecture_variant") == "selfless_siglip":
            required_prefixes += ("model.semantic_encoder.", "model.semantic_input_proj.", "model.image_fusion.")
        if hf_config.get("architecture_variant") == "showo2_unified":
            required_prefixes = ("model.image_token_embedder.", "model.image_time_embedder.", "image_flow_head.")
            if hf_config.get("s2_use_siglip"):
                required_prefixes += ("model.semantic_encoder.", "model.semantic_input_proj.", "model.image_fusion.")
        if str(hf_config.get("architecture_variant", "")).strip().lower() == "dynamic_xt":
            required_prefixes += ("model.backbone_flow_time_embedder.",)
        required_prefixes = tuple(dict.fromkeys(required_prefixes))
        missing_prefixes = [
            prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in keys)
        ]
        if missing_prefixes:
            raise ValueError(
                "final EMA export lacks multimodal weights: "
                f"{missing_prefixes}"
            )
        return EvaluationModelSource(
            path=source,
            kind="hf_final_ema",
            global_step=step,
            world_size=world_size,
            metadata=metadata,
        )

    complete = _read_json(source / "checkpoint_complete.json")
    metadata = _read_json(source / "metadata.json")
    manifest = _read_json(source / "ema_manifest.json")
    if manifest.get("schema") != SHARDED_EMA_SCHEMA:
        raise ValueError(
            "unsupported sharded EMA schema: " f"{manifest.get('schema')!r}"
        )
    steps = {
        int(complete["global_step"]),
        int(metadata["global_step"]),
        int((manifest.get("runtime") or {})["global_step"]),
    }
    if len(steps) != 1:
        raise ValueError(f"checkpoint step fields disagree: {sorted(steps)}")
    world_size = int(manifest.get("world_size", metadata.get("world_size", -1)))
    if world_size <= 0:
        raise ValueError("sharded EMA checkpoint has invalid world size")
    return EvaluationModelSource(
        path=source,
        kind="rank_sharded_ema",
        global_step=steps.pop(),
        world_size=world_size,
        metadata=manifest,
    )


def add_model_source_argument(parser) -> None:
    parser.add_argument(
        "--model_source",
        type=Path,
        required=True,
        help=(
            "Final HF EMA export or rank-sharded EMA checkpoint. The source "
            "metadata selects the model variant and attention contract."
        ),
    )


def model_source_from_args(args) -> EvaluationModelSource:
    path = getattr(args, "model_source", None)
    if path is None:
        raise ValueError("--model_source is required")
    return resolve_evaluation_model_source(path)


def _apply_checkpoint_model_contract(config, saved_model, *, label: str) -> None:
    if not isinstance(saved_model, dict):
        raise ValueError(f"{label} lacks config_contract.model")
    saved_architecture = str(saved_model.get("architecture_variant", ""))
    for field in _REQUIRED_MODEL_CONTRACT_FIELDS:
        if field not in saved_model:
            raise ValueError(f"{label} config lacks required field: {field}")
        saved_value = saved_model[field]
        if field in {
            "architecture_variant",
            "dual_stream_attention_contract",
        } or (saved_architecture == "showo2_unified" and field == "training_objective"):
            # One neutral evaluation YAML serves every ablation. The weight
            # source owns both implementation identity and the backbone A/B
            # mask switch; parameters alone cannot recover either distinction.
            # The separate flow-head mask contract is restored below.
            config.model[field] = str(saved_value)
            continue
        if (
            field == "image_flow_width"
            and saved_architecture == "positionwise_flow_head_on_b"
        ):
            if int(saved_value) != 1936:
                raise ValueError(
                    f"{label} ablation-F image_flow_width must be 1936, "
                    f"got {saved_value!r}"
                )
            config.model[field] = int(saved_value)
            continue
        if field in {"image_flow_width", "image_flow_depth"}:
            # Model capacity belongs to the checkpoint. The shared evaluation
            # YAML supplies dataset/scoring defaults for every scaling arm.
            if isinstance(saved_value, bool) or not isinstance(saved_value, int) or saved_value <= 0:
                raise ValueError(f"{label} has invalid {field}={saved_value!r}")
            config.model[field] = saved_value
            continue
        expected = config.model.get(field)
        if expected is not None and str(saved_value) != str(expected):
            raise ValueError(
                f"{label}/config mismatch for {field}: "
                f"checkpoint={saved_value!r}, evaluation={expected!r}"
            )
        config.model[field] = saved_value

    # Flow-head query/content masks were split after the first A/B runs. A
    # missing field means historical shared-strict behavior for contextual
    # heads. F has no attention at all, so its legacy missing value is N/A.
    default_flow_head_attention_contract = (
        "not_applicable"
        if saved_architecture == "positionwise_flow_head_on_b"
        else "selfless_strict"
    )
    flow_head_attention_contract = str(
        saved_model.get(
            "flow_head_attention_contract",
            default_flow_head_attention_contract,
        )
    ).strip().lower()
    if saved_architecture == "selfless_siglip":
        for field, value in saved_model.items():
            if field.startswith("b_siglip_"):
                config.model[field] = value
    if saved_architecture == "showo2_unified":
        valid_flow_head_contract = flow_head_attention_contract == "showo2_omni_attention"
        for field, value in saved_model.items():
            if field.startswith("s2_"):
                config.model[field] = value
    elif saved_architecture == "positionwise_flow_head_on_b":
        valid_flow_head_contract = flow_head_attention_contract == "not_applicable"
    else:
        valid_flow_head_contract = flow_head_attention_contract in {
            "selfless_strict",
            "xlnet_content_diagonal",
        }
    if not valid_flow_head_contract:
        raise ValueError(
            f"{label} has invalid flow_head_attention_contract="
            f"{flow_head_attention_contract!r} for architecture_variant="
            f"{saved_architecture!r}"
        )
    config.model.flow_head_attention_contract = flow_head_attention_contract
    # AdaLN query/content conditions were split after the original A/B runs.
    # As with the attention contract above, missing checkpoint metadata is an
    # explicit legacy numerical contract rather than permission for the
    # evaluation YAML to select newer behavior.
    default_flow_condition_contract = (
        "not_applicable"
        if saved_architecture == "positionwise_flow_head_on_b"
        else "backbone_xt_shared_query_content"
    )
    flow_condition_contract = str(
        saved_model.get(
            "flow_condition_contract",
            default_flow_condition_contract,
        )
    ).strip().lower()
    valid_flow_condition_contracts = (
        {"backbone_noisy_image_hidden"}
        if saved_architecture == "showo2_unified"
        else {"not_applicable"}
        if saved_architecture == "positionwise_flow_head_on_b"
        else {
            "backbone_xt_shared_query_content",
            "backbone_xt_query_backbone_x0_content",
        }
    )
    if flow_condition_contract not in valid_flow_condition_contracts:
        raise ValueError(
            f"{label} has invalid flow_condition_contract="
            f"{flow_condition_contract!r} for architecture_variant="
            f"{saved_architecture!r}"
        )
    config.model.flow_condition_contract = flow_condition_contract
    if saved_architecture == "positionwise_flow_head_on_b":
        expected_f_fields = {
            "positionwise_reference_flow_width": 1280,
            "positionwise_reference_flow_depth": 8,
            "positionwise_max_parameter_relative_error": 0.005,
        }
        for field, expected in expected_f_fields.items():
            if field not in saved_model:
                raise ValueError(f"{label} ablation-F config lacks {field}")
            value = saved_model[field]
            if float(value) != float(expected):
                raise ValueError(
                    f"{label} ablation-F {field}={value!r}, expected {expected!r}"
                )
            config.model[field] = value
    # New order-sensitive ablations persist the training/generation order in
    # the checkpoint itself.  Keep legacy B/C/D exports compatible by treating
    # this as optional, but make it authoritative whenever present.
    if "training_image_sigma_order" in saved_model:
        order = str(saved_model["training_image_sigma_order"]).lower()
        if order not in {"random", "sequential"}:
            raise ValueError(
                f"{label} has invalid training_image_sigma_order={order!r}"
            )
        config.model.training_image_sigma_order = order
        if config.get("dataset", None) is not None:
            dataset_params = (
                config.dataset.params.image
                if str(config.dataset.class_name) == "UnifiedMixedDataset"
                else config.dataset.params
            )
            dataset_params.image_sigma_order = order
        generation_order = (
            "sequential" if order == "sequential" else "spatial_halton"
        )
        if config.get("experiment", None) is not None:
            config.experiment.validation_single_stream_order_strategies = [
                generation_order
            ]
        if config.get("evaluation", None) is not None:
            config.evaluation.strategies = generation_order


def configure_model_source(config, source: EvaluationModelSource) -> None:
    config.training.from_scratch = False
    config.training.use_gradient_checkpointing = False
    if source.is_hf_final_ema:
        hf_config = _read_json(source.path / "config.json")
        _apply_checkpoint_model_contract(
            config,
            hf_config,
            label="final EMA",
        )
        for field in (
            "mask_token_id",
            "boi_token_id",
            "eoi_token_id",
            "image_mask_token_id",
        ):
            if int(hf_config.get(field, -1)) < 0:
                raise ValueError(f"final EMA config lacks special-token id: {field}")
        config.model.model_path = str(source.path)
        return

    checkpoint_metadata = _read_json(source.path / "metadata.json")
    config_contract = checkpoint_metadata.get("config_contract")
    if not isinstance(config_contract, dict):
        raise ValueError("rank-sharded EMA lacks metadata config_contract")
    _apply_checkpoint_model_contract(
        config,
        config_contract.get("model"),
        label="rank-sharded EMA",
    )


def _validate_dynamic_xt_hf_time_embeddings(
    model, source: EvaluationModelSource
) -> dict[str, Any] | None:
    """Fail closed if HF post-load initialization changed D's learned state.

    Compare values directly after the requested model-dtype cast; no hashing,
    parameter mutation, or second state-dict overlay is involved.
    """
    saved_config = _read_json(source.path / "config.json")
    if saved_config.get("architecture_variant") != "dynamic_xt":
        return None
    if getattr(getattr(model, "config", None), "architecture_variant", None) != "dynamic_xt":
        raise RuntimeError("Dynamic-XT HF export loaded into a different architecture")

    import torch
    from safetensors import safe_open

    names = [
        f"model.backbone_flow_time_embedder.mlp.{layer}.{kind}"
        for layer in (0, 2)
        for kind in ("weight", "bias")
    ]
    with safe_open(
        str(source.path / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        stored_keys = set(handle.keys())
        for name in names:
            if name not in stored_keys:
                raise RuntimeError(f"Dynamic-XT HF export lacks required parameter: {name}")
            try:
                parameter = model.get_parameter(name)
            except (AttributeError, KeyError) as exc:
                raise RuntimeError(f"loaded Dynamic-XT model lacks parameter: {name}") from exc
            if parameter.is_meta:
                raise RuntimeError(f"Dynamic-XT parameter is still on meta device: {name}")
            actual = parameter.detach().cpu()
            expected = handle.get_tensor(name).to(dtype=actual.dtype)
            if not torch.isfinite(expected).all() or not torch.equal(actual, expected):
                raise RuntimeError(
                    "Dynamic-XT post-load weight mismatch: "
                    f"{name}; refusing evaluation with altered time embeddings"
                )
    return {
        "schema": "dynamic_xt_hf_time_embedding_values_v1",
        "complete": True,
        "comparison": "exact_after_model_dtype_cast",
        "checked_parameters": names,
        "runtime_hashing_enabled": False,
    }


def load_model_source_weights(model, source: EvaluationModelSource) -> dict[str, Any]:
    if source.is_hf_final_ema:
        # ``configure_model_source`` made load_model_tokenizer call
        # ``from_pretrained(source.path)``.  Loading a second state here would
        # defeat the final-export contract and can mishandle tied aliases.
        report = source.report()
        validation = _validate_dynamic_xt_hf_time_embeddings(model, source)
        if validation is not None:
            report["post_load_validation"] = validation
        return report

    from utils.sharded_ema import load_sharded_ema_checkpoint

    loaded = load_sharded_ema_checkpoint(model, source.path)
    report = source.report()
    report.update(loaded)
    return report
