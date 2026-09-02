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
        if str(metadata.get("export_kind")) != "training":
            raise ValueError("formal final evaluation requires a training EMA export")
        from safetensors import safe_open

        with safe_open(
            str(source / "model.safetensors"), framework="pt", device="cpu"
        ) as handle:
            keys = list(handle.keys())
            if len(keys) != int(metadata["state_key_count"]):
                raise ValueError("final EMA safetensors key count disagrees with metadata")
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
    for field in _REQUIRED_MODEL_CONTRACT_FIELDS:
        if field not in saved_model:
            raise ValueError(f"{label} config lacks required field: {field}")
        saved_value = saved_model[field]
        if field in {
            "architecture_variant",
            "dual_stream_attention_contract",
        }:
            # One neutral evaluation YAML serves every ablation. The weight
            # source owns both implementation identity and the sole A/B mask
            # switch; parameters alone cannot recover either distinction.
            config.model[field] = str(saved_value)
            continue
        expected = config.model.get(field)
        if expected is not None and str(saved_value) != str(expected):
            raise ValueError(
                f"{label}/config mismatch for {field}: "
                f"checkpoint={saved_value!r}, evaluation={expected!r}"
            )
        config.model[field] = saved_value


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


def load_model_source_weights(model, source: EvaluationModelSource) -> dict[str, Any]:
    if source.is_hf_final_ema:
        # ``configure_model_source`` made load_model_tokenizer call
        # ``from_pretrained(source.path)``.  Loading a second state here would
        # defeat the final-export contract and can mishandle tied aliases.
        return source.report()

    from utils.sharded_ema import load_sharded_ema_checkpoint

    loaded = load_sharded_ema_checkpoint(model, source.path)
    report = source.report()
    report.update(loaded)
    return report
