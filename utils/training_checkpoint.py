"""Training checkpoint, EMA and Hugging Face export lifecycle.

The training entry point re-exports these names for historical callers.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
from safetensors import SafetensorError, safe_open

from utils.checkpoint_transaction import (
    prepare_checkpoint, publish_checkpoint, publish_directory, write_checkpoint_inventory, validate_checkpoint_inventory,
)
from utils.distributed_io import run_io_phase
from utils.selfless_training_runtime import (
    RESUME_SCHEMA, RESUME_CONTRACT_VERSION, build_sampler_resume_state, validate_resume_contract,
)
from utils.sharded_ema import (
    RankShardedEMA, cast_state_dict_floating_dtype, load_ema_manifest,
    mark_hf_ema_config_dtype, merge_sharded_ema_state_dict, read_sharded_ema_rows,
)
from utils.utils import save_checkpoint, prune_training_checkpoints

logger = get_logger(__name__, log_level="INFO")


def _log_info(message):
    try:
        logger.info(message)
    except RuntimeError:
        logging.getLogger(__name__).info(message)


def _is_disabled_path(value):
    return value is None or (
        isinstance(value, str) and value.lower() in {"none", "null", "false", ""}
    )


@dataclass(frozen=True)
class TrainingResumeState:
    global_step: int
    checkpoint_dir: Path | None
    metadata: dict | None
    cumulative_wall_seconds: float
    cumulative_loss_checks: int


def restore_training_state(
    *, config, accelerator, train_dataloader, mixed_source_training,
    config_contract, ema, ema_update_after_step,
) -> TrainingResumeState:
    """Validate the published state before restoring model, data cursor and EMA."""
    checkpoint_dir = None
    metadata = None
    global_step = 0
    resume_path = config.experiment.resume_from_checkpoint
    if not _is_disabled_path(resume_path):
        checkpoint_dir = Path(resume_path)

        def validate_resume():
            if not checkpoint_dir.exists():
                raise FileNotFoundError(
                    f"Specified resume checkpoint does not exist: {checkpoint_dir}"
                )
            metadata_file = checkpoint_dir / "metadata.json"
            if not metadata_file.is_file():
                raise RuntimeError(
                    f"Refusing to resume a checkpoint without metadata: {metadata_file}"
                )
            saved = json.loads(metadata_file.read_text(encoding="utf-8"))
            _validate_checkpoint_complete(checkpoint_dir, expected_global_step=int(saved["global_step"]))
            if int(saved.get("world_size", -1)) != accelerator.num_processes:
                raise RuntimeError(
                    "Caption resume requires the same world size: "
                    f"checkpoint={saved.get('world_size')}, current={accelerator.num_processes}"
                )
            validate_resume_contract(saved, current_contract=config_contract)
            return saved

        metadata = run_io_phase(
            accelerator, validate_resume, description="checkpoint resume preflight", main_process_only=True,
        )
        global_step = int(metadata["global_step"])
        _log_info(f"Resuming training from checkpoint: {checkpoint_dir}")
        # DeepSpeed-internal collectives still depend on backend timeouts and
        # the launcher; file-only phases below share errors before proceeding.
        run_io_phase(
            accelerator, lambda: accelerator.load_state(checkpoint_dir), description="Accelerate state restore",
        )
        run_io_phase(
            accelerator, lambda: _restore_npu_rng_state(checkpoint_dir, accelerator), description="NPU RNG restore",
        )
        if mixed_source_training:
            train_dataloader.load_state(checkpoint_dir, accelerator, global_step)
        _log_info(f"Resumed at global_step={global_step}")
    else:
        _log_info("Starting fresh caption training.")

    if ema is not None:
        if checkpoint_dir is not None:
            ema.load_checkpoint(checkpoint_dir, accelerator, expected_global_step=global_step)
            _log_info(f"Loaded rank {accelerator.process_index} EMA shard at global_step={global_step}.")
        else:
            ema.initialize_from_model(global_step=global_step)
            _log_info("Initialized this rank's FP32 EMA shard from the training model.")
        if ema.started:
            _log_info(f"EMA is active at global_step={global_step}.")
        else:
            _log_info(
                "EMA updates are delayed until "
                f"global_step={ema_update_after_step}; validation and adapter saves will use the training model until then."
            )

    return TrainingResumeState(
        global_step=global_step,
        checkpoint_dir=checkpoint_dir,
        metadata=metadata,
        cumulative_wall_seconds=float((metadata or {}).get("cumulative_training_wall_seconds", 0.0)),
        cumulative_loss_checks=int((metadata or {}).get("cumulative_finite_loss_microbatches_checked", 0)),
    )


def _special_token_ids(config):
    ids = {
        "mask": int(config.model.mask_token_id),
        "boi": int(config.model.boi_token_id),
        "eoi": int(config.model.eoi_token_id),
    }
    image_mask_token_id = config.model.get("image_mask_token_id", None)
    if image_mask_token_id is not None:
        ids["image_mask"] = int(image_mask_token_id)
    return ids


def _write_image_flow_adapter(state, config, global_step):
    path = Path(config.experiment.output_dir) / f"image_flow_adapter-{global_step}.pt"
    torch.save(state, path)
    logger.info(f"Saved image-flow adapter to {path}")


def _image_flow_adapter_save_enabled(config, *, final: bool) -> bool:
    periodic = bool(config.experiment.get("save_image_flow_adapter", False))
    if not final:
        return periodic
    return bool(
        config.experiment.get("save_final_image_flow_adapter", periodic)
    )


def _save_image_flow_adapter(
    model,
    config,
    accelerator,
    global_step,
    *,
    final: bool = False,
):
    if not _image_flow_adapter_save_enabled(config, final=final):
        return
    if not accelerator.is_main_process:
        return

    unwrapped = accelerator.unwrap_model(model)
    token_ids = _special_token_ids(config)
    embed = unwrapped.model.embed_tokens.weight.detach().cpu()
    state = {
        "image_flow_head": {k: v.detach().cpu() for k, v in unwrapped.image_flow_head.state_dict().items()},
        "image_flow_condition_proj": {
            k: v.detach().cpu()
            for k, v in unwrapped.image_flow_condition_proj.state_dict().items()
        },
        "image_token_embedder": {
            k: v.detach().cpu()
            for k, v in unwrapped.image_token_embedder.state_dict().items()
        },
        "special_token_ids": token_ids,
        "special_token_embeddings": {
            name: embed[token_id].clone()
            for name, token_id in token_ids.items()
        },
    }
    backbone_flow_time_embedder = getattr(
        unwrapped.model, "backbone_flow_time_embedder", None
    )
    if backbone_flow_time_embedder is not None:
        state["model_type"] = "selfless_flow_dynamic_xt"
        state["backbone_flow_time_embedder"] = {
            key: value.detach().cpu()
            for key, value in backbone_flow_time_embedder.state_dict().items()
        }
    _write_image_flow_adapter(state, config, global_step)


def _save_ema_image_flow_adapter(
    ema_directory: Path,
    config,
    accelerator,
    global_step,
    *,
    final: bool = False,
) -> None:
    if not _image_flow_adapter_save_enabled(config, final=final):
        return
    if accelerator.is_main_process:
        manifest = load_ema_manifest(ema_directory)
        prefixes = (
            "image_flow_head.",
            "image_flow_condition_proj.",
            "model.image_token_embedder.",
            "model.backbone_flow_time_embedder.",
        )
        selected_names = [
            name
            for name in manifest["state_keys"]
            if name.startswith(prefixes)
        ]
        selected_state = merge_sharded_ema_state_dict(
            ema_directory,
            names=selected_names,
        )
        token_ids = _special_token_ids(config)
        embedding_rows = read_sharded_ema_rows(
            ema_directory,
            "model.embed_tokens.weight",
            token_ids.values(),
        )
        state = {
            "image_flow_head": {
                name.removeprefix("image_flow_head."): value
                for name, value in selected_state.items()
                if name.startswith("image_flow_head.")
            },
            "image_flow_condition_proj": {
                name.removeprefix("image_flow_condition_proj."): value
                for name, value in selected_state.items()
                if name.startswith("image_flow_condition_proj.")
            },
            "image_token_embedder": {
                name.removeprefix("model.image_token_embedder."): value
                for name, value in selected_state.items()
                if name.startswith("model.image_token_embedder.")
            },
            "special_token_ids": token_ids,
            "special_token_embeddings": {
                name: embedding_rows[token_id].clone()
                for name, token_id in token_ids.items()
            },
        }
        backbone_flow_time_state = {
            name.removeprefix("model.backbone_flow_time_embedder."): value
            for name, value in selected_state.items()
            if name.startswith("model.backbone_flow_time_embedder.")
        }
        if backbone_flow_time_state:
            state["model_type"] = "selfless_flow_dynamic_xt"
            state["backbone_flow_time_embedder"] = backbone_flow_time_state
        _write_image_flow_adapter(state, config, global_step)
        del selected_state, embedding_rows, state
    accelerator.wait_for_everyone()


def _ema_enabled(config) -> bool:
    return bool(config.training.get("use_ema", False))


def _ema_decay(config) -> float:
    decay = float(config.training.get("ema_decay", 0.9999))
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
    return decay


def _ema_eval_export_dtype(config) -> torch.dtype:
    name = str(
        config.experiment.get(
            "ema_eval_dtype",
            config.training.get("ema_eval_dtype", "bf16"),
        )
    ).strip().lower()
    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    dtype = dtypes.get(name)
    if dtype is None:
        raise ValueError(
            "training.ema_eval_dtype must be one of "
            f"{sorted(dtypes)}, got {name!r}"
        )
    return dtype


def _ema_state_directory(config, global_step) -> Path:
    output_dir = Path(config.experiment.output_dir)
    if isinstance(global_step, int):
        return output_dir / f"checkpoint-{global_step}"
    return output_dir / f"ema-{global_step}"


def _save_ema_state(
    ema: RankShardedEMA | None,
    config,
    accelerator,
    global_step,
    *,
    directory: Path | None = None,
) -> Path | None:
    if ema is None:
        return None
    directory = (
        Path(directory)
        if directory is not None
        else _ema_state_directory(config, global_step)
    )
    manifest_path = ema.save_checkpoint(
        directory,
        accelerator,
        global_step=int(global_step) if isinstance(global_step, int) else ema.global_step,
    )
    if accelerator.is_main_process:
        logger.info(f"Saved sharded EMA state to {manifest_path}")
    return directory


def _complete_hf_export_exists(
    save_path: Path,
    *,
    metadata_name: str,
    expected_metadata: dict,
    allow_older_step: bool = False,
) -> bool:
    """Validate an export; a complete older final export may be refreshed."""

    save_path = Path(save_path)
    if not save_path.exists():
        return False
    required = (
        save_path / "model.safetensors",
        save_path / "config.json",
        save_path / "tokenizer.json",
        save_path / metadata_name,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Refusing to overwrite an incomplete HF export at {save_path}: "
            f"missing={missing}"
        )
    try:
        metadata = json.loads(
            (save_path / metadata_name).read_text(encoding="utf-8")
        )
        hf_config = json.loads(
            (save_path / "config.json").read_text(encoding="utf-8")
        )
        json.loads((save_path / "tokenizer.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"HF export metadata/config is unreadable: {save_path}") from exc
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    old_step = metadata.get("source_global_step")
    expected_step = expected_metadata.get("source_global_step")
    refreshing_older_step = (
        allow_older_step
        and type(old_step) is int
        and type(expected_step) is int
        and 0 <= old_step < expected_step
    )
    if refreshing_older_step:
        mismatches.pop("source_global_step", None)
    if mismatches:
        raise RuntimeError(
            f"Refusing to overwrite a different HF export at {save_path}: "
            f"{mismatches}"
        )
    dtype_name = str(expected_metadata["floating_dtype"])
    config_dtype = hf_config.get("dtype")
    torch_dtype = hf_config.get("torch_dtype")
    if config_dtype != dtype_name or (
        torch_dtype is not None and torch_dtype != dtype_name
    ):
        raise RuntimeError(
            f"HF export config dtype mismatch at {save_path}: "
            f"dtype={config_dtype!r}, torch_dtype={torch_dtype!r}, "
            f"expected={dtype_name!r}"
        )
    stored_weight_key_count = _validate_hf_safetensors(
        save_path / "model.safetensors",
        floating_dtype=dtype_name,
    )
    recorded_stored_count = metadata.get("stored_weight_key_count")
    if (
        recorded_stored_count is not None
        and int(recorded_stored_count) != stored_weight_key_count
    ):
        raise RuntimeError(
            f"HF export stored key count mismatch at {save_path}: "
            f"metadata={recorded_stored_count}, actual={stored_weight_key_count}"
        )
    return not refreshing_older_step


def _validate_hf_safetensors(
    weight_path: Path,
    *,
    floating_dtype: str,
) -> int:
    """Validate the single-file HF weight header without loading tensor data."""

    expected_dtype = {
        "bfloat16": "BF16",
        "float32": "F32",
    }.get(str(floating_dtype))
    if expected_dtype is None:
        raise ValueError(f"Unsupported HF export dtype: {floating_dtype!r}")
    try:
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if not keys:
                raise RuntimeError(f"HF export contains no weights: {weight_path}")
            floating_dtypes = set()
            for key in keys:
                tensor_dtype = handle.get_slice(key).get_dtype()
                if tensor_dtype.startswith("F") or tensor_dtype == "BF16":
                    floating_dtypes.add(tensor_dtype)
    except (OSError, ValueError, SafetensorError) as exc:
        raise RuntimeError(f"Invalid safetensors export: {weight_path}") from exc
    if floating_dtypes != {expected_dtype}:
        raise RuntimeError(
            f"HF export floating dtype mismatch at {weight_path}: "
            f"actual={sorted(floating_dtypes)}, expected={[expected_dtype]}"
        )
    return len(keys)


def _run_main_process_export_operation(
    accelerator,
    operation,
    *,
    description: str,
):
    """Run filesystem/model export work on rank 0 and share one outcome."""

    payload = None
    if accelerator.is_main_process:
        try:
            payload = {"ok": True, "result": operation()}
        except Exception as exc:
            payload = {
                "ok": False,
                "error": f"{description}: {type(exc).__name__}: {exc}",
            }
    if int(getattr(accelerator, "num_processes", 1)) > 1:
        objects = [payload]
        broadcast_object_list(objects, from_process=0)
        payload = objects[0]
    if not isinstance(payload, dict) or "ok" not in payload:
        raise RuntimeError(f"{description}: rank 0 returned no export outcome")
    if not payload["ok"]:
        raise RuntimeError(str(payload["error"]))
    return payload.get("result")


def _save_model_hf_for_evaluation(
    model,
    tokenizer,
    config,
    accelerator,
    global_step: int,
    *,
    save_name: str | None = None,
    export_kind: str = "evaluation",
    floating_dtype: torch.dtype | None = None,
    refresh: bool = False,
) -> None:
    """Atomically publish the current non-EMA model for offline evaluation."""

    global_step = int(global_step)
    floating_dtype = floating_dtype or _ema_eval_export_dtype(config)
    dtype_name = str(floating_dtype).removeprefix("torch.")
    save_path = (
        Path(config.experiment.output_dir)
        / (save_name or f"hf_model-{global_step}-eval")
    )
    metadata_name = "model_export_metadata.json"
    expected_metadata = {
        "schema": "selfless_model_hf_export_v1",
        "export_kind": export_kind,
        "floating_dtype": dtype_name,
        "source_global_step": global_step,
    }
    partial_save_path = save_path.with_name(f".{save_path.name}.partial")
    previous_save_path = save_path.with_name(f".{save_path.name}.previous")

    def prepare_export():
        if refresh and previous_save_path.exists():
            if not save_path.exists():
                os.replace(previous_save_path, save_path)
            else:
                _complete_hf_export_exists(save_path, metadata_name=metadata_name,
                                          expected_metadata=expected_metadata, allow_older_step=True)
                shutil.rmtree(previous_save_path)
        # Legacy raw final exports lacked metadata; retain them as the backup
        # until a validated replacement is ready, but do not infer their step.
        legacy_raw = refresh and save_path.exists() and not (save_path / metadata_name).is_file()
        if not legacy_raw and _complete_hf_export_exists(
            save_path,
            metadata_name=metadata_name,
            expected_metadata=expected_metadata,
            allow_older_step=refresh,
        ):
            if partial_save_path.exists():
                shutil.rmtree(partial_save_path)
            return "keep"
        if partial_save_path.exists():
            shutil.rmtree(partial_save_path)
        return "write"

    action = _run_main_process_export_operation(
        accelerator,
        prepare_export,
        description=f"preparing current-model evaluation export at {save_path}",
    )
    if action == "keep":
        if accelerator.is_main_process:
            _log_info(
                f"Keeping existing complete model evaluation export: {save_path}"
            )
        return

    def export_current_model():
        state_dict = None
        try:
            # Unified training uses DeepSpeed ZeRO-2.  Parameters are replicated,
            # so gathering on every rank would clone the complete model to every
            # host CPU for no benefit.
            state_dict = accelerator.get_state_dict(model)
            if any(value.is_floating_point() and value.dtype != floating_dtype for value in state_dict.values()):
                source_state = state_dict
                state_dict = cast_state_dict_floating_dtype(
                    source_state,
                    floating_dtype,
                )
                del source_state
            source_state_key_count = len(state_dict)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                partial_save_path,
                save_function=accelerator.save,
                state_dict=state_dict,
                safe_serialization=True,
            )
            mark_hf_ema_config_dtype(partial_save_path, floating_dtype)
            tokenizer.save_pretrained(partial_save_path)
            stored_weight_key_count = _validate_hf_safetensors(
                partial_save_path / "model.safetensors",
                floating_dtype=dtype_name,
            )
            metadata = dict(expected_metadata)
            metadata.update(
                {
                    "state_key_count": source_state_key_count,
                    "stored_weight_key_count": stored_weight_key_count,
                }
            )
            (partial_save_path / metadata_name).write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _complete_hf_export_exists(
                partial_save_path,
                metadata_name=metadata_name,
                expected_metadata=expected_metadata,
            )
            publish_directory(partial_save_path, save_path, replace_existing=refresh)
            return "saved"
        except Exception:
            if partial_save_path.exists():
                shutil.rmtree(partial_save_path, ignore_errors=True)
            raise
        finally:
            if state_dict is not None:
                del state_dict

    _run_main_process_export_operation(
        accelerator,
        export_current_model,
        description=f"exporting current model for evaluation at {save_path}",
    )
    if accelerator.is_main_process:
        _log_info(
            f"Saved evaluation model ({floating_dtype}) to {save_path}"
        )


def _save_ema_hf_model(
    ema: RankShardedEMA | None,
    model,
    tokenizer,
    config,
    accelerator,
    global_step,
    ema_directory: Path | None,
    *,
    floating_dtype: torch.dtype = torch.float32,
    save_name: str | None = None,
    export_kind: str = "training",
) -> None:
    if (
        ema is None
        or not ema.started
        or not bool(config.training.get("ema_save_hf_model", True))
    ):
        return
    resolved_save_name = save_name or f"hf_model-{global_step}-ema"
    save_path = Path(config.experiment.output_dir) / resolved_save_name
    expected_source_step = (
        int(global_step)
        if isinstance(global_step, int)
        else int(ema.global_step)
    )
    dtype_name = str(floating_dtype).removeprefix("torch.")
    expected_metadata = {
        "schema": "selfless_ema_hf_export_v1",
        "export_kind": str(export_kind),
        "floating_dtype": dtype_name,
        "source_global_step": expected_source_step,
    }
    partial_save_path = save_path.with_name(f".{save_path.name}.partial")
    temporary_ema_directory = (
        Path(config.experiment.output_dir)
        / f".{resolved_save_name}.ema-state.partial"
    )
    # Step-named evaluation exports are immutable. The canonical final export
    # advances when a stopped run resumes and reaches a later optimizer step.
    refresh_final = (
        global_step == "final"
        and resolved_save_name == "hf_model-final-ema"
        and export_kind == "training"
    )
    previous_save_path = save_path.with_name(f".{save_path.name}.previous")

    def prepare_export():
        if refresh_final and previous_save_path.exists():
            # Recover an interruption between the two directory renames. If
            # publication finished, validate the new export before cleanup.
            if not save_path.exists():
                os.replace(previous_save_path, save_path)
            else:
                _complete_hf_export_exists(
                    save_path,
                    metadata_name="ema_export_metadata.json",
                    expected_metadata=expected_metadata,
                    allow_older_step=True,
                )
                shutil.rmtree(previous_save_path)
        if _complete_hf_export_exists(
            save_path,
            metadata_name="ema_export_metadata.json",
            expected_metadata=expected_metadata,
            allow_older_step=refresh_final,
        ):
            for stale_path in (partial_save_path, temporary_ema_directory):
                if stale_path.exists():
                    shutil.rmtree(stale_path)
            return {"action": "keep", "use_temporary_state": False}
        for stale_path in (partial_save_path, temporary_ema_directory):
            if stale_path.exists():
                shutil.rmtree(stale_path)
        use_temporary_state = (
            ema_directory is None
            or not (Path(ema_directory) / "ema_manifest.json").is_file()
        )
        return {
            "action": "write",
            "use_temporary_state": use_temporary_state,
        }

    preparation = _run_main_process_export_operation(
        accelerator,
        prepare_export,
        description=f"preparing EMA evaluation export at {save_path}",
    )
    if preparation["action"] == "keep":
        if accelerator.is_main_process:
            _log_info(f"Keeping existing complete EMA export: {save_path}")
        return

    using_temporary_state = bool(preparation["use_temporary_state"])
    source_ema_directory = (
        temporary_ema_directory if using_temporary_state else Path(ema_directory)
    )
    if using_temporary_state:
        source_ema_directory = _save_ema_state(
            ema,
            config,
            accelerator,
            global_step,
            directory=temporary_ema_directory,
        )

    def export_ema_model():
        merged_state = None
        try:
            merged_state = merge_sharded_ema_state_dict(source_ema_directory)
            if floating_dtype != torch.float32:
                source_state = merged_state
                merged_state = cast_state_dict_floating_dtype(
                    source_state,
                    floating_dtype,
                )
                del source_state
            source_state_key_count = len(merged_state)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                partial_save_path,
                state_dict=merged_state,
                safe_serialization=True,
            )
            mark_hf_ema_config_dtype(partial_save_path, floating_dtype)
            tokenizer.save_pretrained(partial_save_path)
            stored_weight_key_count = _validate_hf_safetensors(
                partial_save_path / "model.safetensors",
                floating_dtype=dtype_name,
            )
            manifest = load_ema_manifest(source_ema_directory)
            runtime = manifest.get("runtime") or {}
            metadata = {
                **expected_metadata,
                "source_ema_directory": str(source_ema_directory),
                "source_ema_directory_retained": not using_temporary_state,
                "source_global_step": runtime.get("global_step"),
                "source_world_size": manifest["world_size"],
                "layout_validation": "readable_field_equality",
                "state_key_count": source_state_key_count,
                "stored_weight_key_count": stored_weight_key_count,
            }
            (partial_save_path / "ema_export_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _complete_hf_export_exists(
                partial_save_path,
                metadata_name="ema_export_metadata.json",
                expected_metadata=expected_metadata,
            )
            if save_path.exists():
                if not refresh_final or _complete_hf_export_exists(
                    save_path,
                    metadata_name="ema_export_metadata.json",
                    expected_metadata=expected_metadata,
                    allow_older_step=True,
                ):
                    raise RuntimeError(
                        f"EMA export appeared during publication: {save_path}"
                    )
            publish_directory(partial_save_path, save_path, replace_existing=refresh_final)
            if using_temporary_state and source_ema_directory.exists():
                shutil.rmtree(source_ema_directory)
            return "saved"
        except Exception:
            if partial_save_path.exists():
                shutil.rmtree(partial_save_path, ignore_errors=True)
            if using_temporary_state and source_ema_directory.exists():
                shutil.rmtree(source_ema_directory, ignore_errors=True)
            raise
        finally:
            if merged_state is not None:
                del merged_state

    _run_main_process_export_operation(
        accelerator,
        export_ema_model,
        description=f"exporting EMA model for evaluation at {save_path}",
    )
    if accelerator.is_main_process:
        _log_info(
            f"Saved {export_kind} EMA HF model ({floating_dtype}) to {save_path}"
        )


def _publish_evaluation_model_pair_manifest(
    config,
    accelerator,
    global_step: int,
) -> None:
    """Publish a commit marker only after current and EMA exports both validate."""

    global_step = int(global_step)
    output_dir = Path(config.experiment.output_dir)
    dtype_name = str(_ema_eval_export_dtype(config)).removeprefix("torch.")
    current_name = f"hf_model-{global_step}-eval"
    ema_name = f"hf_model-{global_step}-ema-eval"
    manifest_path = output_dir / f"hf_model-{global_step}-eval-pair.json"
    partial_path = manifest_path.with_name(f".{manifest_path.name}.partial")
    expected_manifest = {
        "schema": "selfless_evaluation_model_pair_v1",
        "complete": True,
        "global_step": global_step,
        "floating_dtype": dtype_name,
        "current_model_directory": current_name,
        "ema_model_directory": ema_name,
    }

    def publish_manifest():
        current_complete = _complete_hf_export_exists(
            output_dir / current_name,
            metadata_name="model_export_metadata.json",
            expected_metadata={
                "schema": "selfless_model_hf_export_v1",
                "export_kind": "evaluation",
                "floating_dtype": dtype_name,
                "source_global_step": global_step,
            },
        )
        ema_complete = _complete_hf_export_exists(
            output_dir / ema_name,
            metadata_name="ema_export_metadata.json",
            expected_metadata={
                "schema": "selfless_ema_hf_export_v1",
                "export_kind": "evaluation",
                "floating_dtype": dtype_name,
                "source_global_step": global_step,
            },
        )
        if not current_complete or not ema_complete:
            raise RuntimeError(
                f"evaluation model pair is incomplete at global_step={global_step}"
            )
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"evaluation model pair manifest is unreadable: {manifest_path}"
                ) from exc
            if existing != expected_manifest:
                raise RuntimeError(
                    "refusing to overwrite a different evaluation model pair "
                    f"manifest at {manifest_path}"
                )
            if partial_path.exists():
                partial_path.unlink()
            return "keep"
        if partial_path.exists():
            partial_path.unlink()
        try:
            partial_path.write_text(
                json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(partial_path, manifest_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
        return "saved"

    action = _run_main_process_export_operation(
        accelerator,
        publish_manifest,
        description=(
            "publishing current-model/EMA evaluation pair manifest at "
            f"{manifest_path}"
        ),
    )
    if accelerator.is_main_process:
        _log_info(
            f"{'Kept' if action == 'keep' else 'Saved'} complete evaluation "
            f"model pair manifest: {manifest_path}"
        )


def _write_training_checkpoint_metadata(
    checkpoint_dir: Path,
    *,
    accelerator,
    global_step: int,
    epoch: int,
    batches_consumed_in_epoch: int,
    sampler_shuffle_seed: int,
    prepared_dataloader_length: int,
    config_contract: dict | None,
    ema_layout,
    cumulative_training_wall_seconds: float,
    cumulative_finite_loss_microbatches_checked: int,
    mixed_data_state_schema: str | None = None,
) -> None:
    def write_metadata():
        payload = {
            "schema": RESUME_SCHEMA,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batches_consumed_in_epoch": int(batches_consumed_in_epoch),
            "sampler_state": (
                None
                if mixed_data_state_schema is not None
                else build_sampler_resume_state(
                    epoch=epoch,
                    batches_consumed_in_epoch=batches_consumed_in_epoch,
                    shuffle_seed=sampler_shuffle_seed,
                    prepared_dataloader_length=prepared_dataloader_length,
                )
            ),
            "mixed_data_state_schema": mixed_data_state_schema,
            "world_size": int(accelerator.num_processes),
            "gradient_accumulation_steps": int(
                accelerator.gradient_accumulation_steps
            ),
            "config_contract": config_contract,
            "config_contract_version": (
                RESUME_CONTRACT_VERSION
                if config_contract is not None
                else None
            ),
            "ema_layout_validation": (
                "readable_field_equality" if ema_layout is not None else None
            ),
            "cumulative_training_wall_seconds": float(
                cumulative_training_wall_seconds
            ),
            "cumulative_finite_loss_microbatches_checked": int(
                cumulative_finite_loss_microbatches_checked
            ),
        }
        path = checkpoint_dir / "metadata.json"
        temp_path = checkpoint_dir / f".{path.name}.tmp-{os.getpid()}"
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    run_io_phase(accelerator, write_metadata, description="training metadata save", main_process_only=True)


def _mark_checkpoint_complete(
    checkpoint_dir: Path,
    *,
    accelerator,
    global_step: int,
) -> None:
    def write_completion():
        path = checkpoint_dir / "checkpoint_complete.json"
        temp_path = checkpoint_dir / f".{path.name}.tmp-{os.getpid()}"
        temp_path.write_text(
            json.dumps(
                {
                    "schema": "selfless_caption_checkpoint_complete_v2",
                    "global_step": int(global_step),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    run_io_phase(accelerator, write_completion, description="checkpoint completion marker", main_process_only=True)


def _begin_checkpoint_write(
    checkpoint_dir: Path,
    *,
    accelerator,
) -> Path:
    """Prepare a private sibling; existing published state remains untouched."""
    return prepare_checkpoint(checkpoint_dir, accelerator, global_step=int(checkpoint_dir.name.split("-")[-1]))


def _save_resumable_training_checkpoint(
    *,
    model,
    config,
    accelerator,
    global_step: int,
    train_dataloader,
    mixed_source_training: bool,
    epoch: int,
    batches_consumed_in_epoch: int,
    sampler_shuffle_seed: int,
    config_contract,
    ema_layout,
    ema,
    cumulative_training_wall_seconds: float,
    cumulative_finite_loss_microbatches_checked: int,
) -> Path:
    destination = (
        Path(config.experiment.output_dir)
        / f"checkpoint-{int(global_step)}"
    )
    checkpoint_dir = _begin_checkpoint_write(
        destination,
        accelerator=accelerator,
    )
    save_checkpoint(model, config, accelerator, int(global_step), defer_retention=True, directory=checkpoint_dir)
    _save_npu_rng_state(checkpoint_dir, accelerator)
    if mixed_source_training:
        train_dataloader.save_state(
            checkpoint_dir,
            accelerator,
            int(global_step),
        )
    _write_training_checkpoint_metadata(
        checkpoint_dir,
        accelerator=accelerator,
        global_step=int(global_step),
        epoch=int(epoch),
        batches_consumed_in_epoch=int(batches_consumed_in_epoch),
        sampler_shuffle_seed=int(sampler_shuffle_seed),
        prepared_dataloader_length=len(train_dataloader),
        config_contract=config_contract,
        ema_layout=ema_layout,
        cumulative_training_wall_seconds=float(
            cumulative_training_wall_seconds
        ),
        cumulative_finite_loss_microbatches_checked=int(
            cumulative_finite_loss_microbatches_checked
        ),
        mixed_data_state_schema=(
            train_dataloader.state_schema
            if mixed_source_training
            else None
        ),
    )
    ema_directory = _save_ema_state(
        ema,
        config,
        accelerator,
        int(global_step),
        directory=checkpoint_dir,
    )
    run_io_phase(
        accelerator,
        lambda: write_checkpoint_inventory(checkpoint_dir, world_size=accelerator.num_processes,
                                           mixed_data=mixed_source_training, ema=ema is not None),
        description="checkpoint payload validation", main_process_only=True,
    )
    _mark_checkpoint_complete(
        checkpoint_dir,
        accelerator=accelerator,
        global_step=int(global_step),
    )
    publish_checkpoint(checkpoint_dir, destination, accelerator, global_step=int(global_step))
    checkpoint_dir = destination
    if ema_directory is not None:
        ema_directory = destination
    _run_main_process_export_operation(
        accelerator,
        lambda: prune_training_checkpoints(config, int(global_step)),
        description=f"applying retention after completed checkpoint {checkpoint_dir}",
    )
    if _image_flow_adapter_save_enabled(config, final=False):
        if (
            ema is not None
            and ema.started
            and bool(config.training.get("ema_save_adapter", True))
        ):
            _save_ema_image_flow_adapter(
                ema_directory,
                config,
                accelerator,
                int(global_step),
            )
        else:
            _save_image_flow_adapter(
                model,
                config,
                accelerator,
                int(global_step),
            )
    if accelerator.is_main_process:
        logger.info(
            "Completed resumable checkpoint at step %d: %s",
            int(global_step),
            checkpoint_dir,
        )
    return checkpoint_dir


def _save_npu_rng_state(checkpoint_dir: Path, accelerator) -> None:
    """Add the per-rank NPU RNG state missing from Accelerate 1.14."""

    if accelerator.device.type != "npu":
        return
    def write_rng():
        rng_path = checkpoint_dir / f"random_states_{accelerator.process_index}.pkl"
        if not rng_path.is_file():
            raise RuntimeError(f"Accelerate did not save the expected RNG file: {rng_path}")
        states = torch.load(str(rng_path), map_location="cpu", weights_only=False)
        states["torch_npu_manual_seed"] = torch.npu.get_rng_state(
            accelerator.device
        ).cpu()
        # Accelerate 1.14 falls through to its CUDA restore branch when no
        # supported accelerator backend is detected. An empty CUDA state is a
        # harmless no-op on an NPU-only host and lets its CPU/Python/NumPy restore
        # path complete before the explicit NPU restore below.
        states.setdefault("torch_cuda_manual_seed", [])
        temp_path = rng_path.with_name(f".{rng_path.name}.tmp-{os.getpid()}")
        torch.save(states, str(temp_path))
        os.replace(temp_path, rng_path)

    run_io_phase(accelerator, write_rng, description="NPU RNG save")


def _restore_npu_rng_state(checkpoint_dir: Path, accelerator) -> None:
    """Strictly restore the per-rank NPU RNG state saved above."""

    if accelerator.device.type != "npu":
        return
    rng_path = checkpoint_dir / f"random_states_{accelerator.process_index}.pkl"
    states = torch.load(str(rng_path), map_location="cpu", weights_only=False)
    npu_state = states.get("torch_npu_manual_seed")
    if not isinstance(npu_state, torch.Tensor) or npu_state.dtype != torch.uint8:
        raise RuntimeError(f"Checkpoint has no valid NPU RNG state: {rng_path}")
    torch.npu.set_rng_state(npu_state, accelerator.device)
    restored = torch.npu.get_rng_state(accelerator.device).cpu()
    if not torch.equal(restored, npu_state.cpu()):
        raise RuntimeError(f"NPU RNG restore verification failed: {rng_path}")
    logger.info(
        "Restored and verified rank %d NPU RNG state from %s",
        accelerator.process_index,
        rng_path,
    )


def _validate_checkpoint_complete(
    checkpoint_dir: Path,
    *,
    expected_global_step: int,
) -> None:
    completion_file = checkpoint_dir / "checkpoint_complete.json"
    if not completion_file.is_file():
        raise RuntimeError(
            f"Refusing to resume an incomplete checkpoint: missing {completion_file}"
        )
    completion = json.loads(completion_file.read_text(encoding="utf-8"))
    expected = {
        "schema": completion.get("schema"),
        "global_step": int(expected_global_step),
    }
    if completion != expected or completion.get("schema") not in {
        "selfless_caption_checkpoint_complete_v1", "selfless_caption_checkpoint_complete_v2",
    }:
        raise RuntimeError(
            "Invalid caption checkpoint completion marker: "
            f"checkpoint={completion}, expected={expected}"
        )
    if completion["schema"] == "selfless_caption_checkpoint_complete_v2":
        validate_checkpoint_inventory(checkpoint_dir)
