import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
try:
    # CANN's embedded compiler initializes more reliably when its Python
    # package is loaded before torch_npu (required for fusion-attention JIT).
    import tbe  # noqa: F401
except ImportError:
    pass
import json
import logging
import math
import shutil
import time
import importlib.util
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.optim import AdamW
import torch.nn.functional as F


from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import (
    broadcast_object_list,
    DistributedType,
    GradientAccumulationPlugin,
    set_seed,
)
from safetensors import SafetensorError, safe_open

from utils.dataset_utils import get_dataloaders
from utils.flow_head_contract import (
    flow_head_attention_report,
    validate_flow_head_attention_contract,
)
from utils.wsd_schedule import get_wsd_schedule
from utils.selfless_flow_optimizer import (
    learning_rate_for_parameter,
    optimizer_parameter_role,
    weight_decay_for_parameter,
)
from utils.selfless_flow_adapter import load_image_flow_adapter
from utils.selfless_training_runtime import (
    RESUME_CONTRACT_VERSION,
    RESUME_SCHEMA,
    TrainingWindow,
    build_resume_contract,
    build_sampler_resume_state,
    gradient_norm_log_payload,
    training_stop_step,
    validate_sampler_resume_state,
    validate_resume_contract,
    validate_wsd_contract,
)
from utils.sharded_ema import (
    DEFAULT_EMA_CHUNK_NUMEL,
    RankShardedEMA,
    build_sharded_ema_layout,
    cast_state_dict_floating_dtype,
    load_ema_manifest,
    mark_hf_ema_config_dtype,
    merge_sharded_ema_state_dict,
    read_sharded_ema_rows,
)
from models.logging import set_verbosity_info, set_verbosity_error
from utils.utils import (
    checkpoint_save_due,
    flatten_omega_conf,
    get_config,
    get_showo_mae_mask,
    get_selfless_mask,
    load_model_tokenizer,
    sample_showo_mae_image_mask,
    save_checkpoint,
    save_hf_model,
)

logger = get_logger(__name__, log_level="INFO")
_VAE_CACHE = None
_MIXED_SOURCE_NAMES = ("climbmix", "t2i", "i2t")


def _active_mixed_source_names(schedule) -> tuple[str, ...]:
    active = tuple(
        dict.fromkeys(str(source).strip().lower() for source in schedule)
    )
    if not active:
        raise ValueError("mixed source schedule must not be empty")
    unsupported = [
        source for source in active if source not in _MIXED_SOURCE_NAMES
    ]
    if unsupported:
        raise ValueError(f"unsupported mixed training sources: {unsupported}")
    return active


def _single_source_global_physical_token_budget(
    config,
    active_sources,
) -> int | None:
    if len(active_sources) != 1:
        return None
    source_name = active_sources[0]
    source = config.dataset.params.sources[source_name]
    raw_budget = source.get(
        "expected_global_physical_tokens_per_optimizer_step", None
    )
    if raw_budget is None:
        raise ValueError(
            "repeated single-source training requires "
            "dataset.params.sources."
            f"{source_name}.expected_global_physical_tokens_per_optimizer_step"
        )
    budget = int(raw_budget)
    if budget <= 0:
        raise ValueError(
            "expected_global_physical_tokens_per_optimizer_step must be "
            f"positive, got {budget}"
        )
    training_budget_raw = config.training.get(
        "physical_tokens_per_optimizer_step", None
    )
    if training_budget_raw is None:
        raise ValueError(
            "repeated single-source training requires "
            "training.physical_tokens_per_optimizer_step"
        )
    training_budget = int(training_budget_raw)
    if training_budget != budget:
        raise ValueError(
            "training.physical_tokens_per_optimizer_step must equal the "
            f"active source budget: training={training_budget}, "
            f"source={budget}"
        )
    target_raw = config.training.get("target_physical_tokens", None)
    if target_raw is None:
        raise ValueError(
            "repeated single-source training requires "
            "training.target_physical_tokens"
        )
    target = int(target_raw)
    max_train_steps = int(config.training.max_train_steps)
    if target <= 0 or max_train_steps <= 0:
        raise ValueError(
            "training target_physical_tokens and max_train_steps must be "
            f"positive, got target={target}, steps={max_train_steps}"
        )
    implied_target = max_train_steps * training_budget
    if implied_target != target:
        raise ValueError(
            "single-source training horizon does not match its exact physical-"
            "token target: "
            f"max_train_steps={max_train_steps}, "
            f"physical_tokens_per_optimizer_step={training_budget}, "
            f"implied={implied_target}, target={target}"
        )
    return budget


def _validate_source_physical_token_budget(
    per_rank_values,
    *,
    expected_global: int,
    source_name: str,
) -> int:
    values = [int(value) for value in per_rank_values]
    if not values:
        raise ValueError("per-rank physical-token values must not be empty")
    expected_global = int(expected_global)
    if expected_global <= 0 or expected_global % len(values):
        raise ValueError(
            "expected global physical-token budget must be positive and "
            f"divisible by rank count: {expected_global}, ranks={len(values)}"
        )
    expected_local = expected_global // len(values)
    actual_global = sum(values)
    if (
        any(value != expected_local for value in values)
        or actual_global != expected_global
    ):
        raise RuntimeError(
            "refusing optimizer.step because the active source "
            "physical-token budget is wrong: "
            f"source={source_name!r}, per_rank_positions={values}, "
            f"expected_per_rank_positions={expected_local}, "
            f"actual_global_positions={actual_global}, "
            f"expected_global_positions={expected_global}"
        )
    return actual_global


def _gradient_accumulation_plugin(
    *,
    gradient_accumulation_steps: int,
    mixed_source_training: bool,
) -> GradientAccumulationPlugin:
    """Keep the fixed mixed-source schedule in charge of step boundaries.

    The T2I and I2T iterators are individually prepared by Accelerate.  Their
    epoch boundaries must not force an early optimizer step while they are
    interleaved inside the infinite four-source loader.
    """

    return GradientAccumulationPlugin(
        num_steps=int(gradient_accumulation_steps),
        sync_with_dataloader=not bool(mixed_source_training),
    )


def _source_task_loss_and_count(
    source_name: str,
    *,
    per_modality_loss,
    text_count: torch.Tensor,
    image_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the normalized task loss and target count for one source."""

    source_name = str(source_name)
    if source_name in {"climbmix", "i2t"}:
        return per_modality_loss["text_loss"], text_count
    if source_name == "t2i":
        return per_modality_loss["image_loss"], image_count
    raise ValueError(f"unsupported mixed training source={source_name!r}")


def _debug_nonfinite_loss_trace_details(
    trace: torch.Tensor,
    *,
    ending_global_step: int,
    gradient_accumulation_steps: int,
    source_schedule: tuple[str, ...],
) -> list[str]:
    """Describe non-finite device-cached loss scalars at a log boundary."""

    if trace.ndim != 3 or trace.shape[-1] != 3:
        raise ValueError(
            "debug loss trace must have shape [world,microbatches,3], got "
            f"{tuple(trace.shape)}"
        )
    accumulation = int(gradient_accumulation_steps)
    if accumulation <= 0 or int(trace.shape[1]) % accumulation:
        raise ValueError(
            "debug loss trace length must contain complete optimizer steps"
        )
    if source_schedule and len(source_schedule) != accumulation:
        raise ValueError(
            "debug loss trace source schedule must match gradient accumulation"
        )

    trace_cpu = trace.detach().float().cpu()
    finite_by_microbatch = torch.isfinite(trace_cpu).all(dim=-1)
    bad_indices = (~finite_by_microbatch).nonzero(as_tuple=False).tolist()
    bad_indices.sort(key=lambda item: (int(item[1]), int(item[0])))
    first_step = int(ending_global_step) - int(trace.shape[1]) // accumulation + 1
    field_names = ("weighted", "text_loss", "image_loss")
    details = []
    for rank, microbatch_index in bad_indices:
        slot = int(microbatch_index) % accumulation
        source = source_schedule[slot] if source_schedule else "unknown"
        values = {
            name: float(value)
            for name, value in zip(
                field_names,
                trace_cpu[int(rank), int(microbatch_index)].tolist(),
            )
        }
        details.append(
            f"rank={int(rank)},step={first_step + int(microbatch_index) // accumulation},"
            f"slot={slot + 1},source={source!r},values={values}"
        )
    return details


def _source_loss_metric_payload(
    reduced_source_totals: torch.Tensor,
    *,
    num_processes: int,
    gradient_accumulation_steps: int,
    active_sources=None,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Convert reduced source totals into raw losses and step contributions.

    Each row stores ``loss * target_count``, ``target_count``, and the sum of
    already task-weighted microbatch losses.  The weighted values are divided
    by data-parallel world size and gradient accumulation, so their sum is
    directly comparable with ``step_loss``.
    """

    if active_sources is None:
        active_sources = _MIXED_SOURCE_NAMES
    active_sources = _active_mixed_source_names(active_sources)
    expected_values = len(active_sources) * 3
    if reduced_source_totals.numel() != expected_values:
        raise ValueError(
            "source loss totals must contain three values per source, got "
            f"{reduced_source_totals.numel()} instead of {expected_values}"
        )
    denominator = int(num_processes) * int(gradient_accumulation_steps)
    if denominator <= 0:
        raise ValueError(
            "num_processes * gradient_accumulation_steps must be positive"
        )

    rows = reduced_source_totals.reshape(len(active_sources), 3)
    logs: dict[str, float] = {}
    display: dict[str, tuple[float, float]] = {}
    for index, source in enumerate(active_sources):
        target_count = float(rows[index, 1].item())
        if target_count <= 0.0:
            raise RuntimeError(
                f"mixed source {source!r} produced no optimization targets"
            )
        raw_loss = float((rows[index, 0] / rows[index, 1]).item())
        weighted_contribution = float(
            (rows[index, 2] / float(denominator)).item()
        )
        if not math.isfinite(raw_loss) or not math.isfinite(
            weighted_contribution
        ):
            raise FloatingPointError(
                f"non-finite source metric for {source}: "
                f"loss={raw_loss}, contribution={weighted_contribution}"
            )
        logs[f"train/loss_{source}"] = raw_loss
        logs[f"train/weighted_contribution_{source}"] = (
            weighted_contribution
        )
        logs[f"train/{source}_target_tokens"] = target_count
        display[source] = (raw_loss, weighted_contribution)
    return logs, display


def _append_training_metrics_jsonl(
    config,
    *,
    global_step: int,
    logs: dict[str, float],
) -> None:
    """Persist rank-zero step metrics without a tracker or any hashing."""

    metrics = {}
    for key, value in logs.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"training metric {key!r} must be scalar, got "
                    f"shape={tuple(value.shape)}"
                )
            value = value.item()
        if isinstance(value, bool):
            metrics[str(key)] = bool(value)
        elif isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
        else:
            raise TypeError(
                f"training metric {key!r} is not JSON scalar: {type(value)}"
            )
    output_path = Path(config.experiment.output_dir) / "training_metrics.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "unified_training_step_metrics_v1",
        "global_step": int(global_step),
        "metrics": metrics,
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _log_info(message):
    try:
        logger.info(message)
    except RuntimeError:
        logging.getLogger(__name__).info(message)


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


def _training_objective(config) -> str:
    objective = str(
        config.model.get("training_objective", "selfless_dual_stream")
    ).strip().lower()
    if objective not in {"selfless_dual_stream", "showo_mae_flow"}:
        raise ValueError(f"unsupported model.training_objective={objective!r}")
    attention_contract = str(
        config.model.get(
            "dual_stream_attention_contract", "selfless_strict"
        )
    ).strip().lower()
    if attention_contract not in {
        "selfless_strict",
        "xlnet_content_diagonal",
    }:
        raise ValueError(
            "unsupported model.dual_stream_attention_contract="
            f"{attention_contract!r}"
        )
    validate_flow_head_attention_contract(config.model)
    if (
        objective == "showo_mae_flow"
        and str(config.model.get("showo_mask_schedule", "cosine")).lower()
        != "cosine"
    ):
        raise ValueError("showo_mae_flow requires showo_mask_schedule=cosine")
    return objective


def _build_backbone_attention_masks(
    *,
    config,
    input_ids,
    token_types,
    sigma,
    segment_ids=None,
    image_uncond_rows=None,
    image_uncond_mask=None,
):
    objective = _training_objective(config)
    if objective == "showo_mae_flow":
        return (
            get_showo_mae_mask(
                input_ids=input_ids,
                token_types=token_types,
                device=input_ids.device,
                boi_token_id=int(config.model.boi_token_id),
                segment_ids=segment_ids,
                image_uncond_rows=image_uncond_rows,
                image_uncond_mask=image_uncond_mask,
            ),
            None,
        )

    mask_kwargs = {
        "sigma": sigma,
        "seq_len": input_ids.shape[1],
        "device": input_ids.device,
        "input_ids": input_ids,
        "token_types": token_types,
        "boi_token_id": int(config.model.boi_token_id),
        "image_uncond_rows": image_uncond_rows,
        "segment_ids": segment_ids,
        "image_uncond_mask": image_uncond_mask,
    }
    query_mask = get_selfless_mask(**mask_kwargs)
    attention_contract = str(
        config.model.get(
            "dual_stream_attention_contract", "selfless_strict"
        )
    ).strip().lower()
    content_mask = (
        get_selfless_mask(**mask_kwargs, include_diagonal=True)
        if attention_contract == "xlnet_content_diagonal"
        else None
    )
    return query_mask, content_mask


def _prepare_showo_image_masks(
    *,
    config,
    token_types,
    image_span_table,
    image_loss_mask,
    mask_generation_images: bool,
):
    if _training_objective(config) != "showo_mae_flow":
        return image_loss_mask, None, None
    image_latent_mask = token_types.eq(1)
    if not mask_generation_images or image_span_table.shape[0] == 0:
        return image_loss_mask, image_latent_mask, None
    sampled_mask, mask_prob = sample_showo_mae_image_mask(
        image_span_table=image_span_table,
        full_image_loss_mask=image_loss_mask,
        image_tokens_per_img=int(config.model.image_tokens_per_img),
        min_masking_rate=float(
            config.model.get("showo_min_masking_rate", 0.0)
        ),
    )
    image_latent_mask = image_latent_mask & ~sampled_mask
    return sampled_mask, image_latent_mask, mask_prob


def _is_disabled_path(value):
    return value is None or (
        isinstance(value, str) and value.lower() in {"none", "null", "false", ""}
    )


def _apply_trainable_scope(model, config) -> dict[str, int | str]:
    """Apply the explicit optimizer scope before EMA/DeepSpeed construction."""

    raw_scope = str(config.training.get("trainable_scope", "full")).strip().lower()
    aliases = {
        "full": "full",
        "all": "full",
        "image_flow_head": "image_flow_head",
        "flow_head": "image_flow_head",
        "flow_head_only": "image_flow_head",
    }
    scope = aliases.get(raw_scope)
    if scope is None:
        raise ValueError(
            "Unknown training.trainable_scope="
            f"{raw_scope!r}; expected 'full' or 'image_flow_head'."
        )

    trainable_prefixes = (
        "image_flow_head.",
        "image_flow_condition_proj.",
    )
    if scope == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))

        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith(trainable_prefixes)
        ]
        if unexpected:
            raise RuntimeError(
                "image_flow_head trainable scope leaked into frozen parameters: "
                f"{unexpected[:8]}"
            )
        expects_tied_embeddings = bool(
            getattr(model.config, "tie_word_embeddings", False)
        )
        if (
            expects_tied_embeddings
            and model.lm_head.weight is not model.model.embed_tokens.weight
        ):
            raise RuntimeError(
                "Expected lm_head.weight and model.embed_tokens.weight to remain tied."
            )
        if (
            model.model.embed_tokens.weight.requires_grad
            or model.lm_head.weight.requires_grad
        ):
            raise RuntimeError("The Qwen token embedding/lm_head was not frozen.")

    total_numel = sum(parameter.numel() for parameter in model.parameters())
    trainable_numel = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_numel <= 0:
        raise RuntimeError(f"trainable_scope={scope!r} selected no parameters")
    frozen_numel = total_numel - trainable_numel
    return {
        "scope": scope,
        "total_numel": int(total_numel),
        "trainable_numel": int(trainable_numel),
        "frozen_numel": int(frozen_numel),
    }


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
) -> bool:
    """Accept a matching completed export and reject ambiguous leftovers."""

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
    return True


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
) -> None:
    """Atomically publish the current non-EMA model for offline evaluation."""

    global_step = int(global_step)
    floating_dtype = _ema_eval_export_dtype(config)
    dtype_name = str(floating_dtype).removeprefix("torch.")
    save_path = (
        Path(config.experiment.output_dir)
        / f"hf_model-{global_step}-eval"
    )
    metadata_name = "model_export_metadata.json"
    expected_metadata = {
        "schema": "selfless_model_hf_export_v1",
        "export_kind": "evaluation",
        "floating_dtype": dtype_name,
        "source_global_step": global_step,
    }
    partial_save_path = save_path.with_name(f".{save_path.name}.partial")

    def prepare_export():
        if _complete_hf_export_exists(
            save_path,
            metadata_name=metadata_name,
            expected_metadata=expected_metadata,
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
            if floating_dtype != torch.float32:
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
            if save_path.exists():
                raise RuntimeError(
                    f"evaluation export appeared during publication: {save_path}"
                )
            os.replace(partial_save_path, save_path)
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

    def prepare_export():
        if _complete_hf_export_exists(
            save_path,
            metadata_name="ema_export_metadata.json",
            expected_metadata=expected_metadata,
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
                raise RuntimeError(
                    f"EMA export appeared during publication: {save_path}"
                )
            os.replace(partial_save_path, save_path)
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


def _unwrap_epoch_dataset(dataset):
    ds = dataset
    if hasattr(ds, "set_epoch"):
        return ds
    while hasattr(ds, "dataset"):
        if hasattr(ds, "set_epoch"):
            return ds
        ds = ds.dataset
    return ds


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
    if accelerator.is_main_process:
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
    accelerator.wait_for_everyone()


def _mark_checkpoint_complete(
    checkpoint_dir: Path,
    *,
    accelerator,
    global_step: int,
) -> None:
    if accelerator.is_main_process:
        path = checkpoint_dir / "checkpoint_complete.json"
        temp_path = checkpoint_dir / f".{path.name}.tmp-{os.getpid()}"
        temp_path.write_text(
            json.dumps(
                {
                    "schema": "selfless_caption_checkpoint_complete_v1",
                    "global_step": int(global_step),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    accelerator.wait_for_everyone()


def _begin_checkpoint_write(
    checkpoint_dir: Path,
    *,
    accelerator,
) -> None:
    """Start from a clean destination so stale partial files cannot survive."""

    if accelerator.is_main_process:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
    accelerator.wait_for_everyone()


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
    checkpoint_dir = (
        Path(config.experiment.output_dir)
        / f"checkpoint-{int(global_step)}"
    )
    _begin_checkpoint_write(
        checkpoint_dir,
        accelerator=accelerator,
    )
    save_checkpoint(model, config, accelerator, int(global_step))
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
    )
    _mark_checkpoint_complete(
        checkpoint_dir,
        accelerator=accelerator,
        global_step=int(global_step),
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
    accelerator.wait_for_everyone()


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
        "schema": "selfless_caption_checkpoint_complete_v1",
        "global_step": int(expected_global_step),
    }
    if completion != expected:
        raise RuntimeError(
            "Invalid caption checkpoint completion marker: "
            f"checkpoint={completion}, expected={expected}"
        )


def _set_caption_dataloader_epoch(dataloader, *, epoch: int, seed: int) -> None:
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(int(epoch))
    for candidate in (
        dataloader,
        getattr(dataloader, "dataloader", None),
        getattr(dataloader, "base_dataloader", None),
    ):
        if candidate is None:
            continue
        generator = getattr(candidate, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(int(seed) + int(epoch))
        sampler = getattr(candidate, "sampler", None)
        sampler_generator = getattr(sampler, "generator", None)
        if isinstance(sampler_generator, torch.Generator):
            sampler_generator.manual_seed(int(seed) + int(epoch))


def main(*, model_loader=None):
    #########################
    #      SETUP Config     #
    #########################
    config = get_config()
    validate_wsd_contract(config)

    save_every = int(config.experiment.save_every)
    checkpoint_milestone_every = int(
        config.experiment.get("checkpoint_milestone_every", 0)
    )
    log_every = int(config.experiment.log_every)
    log_grad_norm_every = int(config.experiment.log_grad_norm_every)
    flow_stats_every = int(config.experiment.get("flow_stats_every", 0))
    backbone_gate_stats_every = int(
        config.experiment.get("backbone_gate_stats_every", 0)
    )
    debug_loss_trace_until_step = int(
        config.experiment.get("debug_loss_trace_until_step", 0)
    )
    deepspeed_bf16_overflow_check_until_step = int(
        config.experiment.get(
            "deepspeed_bf16_overflow_check_until_step",
            0,
        )
    )
    if log_every <= 0:
        raise ValueError(f"experiment.log_every must be positive, got {log_every}")
    if log_grad_norm_every <= 0:
        raise ValueError(
            "experiment.log_grad_norm_every must be positive, got "
            f"{log_grad_norm_every}"
        )
    if debug_loss_trace_until_step < 0:
        raise ValueError(
            "experiment.debug_loss_trace_until_step must be non-negative"
        )
    if deepspeed_bf16_overflow_check_until_step < 0:
        raise ValueError(
            "experiment.deepspeed_bf16_overflow_check_until_step must be "
            "non-negative"
        )
    if (
        debug_loss_trace_until_step
        and debug_loss_trace_until_step % log_every
    ):
        raise ValueError(
            "experiment.debug_loss_trace_until_step must end on a log boundary"
        )
    stop_after_steps = training_stop_step(config)
    mixed_source_training = (
        str(config.dataset.class_name) == "UnifiedMixedDataset"
    )
    active_mixed_sources = (
        _active_mixed_source_names(config.dataset.params.schedule)
        if mixed_source_training
        else ()
    )
    save_ema_eval_every = int(
        config.experiment.get("save_ema_eval_every", 0)
    )
    save_model_with_ema_eval = bool(
        config.experiment.get("save_model_with_ema_eval", False)
    )
    if save_ema_eval_every < 0:
        raise ValueError(
            "experiment.save_ema_eval_every must be non-negative, got "
            f"{save_ema_eval_every}"
        )
    if save_model_with_ema_eval and save_ema_eval_every > 0:
        if not _ema_enabled(config):
            raise ValueError(
                "paired current-model/EMA exports require training.use_ema=true"
            )
        if not bool(config.training.get("ema_save_hf_model", True)):
            raise ValueError(
                "paired current-model/EMA exports require "
                "training.ema_save_hf_model=true"
            )
    expected_global_physical_tokens_per_step = (
        _single_source_global_physical_token_budget(
            config,
            active_mixed_sources,
        )
        if mixed_source_training
        else None
    )
    for name, frequency in (
        ("flow_stats_every", flow_stats_every),
        ("backbone_gate_stats_every", backbone_gate_stats_every),
    ):
        if frequency < 0:
            raise ValueError(f"experiment.{name} must be non-negative, got {frequency}")
        if frequency and frequency % log_every:
            raise ValueError(
                f"experiment.{name} must be zero or a multiple of log_every "
                f"({log_every}), got {frequency}"
            )
        
    total_batch_size_per_gpu = config.training.batch_size
    mixed_precision = str(config.training.mixed_precision).lower()
    gradient_accumulation_dtype = str(
        config.training.get("gradient_accumulation_dtype", "fp32")
    ).lower()
    if mixed_precision != "bf16":
        raise ValueError(
            "Selfless-Flow training requires training.mixed_precision='bf16', "
            f"got {config.training.mixed_precision!r}."
        )
    if gradient_accumulation_dtype != "fp32":
        raise ValueError(
            "Selfless-Flow training requires FP32 gradient accumulation, "
            f"got {gradient_accumulation_dtype!r}."
        )
    
    config.experiment.output_dir = os.path.join(config.experiment.output_dir, config.experiment.project)

    #########################
    # SETUP Accelerator     #
    #########################
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    if num_processes <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {num_processes}")
    if (
        expected_global_physical_tokens_per_step is not None
        and expected_global_physical_tokens_per_step % num_processes
    ):
        raise ValueError(
            "expected global physical-token budget must be divisible by "
            f"WORLD_SIZE: {expected_global_physical_tokens_per_step} % "
            f"{num_processes} != 0"
        )
    if mixed_source_training:
        gradient_accumulation_steps = int(
            config.training.gradient_accumulation_steps
        )
        schedule_length = len(config.dataset.params.schedule)
        if gradient_accumulation_steps != schedule_length:
            raise ValueError(
                "Mixed-source gradient accumulation must equal the fixed "
                f"schedule length: {gradient_accumulation_steps} != "
                f"{schedule_length}"
            )
    else:
        global_micro_batch = int(config.training.batch_size) * num_processes
        if int(config.training.total_batch_size) % global_micro_batch:
            raise ValueError(
                "training.total_batch_size must be divisible by batch_size * world_size: "
                f"{config.training.total_batch_size} % ({config.training.batch_size} * {num_processes}) != 0"
            )
        gradient_accumulation_steps = (
            int(config.training.total_batch_size) // global_micro_batch
        )
    tracker_mode = str(os.environ.get("WANDB_MODE", "disabled")).strip().lower()
    use_wandb_tracker = tracker_mode not in {
        "disabled",
        "none",
        "false",
        "0",
    }
    print(f"Number of processes: {num_processes}")
    print(f"Total batch size: {config.training.total_batch_size}")
    print(f"Batch size per GPU: {total_batch_size_per_gpu}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    accumulation_plugin = _gradient_accumulation_plugin(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_source_training=mixed_source_training,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=accumulation_plugin,
        mixed_precision=mixed_precision,
        log_with="wandb" if use_wandb_tracker else None,
        step_scheduler_with_optimizer=config.training.step_scheduler_with_optimizer,
        dataloader_config=DataLoaderConfiguration(
            non_blocking=True,
            use_seedable_sampler=True,
            data_seed=int(
                config.training.get(
                    "dataloader_shuffle_seed", config.training.seed
                )
            ),
        ),
    )
    print(f"Accelerator state: {accelerator.state}")
    print(f"accelerator.gradient_accumulation_steps: {accelerator.gradient_accumulation_steps}")
    print(
        "accelerator.gradient_accumulation_sync_with_dataloader: "
        f"{accelerator.gradient_state.sync_with_dataloader}"
    )
    if (
        mixed_source_training
        and accelerator.gradient_state.sync_with_dataloader
    ):
        raise RuntimeError(
            "UnifiedMixedDataset requires gradient accumulation independent "
            "of inner DataLoader epoch boundaries."
        )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = (
            accelerator.gradient_accumulation_steps
        )
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] = float(
            config.training.max_grad_norm
        )
        # DeepSpeed 0.18.x reads the accumulation dtype from the nested
        # data_types config. A top-level `gradient_accumulation_dtype` key is
        # accepted but silently ignored.
        accelerator.state.deepspeed_plugin.deepspeed_config.setdefault(
            "data_types", {}
        )["grad_accum_dtype"] = gradient_accumulation_dtype
        if deepspeed_bf16_overflow_check_until_step:
            # Accelerate executes DeepSpeedEngine.step() inside backward() on
            # accumulation boundaries. Enable DeepSpeed's pre-update BF16
            # overflow scan for the bounded startup guard so a bad gradient
            # cannot poison the parameters before the trainer can report it.
            accelerator.state.deepspeed_plugin.deepspeed_config.setdefault(
                "bf16", {}
            )["check_grad_overflow"] = True

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # Initialize trackers
    if accelerator.is_main_process and use_wandb_tracker:
        log_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        log_config.pop("experiment.resume_from_checkpoint", None)

        wandb_init_kwargs = {
            "name": config.experiment.name,
            "resume": "allow",
            "mode": tracker_mode,
        }
        accelerator.init_trackers(
            config.experiment.wandb_project,
            config=log_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    # Set training seed
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    #########################
    # MODELS and TOKENIZER  #
    #########################
    logger.info("Loading tokenizer and model")
    if model_loader is None:
        model_loader = load_model_tokenizer
    model, tokenizer = model_loader(
        config=config,
        logger=logger,
        model_dtype=torch.bfloat16,
    )

    flow_adapter = config.model.get("pretrained_image_flow_adapter", None)
    adapter_initialization = load_image_flow_adapter(
        model, flow_adapter, config, log=_log_info
    )
    if accelerator.is_main_process and adapter_initialization is not None:
        output_dir = Path(config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "selfless_split_initialization_v1",
            "text_model_path": str(config.model.model_path),
            "adapter": adapter_initialization,
        }
        path = output_dir / "initialization_report.json"
        temp_path = output_dir / f".{path.name}.tmp-{os.getpid()}"
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    accelerator.wait_for_everyone()

    trainability = _apply_trainable_scope(model, config)
    logger.info(
        "Trainable scope: "
        f"scope={trainability['scope']}, "
        f"trainable={trainability['trainable_numel']:,}, "
        f"frozen={trainability['frozen_numel']:,}, "
        f"total={trainability['total_numel']:,}"
    )

    ema_layout = None
    ema = None
    ema_decay_value = _ema_decay(config) if _ema_enabled(config) else None
    ema_update_after_step = int(config.training.get("ema_update_after_step", 0))
    if ema_update_after_step < 0:
        raise ValueError(f"ema_update_after_step must be >= 0, got {ema_update_after_step}")
    if _ema_enabled(config):
        if not mixed_source_training and bool(config.training.get("ema_validate", False)):
            raise ValueError(
                "training.ema_validate=true is unsupported with rank-sharded EMA: "
                "the EMA exists as distributed FP32 tensor chunks and is not an "
                "executable model. Save/merge the HF EMA checkpoint and evaluate it offline."
            )
        ema_chunk_numel = int(
            config.training.get("ema_shard_chunk_numel", DEFAULT_EMA_CHUNK_NUMEL)
        )
        ema_layout = build_sharded_ema_layout(
            model,
            world_size=accelerator.num_processes,
            chunk_numel=ema_chunk_numel,
        )
        logger.info(
            "Rank-sharded EMA enabled: "
            f"decay={ema_decay_value:g}, update_after_step={ema_update_after_step}, dtype=fp32, "
            f"chunk_numel={ema_chunk_numel}, "
            f"rank_bytes={ema_layout['rank_bytes']}, "
            f"validate={bool(config.training.get('ema_validate', False))}, "
            f"save_adapter={bool(config.training.get('ema_save_adapter', True))}, "
            f"save_hf_model={bool(config.training.get('ema_save_hf_model', True))}"
        )
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            deepspeed_config = accelerator.state.deepspeed_plugin.deepspeed_config
            zero_stage = int(
                deepspeed_config.get("zero_optimization", {}).get(
                    "stage", deepspeed_config.get("zero_stage", -1)
                )
            )
            if zero_stage != 2:
                raise ValueError(
                    "Rank-sharded EMA currently requires DeepSpeed ZeRO-2 because "
                    f"it reads local complete parameters; resolved ZeRO stage is {zero_stage}."
                )

    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params

    # Use lower LR for the pretrained backbone and higher LR for continuous-
    # image modules. Decay flow-head matrix weights while keeping biases,
    # normalization parameters, and the small input projectors decay-free.
    base_lr = float(optimizer_config.learning_rate)
    backbone_lr = float(optimizer_config.get("backbone_learning_rate", base_lr))
    flow_lr = float(optimizer_config.get("flow_learning_rate", base_lr))
    projector_lr = float(optimizer_config.get("projector_learning_rate", flow_lr))
    special_token_lr = float(optimizer_config.get("special_token_learning_rate", projector_lr))
    global_weight_decay = float(optimizer_config.weight_decay)
    flow_weight_decay = float(
        optimizer_config.get("flow_weight_decay", global_weight_decay)
    )
    grouped = {}
    optimizer_role_numel = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        weight_decay = weight_decay_for_parameter(
            name,
            global_weight_decay=global_weight_decay,
            flow_weight_decay=flow_weight_decay,
        )
        role = optimizer_parameter_role(name)
        optimizer_role_numel[role] = optimizer_role_numel.get(role, 0) + int(
            param.numel()
        )
        learning_rate = learning_rate_for_parameter(
            name,
            backbone_lr=backbone_lr,
            flow_lr=flow_lr,
            projector_lr=projector_lr,
            special_token_lr=special_token_lr,
        )
        key = (learning_rate, weight_decay)
        grouped.setdefault(key, []).append(param)

    optimizer_grouped_parameters = [
        {"params": params, "lr": lr, "weight_decay": weight_decay}
        for (lr, weight_decay), params in grouped.items()
    ]
    logger.info(
        "Optimizer LRs: "
        f"backbone={backbone_lr:g}, image_token_embedder/image_flow_condition_proj={projector_lr:g}, "
        f"image_flow_head={flow_lr:g}; "
        f"special_tokens={special_token_lr:g}; "
        f"weight_decay={global_weight_decay:g}, "
        f"flow_weight_decay={flow_weight_decay:g}"
    )
    tied_embedding = model.lm_head.weight is model.model.embed_tokens.weight
    if not tied_embedding:
        raise RuntimeError(
            "Joint training expects lm_head.weight and embed_tokens.weight to be tied"
        )
    logger.info(
        "Optimizer parameter coverage: "
        + ", ".join(
            f"{role}={optimizer_role_numel.get(role, 0):,}"
            for role in (
                "backbone",
                "tied_lm_head_embedding",
                "image_projector",
                "flow_head",
            )
        )
        + "; lm_head/embed_tokens tied=true; special_token_learning_rate "
        "applies to the complete tied matrix"
    )

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        num_decay_steps=config.lr_scheduler.params.decay_steps,
        num_training_steps=config.training.max_train_steps,
        min_lr_ratio=config.lr_scheduler.params.min_lr_scale
    )
    logger.info(
        "WSD schedule: "
        f"warmup_steps={int(config.lr_scheduler.params.warmup_steps)}, "
        f"stable_steps={int(config.training.max_train_steps) - int(config.lr_scheduler.params.warmup_steps) - int(config.lr_scheduler.params.decay_steps)}, "
        f"warmdown_steps={int(config.lr_scheduler.params.decay_steps)}, "
        f"min_lr_scale={float(config.lr_scheduler.params.min_lr_scale):g}"
    )

    ##################################
    #         DATALOADER             #
    ##################################
    logger.info("Creating dataloaders and lr_scheduler")

    train_dataloader, val_dataloader = get_dataloaders(config, tokenizer)
    if mixed_source_training:
        description = train_dataloader.runtime_description()
        if tuple(train_dataloader.active_sources) != active_mixed_sources:
            raise RuntimeError(
                "mixed DataLoader active sources differ from the schedule: "
                f"runtime={train_dataloader.active_sources}, "
                f"expected={active_mixed_sources}"
            )
        # Training validation uses fixed downstream manifests on all ranks.
        # The paired image loader remains available to standalone full evaluation.
        val_dataloader = None
        logger.info("Mixed DataLoader runtime: %s", description)
    else:
        configured_workers = int(config.training.dataloader_workers)
        for loader_name, dataloader in (
            ("train", train_dataloader),
            ("validation", val_dataloader),
        ):
            if int(dataloader.num_workers) != configured_workers:
                raise RuntimeError(
                    f"{loader_name} DataLoader worker mismatch: "
                    f"configured={configured_workers}, runtime={dataloader.num_workers}"
                )
            if configured_workers == 0 and (
                dataloader.persistent_workers
                or dataloader.prefetch_factor is not None
            ):
                raise RuntimeError(
                    f"{loader_name} DataLoader must disable worker persistence and "
                    "prefetching when dataloader_workers=0: "
                    f"persistent_workers={dataloader.persistent_workers}, "
                    f"prefetch_factor={dataloader.prefetch_factor}"
                )
        logger.info(
            "DataLoader runtime: workers=%d, train_persistent=%s, "
            "train_prefetch=%s, validation_persistent=%s, validation_prefetch=%s",
            configured_workers,
            train_dataloader.persistent_workers,
            train_dataloader.prefetch_factor,
            val_dataloader.persistent_workers,
            val_dataloader.prefetch_factor,
        )

    ##################################
    #       Prepare accelerator     #
    ##################################
    logger.info("Preparing model, optimizer and dataloaders")

    # Store ref to underlying packed dataset for epoch-level reshuffling/repacking.
    ds = (
        None
        if mixed_source_training
        else _unwrap_epoch_dataset(train_dataloader.dataset)
    )
    _is_multimodal_ds = ds is not None and hasattr(ds, 'set_epoch')

    if hasattr(train_dataloader, "prepare_with_accelerator"):
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        train_dataloader = train_dataloader.prepare_with_accelerator(accelerator)
        if val_dataloader is not None:
            val_dataloader = accelerator.prepare_data_loader(val_dataloader)
    else:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    epoch_sample_budget = config.training.get("samples_per_epoch", None)
    if epoch_sample_budget is not None:
        epoch_sample_budget = int(epoch_sample_budget)
        prepared_microbatches = len(train_dataloader)
        expected_microbatches = epoch_sample_budget // (
            int(config.training.batch_size) * accelerator.num_processes
        )
        if prepared_microbatches != expected_microbatches:
            raise RuntimeError(
                "Prepared DataLoader violates the exact epoch sample budget: "
                f"runtime_microbatches={prepared_microbatches}, "
                f"expected_microbatches={expected_microbatches}, "
                f"samples_per_epoch={epoch_sample_budget}"
            )
        accumulation_steps = accelerator.gradient_accumulation_steps
        if prepared_microbatches % accumulation_steps:
            raise RuntimeError(
                "Prepared DataLoader would create a partial gradient-"
                "accumulation step at the epoch boundary: "
                f"microbatches={prepared_microbatches}, "
                f"gradient_accumulation_steps={accumulation_steps}"
            )
        optimizer_steps_per_epoch = prepared_microbatches // accumulation_steps
        configured_steps_per_epoch = int(
            config.training.optimizer_steps_per_epoch
        )
        if optimizer_steps_per_epoch != configured_steps_per_epoch:
            raise RuntimeError(
                "Optimizer steps/epoch mismatch: "
                f"runtime={optimizer_steps_per_epoch}, "
                f"configured={configured_steps_per_epoch}"
            )
        num_train_epochs = int(config.training.num_train_epochs)
        expected_training_steps = optimizer_steps_per_epoch * num_train_epochs
        if int(config.training.max_train_steps) != expected_training_steps:
            raise RuntimeError(
                "Exact epoch contract does not match max_train_steps: "
                f"{optimizer_steps_per_epoch} * {num_train_epochs} = "
                f"{expected_training_steps}, configured="
                f"{int(config.training.max_train_steps)}"
            )
        logger.info(
            "Exact epoch contract: samples=%d, prepared_microbatches/rank=%d, "
            "gradient_accumulation=%d, optimizer_steps/epoch=%d, epochs=%d",
            epoch_sample_budget,
            prepared_microbatches,
            accumulation_steps,
            optimizer_steps_per_epoch,
            num_train_epochs,
        )
    if ema_layout is not None:
        ema = RankShardedEMA(
            ema_layout,
            rank=accelerator.process_index,
            decay=ema_decay_value,
            update_after_step=ema_update_after_step,
        )
        ema.bind(accelerator.unwrap_model(model))

    config_contract = build_resume_contract(
        config,
        world_size=accelerator.num_processes,
        gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
    )

    ##################################
    #       MODEL RESUME         #
    ##################################
    global_step = 0
    resume_step = 0
    resume_checkpoint_dir = None
    resume_metadata = None

    if not _is_disabled_path(config.experiment.resume_from_checkpoint):
        candidate_path = Path(config.experiment.resume_from_checkpoint)
        if candidate_path.exists():
            resume_checkpoint_dir = candidate_path
        else:
            raise FileNotFoundError(
                f"Specified resume checkpoint does not exist: {candidate_path}"
            )

    if resume_checkpoint_dir and resume_checkpoint_dir.exists():
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint_dir}")
        metadata_file = resume_checkpoint_dir / "metadata.json"
        if not metadata_file.is_file():
            raise RuntimeError(
                f"Refusing to resume a checkpoint without metadata: {metadata_file}"
            )
        resume_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        resume_step = int(resume_metadata["global_step"])
        _validate_checkpoint_complete(
            resume_checkpoint_dir,
            expected_global_step=resume_step,
        )
        if int(resume_metadata.get("world_size", -1)) != accelerator.num_processes:
            raise RuntimeError(
                "Caption resume requires the same world size: "
                f"checkpoint={resume_metadata.get('world_size')}, "
                f"current={accelerator.num_processes}"
            )
        validate_resume_contract(
            resume_metadata,
            current_contract=config_contract,
        )
        # Model, optimizer, scheduler, and RNG state are loaded only after the
        # immutable resume contract has been validated.
        accelerator.load_state(resume_checkpoint_dir)
        _restore_npu_rng_state(resume_checkpoint_dir, accelerator)
        if mixed_source_training:
            train_dataloader.load_state(
                resume_checkpoint_dir,
                accelerator,
                resume_step,
            )
        global_step = resume_step
        logger.info(f"Resumed at global_step={global_step}")

    else:
        logger.info("Starting fresh caption training.")
        global_step = 0
        resume_step = 0

    cumulative_wall_seconds_before_run = float(
        (resume_metadata or {}).get("cumulative_training_wall_seconds", 0.0)
    )
    cumulative_loss_checks_before_run = int(
        (resume_metadata or {}).get(
            "cumulative_finite_loss_microbatches_checked", 0
        )
    )
    if ema is not None:
        if resume_checkpoint_dir:
            ema.load_checkpoint(
                resume_checkpoint_dir,
                accelerator,
                expected_global_step=global_step,
            )
            logger.info(
                f"Loaded rank {accelerator.process_index} EMA shard at global_step={global_step}."
            )
        else:
            ema.initialize_from_model(global_step=global_step)
            logger.info("Initialized this rank's FP32 EMA shard from the training model.")
        if ema.started:
            logger.info(f"EMA is active at global_step={global_step}.")
        else:
            logger.info(
                "EMA updates are delayed until "
                f"global_step={ema_update_after_step}; validation and adapter saves will use the training model until then."
            )

    ##################################
    #             Training           #
    ##################################
    if mixed_source_training:
        per_rank_rows_per_update = sum(
            int(config.dataset.params.sources[source].micro_batch_size)
            for source in config.dataset.params.schedule
        )
        total_batch_size = per_rank_rows_per_update * accelerator.num_processes
    else:
        total_batch_size = (
            total_batch_size_per_gpu
            * accelerator.num_processes
            * accelerator.gradient_accumulation_steps
        )
    logger.info("***** Running selfless pretraining *****")
    logger.info(f"  WSD training horizon = {config.training.max_train_steps}")
    logger.info(f"  Stop after step = {stop_after_steps}")
    if mixed_source_training:
        logger.info(
            "  Fixed source schedule = %s",
            list(config.dataset.params.schedule),
        )
        logger.info(
            "  Physical rows per optimizer update across all ranks = %d",
            total_batch_size,
        )
    else:
        logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    if save_ema_eval_every > 0:
        logger.info(
            "  Periodic evaluation export = every %d steps (%s)",
            save_ema_eval_every,
            (
                "current model + EMA model"
                if save_model_with_ema_eval
                else "EMA model only"
            ),
        )
    logger.info(f"  mask_token_id: {config.model.mask_token_id}")
    
    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        if not config_path.exists():
            logging.info(f"Saving immutable run config to {config_path}")
            OmegaConf.save(config, config_path)
        else:
            logging.info(f"Keeping existing immutable run config at {config_path}")

    caption_shuffle_seed = int(
        config.training.get("dataloader_shuffle_seed", config.training.seed)
    )
    batches_to_skip = 0
    resume_epoch = 0
    initial_train_dataloader = train_dataloader
    if resume_step > 0 and not mixed_source_training:
        resume_epoch = int(resume_metadata["epoch"])
        batches_to_skip = int(resume_metadata["batches_consumed_in_epoch"])
        try:
            dataloader_len = len(train_dataloader)
        except TypeError:
            dataloader_len = 0
        if dataloader_len <= 0 or batches_to_skip < 0 or batches_to_skip > dataloader_len:
            raise RuntimeError(
                "Invalid caption dataloader resume offset: "
                f"epoch={resume_epoch}, offset={batches_to_skip}, len={dataloader_len}"
            )
        logger.info(
            f"Resuming from step {resume_step}: dataloader_len={dataloader_len}, "
            f"resume_epoch={resume_epoch}, skipping {batches_to_skip} prepared batches."
        )
        sampler_state = resume_metadata.get("sampler_state")
        if sampler_state is not None:
            validate_sampler_resume_state(
                sampler_state,
                epoch=resume_epoch,
                batches_consumed_in_epoch=batches_to_skip,
                shuffle_seed=caption_shuffle_seed,
                prepared_dataloader_length=dataloader_len,
            )
            logger.info(
                "Validated deterministic sampler state: epoch=%d, offset=%d, seed=%d.",
                resume_epoch,
                batches_to_skip,
                caption_shuffle_seed,
            )
        else:
            logger.info(
                "Checkpoint predates explicit sampler metadata; restoring its "
                "validated epoch/offset cursor deterministically."
            )
        if _is_multimodal_ds:
            ds.set_epoch(resume_epoch)
    if not mixed_source_training:
        _set_caption_dataloader_epoch(
            train_dataloader,
            epoch=resume_epoch,
            seed=caption_shuffle_seed,
        )
        if batches_to_skip > 0:
            initial_train_dataloader = accelerator.skip_first_batches(
                train_dataloader,
                batches_to_skip,
            )
    elif resume_step > 0:
        logger.info(
            "Restored rank-local mixed source cursors at optimizer step %d.",
            resume_step,
        )

    model.train()

    train_iter = iter(initial_train_dataloader)
    training_window = TrainingWindow()
    training_runtime_started_at = time.time()
    training_runtime_start_step = global_step
    if accelerator.device.type == "npu":
        torch.npu.reset_peak_memory_stats(accelerator.device)
    elif accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    # Accumulators across gradient-accumulation micro-batches.
    acc_loss = torch.tensor(0.0, device=accelerator.device)
    acc_text_loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_text_tokens = torch.tensor(0.0, device=accelerator.device)
    acc_image_loss_sum = torch.tensor(0.0, device=accelerator.device)
    acc_image_tokens = torch.tensor(0.0, device=accelerator.device)
    acc_physical_token_positions = (
        0 if expected_global_physical_tokens_per_step is not None else None
    )
    acc_source_loss_sum = {
        source: torch.tensor(0.0, device=accelerator.device)
        for source in active_mixed_sources
    }
    acc_source_target_count = {
        source: torch.tensor(0.0, device=accelerator.device)
        for source in active_mixed_sources
    }
    acc_source_weighted_loss_sum = {
        source: torch.tensor(0.0, device=accelerator.device)
        for source in active_mixed_sources
    }
    acc_flow_stats = {}
    acc_flow_stat_batches = torch.tensor(0.0, device=accelerator.device)
    acc_backbone_gate_stats = {}
    acc_backbone_gate_stat_batches = torch.tensor(
        0.0, device=accelerator.device
    )
    finite_loss_microbatches_checked = 0
    debug_loss_trace: list[torch.Tensor] = []
    last_logged_loss = None
    actual_global_physical_tokens_per_step = None
    source_microbatches_window = {
        source: 0 for source in active_mixed_sources
    }
    source_data_wait_window = {
        source: 0.0 for source in active_mixed_sources
    }

    epoch = resume_epoch
    batches_consumed_in_epoch = batches_to_skip
    while global_step < stop_after_steps:
        data_wait_started = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            batches_consumed_in_epoch = 0
            if _is_multimodal_ds:
                ds.set_epoch(epoch)
            _set_caption_dataloader_epoch(
                train_dataloader,
                epoch=epoch,
                seed=caption_shuffle_seed,
            )
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        batches_consumed_in_epoch += 1
        data_wait_seconds = time.perf_counter() - data_wait_started
        
        # *-------*-------*-------*-------*-------*-------*
        # Data Processing
        # *-------*-------*-------*-------*-------*-------*
        source_name = str(batch.get("source_name", "imagenet"))
        if (
            mixed_source_training
            and source_name not in active_mixed_sources
        ):
            raise RuntimeError(
                "batch source is not active in the mixed training schedule: "
                f"source_name={source_name!r}, "
                f"active_sources={active_mixed_sources}"
            )
        if mixed_source_training:
            source_microbatches_window[source_name] += 1
            source_data_wait_window[source_name] += data_wait_seconds
        is_multimodal = "token_types" in batch
        if not is_multimodal:
            raise ValueError(
                "Selfless-Flow training requires a token_types-aware text or "
                "image batch."
            )
        if is_multimodal:
            input_ids = batch["input_ids"].contiguous().to(
                accelerator.device, non_blocking=True
            )  # [B, L] — no shift for selfless
            token_types = batch["token_types"].to(
                accelerator.device, non_blocking=True
            )  # [B, L]
            sigma = batch["sigma"].to(
                accelerator.device, non_blocking=True
            )  # [B, L], pre-computed by dataloader
            labels = batch["labels"].to(
                accelerator.device, non_blocking=True
            )  # [B, L], pre-computed by dataloader
            image_loss_mask = batch["image_loss_mask"].to(
                accelerator.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            segment_ids = batch.get("segment_ids", None)
            if segment_ids is not None:
                segment_ids = segment_ids.to(accelerator.device, non_blocking=True)
            position_ids = batch.get("position_ids", None)
            if position_ids is not None:
                position_ids = position_ids.to(accelerator.device, non_blocking=True)
            image_local_positions = batch.get("image_local_positions", None)
            if image_local_positions is not None:
                image_local_positions = image_local_positions.to(
                    accelerator.device, non_blocking=True
                )
            image_span_table = batch.get("image_span_table", None)
            logical_images = int(image_span_table.shape[0]) if image_span_table is not None else 0
            if image_span_table is not None:
                image_span_table = image_span_table.to(
                    accelerator.device, non_blocking=True
                )
            image_latents = batch.get("image_latents", None)
            if image_latents is not None:
                image_latents = image_latents.to(
                    accelerator.device, non_blocking=True
                )
            pack_stats = batch.get("pack_stats", None)
            B, L = input_ids.shape
            training_window.record_batch(
                rows=B,
                sequence_length=L,
                logical_images=logical_images,
                pack_stats=pack_stats,
                data_wait_seconds=data_wait_seconds,
            )
            if acc_physical_token_positions is not None:
                acc_physical_token_positions += int(B * L)

            image_uncond_rows = None
            image_uncond_mask = batch.get("image_uncond_mask", None)
            if image_uncond_mask is not None:
                image_uncond_mask = image_uncond_mask.to(
                    accelerator.device, dtype=torch.bool, non_blocking=True
                )
            image_uncond_prob = float(config.model.get("image_uncond_prob", 0.0))
            if image_uncond_mask is None and image_uncond_prob > 0.0:
                has_image = (token_types == 1).any(dim=1)
                sampled_rows = (
                    torch.rand(B, device=accelerator.device) < image_uncond_prob
                ) & has_image
                image_uncond_rows = sampled_rows

            image_loss_mask, image_latent_mask, showo_mask_prob = (
                _prepare_showo_image_masks(
                    config=config,
                    token_types=token_types,
                    image_span_table=image_span_table,
                    image_loss_mask=image_loss_mask,
                    mask_generation_images=(source_name == "t2i"),
                )
            )
            selfless_attention_mask, content_attention_mask = (
                _build_backbone_attention_masks(
                    config=config,
                    input_ids=input_ids,
                    token_types=token_types,
                    sigma=sigma,
                    segment_ids=segment_ids,
                    image_uncond_rows=image_uncond_rows,
                    image_uncond_mask=image_uncond_mask,
                )
            )

            if global_step == 0 and accelerator.is_main_process and not hasattr(main, '_logged_first_batch'):
                main._logged_first_batch = True
                logger.info(f"Input ids shape: {input_ids.shape}, multimodal mode")
                logger.info(f"token type counts: text={(token_types==0).sum().item()}, "
                           f"image={(token_types==1).sum().item()}, "
                           f"special={(token_types==2).sum().item()}, "
                           f"padding={(token_types==3).sum().item()}")
                logger.info(f"sigma range: [{sigma.min().item()}, {sigma.max().item()}], "
                           f"labels -100 ratio: {(labels==-100).sum().item() / labels.numel():.3f}")
                if image_uncond_rows is not None:
                    logger.info(
                        f"image-uncond attention rows in first batch: "
                        f"{int(image_uncond_rows.sum().item())}/{B}"
                    )
                if image_uncond_mask is not None:
                    logger.info(
                        "image-uncond packed image tokens in first batch: "
                        f"{int(image_uncond_mask.sum().item())}"
                    )
                if showo_mask_prob is not None:
                    logger.info(
                        "Show-O MAE mask ratio in first batch: "
                        f"mean={showo_mask_prob.mean().item():.4f}, "
                        f"masked={int(image_loss_mask.sum().item())}"
                    )
                if pack_stats is not None:
                    valid_tokens, image_tokens, padding_tokens, packed_len = map(
                        int, pack_stats
                    )
                    logger.info(
                        f"pack stats: valid={valid_tokens}, image={image_tokens}, "
                        f"padding={padding_tokens}, L={packed_len}, "
                        f"padding ratio={padding_tokens / max(1, B * L):.3f}"
                    )
                if segment_ids is not None:
                    pack_details = batch.get("pack_details", None)
                    if pack_details is not None:
                        (
                            image_count,
                            row_count,
                            pack_capacity,
                            overflow_count,
                        ) = map(int, pack_details)
                        logger.info(
                            "segment pack: "
                            f"images={image_count}, rows={row_count}, "
                            f"capacity={pack_capacity}, "
                            f"overflow_rows={overflow_count}"
                        )
        # *-------*-------*-------*-------*-------*-------*
        # Forward & Backward
        # *-------*-------*-------*-------*-------*-------*
        grad_norm_value = None
        with accelerator.accumulate(model):
            forward_kwargs = {
                "X0_input_ids": input_ids,
                "labels": labels if is_multimodal else input_ids,
                "attention_mask": selfless_attention_mask,
            }
            if content_attention_mask is not None:
                forward_kwargs["content_attention_mask"] = (
                    content_attention_mask
                )
            if segment_ids is not None:
                forward_kwargs["_text_segment_ids"] = segment_ids
            if token_types is not None:
                forward_kwargs["token_types"] = token_types
                forward_kwargs["flow_sigma"] = sigma
                if mixed_source_training:
                    forward_kwargs["compute_text_loss"] = source_name in {
                        "climbmix",
                        "i2t",
                    }
                    forward_kwargs["compute_image_loss"] = (
                        source_name == "t2i"
                    )
                if position_ids is not None:
                    forward_kwargs["position_ids"] = position_ids
                if image_local_positions is not None:
                    forward_kwargs["image_local_positions"] = image_local_positions
                if image_span_table is not None:
                    forward_kwargs["image_span_table"] = image_span_table
                forward_kwargs["image_loss_mask"] = image_loss_mask
                if image_latent_mask is not None:
                    forward_kwargs["image_latent_mask"] = image_latent_mask
            if is_multimodal and image_latents is not None:
                forward_kwargs["image_latents"] = image_latents
            record_backbone_gate_stats = (
                str(config.model.get("backbone_attention_output_gate", "none"))
                != "none"
                and backbone_gate_stats_every > 0
                and accelerator.sync_gradients
                and (global_step + 1) % backbone_gate_stats_every == 0
            )
            if record_backbone_gate_stats:
                forward_kwargs["record_backbone_gate_stats"] = True
                forward_kwargs["backbone_gate_stats_level"] = "summary"
            forward_kwargs["record_flow_stats"] = (
                flow_stats_every > 0
                and accelerator.sync_gradients
                and (global_step + 1) % flow_stats_every == 0
            )
            model_output = model(**forward_kwargs)
            loss = model_output.loss
            # Every microbatch loss is accumulated below and the complete
            # window is checked when it is reduced for logging.  Do not use
            # torch._assert_async here: torch-npu 2.6.0 falls back to CPU for
            # aten::_assert_async.msg, introducing a host/device synchronization
            # on every microbatch.
            finite_loss_microbatches_checked += 1

            per_modality_loss = getattr(
                model_output, "per_modality_loss", None
            )
            per_modality_count = getattr(
                model_output, "per_modality_count", None
            )
            if per_modality_loss is None or per_modality_count is None:
                raise RuntimeError(
                    "joint image-text training requires per-modality loss/count output"
                )
            if 0 < global_step + 1 <= debug_loss_trace_until_step:
                debug_loss_trace.append(
                    torch.stack(
                        (
                            loss.detach().float(),
                            per_modality_loss["text_loss"].detach().float(),
                            per_modality_loss["image_loss"].detach().float(),
                        )
                    )
                )
            text_count = per_modality_count["text_tokens"].detach().to(
                accelerator.device, dtype=torch.float32
            )
            image_count = per_modality_count["image_tokens"].detach().to(
                accelerator.device, dtype=torch.float32
            )
            acc_text_loss_sum += (
                per_modality_loss["text_loss"].detach().float() * text_count
            )
            acc_text_tokens += text_count
            acc_image_loss_sum += (
                per_modality_loss["image_loss"].detach().float() * image_count
            )
            acc_image_tokens += image_count
            if mixed_source_training:
                source_task_loss, source_target_count = (
                    _source_task_loss_and_count(
                        source_name,
                        per_modality_loss=per_modality_loss,
                        text_count=text_count,
                        image_count=image_count,
                    )
                )
                acc_source_loss_sum[source_name] += (
                    source_task_loss.detach().float() * source_target_count
                )
                acc_source_target_count[source_name] += source_target_count
                acc_source_weighted_loss_sum[source_name] += (
                    loss.detach().float()
                )

            flow_stats = getattr(model_output, "flow_debug_stats", None)
            if flow_stats:
                for key, value in flow_stats.items():
                    acc_flow_stats[key] = acc_flow_stats.get(
                        key, torch.tensor(0.0, device=accelerator.device)
                    ) + value.detach().to(accelerator.device)
                acc_flow_stat_batches += 1
            gate_stats = getattr(model_output, "backbone_gate_stats", None)
            if gate_stats:
                for key, value in gate_stats.items():
                    acc_backbone_gate_stats[key] = (
                        acc_backbone_gate_stats.get(
                            key,
                            torch.tensor(
                                0.0, device=accelerator.device
                            ),
                        )
                        + value.detach().to(accelerator.device)
                    )
                acc_backbone_gate_stat_batches += 1
            acc_loss += loss.detach()

            # Resolve the bounded startup trace before the final backward of
            # an accumulation window.  A non-finite forward value must be
            # reported before DeepSpeed's gradient-overflow guard can obscure
            # its source, and every rank must enter the gather together.
            if accelerator.sync_gradients and debug_loss_trace:
                local_trace = torch.stack(debug_loss_trace)
                gathered_trace = accelerator.gather(local_trace).reshape(
                    accelerator.num_processes,
                    local_trace.shape[0],
                    local_trace.shape[1],
                )
                trace_details = _debug_nonfinite_loss_trace_details(
                    gathered_trace,
                    ending_global_step=global_step + 1,
                    gradient_accumulation_steps=(
                        accelerator.gradient_accumulation_steps
                    ),
                    source_schedule=(
                        tuple(train_dataloader.schedule)
                        if mixed_source_training
                        else ()
                    ),
                )
                debug_loss_trace.clear()
                if trace_details:
                    raise FloatingPointError(
                        "non-finite training loss trace before backward at "
                        f"global_step={global_step + 1}: "
                        + "; ".join(trace_details[:32])
                    )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if expected_global_physical_tokens_per_step is not None:
                    if acc_physical_token_positions is None:
                        raise AssertionError(
                            "physical-token accumulator was not initialized"
                        )
                    per_rank_physical_tokens = accelerator.gather(
                        torch.tensor(
                            [acc_physical_token_positions],
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                    )
                    per_rank_values = [
                        int(value)
                        for value in per_rank_physical_tokens.tolist()
                    ]
                    actual_global_physical_tokens_per_step = (
                        _validate_source_physical_token_budget(
                            per_rank_values,
                            expected_global=(
                                expected_global_physical_tokens_per_step
                            ),
                            source_name=active_mixed_sources[0],
                        )
                    )
                if (
                    accelerator.distributed_type != DistributedType.DEEPSPEED
                    and config.training.max_grad_norm
                ):
                    grad_norm_value = accelerator.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm
                    )
                    if (
                        (global_step + 1)
                        % log_grad_norm_every
                        == 0
                        and not bool(
                            torch.isfinite(torch.as_tensor(grad_norm_value)).all()
                        )
                    ):
                        raise FloatingPointError(
                            f"non-finite gradient norm at global_step={global_step}"
                        )
                
                optimizer.step()
                if (
                    0
                    < global_step + 1
                    <= deepspeed_bf16_overflow_check_until_step
                ):
                    # Accelerate invokes DeepSpeedEngine.step() inside
                    # accelerator.backward() at a synchronization boundary;
                    # its optimizer wrapper's step() above is intentionally a
                    # no-op. ZeRO has therefore already populated ``overflow``
                    # here, before scheduler/EMA state advances.
                    zero_optimizer = getattr(model, "optimizer", None)
                    if zero_optimizer is None:
                        raise RuntimeError(
                            "startup BF16 overflow guard requires the "
                            "DeepSpeed ZeRO optimizer"
                        )
                    if bool(getattr(zero_optimizer, "overflow", False)):
                        raise FloatingPointError(
                            "DeepSpeed detected a non-finite BF16 gradient and "
                            "skipped the optimizer update: "
                            f"next_global_step={global_step + 1}"
                        )
                    if (
                        global_step + 1
                        == deepspeed_bf16_overflow_check_until_step
                        and hasattr(zero_optimizer, "check_grad_overflow")
                    ):
                        zero_optimizer.check_grad_overflow = False
                        if accelerator.is_main_process:
                            logger.info(
                                "Startup BF16 overflow-check window passed; "
                                "disabled its extra scan for subsequent "
                                "optimizer steps."
                            )
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if (
                    accelerator.distributed_type == DistributedType.DEEPSPEED
                    and (global_step + 1)
                    % log_grad_norm_every
                    == 0
                ):
                    if hasattr(model, "get_global_grad_norm"):
                        deepspeed_grad_norm = model.get_global_grad_norm()
                    else:
                        deepspeed_grad_norm = None
                    if deepspeed_grad_norm is None:
                        raise RuntimeError("DeepSpeed gradient norm is unavailable")
                    if not bool(
                        torch.isfinite(torch.as_tensor(deepspeed_grad_norm)).all()
                    ):
                        raise FloatingPointError(
                            f"non-finite DeepSpeed gradient norm at global_step={global_step}"
                        )
                    grad_norm_value = deepspeed_grad_norm
                if ema is not None:
                    next_step = global_step + 1
                    was_ema_started = ema.started
                    ema.maybe_update(next_step)
                    if ema.started and not was_ema_started and accelerator.is_main_process:
                        logger.info(f"Started EMA at global_step={next_step} by syncing current model weights.")

        # *-------*-------*-------*-------*-------*-------*
        # Logging & Saving & Validation
        # *-------*-------*-------*-------*-------*-------*
        if accelerator.sync_gradients:
            global_step += 1
            training_window.record_optimizer_step()
            did_log = global_step % log_every == 0
            sampled_grad_norm_value = None
            grad_norm_logs = (
                gradient_norm_log_payload(
                    global_step=global_step,
                    every=log_grad_norm_every,
                    pre_clip_norm=grad_norm_value,
                    max_norm=float(config.training.max_grad_norm),
                )
                if grad_norm_value is not None
                else None
            )
            if grad_norm_logs is not None:
                sampled_grad_norm_value = grad_norm_logs[
                    "train/global_grad_norm_pre_clip"
                ]
                clip_limit = grad_norm_logs["train/grad_clip_max_norm"]
                accelerator.log(grad_norm_logs, step=global_step)
                if accelerator.is_main_process:
                    logger.info(
                        "GradientNorm: Step: %d | PreClip: %.6f | MaxNorm: %.6f | Clipped: %s",
                        global_step,
                        sampled_grad_norm_value,
                        clip_limit,
                        bool(grad_norm_logs["train/grad_clip_applied"]),
                    )

            # Logging
            if did_log:
                grad_accum = accelerator.gradient_accumulation_steps
                # Average loss across all micro-batches and all ranks
                avg_loss_per_step = acc_loss / grad_accum
                global_avg_loss = accelerator.reduce(avg_loss_per_step, reduction="mean")
                global_avg_loss_value = float(global_avg_loss.item())
                if not math.isfinite(global_avg_loss_value):
                    raise FloatingPointError(
                        "non-finite accumulated training loss at "
                        f"global_step={global_step}"
                    )
                last_logged_loss = global_avg_loss_value

                window_rows = accelerator.gather(
                    training_window.as_tensor(accelerator.device)
                ).reshape(-1, 10)
                totals = window_rows[:, :9].sum(dim=0)
                elapsed = window_rows[:, 9].max()
                summary_values = torch.cat((totals, elapsed.unsqueeze(0))).tolist()
                (
                    optimizer_steps_total,
                    micro_batches_total,
                    logical_images_total,
                    physical_rows_total,
                    physical_tokens_total,
                    valid_tokens_total,
                    image_tokens_total,
                    padding_tokens_total,
                    data_wait_total,
                    window_seconds,
                ) = summary_values
                world_size = float(accelerator.num_processes)
                window_seconds = max(window_seconds, 1.0e-12)
                optimizer_steps_per_rank = max(
                    optimizer_steps_total / world_size,
                    1.0,
                )
                micro_batches_total = max(micro_batches_total, 1.0)
                samples_per_second_per_gpu = (
                    logical_images_total / world_size / window_seconds
                )

                logs = {
                    "step_loss": global_avg_loss_value,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "physical_rows/sec/gpu": (
                        physical_rows_total / world_size / window_seconds
                    ),
                    "tokens/sec/gpu": (
                        physical_tokens_total / world_size / window_seconds
                    ),
                    "valid_tokens/sec/gpu": (
                        valid_tokens_total / world_size / window_seconds
                    ),
                    "image_tokens/sec/gpu": (
                        image_tokens_total / world_size / window_seconds
                    ),
                    "seconds/optimizer_step": (
                        window_seconds / optimizer_steps_per_rank
                    ),
                    "data_wait_ms/microbatch": (
                        1000.0 * data_wait_total / micro_batches_total
                    ),
                    "data_wait_fraction": (
                        data_wait_total / world_size / window_seconds
                    ),
                }
                if actual_global_physical_tokens_per_step is not None:
                    logs[
                        "train/global_physical_tokens_per_optimizer_step"
                    ] = float(actual_global_physical_tokens_per_step)
                source_metric_display = {}
                if mixed_source_training:
                    local_source_window = torch.tensor(
                        [
                            value
                            for source in active_mixed_sources
                            for value in (
                                source_microbatches_window[source],
                                source_data_wait_window[source],
                            )
                        ],
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    global_source_window = accelerator.reduce(
                        local_source_window, reduction="sum"
                    ).tolist()
                    for index, source in enumerate(active_mixed_sources):
                        microbatches = global_source_window[2 * index]
                        wait_seconds = global_source_window[2 * index + 1]
                        logs[f"source/{source}_microbatches"] = microbatches
                        logs[f"source/{source}_data_wait_ms"] = (
                            1000.0 * wait_seconds / max(microbatches, 1.0)
                        )
                    local_source_loss_totals = torch.stack(
                        [
                            value
                            for source in active_mixed_sources
                            for value in (
                                acc_source_loss_sum[source],
                                acc_source_target_count[source],
                                acc_source_weighted_loss_sum[source],
                            )
                        ]
                    )
                    reduced_source_loss_totals = accelerator.reduce(
                        local_source_loss_totals,
                        reduction="sum",
                    )
                    source_logs, source_metric_display = (
                        _source_loss_metric_payload(
                            reduced_source_loss_totals,
                            num_processes=accelerator.num_processes,
                            gradient_accumulation_steps=grad_accum,
                            active_sources=active_mixed_sources,
                        )
                    )
                    logs.update(source_logs)
                    weighted_contribution_total = sum(
                        contribution
                        for _, contribution in source_metric_display.values()
                    )
                    logs["train/weighted_contribution_total"] = (
                        weighted_contribution_total
                    )
                    logs["train/weighted_contribution_residual"] = (
                        global_avg_loss_value - weighted_contribution_total
                    )
                modality_totals = torch.stack(
                    (
                        acc_text_loss_sum,
                        acc_text_tokens,
                        acc_image_loss_sum,
                        acc_image_tokens,
                    )
                )
                modality_totals = accelerator.reduce(
                    modality_totals,
                    reduction="sum",
                )
                global_text_loss = (
                    modality_totals[0]
                    / modality_totals[1].clamp_min(1.0)
                )
                global_image_loss = (
                    modality_totals[2]
                    / modality_totals[3].clamp_min(1.0)
                )
                logs.update(
                    {
                        "train/loss_text": float(global_text_loss.item()),
                        "train/ppl_text": math.exp(
                            min(float(global_text_loss.item()), 100.0)
                        ),
                        "train/loss_image_flow": float(
                            global_image_loss.item()
                        ),
                        "train/text_target_tokens": float(
                            modality_totals[1].item()
                        ),
                        "train/image_target_tokens": float(
                            modality_totals[3].item()
                        ),
                    }
                )
                if ema is not None:
                    logs["ema/decay"] = ema_decay_value
                    logs["ema/started"] = float(ema.started)
                if physical_rows_total > 0:
                    logs["pack/seq_len"] = (
                        physical_tokens_total / physical_rows_total
                    )
                if valid_tokens_total + padding_tokens_total > 0:
                    logs["pack/padding_ratio"] = (
                        padding_tokens_total
                        / (valid_tokens_total + padding_tokens_total)
                    )
                    logs["pack/valid_tokens/microbatch/gpu"] = (
                        valid_tokens_total / micro_batches_total
                    )
                    logs["pack/image_tokens/microbatch/gpu"] = (
                        image_tokens_total / micro_batches_total
                    )
                    logs["pack/padding_tokens/microbatch/gpu"] = (
                        padding_tokens_total / micro_batches_total
                    )

                if acc_flow_stats:
                    flow_stat_count = acc_flow_stat_batches.clamp_min(1.0)
                    flow_stat_keys = sorted(acc_flow_stats)
                    local_flow_stats = torch.stack(
                        [
                            acc_flow_stats[key] / flow_stat_count
                            for key in flow_stat_keys
                        ]
                    )
                    reduced_flow_stats = accelerator.reduce(
                        local_flow_stats,
                        reduction="mean",
                    )
                    flow_stat_values = reduced_flow_stats.tolist()
                    global_flow_stats = dict(
                        zip(flow_stat_keys, flow_stat_values)
                    )
                    logs.update(
                        {
                            f"train/{key}": value
                            for key, value in global_flow_stats.items()
                        }
                    )
                if acc_backbone_gate_stats:
                    gate_stat_count = (
                        acc_backbone_gate_stat_batches.clamp_min(1.0)
                    )
                    gate_stat_keys = list(acc_backbone_gate_stats)
                    local_gate_stats = torch.stack(
                        [
                            acc_backbone_gate_stats[key]
                            / gate_stat_count
                            for key in gate_stat_keys
                        ]
                    )
                    reduced_gate_stats = accelerator.reduce(
                        local_gate_stats,
                        reduction="mean",
                    )
                    for key, stat in zip(
                        gate_stat_keys,
                        reduced_gate_stats.tolist(),
                    ):
                        logs[f"train/{key}"] = stat

                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    if mixed_source_training:
                        _append_training_metrics_jsonl(
                            config,
                            global_step=global_step,
                            logs=logs,
                        )
                    msg = (
                        f"Step: {global_step} | "
                        f"Loss: {global_avg_loss_value:0.4f}"
                        f" | Text: {float(global_text_loss.item()):0.4f}"
                        f" | Image: {float(global_image_loss.item()):0.4f}"
                    )
                    if source_metric_display:
                        msg += " | Sources: " + ", ".join(
                            (
                                f"{source}={raw_loss:0.4f}"
                                f"/{contribution:0.4f}w"
                            )
                            for source, (
                                raw_loss,
                                contribution,
                            ) in source_metric_display.items()
                        )
                    if acc_flow_stats:
                        msg += (
                            f" | FlowMSE: {global_flow_stats.get('flow/v_mse', 0.0):0.4f}"
                            f" | FlowPredVRMS: {global_flow_stats.get('flow/v_pred_rms', 0.0):0.4f}"
                        )
                    if sampled_grad_norm_value is not None:
                        msg += f" | GradNormPreClip: {sampled_grad_norm_value:0.4f}"
                    msg += (
                        f" | LR: {lr_scheduler.get_last_lr()[0]:0.6f} | "
                        f"Sec/Step: {logs['seconds/optimizer_step']:0.4f} | "
                        f"Data/Microbatch: {logs['data_wait_ms/microbatch']:0.2f}ms"
                    )
                    logger.info(msg)

            post_step_maintenance_started = time.perf_counter()

            # Checkpointing
            if checkpoint_save_due(
                global_step,
                save_every=save_every,
                milestone_every_steps=checkpoint_milestone_every,
            ):
                _save_resumable_training_checkpoint(
                    model=model,
                    config=config,
                    accelerator=accelerator,
                    global_step=global_step,
                    train_dataloader=train_dataloader,
                    mixed_source_training=mixed_source_training,
                    epoch=epoch,
                    batches_consumed_in_epoch=batches_consumed_in_epoch,
                    sampler_shuffle_seed=caption_shuffle_seed,
                    config_contract=config_contract,
                    ema_layout=ema_layout,
                    ema=ema,
                    cumulative_training_wall_seconds=(
                        cumulative_wall_seconds_before_run
                        + time.time()
                        - training_runtime_started_at
                    ),
                    cumulative_finite_loss_microbatches_checked=(
                        cumulative_loss_checks_before_run
                        + finite_loss_microbatches_checked
                    ),
                )

            if save_ema_eval_every > 0 and global_step % save_ema_eval_every == 0:
                if save_model_with_ema_eval:
                    _save_model_hf_for_evaluation(
                        model,
                        tokenizer,
                        config,
                        accelerator,
                        global_step,
                    )
                ema_directory = (
                    _ema_state_directory(config, global_step)
                    if ema is not None
                    else None
                )
                _save_ema_hf_model(
                    ema,
                    model,
                    tokenizer,
                    config,
                    accelerator,
                    global_step,
                    ema_directory,
                    floating_dtype=_ema_eval_export_dtype(config),
                    save_name=f"hf_model-{global_step}-ema-eval",
                    export_kind="evaluation",
                )
                if save_model_with_ema_eval:
                    _publish_evaluation_model_pair_manifest(
                        config,
                        accelerator,
                        global_step,
                    )

            if global_step % config.experiment.save_hfmodel_every == 0:
                save_hf_model(model, tokenizer, config, accelerator, global_step)
                ema_directory = (
                    _ema_state_directory(config, global_step)
                    if ema is not None
                    else None
                )
                _save_ema_hf_model(
                    ema,
                    model,
                    tokenizer,
                    config,
                    accelerator,
                    global_step,
                    ema_directory,
                )
                
            # Validation
            if int(config.experiment.val_every) > 0 and global_step % int(config.experiment.val_every) == 0:
                if mixed_source_training:
                    validation_started = time.monotonic()
                    from utils.training_downstream_validation import (
                        ValidationProfile, run_downstream_validation,
                    )

                    validation = run_downstream_validation(
                        accelerator.unwrap_model(model), tokenizer,
                        device=accelerator.device,
                        output_dir=Path(config.experiment.output_dir) / "downstream_validation" / f"step-{global_step}",
                        step=global_step, ema=ema,
                        profile=ValidationProfile.from_config(config),
                        started=validation_started,
                    )
                    validation_logs = {
                        "val/downstream_complete": int(validation["complete"]),
                        "val/downstream_seconds": validation["wall_seconds"],
                        "val/downstream_within_budget": int(validation["within_time_budget"]),
                    }
                    validation_logs.update({
                        f"val/downstream/{task}": result["primary"]
                        for task, result in validation["tasks"].items() if result["complete"]
                    })
                    if "text_mean" in validation:
                        validation_logs["val/downstream/text_mean"] = validation["text_mean"]
                    accelerator.log(validation_logs, step=global_step)
                elif val_dataloader is not None:
                    validate(
                        model,
                        val_dataloader,
                        accelerator,
                        global_step,
                        config,
                        tokenizer,
                    )
                model.train()

            # Exclude checkpointing and validation from the next training
            # throughput window. The current window was sampled before these
            # cold-path operations started.
            if did_log:
                training_window.reset()
                for source in source_microbatches_window:
                    source_microbatches_window[source] = 0
                    source_data_wait_window[source] = 0.0
            else:
                training_window.exclude_elapsed(
                    time.perf_counter() - post_step_maintenance_started
                )

            # Reset per-step accumulators for the next optimizer step
            acc_loss.zero_()
            acc_text_loss_sum.zero_()
            acc_text_tokens.zero_()
            acc_image_loss_sum.zero_()
            acc_image_tokens.zero_()
            if acc_physical_token_positions is not None:
                acc_physical_token_positions = 0
            for source in active_mixed_sources:
                acc_source_loss_sum[source].zero_()
                acc_source_target_count[source].zero_()
                acc_source_weighted_loss_sum[source].zero_()
            acc_flow_stats.clear()
            acc_flow_stat_batches.zero_()
            acc_backbone_gate_stats.clear()
            acc_backbone_gate_stat_batches.zero_()
            if global_step >= stop_after_steps:
                break

    training_runtime_elapsed = time.time() - training_runtime_started_at
    if (
        bool(config.experiment.get("save_final_checkpoint", False))
        and global_step > 0
        and not checkpoint_save_due(
            global_step,
            save_every=save_every,
            milestone_every_steps=checkpoint_milestone_every,
        )
    ):
        _save_resumable_training_checkpoint(
            model=model,
            config=config,
            accelerator=accelerator,
            global_step=global_step,
            train_dataloader=train_dataloader,
            mixed_source_training=mixed_source_training,
            epoch=epoch,
            batches_consumed_in_epoch=batches_consumed_in_epoch,
            sampler_shuffle_seed=caption_shuffle_seed,
            config_contract=config_contract,
            ema_layout=ema_layout,
            ema=ema,
            cumulative_training_wall_seconds=(
                cumulative_wall_seconds_before_run
                + training_runtime_elapsed
            ),
            cumulative_finite_loss_microbatches_checked=(
                cumulative_loss_checks_before_run
                + finite_loss_microbatches_checked
            ),
        )
    if accelerator.device.type == "npu":
        memory_backend = "npu"
        local_memory = torch.tensor(
            [
                int(torch.npu.max_memory_allocated(accelerator.device)),
                int(torch.npu.max_memory_reserved(accelerator.device)),
            ],
            device=accelerator.device,
            dtype=torch.int64,
        )
    elif accelerator.device.type == "cuda":
        memory_backend = "cuda"
        local_memory = torch.tensor(
            [
                int(torch.cuda.max_memory_allocated(accelerator.device)),
                int(torch.cuda.max_memory_reserved(accelerator.device)),
            ],
            device=accelerator.device,
            dtype=torch.int64,
        )
    else:
        memory_backend = accelerator.device.type
        local_memory = torch.zeros(
            2,
            device=accelerator.device,
            dtype=torch.int64,
        )
    local_elapsed = torch.tensor(
        [float(training_runtime_elapsed)],
        device=accelerator.device,
        dtype=torch.float32,
    )
    gathered_memory = accelerator.gather(local_memory).reshape(-1, 2)
    gathered_elapsed = accelerator.gather(local_elapsed).reshape(-1)
    memory_max = gathered_memory.max(dim=0).values
    elapsed_max = gathered_elapsed.max()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        runtime_payload = {
            "schema": "selfless_training_runtime_metrics_v1",
            "global_step": int(global_step),
            "run_start_global_step": int(training_runtime_start_step),
            "world_size": int(accelerator.num_processes),
            "total_batch_size": int(total_batch_size),
            "steps_this_run": int(global_step - training_runtime_start_step),
            "finite_loss_microbatches_checked": int(
                finite_loss_microbatches_checked
            ),
            "last_logged_loss": last_logged_loss,
            "training_wall_seconds": float(elapsed_max.item()),
            "cumulative_training_wall_seconds": float(
                cumulative_wall_seconds_before_run + elapsed_max.item()
            ),
            "cumulative_finite_loss_microbatches_checked": int(
                cumulative_loss_checks_before_run
                + finite_loss_microbatches_checked
            ),
            "train_samples_per_second": float(
                (global_step - training_runtime_start_step)
                * total_batch_size
                / max(float(elapsed_max.item()), 1e-12)
            ),
            "memory_backend": memory_backend,
            "peak_memory_allocated_bytes_per_rank": int(
                memory_max[0].item()
            ),
            "peak_memory_reserved_bytes_per_rank": int(
                memory_max[1].item()
            ),
            "trainability": trainability,
        }
        if ema_layout is not None:
            full_ema_bytes = int(
                sum(chunk["bytes"] for chunk in ema_layout["chunks"].values())
            )
            max_shard_bytes = int(max(ema_layout["rank_bytes"]))
            runtime_payload["ema"] = {
                "full_fp32_replica_bytes": full_ema_bytes,
                "shard_bytes_by_rank": ema_layout["rank_bytes"],
                "max_shard_bytes": max_shard_bytes,
                "minimum_bytes_saved_per_rank": full_ema_bytes
                - max_shard_bytes,
                "minimum_fraction_saved_per_rank": (
                    (full_ema_bytes - max_shard_bytes) / full_ema_bytes
                    if full_ema_bytes
                    else 0.0
                ),
            }
        runtime_root = Path(config.experiment.output_dir)
        runtime_paths = (
            runtime_root / "training_runtime_metrics.json",
            runtime_root
            / (
                "training_runtime_metrics_"
                f"step-{training_runtime_start_step}-to-{global_step}.json"
            ),
        )
        for runtime_path in runtime_paths:
            runtime_temp_path = runtime_path.with_name(
                f".{runtime_path.name}.tmp-{os.getpid()}"
            )
            runtime_temp_path.write_text(
                json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(runtime_temp_path, runtime_path)
    if bool(config.experiment.get("save_final", True)):
        save_hf_model(model, tokenizer, config, accelerator, "final")
        ema_directory = _save_ema_state(ema, config, accelerator, "final")
        _save_ema_hf_model(
            ema,
            model,
            tokenizer,
            config,
            accelerator,
            "final",
            ema_directory,
        )
        if _image_flow_adapter_save_enabled(config, final=True):
            if ema is not None and ema.started and bool(config.training.get("ema_save_adapter", True)):
                _save_ema_image_flow_adapter(
                    ema_directory,
                    config,
                    accelerator,
                    "final",
                    final=True,
                )
            else:
                _save_image_flow_adapter(
                    model,
                    config,
                    accelerator,
                    "final",
                    final=True,
                )
    accelerator.end_training()


@torch.no_grad()
def validate(
    model,
    val_dataloader,
    accelerator,
    global_step,
    config=None,
    tokenizer=None,
):
    validation_seed = int(
        config.experiment.get("validation_seed", config.training.seed)
    ) + int(accelerator.process_index)
    npu_devices = (
        [int(accelerator.device.index)]
        if accelerator.device.type == "npu"
        else []
    )
    # Validation must not perturb the training RNG stream, and every checkpoint
    # must see the same flow times/noise on each rank.
    with torch.random.fork_rng(devices=npu_devices, device_type="npu"):
        torch.default_generator.manual_seed(validation_seed)
        if accelerator.device.type == "npu":
            with torch.npu.device(accelerator.device):
                torch.npu.manual_seed(validation_seed)
        model.eval()  # DeepSpeed requires explicit eval mode for no_grad forward
        try:
            _validate_multimodal(
                model,
                val_dataloader,
                accelerator,
                global_step,
                config,
                tokenizer,
            )
        finally:
            model.train()


@torch.no_grad()
def _load_vae_decoder(config, accelerator):
    global _VAE_CACHE
    if _VAE_CACHE is None:
        vae_path = Path(config.experiment.get("validation_vae_path", "public/vae/mar-kl16/kl16.ckpt"))
        if not vae_path.exists():
            logger.warning(f"Skipping validation image decode; missing VAE checkpoint: {vae_path}")
            return None

        vae_module_root = Path(
            config.experiment.get(
                "validation_vae_module_root",
                "/inspire/hdd/global_user/wanjiaxin-253108030048/code/mar",
            )
        )
        vae_module_path = vae_module_root / "models" / "vae.py"
        if not vae_module_path.exists():
            logger.warning(f"Skipping validation image decode; missing VAE module: {vae_module_path}")
            return None
        spec = importlib.util.spec_from_file_location("kl16_vae", vae_module_path)
        vae_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vae_module)
        AutoencoderKL = vae_module.AutoencoderKL

        vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=str(vae_path))
        vae = vae.eval()
        for param in vae.parameters():
            param.requires_grad_(False)
        _VAE_CACHE = vae

    vae_dtype_name = str(config.experiment.get("validation_vae_dtype", "fp32")).lower()
    dtype = (
        torch.float16
        if vae_dtype_name in {"fp16", "float16", "half"}
        and accelerator.device.type in {"cuda", "npu"}
        else torch.float32
    )
    return _VAE_CACHE.to(device=accelerator.device, dtype=dtype).eval()


def _empty_validation_device_cache(accelerator) -> None:
    """Release inactive allocator blocks around cold-path validation decode."""

    device_type = str(getattr(accelerator.device, "type", ""))
    if device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _image_spans_from_table(
    image_span_table: torch.Tensor,
    image_tokens_per_img: int,
) -> list[tuple[int, int, int]]:
    """Read validation spans without scanning an accelerator tensor token by token."""

    rows = image_span_table.detach().cpu().tolist()
    spans = []
    for row in rows:
        batch_index, _, start, end, _ = map(int, row)
        if end - start != image_tokens_per_img:
            raise ValueError(
                "validation image span length does not match model config: "
                f"start={start}, end={end}, expected={image_tokens_per_img}"
            )
        spans.append((batch_index, start, end))
    return spans


def _log_wandb_validation_images(accelerator, image_paths: dict[str, Path], global_step: int) -> None:
    if not image_paths:
        return
    try:
        import wandb
    except Exception:
        return
    logs = {}
    for key, path in image_paths.items():
        if path.exists():
            logs[key] = wandb.Image(str(path), caption=path.name)
    if logs:
        accelerator.log(logs, step=global_step)


def _build_i2t_generation_prefix(
    tokenizer,
    *,
    text_prefix: str,
    boi_token_id: int,
    eoi_token_id: int,
    image_mask_token_id: int,
    image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prefix_ids = torch.tensor(
        tokenizer.encode(
            str(text_prefix).strip(),
            add_special_tokens=False,
        ),
        dtype=torch.long,
    )
    if prefix_ids.numel() == 0:
        raise ValueError("I2T validation prefix tokenized to an empty sequence")
    image_start = int(prefix_ids.numel()) + 1
    input_ids = torch.cat(
        [
            prefix_ids,
            torch.tensor([int(boi_token_id)], dtype=torch.long),
            torch.full(
                (int(image_tokens),),
                int(image_mask_token_id),
                dtype=torch.long,
            ),
            torch.tensor([int(eoi_token_id)], dtype=torch.long),
        ]
    )
    token_types = torch.cat(
        [
            torch.zeros(prefix_ids.numel(), dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
            torch.ones(int(image_tokens), dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
        ]
    )
    prefix_length = int(prefix_ids.numel())
    sigma = torch.empty(input_ids.numel(), dtype=torch.float32)
    sigma[:prefix_length] = torch.arange(prefix_length, dtype=torch.float32)
    sigma[prefix_length] = float(prefix_length)
    sigma[-1] = float(prefix_length + 1)
    sigma[image_start : image_start + int(image_tokens)] = torch.arange(
        prefix_length + 2,
        prefix_length + 2 + int(image_tokens),
        dtype=torch.float32,
    )
    return input_ids, token_types, sigma, image_start


@torch.inference_mode()
def _generate_i2t_caption_batch(
    model,
    tokenizer,
    image_batch: torch.Tensor,
    *,
    text_prefix: str,
    max_new_tokens: int,
    temperature: float,
    base_sigma_batch: torch.Tensor | None = None,
) -> tuple[list[str], list[list[int]], list[str]]:
    """Generate captions from cached image latents without any hashing."""

    if image_batch.ndim != 3:
        raise ValueError(
            "image_batch must be [batch, image_tokens, latent_dim], got "
            f"{tuple(image_batch.shape)}"
        )
    batch_size, image_tokens, latent_dim = image_batch.shape
    if batch_size <= 0:
        return [], [], []
    if int(max_new_tokens) <= 0:
        raise ValueError("validation_i2t_max_new_tokens must be positive")
    if not math.isfinite(float(temperature)) or float(temperature) < 0.0:
        raise ValueError(
            "validation_i2t_temperature must be finite and non-negative"
        )

    base_ids, base_types, base_sigma, image_start = (
        _build_i2t_generation_prefix(
            tokenizer,
            text_prefix=text_prefix,
            boi_token_id=int(model.config.boi_token_id),
            eoi_token_id=int(model.config.eoi_token_id),
            image_mask_token_id=int(model.config.image_mask_token_id),
            image_tokens=int(image_tokens),
        )
    )
    device = image_batch.device
    input_ids = base_ids.unsqueeze(0).expand(batch_size, -1).clone().to(device)
    token_types = (
        base_types.unsqueeze(0).expand(batch_size, -1).clone().to(device)
    )
    if base_sigma_batch is None:
        sigma = base_sigma.unsqueeze(0).expand(batch_size, -1).clone()
    else:
        if tuple(base_sigma_batch.shape) != (
            batch_size,
            int(base_ids.numel()),
        ):
            raise ValueError(
                "base_sigma_batch must align with the serialized I2T prefix: "
                f"got {tuple(base_sigma_batch.shape)}, expected "
                f"{(batch_size, int(base_ids.numel()))}"
            )
        sigma = base_sigma_batch.clone()
    sigma = sigma.to(device=device, dtype=torch.float32)
    aligned_latents = torch.zeros(
        batch_size,
        input_ids.shape[1],
        latent_dim,
        device=device,
        dtype=image_batch.dtype,
    )
    image_latent_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    image_end = image_start + int(image_tokens)
    aligned_latents[:, image_start:image_end] = image_batch
    image_latent_mask[:, image_start:image_end] = True

    eos_id = int(tokenizer.eos_token_id)
    stop_ids = {eos_id}
    image_end_id = getattr(model.config, "im_end_token_id", None)
    if image_end_id is not None:
        stop_ids.add(int(image_end_id))
    output_ids, trace = model.generate(
        "i2t",
        input_ids=input_ids,
        token_types=token_types,
        sigma=sigma,
        image_latents=aligned_latents,
        image_latent_mask=image_latent_mask,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        eos_token_id=sorted(stop_ids),
        use_cache=True,
        return_trace=True,
    )
    if trace.get("backbone_kv_cache_enabled") is not True:
        raise RuntimeError("I2T validation must use the model KV cache")

    generated: list[list[int]] = []
    stop_reasons: list[str] = []
    prompt_length = int(input_ids.shape[1])
    for suffix in output_ids[:, prompt_length:].detach().cpu().tolist():
        tokens: list[int] = []
        reason = "max_new_tokens"
        for token in suffix:
            token = int(token)
            if token in stop_ids:
                reason = "eos" if token == eos_id else "im_end"
                break
            tokens.append(token)
        generated.append(tokens)
        stop_reasons.append(reason)

    texts = [
        tokenizer.decode(tokens, skip_special_tokens=True).strip()
        for tokens in generated
    ]
    return texts, generated, stop_reasons


@torch.no_grad()
def _save_validation_i2t_captions(
    *,
    model,
    tokenizer,
    labels,
    task_modes,
    image_span_table,
    image_latents,
    sigma,
    accelerator,
    global_step: int,
    config,
) -> bool:
    if config is None or tokenizer is None or image_latents is None:
        return False
    caption_every = int(
        config.experiment.get(
            "validation_i2t_every",
            config.experiment.get("validation_image_every", 0),
        )
    )
    if caption_every <= 0 or int(global_step) % caption_every:
        return False

    spans_by_row = {
        int(span[0]): tuple(int(value) for value in span)
        for span in image_span_table.detach().cpu().tolist()
    }
    candidates = [
        row
        for row, mode in enumerate(task_modes)
        if str(mode) == "i2t" and row in spans_by_row
    ]
    local_count = torch.tensor(
        [len(candidates)],
        device=accelerator.device,
        dtype=torch.long,
    )
    common_count = int(accelerator.gather(local_count).min().item())
    sample_count = min(
        int(config.experiment.get("validation_i2t_samples", 2)),
        common_count,
    )
    if sample_count <= 0:
        return False

    selected_rows = candidates[:sample_count]
    selected_latents = []
    selected_base_sigmas = []
    image_ids = []
    references = []
    for row in selected_rows:
        _, _, start, end, image_id, *_ = spans_by_row[row]
        selected_latents.append(image_latents[row, start:end])
        selected_base_sigmas.append(sigma[row, : end + 1])
        image_ids.append(int(image_id))
        reference_ids = labels[row][labels[row].ne(-100)].detach().cpu().tolist()
        references.append(
            tokenizer.decode(
                reference_ids,
                skip_special_tokens=True,
            ).strip()
        )
    image_batch = torch.stack(selected_latents)
    unwrapped = accelerator.unwrap_model(model)
    generated_texts, generated_ids, stop_reasons = (
        _generate_i2t_caption_batch(
            unwrapped,
            tokenizer,
            image_batch,
            text_prefix=str(config.dataset.params.image.caption_i2t_prefix),
            max_new_tokens=int(
                config.experiment.get("validation_i2t_max_new_tokens", 64)
            ),
            temperature=float(
                config.experiment.get("validation_i2t_temperature", 0.0)
            ),
            base_sigma_batch=torch.stack(selected_base_sigmas),
        )
    )

    if accelerator.is_main_process:
        output_directory = (
            Path(config.experiment.output_dir)
            / "validation_i2t_captions"
            / f"step-{int(global_step):08d}"
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        vae = _load_vae_decoder(config, accelerator)
        image_names = [None] * sample_count
        if vae is not None:
            image_tokens = int(image_batch.shape[1])
            side = int(image_tokens**0.5)
            if side * side != image_tokens:
                raise ValueError(
                    f"image_tokens_per_img={image_tokens} is not square"
                )
            vae_latents = image_batch.view(
                sample_count,
                side,
                side,
                image_batch.shape[-1],
            ).permute(0, 3, 1, 2)
            vae_dtype = next(vae.parameters()).dtype
            scaling_factor = float(
                config.experiment.get(
                    "validation_vae_scaling_factor",
                    0.2325,
                )
            )
            decoded = vae.decode(
                vae_latents.to(dtype=vae_dtype) / scaling_factor
            ).float().clamp(-1, 1)
            from torchvision.utils import save_image

            for index, decoded_image in enumerate(decoded):
                image_name = (
                    f"sample-{index:02d}-img-{image_ids[index]}.png"
                )
                save_image(
                    (decoded_image + 1.0) / 2.0,
                    output_directory / image_name,
                )
                image_names[index] = image_name
            if bool(
                config.experiment.get("validation_release_vae_gpu", True)
            ):
                vae.to(device="cpu")

        rows = []
        readable_lines = []
        for index in range(sample_count):
            row = {
                "schema": "unified_i2t_qualitative_sample_v1",
                "global_step": int(global_step),
                "sample_index": int(index),
                "img_id": int(image_ids[index]),
                "image_file": image_names[index],
                "reference_caption": references[index],
                "generated_caption": generated_texts[index],
                "generated_token_ids": generated_ids[index],
                "generated_token_count": len(generated_ids[index]),
                "stop_reason": stop_reasons[index],
                "generation_entry": "model.generate",
                "backbone_kv_cache_enabled": True,
                "dual_stream_attention_contract": str(
                    config.model.get(
                        "dual_stream_attention_contract",
                        "selfless_strict",
                    )
                ),
                "single_stream_visible_content_diagonal": (
                    str(
                        config.model.get(
                            "dual_stream_attention_contract",
                            "selfless_strict",
                        )
                    ).strip().lower()
                    == "xlnet_content_diagonal"
                ),
                "single_stream_current_query_diagonal": False,
                "sigma_source": "validation_batch_training_contract",
            }
            rows.append(row)
            readable_lines.extend(
                [
                    f"sample {index} | img_id={image_ids[index]}",
                    f"reference: {references[index]}",
                    f"generated: {generated_texts[index]}",
                    "",
                ]
            )
        jsonl_path = output_directory / "captions.jsonl"
        jsonl_temp = jsonl_path.with_name(
            f".{jsonl_path.name}.tmp-{os.getpid()}"
        )
        jsonl_temp.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        os.replace(jsonl_temp, jsonl_path)
        text_path = output_directory / "captions.txt"
        text_temp = text_path.with_name(
            f".{text_path.name}.tmp-{os.getpid()}"
        )
        text_temp.write_text("\n".join(readable_lines), encoding="utf-8")
        os.replace(text_temp, text_path)
        logger.info(
            "Saved %d validation I2T captions and source images to %s",
            sample_count,
            output_directory,
        )
    accelerator.wait_for_everyone()
    return True


@torch.no_grad()
def _save_validation_flow_images(
    model,
    output,
    input_ids,
    token_types,
    sigma,
    image_span_table,
    image_latents,
    accelerator,
    global_step,
    config,
) -> None:
    """Generate held-out images exclusively through the cached public API."""

    del output
    if config is None:
        return
    image_every = int(
        config.experiment.get(
            "validation_image_every",
            config.experiment.get("val_every", 0),
        )
    )
    if (
        image_every <= 0
        or global_step % image_every != 0
        or not bool(
            config.experiment.get("validation_single_stream_images", True)
        )
    ):
        return

    unwrapped = accelerator.unwrap_model(model)
    image_tokens_per_img = int(
        getattr(
            unwrapped.config,
            "image_tokens_per_img",
            config.model.get("image_tokens_per_img", 256),
        )
    )
    side = math.isqrt(image_tokens_per_img)
    if side * side != image_tokens_per_img:
        raise ValueError(
            "validation image token count must form a square grid, got "
            f"{image_tokens_per_img}"
        )
    spans = _image_spans_from_table(
        image_span_table,
        image_tokens_per_img,
    )

    # ZeRO-2 keeps complete parameters on every rank. Direct generation is not
    # valid when ZeRO-3 partitions the model.
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        deepspeed_config = (
            accelerator.state.deepspeed_plugin.deepspeed_config
        )
        zero_stage = int(
            deepspeed_config.get("zero_optimization", {}).get(
                "stage",
                deepspeed_config.get("zero_stage", -1),
            )
        )
        if zero_stage != 2:
            raise RuntimeError(
                "validation generation requires replicated parameters; "
                f"resolved DeepSpeed stage is {zero_stage}"
            )

    span_counts = accelerator.gather(
        torch.tensor(
            [len(spans)],
            device=accelerator.device,
            dtype=torch.long,
        )
    )
    common_span_count = int(span_counts.min().item())
    requested_samples = int(
        config.experiment.get("validation_image_samples", 4)
    )
    sample_count = min(requested_samples, common_span_count)
    if sample_count <= 0:
        if accelerator.is_main_process:
            logger.warning(
                "Skipping validation generation because at least one rank "
                "has no complete image span."
            )
        return

    raw_strategies = config.experiment.get(
        "validation_single_stream_order_strategies",
        ["spatial_halton"],
    )
    if isinstance(raw_strategies, str):
        strategies = [
            item.strip()
            for item in raw_strategies.split(",")
            if item.strip()
        ]
    else:
        strategies = [str(item).strip() for item in raw_strategies]
        strategies = [item for item in strategies if item]
    if not strategies:
        raise ValueError(
            "validation_single_stream_order_strategies must not be empty"
        )

    selected_spans = spans[:sample_count]
    target_latents = torch.stack(
        [
            image_latents[batch_idx, start:end]
            .view(side, side, -1)
            .permute(2, 0, 1)
            for batch_idx, start, end in selected_spans
        ]
    )
    flow_temperature = float(
        config.experiment.get("validation_flow_temperature", 1.0)
    )
    flow_cfg = float(
        config.experiment.get("validation_flow_cfg", 3.5)
    )
    flow_cfg_schedule = str(
        config.experiment.get(
            "validation_flow_cfg_schedule",
            "constant",
        )
    )
    flow_solver = config.experiment.get(
        "validation_flow_solver",
        config.model.get("image_flow_solver", None),
    )
    parallel_rate = int(
        config.experiment.get(
            "validation_single_stream_parallel_rate",
            1,
        )
    )

    generated = {}
    local_logs = {
        "val/generation/target_latent_rms": (
            target_latents.float().pow(2).mean().sqrt().item()
        )
    }
    report_strategies = {}
    for strategy in strategies:
        pred_latents, trace = unwrapped.generate(
            "t2i",
            input_ids=input_ids,
            token_types=token_types,
            sigma=sigma,
            spans=selected_spans,
            image_latent_dim=image_latents.shape[-1],
            flow_temperature=flow_temperature,
            flow_cfg=flow_cfg,
            flow_cfg_schedule=flow_cfg_schedule,
            flow_solver=flow_solver,
            parallel_rate=parallel_rate,
            order_strategy=strategy,
            use_cache=True,
            return_trace=True,
        )
        if trace.get("backbone_kv_cache_enabled") is not True:
            raise RuntimeError(
                "validation generation unexpectedly disabled backbone cache"
            )
        if tuple(pred_latents.shape) != tuple(target_latents.shape):
            raise RuntimeError(
                "validation generation shape mismatch: "
                f"generated={tuple(pred_latents.shape)}, "
                f"target={tuple(target_latents.shape)}"
            )
        generated[strategy] = pred_latents
        prefix = f"val/generation/{strategy}"
        local_logs.update(
            {
                f"{prefix}/latent_mse_to_target": F.mse_loss(
                    pred_latents.float(),
                    target_latents.float(),
                ).item(),
                f"{prefix}/latent_rms": (
                    pred_latents.float().pow(2).mean().sqrt().item()
                ),
                f"{prefix}/generation_step_max": (
                    trace["generation_step"].float().max().item()
                ),
                f"{prefix}/backbone_kv_cache_peak_mib": (
                    float(trace.get("backbone_kv_cache_peak_bytes", 0))
                    / (1024.0 * 1024.0)
                ),
            }
        )
        report_strategies[strategy] = {
            "attention_contract": trace.get("attention_contract"),
            "content_self_diagonal": trace.get(
                "single_stream_content_self_diagonal"
            ),
            "backbone_kv_cache_enabled": True,
            "backbone_kv_cache_peak_bytes": int(
                trace.get("backbone_kv_cache_peak_bytes", 0)
            ),
            "generation_step_max": int(
                trace["generation_step"].max().item()
            ),
        }

    metric_keys = sorted(local_logs)
    metric_values = torch.tensor(
        [local_logs[key] for key in metric_keys],
        device=accelerator.device,
        dtype=torch.float32,
    )
    global_values = accelerator.reduce(metric_values, reduction="mean")
    global_logs = dict(zip(metric_keys, global_values.tolist()))

    write_images = accelerator.is_main_process
    vae = None
    if write_images:
        image_dir = (
            Path(config.experiment.output_dir) / "validation_flow_images"
        )
        image_dir.mkdir(parents=True, exist_ok=True)
        _empty_validation_device_cache(accelerator)
        vae = _load_vae_decoder(config, accelerator)
        if vae is not None:
            from torchvision.utils import make_grid, save_image

            scaling_factor = float(
                config.experiment.get(
                    "validation_vae_scaling_factor",
                    0.2325,
                )
            )
            vae_dtype = next(vae.parameters()).dtype

            def decode(latents):
                return (
                    vae.decode(
                        latents.to(dtype=vae_dtype) / scaling_factor
                    )
                    .float()
                    .clamp(-1, 1)
                    .add(1.0)
                    .div(2.0)
                )

            target_images = decode(target_latents)
            target_path = (
                image_dir / f"step-{global_step:08d}-target.png"
            )
            save_image(target_images, target_path)
            wandb_images = {"val/generation/target": target_path}
            overview_columns = [target_images]
            for strategy, pred_latents in generated.items():
                strategy_tag = strategy.replace("/", "_")
                pred_images = decode(pred_latents)
                pred_path = (
                    image_dir
                    / (
                        f"step-{global_step:08d}-"
                        f"single_stream_pred_{strategy_tag}.png"
                    )
                )
                save_image(pred_images, pred_path)
                comparison = torch.stack(
                    [target_images, pred_images],
                    dim=1,
                ).flatten(0, 1)
                comparison_path = (
                    image_dir
                    / f"step-{global_step:08d}-strategy_{strategy_tag}.png"
                )
                save_image(
                    make_grid(comparison, nrow=2),
                    comparison_path,
                )
                wandb_images[
                    f"val/generation/{strategy}"
                ] = comparison_path
                overview_columns.append(pred_images)

            overview = torch.stack(
                overview_columns,
                dim=1,
            ).flatten(0, 1)
            overview_path = (
                image_dir / f"step-{global_step:08d}-overview.png"
            )
            save_image(
                make_grid(overview, nrow=len(overview_columns)),
                overview_path,
            )
            wandb_images["val/generation/overview"] = overview_path
            _log_wandb_validation_images(
                accelerator,
                wandb_images,
                global_step,
            )
            logger.info(
                "Validation overview columns: target, "
                + ", ".join(strategies)
            )

        report = {
            "schema": "selfless_cached_validation_generation_v1",
            "global_step": int(global_step),
            "generation_entry": "model.generate",
            "task": "t2i",
            "use_cache": True,
            "cfg": flow_cfg,
            "cfg_schedule": flow_cfg_schedule,
            "flow_solver": flow_solver,
            "parallel_rate": parallel_rate,
            "samples": sample_count,
            "strategies": report_strategies,
            "metrics": global_logs,
        }
        report_path = (
            Path(config.experiment.output_dir)
            / f"validation_generation_step_{global_step}.json"
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        accelerator.log(global_logs, step=global_step)

    if bool(
        config.experiment.get("validation_release_vae_gpu", True)
    ):
        if vae is not None:
            vae.to(device="cpu")
        _empty_validation_device_cache(accelerator)


@torch.no_grad()
def _validate_multimodal(
    model,
    val_dataloader,
    accelerator,
    global_step,
    config=None,
    tokenizer=None,
):
    unified_active_sources = (
        _active_mixed_source_names(config.dataset.params.schedule)
        if config is not None
        and str(config.dataset.class_name) == "UnifiedMixedDataset"
        else None
    )
    validate_text_source = (
        unified_active_sources is None or "i2t" in unified_active_sources
    )
    validate_image_source = (
        unified_active_sources is None or "t2i" in unified_active_sources
    )
    local_weighted_text = torch.tensor(0.0, device=accelerator.device)
    local_text_tokens = torch.tensor(0.0, device=accelerator.device)
    local_weighted_image = torch.tensor(0.0, device=accelerator.device)
    local_image_tokens = torch.tensor(0.0, device=accelerator.device)
    local_flow_stat_sums = {}
    local_flow_stat_counts = {}
    saved_validation_images = False
    saved_validation_i2t = False
    diagnostic_batches = int(
        config.experiment.get("flow_head_attention_diagnostic_batches", 0)
        if config is not None
        else 0
    )
    unwrapped_model = accelerator.unwrap_model(model)
    diagnostic_head = getattr(
        getattr(unwrapped_model, "image_flow_head", None),
        "net",
        None,
    )
    validation_max_batches = int(
        config.experiment.get("validation_max_batches", 0)
        if config is not None
        else 0
    )
    if validation_max_batches < 0:
        raise ValueError("validation_max_batches must be non-negative")

    for validation_batch_idx, batch in enumerate(val_dataloader):
        if (
            validation_max_batches > 0
            and validation_batch_idx >= validation_max_batches
        ):
            break
        if "segment_ids" in batch:
            raise RuntimeError(
                "Packed multimodal batches are training-only. Validation loss "
                "and validation image generation require one logical sample "
                "per physical row."
            )
        if hasattr(diagnostic_head, "set_attention_diagnostics"):
            diagnostic_head.set_attention_diagnostics(
                validation_batch_idx < diagnostic_batches
            )
        input_ids = batch["input_ids"].contiguous().to(
            accelerator.device, non_blocking=True
        )
        token_types = batch["token_types"].to(
            accelerator.device, non_blocking=True
        )
        sigma = batch["sigma"].to(accelerator.device, non_blocking=True)
        labels = batch["labels"].to(accelerator.device, non_blocking=True)
        host_image_loss_mask = batch["image_loss_mask"]
        image_loss_mask = host_image_loss_mask.to(
            accelerator.device,
            dtype=torch.bool,
            non_blocking=True,
        )
        position_ids = batch["position_ids"].to(
            accelerator.device, non_blocking=True
        )
        image_local_positions = batch["image_local_positions"].to(
            accelerator.device, non_blocking=True
        )
        host_image_span_table = batch["image_span_table"]
        image_span_table = host_image_span_table.to(
            accelerator.device, non_blocking=True
        )
        image_latents = batch.get("image_latents", None)
        if image_latents is not None:
            image_latents = image_latents.to(
                accelerator.device, non_blocking=True
            )
        B, L = input_ids.shape

        image_loss_mask, image_latent_mask, _ = _prepare_showo_image_masks(
            config=config,
            token_types=token_types,
            image_span_table=image_span_table,
            image_loss_mask=image_loss_mask,
            mask_generation_images=True,
        )
        selfless_attention_mask, content_attention_mask = (
            _build_backbone_attention_masks(
                config=config,
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
            )
        )
        forward_kwargs = dict(
            X0_input_ids=input_ids,
            labels=labels,
            attention_mask=selfless_attention_mask,
            token_types=token_types,
            position_ids=position_ids,
            image_local_positions=image_local_positions,
            image_span_table=image_span_table,
            image_loss_mask=image_loss_mask,
            image_latents=image_latents,
            flow_sigma=sigma,
            calculate_likelihood=True,
            record_flow_stats=(validation_batch_idx < diagnostic_batches),
        )
        if unified_active_sources is not None:
            forward_kwargs["compute_text_loss"] = validate_text_source
            forward_kwargs["compute_image_loss"] = validate_image_source
        if content_attention_mask is not None:
            forward_kwargs["content_attention_mask"] = content_attention_mask
        if image_latent_mask is not None:
            forward_kwargs["image_latent_mask"] = image_latent_mask
        output = model(**forward_kwargs)
        per_modality_loss = getattr(output, "per_modality_loss", None)
        per_modality_count = getattr(output, "per_modality_count", None)
        if per_modality_loss is None or per_modality_count is None:
            raise RuntimeError(
                "joint validation requires per-modality loss/count output"
            )
        text_count = per_modality_count["text_tokens"].to(
            accelerator.device, dtype=torch.float32
        )
        image_count = per_modality_count["image_tokens"].to(
            accelerator.device, dtype=torch.float32
        )
        local_weighted_text += (
            per_modality_loss["text_loss"].float() * text_count
        )
        local_text_tokens += text_count
        local_weighted_image += (
            per_modality_loss["image_loss"].float() * image_count
        )
        local_image_tokens += image_count
        flow_stats = getattr(output, "flow_debug_stats", None) or {}
        for key, value in flow_stats.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                continue
            scalar = value.detach().to(device=accelerator.device, dtype=torch.float32)
            if not bool(torch.isfinite(scalar).all().item()):
                raise FloatingPointError(
                    "non-finite validation flow diagnostic: "
                    f"global_step={global_step}, key={key!r}, value={scalar}"
                )
            local_flow_stat_sums[key] = local_flow_stat_sums.get(
                key, torch.tensor(0.0, device=accelerator.device)
            ) + scalar
            local_flow_stat_counts[key] = local_flow_stat_counts.get(
                key, torch.tensor(0.0, device=accelerator.device)
            ) + 1.0
        if (
            not saved_validation_images
            and image_latents is not None
        ):
            active_spans = []
            for span in host_image_span_table.tolist():
                row, _, start, end, *_ = map(int, span)
                if bool(host_image_loss_mask[row, start:end].any()):
                    active_spans.append(span)
            active_span_table = torch.tensor(
                active_spans,
                dtype=host_image_span_table.dtype,
            ).reshape(-1, host_image_span_table.shape[1])
            if active_span_table.shape[0] > 0:
                _save_validation_flow_images(
                    model=model,
                    output=output,
                    input_ids=input_ids,
                    token_types=token_types,
                    sigma=sigma,
                    image_span_table=active_span_table,
                    image_latents=image_latents,
                    accelerator=accelerator,
                    global_step=global_step,
                    config=config,
                )
                saved_validation_images = True
        if not saved_validation_i2t and image_latents is not None:
            saved_validation_i2t = _save_validation_i2t_captions(
                model=model,
                tokenizer=tokenizer,
                labels=labels,
                task_modes=batch.get("task_modes", []),
                image_span_table=host_image_span_table,
                image_latents=image_latents,
                sigma=sigma,
                accelerator=accelerator,
                global_step=global_step,
                config=config,
            )

    if hasattr(diagnostic_head, "set_attention_diagnostics"):
        diagnostic_head.set_attention_diagnostics(False)

    global_weighted_text = accelerator.reduce(
        local_weighted_text, reduction="sum"
    )
    global_text_tokens = accelerator.reduce(
        local_text_tokens, reduction="sum"
    )
    global_weighted_image = accelerator.reduce(
        local_weighted_image, reduction="sum"
    )
    global_image_tokens = accelerator.reduce(local_image_tokens, reduction="sum")
    if validate_image_source and global_image_tokens.item() <= 0:
        raise RuntimeError("validation dataloader produced no image tokens")
    lambda_text = float(getattr(unwrapped_model, "lambda_text", 0.0))
    lambda_image = float(getattr(unwrapped_model, "lambda_image", 1.0))
    require_text_targets = (
        validate_text_source
        if unified_active_sources is not None
        else lambda_text > 0.0
    )
    if require_text_targets and global_text_tokens.item() <= 0:
        raise RuntimeError(
            "validation dataloader produced no active caption targets"
        )
    avg_text = (
        global_weighted_text / global_text_tokens.clamp_min(1.0)
    )
    avg_image = global_weighted_image / global_image_tokens.clamp_min(1.0)
    weighted_i2t = float((lambda_text * avg_text).item())
    weighted_t2i = float((lambda_image * avg_image).item())
    avg_loss = float(
        (lambda_text * avg_text + lambda_image * avg_image).item()
    )

    logs = {
        "val/loss": avg_loss,
        "val/loss_text": float(avg_text.item()),
        "val/loss_i2t": float(avg_text.item()),
        "val/ppl_text": math.exp(min(float(avg_text.item()), 100.0)),
        "val/loss_image_flow": float(avg_image.item()),
        "val/loss_t2i": float(avg_image.item()),
        "val/weighted_contribution_i2t": weighted_i2t,
        "val/weighted_contribution_t2i": weighted_t2i,
        "val/weighted_contribution_total": weighted_i2t + weighted_t2i,
        "val/text_target_tokens": float(global_text_tokens.item()),
        "val/image_target_tokens": float(global_image_tokens.item()),
    }
    for key in sorted(local_flow_stat_sums):
        global_sum = accelerator.reduce(local_flow_stat_sums[key], reduction="sum")
        global_count = accelerator.reduce(
            local_flow_stat_counts[key], reduction="sum"
        )
        logs[f"val/{key}"] = (global_sum / global_count.clamp_min(1.0)).item()

    if accelerator.is_main_process:
        accelerator.log(logs, step=global_step)
        if config is not None:
            metrics_path = (
                Path(config.experiment.output_dir)
                / f"validation_metrics_step_{int(global_step)}.json"
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "schema": "selfless_flow_validation_metrics_v1",
                        "global_step": int(global_step),
                        "backbone_attention": {
                            "training_objective": str(
                                config.model.get(
                                    "training_objective",
                                    "selfless_dual_stream",
                                )
                            ),
                            "dual_stream_attention_contract": str(
                                config.model.get(
                                    "dual_stream_attention_contract",
                                    "selfless_strict",
                                )
                            ),
                            "query_stream_diagonal": False,
                            "content_stream_diagonal": (
                                str(
                                    config.model.get(
                                        "dual_stream_attention_contract",
                                        "selfless_strict",
                                    )
                                ).strip().lower()
                                == "xlnet_content_diagonal"
                            ),
                        },
                        "flow_head_attention": flow_head_attention_report(
                            config.model
                        ),
                        "training_seed": int(config.training.seed),
                        "validation_seed": int(
                            config.experiment.get(
                                "validation_seed", config.training.seed
                            )
                        ),
                        "metrics": logs,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        logger.info(
            f"[Validation] Step {global_step} | "
            f"Loss: {avg_loss:.4f} | Text: {float(avg_text.item()):.4f} | "
            f"ImageFlow: {float(avg_image.item()):.4f}"
        )

    return avg_loss


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error", message="None of the inputs have requires_grad=True")
    main() 
