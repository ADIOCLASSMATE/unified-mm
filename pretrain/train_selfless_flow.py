import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import json
import logging
import math
import time
import importlib.util
import colorsys
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.optim import AdamW
import torch.nn.functional as F


from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from utils.dataset_utils import get_dataloaders
from utils.wsd_schedule import get_wsd_schedule
from utils.selfless_flow_optimizer import weight_decay_for_parameter
from utils.selfless_training_runtime import (
    RESUME_SCHEMA,
    RESUME_SIGNATURE_VERSION,
    TrainingWindow,
    build_resume_signature,
    validate_resume_metadata,
    validate_wsd_contract,
)
from utils.sharded_ema import (
    DEFAULT_EMA_CHUNK_NUMEL,
    RankShardedEMA,
    build_sharded_ema_layout,
    load_ema_manifest,
    mark_hf_ema_config_fp32,
    merge_sharded_ema_state_dict,
    read_sharded_ema_rows,
)
from models.logging import set_verbosity_info, set_verbosity_error
from utils.utils import (
    flatten_omega_conf,
    get_config,
    get_selfless_mask,
    load_model_tokenizer,
    save_checkpoint,
    save_hf_model,
)

logger = get_logger(__name__, log_level="INFO")
_VAE_CACHE = None


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


def _is_disabled_path(value):
    return value is None or (
        isinstance(value, str) and value.lower() in {"none", "null", "false", ""}
    )


def _load_image_flow_adapter(model, adapter_path, config):
    if _is_disabled_path(adapter_path):
        return

    adapter_path = Path(adapter_path)
    if adapter_path.is_dir():
        adapter_path = adapter_path / "model.safetensors"
    if adapter_path.suffix == ".safetensors":
        from safetensors import safe_open

        head_state = {}
        condition_proj_state = {}
        projector_state = {}

        with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_flow_head."):
                    head_state[key[len("image_flow_head."):]] = f.get_tensor(key)
                elif key.startswith("image_flow_condition_proj."):
                    condition_proj_state[key[len("image_flow_condition_proj."):]] = f.get_tensor(key)
                elif key.startswith("model.image_token_embedder."):
                    name = key[len("model.image_token_embedder."):]
                    projector_state[name] = f.get_tensor(key)
        model.image_flow_head.load_state_dict(head_state, strict=True)
        model.image_flow_condition_proj.load_state_dict(
            condition_proj_state, strict=True
        )
        model.image_token_embedder.load_state_dict(
            projector_state, strict=True
        )
        _log_info(f"Loaded finalized image-flow modules from {adapter_path}")
        return

    state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    required = {
        "image_flow_head",
        "image_flow_condition_proj",
        "image_token_embedder",
        "special_token_embeddings",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(
            f"Final image-flow adapter {adapter_path} is missing {sorted(missing)}"
        )
    model.image_flow_head.load_state_dict(state["image_flow_head"], strict=True)
    model.image_flow_condition_proj.load_state_dict(
        state["image_flow_condition_proj"], strict=True
    )
    model.image_token_embedder.load_state_dict(
        state["image_token_embedder"], strict=True
    )
    token_ids = _special_token_ids(config)
    if set(state["special_token_embeddings"]) != set(token_ids):
        raise ValueError(
            "Adapter special-token set does not match the finalized model: "
            f"adapter={sorted(state['special_token_embeddings'])}, "
            f"model={sorted(token_ids)}"
        )
    with torch.no_grad():
        embed = model.model.embed_tokens.weight
        for name, token_id in token_ids.items():
            value = state["special_token_embeddings"][name].to(
                device=embed.device,
                dtype=embed.dtype,
            )
            embed[token_id].copy_(value)
    _log_info(f"Loaded finalized image-flow adapter from {adapter_path}")


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


def _save_image_flow_adapter(model, config, accelerator, global_step):
    if not config.training.get("save_image_flow_adapter", False):
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
    _write_image_flow_adapter(state, config, global_step)


def _save_ema_image_flow_adapter(
    ema_directory: Path,
    config,
    accelerator,
    global_step,
) -> None:
    if not config.training.get("save_image_flow_adapter", False):
        return
    if accelerator.is_main_process:
        manifest = load_ema_manifest(ema_directory)
        prefixes = (
            "image_flow_head.",
            "image_flow_condition_proj.",
            "model.image_token_embedder.",
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
) -> Path | None:
    if ema is None:
        return None
    directory = _ema_state_directory(config, global_step)
    manifest_path = ema.save_checkpoint(
        directory,
        accelerator,
        global_step=int(global_step) if isinstance(global_step, int) else ema.global_step,
    )
    if accelerator.is_main_process:
        logger.info(f"Saved sharded EMA state to {manifest_path}")
    return directory


def _save_ema_hf_model(
    ema: RankShardedEMA | None,
    model,
    tokenizer,
    config,
    accelerator,
    global_step,
    ema_directory: Path | None,
) -> None:
    if ema is None or not ema.started or not bool(config.training.get("ema_save_hf_model", True)):
        return
    if ema_directory is None or not (ema_directory / "ema_manifest.json").is_file():
        ema_directory = _save_ema_state(ema, config, accelerator, global_step)
    if accelerator.is_main_process:
        merged_state = merge_sharded_ema_state_dict(ema_directory)
        save_path = Path(config.experiment.output_dir) / f"hf_model-{global_step}-ema"
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            save_path,
            state_dict=merged_state,
            safe_serialization=True,
        )
        mark_hf_ema_config_fp32(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved EMA HF model to {save_path}")
        del merged_state
    accelerator.wait_for_everyone()


def _unwrap_epoch_dataset(dataset):
    ds = dataset
    if hasattr(ds, "set_epoch"):
        return ds
    while hasattr(ds, "dataset"):
        if hasattr(ds, "set_epoch"):
            return ds
        ds = ds.dataset
    return ds


_resume_signature = build_resume_signature


def _write_training_checkpoint_metadata(
    checkpoint_dir: Path,
    *,
    accelerator,
    global_step: int,
    epoch: int,
    batches_consumed_in_epoch: int,
    config_signature: str,
    ema_layout,
) -> None:
    if accelerator.is_main_process:
        payload = {
            "schema": RESUME_SCHEMA,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batches_consumed_in_epoch": int(batches_consumed_in_epoch),
            "world_size": int(accelerator.num_processes),
            "gradient_accumulation_steps": int(
                accelerator.gradient_accumulation_steps
            ),
            "config_signature": config_signature,
            "config_signature_version": RESUME_SIGNATURE_VERSION,
            "ema_layout_fingerprint": (
                ema_layout["layout_fingerprint"] if ema_layout is not None else None
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
    """Invalidate an existing commit marker before replacing checkpoint files."""

    if accelerator.is_main_process:
        (checkpoint_dir / "checkpoint_complete.json").unlink(missing_ok=True)
    accelerator.wait_for_everyone()


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


def main():
    #########################
    #      SETUP Config     #
    #########################
    config = get_config()
    validate_wsd_contract(config)

    log_every = int(config.experiment.log_every)
    flow_stats_every = int(config.experiment.get("flow_stats_every", 0))
    backbone_gate_stats_every = int(
        config.experiment.get("backbone_gate_stats_every", 0)
    )
    if log_every <= 0:
        raise ValueError(f"experiment.log_every must be positive, got {log_every}")
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
    global_micro_batch = int(config.training.batch_size) * num_processes
    if int(config.training.total_batch_size) % global_micro_batch:
        raise ValueError(
            "training.total_batch_size must be divisible by batch_size * world_size: "
            f"{config.training.total_batch_size} % ({config.training.batch_size} * {num_processes}) != 0"
        )
    gradient_accumulation_steps = int(config.training.total_batch_size) // global_micro_batch
    print(f"Number of processes: {num_processes}")
    print(f"Total batch size: {config.training.total_batch_size}")
    print(f"Batch size per GPU: {total_batch_size_per_gpu}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb",
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
    if accelerator.is_main_process:
        log_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        log_config.pop("experiment.resume_from_checkpoint", None)

        wandb_init_kwargs = {
            "name": config.experiment.name,
            "resume": "allow",
            "mode": os.environ.get("WANDB_MODE", "offline"),
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
    model, tokenizer = load_model_tokenizer(
        config=config,
        logger=logger,
        model_dtype=torch.bfloat16,
    )

    flow_adapter = config.model.get("pretrained_image_flow_adapter", None)
    _load_image_flow_adapter(model, flow_adapter, config)

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
        if bool(config.training.get("ema_validate", False)):
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
    def lr_for_param(name):
        if name.startswith("image_flow_head."):
            return flow_lr
        if "image_token_embedder" in name or name.startswith("image_flow_condition_proj."):
            return projector_lr
        if "embed_tokens.weight" in name:
            return special_token_lr
        return backbone_lr

    grouped = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        weight_decay = weight_decay_for_parameter(
            name,
            global_weight_decay=global_weight_decay,
            flow_weight_decay=flow_weight_decay,
        )
        key = (lr_for_param(name), weight_decay)
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

    ##################################
    #       Prepare accelerator     #
    ##################################
    logger.info("Preparing model, optimizer and dataloaders")

    # Store ref to underlying packed dataset for epoch-level reshuffling/repacking.
    ds = _unwrap_epoch_dataset(train_dataloader.dataset)
    _is_multimodal_ds = hasattr(ds, 'set_epoch')

    if hasattr(train_dataloader, "prepare_with_accelerator"):
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        train_dataloader = train_dataloader.prepare_with_accelerator(accelerator)
        val_dataloader = val_dataloader.prepare_with_accelerator(accelerator)
    else:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)
    if ema_layout is not None:
        ema = RankShardedEMA(
            ema_layout,
            rank=accelerator.process_index,
            decay=ema_decay_value,
            update_after_step=ema_update_after_step,
        )
        ema.bind(accelerator.unwrap_model(model))

    config_signature = _resume_signature(
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
        validate_resume_metadata(
            resume_metadata,
            checkpoint_dir=resume_checkpoint_dir,
            config=config,
            world_size=accelerator.num_processes,
            gradient_accumulation_steps=(
                accelerator.gradient_accumulation_steps
            ),
            current_signature=config_signature,
        )
        expected_ema_fingerprint = (
            ema_layout["layout_fingerprint"] if ema_layout is not None else None
        )
        if resume_metadata.get("ema_layout_fingerprint") != expected_ema_fingerprint:
            raise RuntimeError("Resume EMA layout fingerprint mismatch")

        # Model, optimizer, scheduler, and RNG state are loaded only after the
        # immutable resume contract has been validated.
        accelerator.load_state(resume_checkpoint_dir)
        global_step = resume_step
        logger.info(f"Resumed at global_step={global_step}")

    else:
        logger.info("Starting fresh caption training.")
        global_step = 0
        resume_step = 0

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
    total_batch_size = (
        total_batch_size_per_gpu
        * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )
    logger.info("***** Running selfless pretraining *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  mask_token_id: {config.model.mask_token_id}")
    
    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        if not config_path.exists():
            logging.info(f"Saving immutable run config to {config_path}")
            OmegaConf.save(config, config_path)
        else:
            logging.info(f"Keeping existing immutable run config at {config_path}")

    batches_to_skip = 0
    resume_epoch = 0
    initial_train_dataloader = train_dataloader
    if resume_step > 0:
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
        if _is_multimodal_ds:
            ds.set_epoch(resume_epoch)
    caption_shuffle_seed = int(
        config.training.get("dataloader_shuffle_seed", config.training.seed)
    )
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

    model.train()

    train_iter = iter(initial_train_dataloader)
    training_window = TrainingWindow()
    training_runtime_started_at = time.time()
    training_runtime_start_step = global_step
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    # Accumulators across gradient-accumulation micro-batches.
    acc_loss = torch.tensor(0.0, device=accelerator.device)
    acc_flow_stats = {}
    acc_flow_stat_batches = torch.tensor(0.0, device=accelerator.device)
    acc_backbone_gate_stats = {}
    acc_backbone_gate_stat_batches = torch.tensor(
        0.0, device=accelerator.device
    )

    epoch = resume_epoch
    batches_consumed_in_epoch = batches_to_skip
    while global_step < config.training.max_train_steps:
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
        is_multimodal = "token_types" in batch
        if not is_multimodal:
            raise ValueError(
                "Selfless-Flow training requires an image batch from "
                "ImageNetFlowCacheDataset."
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

            selfless_attention_mask = get_selfless_mask(
                sigma=sigma,
                seq_len=L,
                device=accelerator.device,
                input_ids=input_ids,
                token_types=token_types,
                boi_token_id=int(config.model.boi_token_id),
                image_uncond_rows=image_uncond_rows,
                segment_ids=segment_ids,
                image_uncond_mask=image_uncond_mask,
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
                    if accelerator.is_main_process:
                        first_pack_path = (
                            Path(config.experiment.output_dir)
                            / "first_batch_pack_manifest.json"
                        )
                        first_pack_payload = {
                            "pack_manifest_sha256": batch.get(
                                "pack_manifest_sha256"
                            ),
                            "pack_manifest": batch.get("pack_manifest"),
                            "sample_img_ids": batch.get(
                                "sample_img_ids", []
                            ),
                            "sample_token_sha256": batch.get(
                                "sample_token_sha256", []
                            ),
                            "augmentation_sha256": batch.get(
                                "augmentation_sha256", []
                            ),
                        }
                        first_pack_path.write_text(
                            json.dumps(
                                first_pack_payload,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
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
            if token_types is not None:
                forward_kwargs["token_types"] = token_types
                forward_kwargs["flow_sigma"] = sigma
                if position_ids is not None:
                    forward_kwargs["position_ids"] = position_ids
                if image_local_positions is not None:
                    forward_kwargs["image_local_positions"] = image_local_positions
                if image_span_table is not None:
                    forward_kwargs["image_span_table"] = image_span_table
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
            finite_loss = torch.isfinite(loss.detach()).all()
            if hasattr(torch, "_assert_async"):
                torch._assert_async(
                    finite_loss,
                    f"non-finite caption training loss at global_step={global_step}",
                )
            elif (global_step + 1) % int(config.experiment.log_every) == 0:
                if not bool(finite_loss.item()):
                    raise FloatingPointError(
                        f"non-finite caption training loss at global_step={global_step}"
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

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if (
                    accelerator.distributed_type != DistributedType.DEEPSPEED
                    and config.training.max_grad_norm
                ):
                    grad_norm_value = accelerator.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm
                    )
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if (
                    accelerator.distributed_type == DistributedType.DEEPSPEED
                    and (global_step + 1)
                    % int(config.experiment.log_grad_norm_every)
                    == 0
                    and hasattr(model, "get_global_grad_norm")
                ):
                    grad_norm_value = model.get_global_grad_norm()
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

            # Logging
            if did_log:
                grad_accum = accelerator.gradient_accumulation_steps
                # Average loss across all micro-batches and all ranks
                avg_loss_per_step = acc_loss / grad_accum
                global_avg_loss = accelerator.reduce(avg_loss_per_step, reduction="mean")
                global_avg_loss_value = float(global_avg_loss.item())

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
                    "train/loss_image_flow": global_avg_loss_value,
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

                if (
                    grad_norm_value is not None
                    and global_step
                    % int(config.experiment.log_grad_norm_every)
                    == 0
                ):
                    logs["train/global_grad_norm"] = float(
                        torch.as_tensor(grad_norm_value).detach().float().item()
                    )

                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    msg = (
                        f"Step: {global_step} | "
                        f"Loss: {global_avg_loss_value:0.4f}"
                    )
                    if acc_flow_stats:
                        msg += (
                            f" | FlowMSE: {global_flow_stats.get('flow/v_mse', 0.0):0.4f}"
                            f" | FlowPredVRMS: {global_flow_stats.get('flow/v_pred_rms', 0.0):0.4f}"
                        )
                    msg += (
                        f" | LR: {lr_scheduler.get_last_lr()[0]:0.6f} | "
                        f"Sec/Step: {logs['seconds/optimizer_step']:0.4f} | "
                        f"Data/Microbatch: {logs['data_wait_ms/microbatch']:0.2f}ms"
                    )
                    logger.info(msg)

            post_step_maintenance_started = time.perf_counter()

            # Checkpointing
            if global_step % config.experiment.save_every == 0:
                checkpoint_dir = (
                    Path(config.experiment.output_dir)
                    / f"checkpoint-{global_step}"
                )
                _begin_checkpoint_write(
                    checkpoint_dir,
                    accelerator=accelerator,
                )
                save_checkpoint(model, config, accelerator, global_step)
                _write_training_checkpoint_metadata(
                    checkpoint_dir,
                    accelerator=accelerator,
                    global_step=global_step,
                    epoch=epoch,
                    batches_consumed_in_epoch=batches_consumed_in_epoch,
                    config_signature=config_signature,
                    ema_layout=ema_layout,
                )
                ema_directory = _save_ema_state(ema, config, accelerator, global_step)
                _mark_checkpoint_complete(
                    checkpoint_dir,
                    accelerator=accelerator,
                    global_step=global_step,
                )
                if ema is not None and ema.started and bool(config.training.get("ema_save_adapter", True)):
                    _save_ema_image_flow_adapter(
                        ema_directory,
                        config,
                        accelerator,
                        global_step,
                    )
                else:
                    _save_image_flow_adapter(model, config, accelerator, global_step)
            
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
            if global_step % config.experiment.val_every == 0:
                validate(
                    model,
                    val_dataloader,
                    accelerator,
                    global_step,
                    config,
                )

                model.train()

            # Exclude checkpointing and validation from the next training
            # throughput window. The current window was sampled before these
            # cold-path operations started.
            if did_log:
                training_window.reset()
            else:
                training_window.exclude_elapsed(
                    time.perf_counter() - post_step_maintenance_started
                )

            # Reset per-step accumulators for the next optimizer step
            acc_loss.zero_()
            acc_flow_stats.clear()
            acc_flow_stat_batches.zero_()
            acc_backbone_gate_stats.clear()
            acc_backbone_gate_stat_batches.zero_()
            if global_step >= config.training.max_train_steps:
                break

    training_runtime_elapsed = time.time() - training_runtime_started_at
    if accelerator.device.type == "cuda":
        local_runtime = torch.tensor(
            [
                float(torch.cuda.max_memory_allocated(accelerator.device)),
                float(torch.cuda.max_memory_reserved(accelerator.device)),
                float(training_runtime_elapsed),
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )
    else:
        local_runtime = torch.tensor(
            [0.0, 0.0, float(training_runtime_elapsed)],
            device=accelerator.device,
            dtype=torch.float64,
        )
    gathered_runtime = accelerator.gather(local_runtime).reshape(-1, 3)
    runtime_max = gathered_runtime.max(dim=0).values
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        runtime_payload = {
            "schema": "selfless_training_runtime_metrics_v1",
            "global_step": int(global_step),
            "world_size": int(accelerator.num_processes),
            "total_batch_size": int(total_batch_size),
            "steps_this_run": int(global_step - training_runtime_start_step),
            "training_wall_seconds": float(runtime_max[2].item()),
            "train_samples_per_second": float(
                (global_step - training_runtime_start_step)
                * total_batch_size
                / max(float(runtime_max[2].item()), 1e-12)
            ),
            "peak_cuda_allocated_bytes_per_rank": int(
                runtime_max[0].item()
            ),
            "peak_cuda_reserved_bytes_per_rank": int(
                runtime_max[1].item()
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
        runtime_path = (
            Path(config.experiment.output_dir)
            / "training_runtime_metrics.json"
        )
        runtime_path.write_text(
            json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
        if ema is not None and ema.started and bool(config.training.get("ema_save_adapter", True)):
            _save_ema_image_flow_adapter(
                ema_directory,
                config,
                accelerator,
                "final",
            )
        else:
            _save_image_flow_adapter(model, config, accelerator, "final")
    accelerator.end_training()


@torch.no_grad()
def validate(model, val_dataloader, accelerator, global_step, config=None):
    validation_seed = int(
        config.experiment.get("validation_seed", config.training.seed)
    ) + int(accelerator.process_index)
    cuda_devices = (
        [int(accelerator.device.index)]
        if accelerator.device.type == "cuda"
        else []
    )
    # Validation must not perturb the training RNG stream, and every checkpoint
    # must see the same flow times/noise on each rank.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.default_generator.manual_seed(validation_seed)
        if accelerator.device.type == "cuda":
            with torch.cuda.device(accelerator.device):
                torch.cuda.manual_seed(validation_seed)
        model.eval()  # DeepSpeed requires explicit eval mode for no_grad forward
        try:
            _validate_multimodal(
                model,
                val_dataloader,
                accelerator,
                global_step,
                config,
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
    dtype = torch.float16 if vae_dtype_name in {"fp16", "float16", "half"} and accelerator.device.type == "cuda" else torch.float32
    return _VAE_CACHE.to(device=accelerator.device, dtype=dtype).eval()


def _image_spans_from_table(
    image_span_table: torch.Tensor,
    image_tokens_per_img: int,
) -> list[tuple[int, int, int]]:
    """Read validation spans without scanning a CUDA tensor token by token."""

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


def _heatmap_images(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float().cpu()
    valid = torch.isfinite(values) & (values != 0)
    if valid.any():
        min_val = values[valid].min()
        max_val = values[valid].max()
        values = (values - min_val) / (max_val - min_val).clamp_min(1e-6)
    else:
        values = torch.zeros_like(values)
    values = values.clamp(0, 1)
    red = values
    green = 1.0 - (values - 0.5).abs() * 2.0
    blue = 1.0 - values
    heatmap = torch.stack([red, green.clamp(0, 1), blue], dim=1)
    heatmap = heatmap * valid.unsqueeze(1).float()
    return heatmap


def _generation_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, float(value)))
    hue = (2.0 / 3.0) * (1.0 - value)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _save_readable_generation_map(
    values: torch.Tensor,
    path: Path,
    *,
    title: str,
    label_prefix: str = "",
    normalize_labels: bool = False,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    values = values.detach().float().cpu()
    if values.dim() == 2:
        values = values.unsqueeze(0)
    if values.dim() != 3:
        raise ValueError(f"generation map must be [N,H,W] or [H,W], got {tuple(values.shape)}")

    valid = torch.isfinite(values)
    if valid.any():
        min_val = float(values[valid].min().item())
        max_val = float(values[valid].max().item())
    else:
        min_val, max_val = 0.0, 1.0
    span = max(max_val - min_val, 1e-6)

    def font(size: int):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    title_font = font(16)
    cell_font = font(10)
    caption_font = font(11)
    cell = 34
    top = 36
    left = 12
    right = 96
    bottom = 46
    gap = 18
    panels = []

    for sample_idx, sample in enumerate(values):
        height, width = sample.shape
        panel_w = left + width * cell + right
        panel_h = top + height * cell + bottom
        image = Image.new("RGB", (panel_w, panel_h), "white")
        draw = ImageDraw.Draw(image)
        draw.text((left, 8), f"{title} | sample {sample_idx + 1}", fill=(20, 20, 20), font=title_font)

        for row in range(height):
            for col in range(width):
                raw = float(sample[row, col].item())
                is_valid = math.isfinite(raw)
                norm = 0.0 if not is_valid else (raw - min_val) / span
                x0 = left + col * cell
                y0 = top + row * cell
                color = _generation_color(norm) if is_valid else (235, 235, 235)
                draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color, outline=(75, 75, 75))
                if is_valid:
                    label_value = raw - min_val if normalize_labels else raw
                    label = f"{label_prefix}{int(round(label_value)) + 1}"
                    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                    text_color = (0, 0, 0) if luminance > 145 else (255, 255, 255)
                    bbox = draw.textbbox((0, 0), label, font=cell_font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    draw.text(
                        (x0 + (cell - text_w) / 2, y0 + (cell - text_h) / 2 - 1),
                        label,
                        fill=text_color,
                        font=cell_font,
                    )

        bar_x = left + width * cell + 22
        bar_y = top
        bar_w = 18
        bar_h = height * cell
        for offset in range(bar_h):
            norm = 1.0 - offset / max(1, bar_h - 1)
            draw.line(
                [(bar_x, bar_y + offset), (bar_x + bar_w, bar_y + offset)],
                fill=_generation_color(norm),
            )
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=(75, 75, 75))
        draw.text((bar_x + bar_w + 6, bar_y - 2), "late", fill=(30, 30, 30), font=caption_font)
        draw.text((bar_x + bar_w + 6, bar_y + bar_h - 12), "early", fill=(30, 30, 30), font=caption_font)
        draw.text((left, top + height * cell + 10), "Numbers are 1-indexed: 1 = first generated.", fill=(45, 45, 45), font=caption_font)
        panels.append(image)

    total_w = sum(panel.width for panel in panels) + gap * max(0, len(panels) - 1)
    total_h = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (total_w, total_h), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


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


def _validation_sequence_mixer_context(
    target: torch.Tensor,
    span_sigma: torch.Tensor,
    local_positions: torch.Tensor,
    conditions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sequence_length = target.shape[0]
    if conditions.shape[0] != sequence_length:
        raise ValueError(
            "validation content conditions must align with target tokens, "
            f"got target={tuple(target.shape)}, conditions={tuple(conditions.shape)}"
        )
    sigma_row = span_sigma.to(
        device=target.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    positions = local_positions.to(
        device=target.device,
        dtype=torch.long,
    ).unsqueeze(0)
    return {
        "context_latents": target.unsqueeze(0),
        "context_mask": sigma_row.unsqueeze(1) < sigma_row.unsqueeze(2),
        "query_positions": positions,
        "context_positions": positions,
        "context_conditions": conditions.to(device=target.device).unsqueeze(0),
    }


def _validation_flat_query_mixer_context(
    target: torch.Tensor,
    span_sigma: torch.Tensor,
    local_positions: torch.Tensor,
    conditions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    query_count = target.shape[0]
    if conditions.shape[0] != query_count:
        raise ValueError(
            "validation content conditions must align with target tokens, "
            f"got target={tuple(target.shape)}, conditions={tuple(conditions.shape)}"
        )
    sigma_values = span_sigma.to(
        device=target.device,
        dtype=torch.float32,
    )
    positions = local_positions.to(
        device=target.device,
        dtype=torch.long,
    )
    return {
        "context_latents": target.unsqueeze(0)
        .expand(query_count, -1, -1)
        .contiguous(),
        "context_mask": (
            sigma_values.unsqueeze(0) < sigma_values.unsqueeze(1)
        ).unsqueeze(1),
        "query_positions": positions,
        "context_positions": positions.unsqueeze(0)
        .expand(query_count, -1)
        .contiguous(),
        "context_conditions": conditions.to(device=target.device)
        .unsqueeze(0)
        .expand(query_count, -1, -1)
        .contiguous(),
    }


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
    if config is None:
        return
    image_every = config.experiment.get("validation_image_every", config.experiment.get("val_every", 0))
    if not image_every or global_step % image_every != 0:
        return
    if image_latents is None or not hasattr(output, "last_hidden_state"):
        return

    unwrapped = accelerator.unwrap_model(model)
    image_tokens_per_img = int(getattr(unwrapped.config, "image_tokens_per_img", config.model.get("image_tokens_per_img", 256)))
    spans = _image_spans_from_table(
        image_span_table,
        image_tokens_per_img,
    )

    # ZeRO-2 keeps complete parameters on every rank, so validation can run in
    # parallel without gathering the flow head onto rank 0.  Direct execution
    # is not valid for partitioned ZeRO-3 parameters.
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        deepspeed_config = accelerator.state.deepspeed_plugin.deepspeed_config
        zero_stage = int(
            deepspeed_config.get("zero_optimization", {}).get(
                "stage", deepspeed_config.get("zero_stage", -1)
            )
        )
        if zero_stage != 2:
            raise RuntimeError(
                "Parallel validation image generation requires replicated "
                f"parameters (DeepSpeed ZeRO-2); resolved stage is {zero_stage}."
            )

    span_counts = accelerator.gather(
        torch.tensor([len(spans)], device=accelerator.device, dtype=torch.long)
    )
    common_span_count = int(span_counts.min().item())

    with torch.no_grad():
        if common_span_count <= 0:
            if accelerator.is_main_process:
                logger.warning("Skipping validation image decode; at least one rank has no complete image span.")
            return
        if not spans:
            logger.warning("Skipping validation image decode; no complete image span in validation batch.")
            return

        vae = _load_vae_decoder(config, accelerator)
        if vae is None:
            return

        sample_count = min(
            int(config.experiment.get("validation_image_samples", 4)),
            common_span_count,
        )
        flow_temperature = float(config.experiment.get("validation_flow_temperature", 1.0))
        flow_cfg = float(config.experiment.get("validation_flow_cfg", 1.0))
        flow_cfg_schedule = str(config.experiment.get("validation_flow_cfg_schedule", "constant"))
        flow_solver = config.experiment.get("validation_flow_solver", config.model.get("image_flow_solver", None))
        probe_config = config.experiment.get("validation_flow_probe_times", [0.25, 0.5, 0.75, 0.95])
        if isinstance(probe_config, str):
            probe_times = [float(item.strip()) for item in probe_config.split(",") if item.strip()]
        elif isinstance(probe_config, (int, float)):
            probe_times = [float(probe_config)]
        else:
            probe_times = [float(value) for value in probe_config]
        probe_times = [min(1.0 - 1.0e-4, max(1.0e-4, value)) for value in probe_times]
        scaling_factor = float(config.experiment.get("validation_vae_scaling_factor", 0.2325))
        side = int(image_tokens_per_img ** 0.5)
        if side * side != image_tokens_per_img:
            logger.warning(f"Skipping validation image decode; image_tokens_per_img={image_tokens_per_img} is not square.")
            return

        hidden_states = output.last_hidden_state
        pred_latents = []
        probe_x0_latents = {time_value: [] for time_value in probe_times}
        probe_v_mse = {time_value: [] for time_value in probe_times}
        probe_x0_mse = {time_value: [] for time_value in probe_times}
        target_latents = []
        selected_spans = spans[:sample_count]

        for b, start, end in selected_spans:
            local_positions = torch.arange(
                end - start,
                device=accelerator.device,
                dtype=torch.long,
            )
            target = image_latents[b, start:end].to(device=accelerator.device)
            span_sigma = sigma[b, start:end].to(device=accelerator.device, dtype=torch.float32)
            z = unwrapped._prepare_image_flow_condition(
                hidden_states[b, start:end].to(device=accelerator.device)
            )
            pred = unwrapped.sample_image_flow_with_cfg(
                z,
                z_uncond=None,
                temperature=flow_temperature,
                cfg=1.0,
                cfg_schedule="constant",
                solver=flow_solver,
                **_validation_flat_query_mixer_context(
                    target,
                    span_sigma,
                    local_positions,
                    z,
                ),
            )
            target = target.to(dtype=pred.dtype)
            sequence_context = _validation_sequence_mixer_context(
                target,
                span_sigma,
                local_positions,
                z,
            )
            for time_value in probe_times:
                t = torch.full(
                    (target.shape[0],),
                    float(time_value),
                    device=target.device,
                    dtype=torch.float32,
                )
                noise = torch.randn_like(target)
                t_view = t.view(-1, 1).to(dtype=target.dtype)
                x_t = (1.0 - t_view) * noise + t_view * target
                v_target = target - noise
                v_pred = unwrapped.image_flow_head.velocity(
                    x_t.unsqueeze(0),
                    t.unsqueeze(0),
                    z.unsqueeze(0),
                    **sequence_context,
                ).squeeze(0).to(dtype=target.dtype)
                x0_est = x_t + (1.0 - t_view) * v_pred
                probe_x0_latents[time_value].append(x0_est.view(side, side, -1).permute(2, 0, 1))
                probe_v_mse[time_value].append(F.mse_loss(v_pred.float(), v_target.float()).detach().float())
                probe_x0_mse[time_value].append(F.mse_loss(x0_est.float(), target.float()).detach().float())
            pred_latents.append(pred.view(side, side, -1).permute(2, 0, 1))
            target_latents.append(target.view(side, side, -1).permute(2, 0, 1))

        single_stream_results = {}
        if config.experiment.get("validation_single_stream_images", True):
            default_strategies = [
                "hidden_norm",
                "latent_proj_cosine",
                "spatial_halton",
            ]
            strategies = config.experiment.get("validation_single_stream_order_strategies", None)
            if strategies is None:
                strategies = default_strategies
            if isinstance(strategies, str):
                strategies = [item.strip() for item in strategies.split(",") if item.strip()]
            for order_strategy in strategies:
                single_stream_result = unwrapped.sample_image_latents_single_stream(
                    input_ids=input_ids,
                    token_types=token_types,
                    sigma=sigma,
                    spans=selected_spans,
                    image_latent_dim=image_latents.shape[-1],
                    flow_temperature=flow_temperature,
                    flow_cfg=flow_cfg,
                    flow_cfg_schedule=flow_cfg_schedule,
                    flow_solver=flow_solver,
                    parallel_rate=int(config.experiment.get("validation_single_stream_parallel_rate", 1)),
                    order_strategy=str(order_strategy),
                    return_trace=True,
                )
                single_stream_latents, single_stream_trace = single_stream_result
                if single_stream_latents is not None:
                    single_stream_results[str(order_strategy)] = (single_stream_latents, single_stream_trace)

        raw_pred_latents = torch.stack(pred_latents)
        raw_probe_x0_latents = {
            time_value: torch.stack(latents)
            for time_value, latents in probe_x0_latents.items()
            if latents
        }
        raw_target_latents = torch.stack(target_latents)
        vae_pred_latents = raw_pred_latents
        vae_probe_x0_latents = raw_probe_x0_latents
        vae_target_latents = raw_target_latents
        vae_dtype = next(vae.parameters()).dtype
        decoded_pred = vae.decode(vae_pred_latents.to(dtype=vae_dtype) / scaling_factor).float().clamp(-1, 1)
        decoded_probe_x0 = {
            time_value: vae.decode(latents.to(dtype=vae_dtype) / scaling_factor).float().clamp(-1, 1)
            for time_value, latents in vae_probe_x0_latents.items()
        }
        decoded_target = vae.decode(vae_target_latents.to(dtype=vae_dtype) / scaling_factor).float().clamp(-1, 1)

        from torchvision.utils import make_grid, save_image

        image_dir = Path(config.experiment.output_dir) / "validation_flow_images"
        write_images = accelerator.is_main_process
        if write_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        wandb_images = {}
        save_debug_images = write_images and bool(
            config.experiment.get("validation_save_debug_images", False)
        )
        pred_img = (decoded_pred + 1.0) / 2.0
        probe_x0_imgs = {
            time_value: (decoded + 1.0) / 2.0
            for time_value, decoded in decoded_probe_x0.items()
        }
        target_img = (decoded_target + 1.0) / 2.0
        if save_debug_images:
            pred_path = image_dir / f"step-{global_step:08d}-full_sample.png"
            target_path = image_dir / f"step-{global_step:08d}-target.png"
            save_image(pred_img, pred_path)
            save_image(target_img, target_path)
            wandb_images["val/debug/full_sample"] = pred_path
            wandb_images["val/debug/target"] = target_path
            for time_value, probe_img in probe_x0_imgs.items():
                probe_path = image_dir / f"step-{global_step:08d}-flow_x0_est_t{time_value:g}.png"
                save_image(probe_img, probe_path)
                wandb_images[f"val/debug/flow_x0_est_t{time_value:g}"] = probe_path

        target_rms = raw_target_latents.float().pow(2).mean().sqrt().item()
        logs = {
            "val/flow_full_sample_cfg": 1.0,
            "val/flow_full_sample_latent_mse": F.mse_loss(raw_pred_latents.float(), raw_target_latents.float()).item(),
            "val/flow_full_sample_latent_rms": raw_pred_latents.float().pow(2).mean().sqrt().item(),
            "val/flow_target_latent_rms": target_rms,
        }
        for time_value, probe_latents in raw_probe_x0_latents.items():
            tag = f"t{time_value:g}".replace(".", "p")
            probe_rms = probe_latents.float().pow(2).mean().sqrt().item()
            logs.update(
                {
                    f"val/flow_x0_est_{tag}_latent_mse": F.mse_loss(
                        probe_latents.float(), raw_target_latents.float()
                    ).item(),
                    f"val/flow_x0_est_{tag}_latent_rms": probe_rms,
                    f"val/flow_x0_est_{tag}_rms_ratio_to_target": probe_rms / max(target_rms, 1.0e-12),
                    f"val/flow_x0_est_{tag}_abs_p99": torch.quantile(
                        probe_latents.float().abs().flatten(),
                        torch.tensor(0.99, device=probe_latents.device),
                    ).item(),
                    f"val/flow_v_mse_{tag}": torch.stack(probe_v_mse[time_value]).mean().item(),
                    f"val/flow_x0_est_mse_{tag}": torch.stack(probe_x0_mse[time_value]).mean().item(),
                }
            )
        if single_stream_results:
            comparison_tiles = []
            comparison_names = []
            for order_strategy, (single_stream_latents, single_stream_trace) in single_stream_results.items():
                vae_single_stream_latents = single_stream_latents
                decoded_single_stream = vae.decode(
                    vae_single_stream_latents.to(dtype=vae_dtype) / scaling_factor
                ).float().clamp(-1, 1)
                strategy_tag = str(order_strategy).replace("/", "_")
                log_prefix = f"val/single_stream/{strategy_tag}"

                logs.update(
                    {
                        f"{log_prefix}/latent_mse_to_target": F.mse_loss(
                            single_stream_latents.float(), raw_target_latents.float()
                        ).item(),
                        f"{log_prefix}/latent_mse_to_teacher": F.mse_loss(
                            single_stream_latents.float(), raw_pred_latents.float()
                        ).item(),
                        f"{log_prefix}/latent_rms": single_stream_latents.float().pow(2).mean().sqrt().item(),
                    }
                )

                single_stream_img = (decoded_single_stream + 1.0) / 2.0
                if save_debug_images:
                    single_stream_path = image_dir / f"step-{global_step:08d}-single_stream_pred_{strategy_tag}.png"
                    save_image(single_stream_img, single_stream_path)
                    wandb_images[f"{log_prefix}/debug/pred"] = single_stream_path

                if write_images:
                    comparison = torch.stack([target_img, single_stream_img], dim=1).flatten(0, 1)
                    comparison_grid = make_grid(comparison, nrow=2)
                    comparison_path = image_dir / f"step-{global_step:08d}-strategy_{strategy_tag}.png"
                    save_image(comparison_grid, comparison_path)
                    wandb_images[f"{log_prefix}/target_strategy_grid"] = comparison_path

                    comparison_tiles.append(single_stream_img)
                    comparison_names.append(strategy_tag)

                if single_stream_trace:
                    trace_strategy = single_stream_trace.get("order_strategy", strategy_tag)
                    order_map = single_stream_trace.get("generation_order", None)
                    step_map = single_stream_trace.get("generation_step", None)
                    score_map = single_stream_trace.get("generation_score", None)
                    if isinstance(order_map, torch.Tensor):
                        logs[f"{log_prefix}/generation_order_max"] = order_map.float().max().item()
                        if save_debug_images:
                            order_path = image_dir / f"step-{global_step:08d}-single_stream_order_{trace_strategy}.png"
                            _save_readable_generation_map(
                                order_map,
                                order_path,
                                title=f"{trace_strategy} generation order",
                                normalize_labels=True,
                            )
                            wandb_images[f"{log_prefix}/debug/generation_order"] = order_path
                    if isinstance(step_map, torch.Tensor):
                        logs[f"{log_prefix}/generation_step_max"] = step_map.float().max().item()
                        if save_debug_images:
                            step_path = image_dir / f"step-{global_step:08d}-single_stream_steps_{trace_strategy}.png"
                            _save_readable_generation_map(
                                step_map,
                                step_path,
                                title=f"{trace_strategy} generation round",
                                label_prefix="R",
                                normalize_labels=True,
                            )
                            wandb_images[f"{log_prefix}/debug/generation_steps"] = step_path
                    if isinstance(score_map, torch.Tensor):
                        valid_scores = score_map[score_map != 0].float()
                        if valid_scores.numel() > 0:
                            score_mean = valid_scores.mean().item()
                            score_std = valid_scores.std(unbiased=False).item()
                            logs[f"{log_prefix}/generation_score_mean"] = score_mean
                            logs[f"{log_prefix}/generation_score_std"] = score_std
                        if save_debug_images:
                            score_grid = make_grid(_heatmap_images(score_map), nrow=sample_count)
                            score_path = image_dir / f"step-{global_step:08d}-single_stream_scores_{trace_strategy}.png"
                            save_image(score_grid, score_path)
                            wandb_images[f"{log_prefix}/debug/generation_scores"] = score_path

        else:
            comparison_tiles = []
            comparison_names = []

        metric_keys = sorted(logs)
        local_metric_values = torch.tensor(
            [logs[key] for key in metric_keys],
            device=accelerator.device,
            dtype=torch.float64,
        )
        global_metric_values = accelerator.reduce(
            local_metric_values,
            reduction="mean",
        )
        global_logs = dict(zip(metric_keys, global_metric_values.tolist()))

        if write_images:
            overview_tiles = [target_img] + list(probe_x0_imgs.values()) + [pred_img] + comparison_tiles
            overview_grid = make_grid(torch.stack(overview_tiles, dim=1).flatten(0, 1), nrow=len(overview_tiles))
            overview_path = image_dir / f"step-{global_step:08d}-overview.png"
            save_image(overview_grid, overview_path)
            wandb_images["val/overview_target_flow_fullsample"] = overview_path
            probe_column_names = [f"flow_x0_est_t{time_value:g}" for time_value in probe_x0_imgs]
            logger.info(
                "Validation overview columns: target, "
                + ", ".join(probe_column_names)
                + ", full_sample"
                + (f", {', '.join(comparison_names)}" if comparison_names else "")
            )
            accelerator.log(global_logs, step=global_step)
            _log_wandb_validation_images(accelerator, wandb_images, global_step)
            logger.info(f"Saved validation flow images to {image_dir}")

        if bool(config.experiment.get("validation_release_vae_gpu", True)):
            vae.to(device="cpu")


@torch.no_grad()
def _validate_multimodal(model, val_dataloader, accelerator, global_step, config=None):
    local_weighted_loss = torch.tensor(0.0, device=accelerator.device)
    local_image_tokens = torch.tensor(0.0, device=accelerator.device)
    local_flow_stat_sums = {}
    local_flow_stat_counts = {}
    saved_validation_images = False
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

    for validation_batch_idx, batch in enumerate(val_dataloader):
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

        image_mask = token_types == 1
        n_image = image_mask.sum().float()

        # Sigma and labels pre-computed by dataloader
        selfless_attention_mask = get_selfless_mask(
            sigma=sigma, seq_len=L, device=accelerator.device
        )
        output = model(
            X0_input_ids=input_ids, labels=labels,
            attention_mask=selfless_attention_mask,
            token_types=token_types,
            position_ids=position_ids,
            image_local_positions=image_local_positions,
            image_span_table=image_span_table,
            image_latents=image_latents,
            flow_sigma=sigma,
            calculate_likelihood=True,
            record_flow_stats=(validation_batch_idx < diagnostic_batches),
        )
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
            _save_validation_flow_images(
                model=model,
                output=output,
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                image_span_table=host_image_span_table,
                image_latents=image_latents,
                accelerator=accelerator,
                global_step=global_step,
                config=config,
            )
            saved_validation_images = True
        local_weighted_loss += output.loss.detach() * n_image
        local_image_tokens += n_image

    if hasattr(diagnostic_head, "set_attention_diagnostics"):
        diagnostic_head.set_attention_diagnostics(False)

    # Reduce across ranks using the number of image tokens as the weight.
    global_weighted_loss = accelerator.reduce(local_weighted_loss, reduction="sum")
    global_image_tokens = accelerator.reduce(local_image_tokens, reduction="sum")
    if global_image_tokens.item() <= 0:
        raise RuntimeError("validation dataloader produced no image tokens")
    avg_loss = (global_weighted_loss / global_image_tokens).item()

    logs = {
        "val/loss": avg_loss,
        "val/loss_image_flow": avg_loss,
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
            f"ImageFlow: {avg_loss:.4f}"
        )

    return avg_loss


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error", message="None of the inputs have requires_grad=True")
    main() 
