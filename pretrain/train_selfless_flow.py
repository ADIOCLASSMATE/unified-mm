import os
import copy
import random
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer


from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from utils.dataset_utils import get_dataloaders
from utils.selfless_utils import SelflessSampler
from utils.wsd_schedule import get_wsd_schedule
from models.logging import set_verbosity_info, set_verbosity_error

from utils.utils import get_config, flatten_omega_conf, get_selfless_mask, log_grad_norm, AverageMeter, save_checkpoint, save_hf_model
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM as FlowQwen3ForCausalLM

logger = get_logger(__name__, log_level="INFO")
_VAE_CACHE = None


def load_model_tokenizer(config: OmegaConf, logger=None):
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, fix_mistral_regex=True)
    mask_token = "<|mdm_mask|>"
    if mask_token in tokenizer.get_vocab():
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    else:
        tokenizer.add_special_tokens({"mask_token": mask_token})
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    config.model.mask_token_id = mask_token_id

    boi_token = "<|boi|>"
    eoi_token = "<|eoi|>"
    image_mask_token = "<|img_mask|>"
    tokens_to_add = [
        token
        for token in (boi_token, eoi_token, image_mask_token)
        if token not in tokenizer.get_vocab()
    ]
    added_image_mask_token = image_mask_token in tokens_to_add
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)

    config.model.boi_token_id = tokenizer.convert_tokens_to_ids(boi_token)
    config.model.eoi_token_id = tokenizer.convert_tokens_to_ids(eoi_token)
    config.model.image_mask_token_id = tokenizer.convert_tokens_to_ids(image_mask_token)
    unified_head = getattr(config.model, "unified_head", False)
    config.model.image_offset = getattr(config.model, "image_offset", None) or (len(tokenizer) if unified_head else 200000)

    if logger is not None:
        logger.info("Using flow model implementation.")
        logger.info('special tokens : \n', tokenizer.special_tokens_map)
        logger.info(
            f"BOI token id: {config.model.boi_token_id}, "
            f"EOI token id: {config.model.eoi_token_id}, "
            f"IMG_MASK token id: {config.model.image_mask_token_id}"
        )

    multimodal_config_keys = (
        "image_vocab_size", "image_offset", "lambda_image", "lambda_text",
        "boi_token_id", "eoi_token_id", "image_mask_token_id", "unified_head", "image_tokens_per_img",
        "image_latent_dim", "continuous_image_latents",
        "image_generation_head_type", "image_flow_width", "image_flow_depth",
        "image_flow_num_sampling_steps", "image_flow_batch_mul",
        "image_flow_grad_checkpointing", "image_flow_condition_norm",
        "image_flow_condition_norm_eps", "image_flow_time_scale",
        "image_flow_time_sampling", "image_flow_logit_mean", "image_flow_logit_std",
        "image_flow_time_eps", "image_flow_time_uniform_mix", "image_flow_solver",
        "image_flow_mlp_ratio",
        "image_flow_latent_mixer_heads", "image_flow_latent_mixer_dropout",
        "image_flow_latent_mixer_zero_init_gate",
        "image_input_noise_strength", "image_input_noise_strength_std",
        "image_input_noise_strength_min", "image_input_noise_strength_max",
        "image_uncond_prob", "image_projector_width",
    )

    model_config = AutoConfig.from_pretrained(config.model.model_path, trust_remote_code=True)
    model_config.mask_token_id = config.model.mask_token_id
    model_config.use_flex_attention = config.model.use_flex_attention
    model_config.eos_token_id = tokenizer.eos_token_id
    for key in multimodal_config_keys:
        val = config.model.get(key)
        if val is not None:
            setattr(model_config, key, val)

    if hasattr(tokenizer, "im_end_token_id") and tokenizer.im_end_token_id is not None:
        model_config.im_end_token_id = tokenizer.im_end_token_id
    else:
        try:
            im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            model_config.im_end_token_id = im_end_ids[0] if len(im_end_ids) > 0 else None
        except Exception:
            model_config.im_end_token_id = None

    if config.training.from_scratch:
        if logger is not None:
            logger.info(f"Initializing flow model from scratch using config from: {config.model.model_path}")
        model = FlowQwen3ForCausalLM(model_config).to(dtype=torch.bfloat16)
    else:
        if logger is not None:
            logger.info(f"Loading pretrained weights into flow model from: {config.model.model_path}")
        model = FlowQwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model.model_path,
            config=model_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    image_mask_token_id = getattr(model.config, "image_mask_token_id", None)
    if image_mask_token_id is not None and added_image_mask_token:
        with torch.no_grad():
            embed = model.model.embed_tokens.weight
            mask_token_id = int(model.config.mask_token_id)
            image_mask_token_id = int(image_mask_token_id)
            if 0 <= mask_token_id < embed.shape[0] and 0 <= image_mask_token_id < embed.shape[0]:
                embed[image_mask_token_id].copy_(embed[mask_token_id])
                if logger is not None:
                    logger.info(
                        f"Initialized newly added image mask token id={image_mask_token_id} "
                        f"from text mask token id={mask_token_id}"
                    )

    if config.training.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if logger is not None:
            logger.info("Gradient checkpointing enabled")

    return model, tokenizer


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


def _apply_image_flow_warmup_freeze(model, config):
    freeze_for_refine = config.training.get("freeze_backbone_for_image_flow_refine", False)
    freeze_for_warmup = config.training.get("freeze_backbone_for_image_flow_warmup", False)
    if not (freeze_for_refine or freeze_for_warmup):
        return

    for param in model.parameters():
        param.requires_grad = False

    for param in model.image_flow_head.parameters():
        param.requires_grad = True
    for param in model.image_token_embedder.parameters():
        param.requires_grad = True
    for param in model.image_flow_condition_proj.parameters():
        param.requires_grad = True

    embed_weight = model.model.embed_tokens.weight
    embed_weight.requires_grad = True
    train_ids = torch.tensor(
        list(_special_token_ids(config).values()),
        device=embed_weight.device,
        dtype=torch.long,
    )

    def _mask_special_token_grads(grad):
        keep = torch.zeros((grad.shape[0], 1), device=grad.device, dtype=grad.dtype)
        keep[train_ids.to(grad.device)] = 1
        return grad * keep

    embed_weight.register_hook(_mask_special_token_grads)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    stage_name = "refine" if freeze_for_refine else "warmup"
    logger.info(
        f"Enabled frozen-Qwen image-flow {stage_name}: "
        f"trainable={trainable:,}/{total:,} params; "
        f"special_token_ids={_special_token_ids(config)}"
    )


def _migrate_image_flow_head_state(model, head_state, source):
    target_state = model.image_flow_head.state_dict()
    load_state = {}
    skipped = {}
    for name, value in head_state.items():
        if name not in target_state:
            skipped[name] = (tuple(value.shape), None)
            continue
        target_shape = tuple(target_state[name].shape)
        value_shape = tuple(value.shape)
        if value_shape == target_shape:
            load_state[name] = value
            continue
        skipped[name] = (value_shape, target_shape)

    missing, unexpected = model.image_flow_head.load_state_dict(load_state, strict=False)
    logger.info(
        f"Migrated image_flow_head from {source}: loaded={len(load_state)}, "
        f"missing={list(missing)}, unexpected={list(unexpected)}, skipped={skipped}"
    )


def _migrate_image_flow_condition_proj_state(model, projector_state, source):
    projector = getattr(model, "image_flow_condition_proj", None)
    if projector is None:
        logger.info(f"Skipping image_flow_condition_proj load from {source}: model has no projector")
        return
    target_state = projector.state_dict()
    if not target_state:
        logger.info(f"Skipping image_flow_condition_proj load from {source}: projector has no parameters")
        return

    load_state = {}
    skipped = {}
    for name, value in projector_state.items():
        if name not in target_state:
            skipped[name] = (tuple(value.shape), None)
            continue
        target_shape = tuple(target_state[name].shape)
        value_shape = tuple(value.shape)
        if value_shape == target_shape:
            load_state[name] = value
            continue
        skipped[name] = (value_shape, target_shape)

    missing, unexpected = projector.load_state_dict(load_state, strict=False)
    logger.info(
        f"Migrated image_flow_condition_proj from {source}: loaded={len(load_state)}, "
        f"missing={list(missing)}, unexpected={list(unexpected)}, skipped={skipped}"
    )


def _is_disabled_path(value):
    return value is None or (
        isinstance(value, str) and value.lower() in {"none", "null", "false", ""}
    )


def _reinitialize_image_modules(model, config):
    if not config.model.get("reinitialize_image_modules", False):
        return False

    reset_modules = []
    if hasattr(model, "image_flow_head") and hasattr(model.image_flow_head, "net"):
        model.image_flow_head.net.initialize_weights()
        reset_modules.append("image_flow_head")
    if hasattr(model, "image_token_embedder") and hasattr(model.image_token_embedder, "_reset_parameters"):
        model.image_token_embedder._reset_parameters()
        reset_modules.append("image_token_embedder")
    if hasattr(model, "_reset_image_flow_condition_proj"):
        model._reset_image_flow_condition_proj()
        reset_modules.append("image_flow_condition_proj")

    _log_info(
        "Reinitialized image modules after loading the base model: "
        f"{', '.join(reset_modules) if reset_modules else 'none'}"
    )
    return True


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
        projector_target = model.image_token_embedder.state_dict()
        projector_skipped = {}
        adapter_mask_token = None

        def _canonical_projector_key(name):
            if name == "diffusion_pos_embed":
                return "flow_pos_embed"
            return name

        def _maybe_add_projector_key(name, value):
            name = _canonical_projector_key(name)
            if name not in projector_target:
                projector_skipped[name] = (tuple(value.shape), None)
                return
            target_shape = tuple(projector_target[name].shape)
            if tuple(value.shape) != target_shape:
                projector_skipped[name] = (tuple(value.shape), target_shape)
                return
            projector_state[name] = value

        with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("diffloss."):
                    head_state[key[len("diffloss."):]] = f.get_tensor(key)
                elif key.startswith("image_flow_head."):
                    head_state[key[len("image_flow_head."):]] = f.get_tensor(key)
                elif key.startswith("image_flow_condition_proj."):
                    condition_proj_state[key[len("image_flow_condition_proj."):]] = f.get_tensor(key)
                elif key.startswith("model.image_flow_condition_proj."):
                    condition_proj_state[key[len("model.image_flow_condition_proj."):]] = f.get_tensor(key)
                elif key.startswith("model.image_token_embedder."):
                    name = key[len("model.image_token_embedder."):]
                    _maybe_add_projector_key(name, f.get_tensor(key))
                elif key in {"z_proj.weight", "z_proj.bias", "z_proj_ln.weight", "z_proj_ln.bias"}:
                    _maybe_add_projector_key(key, f.get_tensor(key))
                elif key == "encoder_pos_embed_learned":
                    pos = f.get_tensor(key).squeeze(0)
                    _maybe_add_projector_key("image_pos_embed", pos[-model.image_token_embedder.image_tokens_per_img:])
                elif key == "diffusion_pos_embed_learned":
                    _maybe_add_projector_key("flow_pos_embed", f.get_tensor(key).squeeze(0))
                elif key == "mask_token":
                    adapter_mask_token = f.get_tensor(key).reshape(-1)
        _migrate_image_flow_head_state(model, head_state, adapter_path)
        if condition_proj_state:
            _migrate_image_flow_condition_proj_state(model, condition_proj_state, adapter_path)
        if projector_state:
            missing, unexpected = model.image_token_embedder.load_state_dict(projector_state, strict=False)
            logger.info(
                f"Loaded image_token_embedder from {adapter_path}: "
                f"keys={len(projector_state)}, missing={missing}, unexpected={unexpected}, "
                f"skipped={projector_skipped}"
            )
        else:
            logger.warning(
                f"No image_token_embedder keys were loaded from {adapter_path}; skipped={projector_skipped}"
            )
        image_mask_token_id = config.model.get("image_mask_token_id", None)
        if adapter_mask_token is not None and image_mask_token_id is not None:
            with torch.no_grad():
                embed = model.model.embed_tokens.weight
                if adapter_mask_token.numel() != embed.shape[1]:
                    logger.warning(
                        f"Skipping adapter mask_token load: shape={tuple(adapter_mask_token.shape)} "
                        f"does not match embedding dim={embed.shape[1]}"
                    )
                else:
                    embed[int(image_mask_token_id)].copy_(
                        adapter_mask_token.to(device=embed.device, dtype=embed.dtype)
                    )
                    logger.info(
                        f"Loaded adapter mask_token into image_mask_token_id={int(image_mask_token_id)}"
                    )
        return

    state = torch.load(adapter_path, map_location="cpu")
    if "image_flow_head" in state:
        _migrate_image_flow_head_state(model, state["image_flow_head"], adapter_path)
    else:
        flat_head = {}
        for key, value in (state.items() if isinstance(state, dict) else []):
            if key.startswith("image_flow_head."):
                flat_head[key[len("image_flow_head."):]] = value
        if flat_head:
            _migrate_image_flow_head_state(model, flat_head, adapter_path)
    if "image_flow_condition_proj" in state:
        _migrate_image_flow_condition_proj_state(model, state["image_flow_condition_proj"], adapter_path)
    else:
        flat_condition_proj = {}
        for key, value in (state.items() if isinstance(state, dict) else []):
            if key.startswith("image_flow_condition_proj."):
                flat_condition_proj[key[len("image_flow_condition_proj."):]] = value
        if flat_condition_proj:
            _migrate_image_flow_condition_proj_state(model, flat_condition_proj, adapter_path)
    if "image_token_embedder" in state:
        missing, unexpected = model.image_token_embedder.load_state_dict(
            state["image_token_embedder"], strict=False
        )
        logger.info(
            f"Loaded adapter image_token_embedder from {adapter_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if "special_token_embeddings" in state:
        token_ids = _special_token_ids(config)
        with torch.no_grad():
            embed = model.model.embed_tokens.weight
            for name, token_id in token_ids.items():
                if name not in state["special_token_embeddings"]:
                    continue
                value = state["special_token_embeddings"][name].to(
                    device=embed.device,
                    dtype=embed.dtype,
                )
                embed[token_id].copy_(value)
        logger.info(f"Loaded adapter special token embeddings from {adapter_path}")


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
    path = Path(config.experiment.output_dir) / f"image_flow_adapter-{global_step}.pt"
    torch.save(state, path)
    logger.info(f"Saved image-flow adapter to {path}")


def _ema_enabled(config) -> bool:
    return bool(config.training.get("use_ema", False))


def _ema_decay(config) -> float:
    decay = float(config.training.get("ema_decay", 0.9999))
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
    return decay


def _create_ema_model(model, config):
    if not _ema_enabled(config):
        return None
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _sync_ema_model(ema_model, model, accelerator) -> None:
    source_state = accelerator.unwrap_model(model).state_dict()
    ema_state = ema_model.state_dict()
    if set(source_state.keys()) != set(ema_state.keys()):
        missing = sorted(set(ema_state.keys()) - set(source_state.keys()))
        unexpected = sorted(set(source_state.keys()) - set(ema_state.keys()))
        raise RuntimeError(f"EMA state mismatch: missing={missing}, unexpected={unexpected}")

    for name, ema_value in ema_state.items():
        source_value = source_state[name].detach().to(
            device=ema_value.device,
            dtype=ema_value.dtype,
            non_blocking=True,
        )
        ema_value.copy_(source_value)


@torch.no_grad()
def _update_ema_model(ema_model, model, accelerator, decay: float) -> None:
    source_state = accelerator.unwrap_model(model).state_dict()
    ema_state = ema_model.state_dict()
    for name, ema_value in ema_state.items():
        source_value = source_state[name].detach().to(
            device=ema_value.device,
            dtype=ema_value.dtype,
            non_blocking=True,
        )
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(source_value)


def _ema_state_path(config, global_step) -> Path:
    output_dir = Path(config.experiment.output_dir)
    if isinstance(global_step, int):
        return output_dir / f"checkpoint-{global_step}" / "ema_state.pt"
    return output_dir / f"ema_state-{global_step}.pt"


def _save_ema_state(ema_model, config, accelerator, global_step) -> None:
    if ema_model is None or not accelerator.is_main_process:
        return
    path = _ema_state_path(config, global_step)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "global_step": global_step,
        "decay": _ema_decay(config),
        "state_dict": {k: v.detach().cpu() for k, v in ema_model.state_dict().items()},
    }
    torch.save(state, path)
    logger.info(f"Saved EMA state to {path}")


def _load_ema_state_if_available(ema_model, checkpoint_dir: Path) -> bool:
    path = Path(checkpoint_dir) / "ema_state.pt"
    if not path.exists():
        return False
    state = torch.load(path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    ema_model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded EMA state from {path}")
    return True


def _save_ema_hf_model(ema_model, tokenizer, config, accelerator, global_step) -> None:
    if ema_model is None or not bool(config.training.get("ema_save_hf_model", True)):
        return
    if not accelerator.is_main_process:
        return
    save_path = Path(config.experiment.output_dir) / f"hf_model-{global_step}-ema"
    ema_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved EMA HF model to {save_path}")


def _unwrap_omnicorpus_dataset(dataset):
    ds = dataset
    if hasattr(ds, "set_epoch"):
        return ds
    while hasattr(ds, "dataset"):
        if hasattr(ds, "set_epoch") or hasattr(ds, "_packs") or ds.__class__.__name__ == "OmniCorpusPackedDataset":
            return ds
        ds = ds.dataset
    return ds


def main():
    #########################
    #      SETUP Config     #
    #########################
    config = get_config()
        
    total_batch_size_per_gpu = config.training.batch_size
    
    config.experiment.output_dir = os.path.join(config.experiment.output_dir, config.experiment.project)

    #########################
    # SETUP Accelerator     #
    #########################
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    assert num_processes != -1
    print(f"Number of processes: {num_processes}")
    print(f"Total batch size: {config.training.total_batch_size}")
    print(f"Batch size per GPU: {total_batch_size_per_gpu}")
    print(f"Gradient accumulation steps: {(config.training.total_batch_size // config.training.batch_size) // num_processes}")
    accelerator = Accelerator(
        gradient_accumulation_steps=((config.training.total_batch_size // config.training.batch_size) // num_processes),
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        step_scheduler_with_optimizer=config.training.step_scheduler_with_optimizer,
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
            "mode": os.environ.get("WANDB_MODE", "online"),
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
    model, tokenizer = load_model_tokenizer(config=config, logger=logger)

    reinitialized_image_modules = _reinitialize_image_modules(model, config)
    flow_adapter = config.model.get("pretrained_image_flow_adapter", None)
    if reinitialized_image_modules and not _is_disabled_path(flow_adapter):
        logger.warning(
            "Skipping pretrained_image_flow_adapter because reinitialize_image_modules=true: "
            f"{flow_adapter}"
        )
    else:
        _load_image_flow_adapter(model, flow_adapter, config)
    _apply_image_flow_warmup_freeze(model, config)

    if config.training.get("use_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    ema_model = _create_ema_model(model, config)
    ema_decay_value = _ema_decay(config) if ema_model is not None else None
    ema_update_after_step = int(config.training.get("ema_update_after_step", 0))
    if ema_model is not None:
        logger.info(
            "EMA enabled: "
            f"decay={ema_decay_value:g}, update_after_step={ema_update_after_step}, "
            f"validate={bool(config.training.get('ema_validate', True))}, "
            f"save_adapter={bool(config.training.get('ema_save_adapter', True))}, "
            f"save_hf_model={bool(config.training.get('ema_save_hf_model', True))}"
        )
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            logger.warning(
                "EMA keeps an unsharded shadow model. This may require extra memory with DeepSpeed."
            )

    selfless_sampler = SelflessSampler(mask_token_id=model.config.mask_token_id, config=config)
    
    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params

    # Use lower LR for pretrained backbone and higher LR for continuous-image
    # modules. Keep flow head normalization/projection parameters out of weight decay.
    base_lr = float(optimizer_config.learning_rate)
    backbone_lr = float(optimizer_config.get("backbone_learning_rate", base_lr))
    flow_lr = float(optimizer_config.get("flow_learning_rate", base_lr))
    projector_lr = float(optimizer_config.get("projector_learning_rate", flow_lr))
    special_token_lr = float(optimizer_config.get("special_token_learning_rate", projector_lr))
    no_decay = [
        "bias",
        "layer_norm.weight",
        "layernorm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "norm.weight",
        "embed_tokens.weight",
        "lm_head.weight",
    ]

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
        is_image_module = (
            name.startswith("image_flow_head.")
            or "image_token_embedder" in name
            or name.startswith("image_flow_condition_proj.")
        )
        weight_decay = 0.0 if is_image_module or any(nd in name for nd in no_decay) else optimizer_config.weight_decay
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
        f"weight_decay={optimizer_config.weight_decay:g}"
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

    ##################################
    #         DATALOADER             #
    ##################################
    logger.info("Creating dataloaders and lr_scheduler")

    seq_len = config.dataset.preprocessing.max_seq_length
    
    train_dataloader, val_dataloader = get_dataloaders(config, tokenizer)

    ##################################
    #       Prepare accelerator     #
    ##################################
    logger.info("Preparing model, optimizer and dataloaders")

    # Store ref to underlying packed dataset for epoch-level reshuffling/repacking.
    ds = _unwrap_omnicorpus_dataset(train_dataloader.dataset)
    _is_multimodal_ds = hasattr(ds, 'set_epoch')

    if hasattr(train_dataloader, "prepare_with_accelerator"):
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        train_dataloader = train_dataloader.prepare_with_accelerator(accelerator)
        val_dataloader = val_dataloader.prepare_with_accelerator(accelerator)
    else:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)
    if ema_model is not None:
        ema_model.to(accelerator.device)

    ##################################
    #       MODEL RESUME         #
    ##################################
    global_step = 0
    resume_step = 0
    resume_checkpoint_dir = None

    if config.experiment.resume_from_checkpoint is not None:
        candidate_path = Path(config.experiment.resume_from_checkpoint)
        if candidate_path.exists():
            resume_checkpoint_dir = candidate_path
        else:
            logger.warning(f"Specified checkpoint not found: {candidate_path}")

    if resume_checkpoint_dir and resume_checkpoint_dir.exists():
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint_dir}")
        
        # 加载模型权重、优化器状态、RNG 状态
        accelerator.load_state(resume_checkpoint_dir)
        
        metadata_file = resume_checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            resume_step = metadata.get("global_step", 0)
        else:
            logger.error(f"Error loading metadata from {metadata_file}")
        
        global_step = resume_step
        logger.info(f"Resumed at global_step={global_step}")

    else:
        logger.warning("No valid checkpoint found or specified, starting fresh training.")
        global_step = 0
        resume_step = 0

    if ema_model is not None:
        if resume_checkpoint_dir and _load_ema_state_if_available(ema_model, resume_checkpoint_dir):
            ema_model.to(accelerator.device)
        else:
            _sync_ema_model(ema_model, model, accelerator)
            logger.info("Initialized EMA weights from the current training model.")
        ema_model.eval()

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
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    batches_to_skip = 0
    resume_epoch = 0
    initial_train_dataloader = train_dataloader
    if resume_step > 0:
        raw_batches_to_skip = resume_step * accelerator.gradient_accumulation_steps
        try:
            dataloader_len = len(train_dataloader)
        except TypeError:
            dataloader_len = 0
        if dataloader_len > 0:
            resume_epoch = raw_batches_to_skip // dataloader_len
            batches_to_skip = raw_batches_to_skip % dataloader_len
            logger.info(
                f"Resuming from step {resume_step}: dataloader_len={dataloader_len}, "
                f"resume_epoch={resume_epoch}, skipping {batches_to_skip} batches in the first resumed epoch."
            )
            if _is_multimodal_ds:
                ds.set_epoch(resume_epoch)
        else:
            batches_to_skip = raw_batches_to_skip
            logger.info(f"Resuming from step {resume_step}, skipping {batches_to_skip} batches...")
        if batches_to_skip > 0:
            initial_train_dataloader = accelerator.skip_first_batches(train_dataloader, batches_to_skip)

    model.train()
    if ema_model is not None:
        ema_model.eval()

    train_iter = iter(initial_train_dataloader)

    # Accumulators for per-modality loss across gradient-accumulation micro-batches.
    # Reset after each optimizer step (sync_gradients=True).
    acc_loss = torch.tensor(0.0, device=accelerator.device)
    acc_text_loss = torch.tensor(0.0, device=accelerator.device)
    acc_image_loss = torch.tensor(0.0, device=accelerator.device)
    acc_text_batches = torch.tensor(0.0, device=accelerator.device)
    acc_image_batches = torch.tensor(0.0, device=accelerator.device)
    acc_flow_stats = {}
    acc_flow_stat_batches = torch.tensor(0.0, device=accelerator.device)

    epoch = resume_epoch
    while global_step < config.training.max_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if _is_multimodal_ds:
                ds.set_epoch(epoch)
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        
        # *-------*-------*-------*-------*-------*-------*
        # Data Processing
        # *-------*-------*-------*-------*-------*-------*
        is_multimodal = "token_types" in batch
        t_1 = 0.0

        if is_multimodal:
            input_ids = batch["input_ids"].contiguous().to(accelerator.device)  # [B, L] — no shift for selfless
            token_types = batch["token_types"].to(accelerator.device)  # [B, L]
            sigma = batch["sigma"].to(accelerator.device)  # [B, L], pre-computed by dataloader
            labels = batch["labels"].to(accelerator.device)  # [B, L], pre-computed by dataloader
            image_latents = batch.get("image_latents", None)
            if image_latents is not None:
                image_latents = image_latents.to(accelerator.device)
            pack_stats = batch.get("pack_stats", None)
            if pack_stats is not None:
                pack_stats = pack_stats.to(accelerator.device)
            B, L = input_ids.shape

            image_uncond_rows = None
            image_uncond_prob = float(config.model.get("image_uncond_prob", 0.0))
            if image_uncond_prob > 0.0:
                has_image = (token_types == 1).any(dim=1)
                sampled_rows = (
                    torch.rand(B, device=accelerator.device) < image_uncond_prob
                ) & has_image
                if sampled_rows.any():
                    image_uncond_rows = sampled_rows

            selfless_attention_mask = get_selfless_mask(
                sigma=sigma,
                seq_len=L,
                device=accelerator.device,
                input_ids=input_ids,
                token_types=token_types,
                boi_token_id=int(config.model.boi_token_id),
                image_uncond_rows=image_uncond_rows,
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
                if pack_stats is not None:
                    valid_tokens, image_tokens, padding_tokens, packed_len = pack_stats.tolist()
                    logger.info(
                        f"pack stats: valid={valid_tokens}, image={image_tokens}, "
                        f"padding={padding_tokens}, L={packed_len}, "
                        f"padding ratio={padding_tokens / max(1, B * L):.3f}"
                    )

        else:
            # Legacy text-only path
            text_ids = batch["input_ids"][:, :-1].contiguous()
            token_types = None
            B, L = text_ids.shape

            t_sample, v_sample = selfless_sampler.sample_v(text_ids)
            t_1 = t_sample[0, 0].item()

            v_sample = v_sample.to(accelerator.device)
            selfless_attention_mask = get_selfless_mask(
                sigma=v_sample, seq_len=L, device=accelerator.device
            )
            del t_sample

            if global_step == 0 and accelerator.is_main_process and not hasattr(main, '_logged_first_batch'):
                main._logged_first_batch = True
                logger.info(f"Input ids shape: {text_ids.shape}, text-only mode")
            input_ids = text_ids

        # *-------*-------*-------*-------*-------*-------*
        # Forward & Backward
        # *-------*-------*-------*-------*-------*-------*
        with accelerator.accumulate(model):
            forward_kwargs = {
                "X0_input_ids": input_ids,
                "labels": labels if is_multimodal else input_ids,
                "attention_mask": selfless_attention_mask,
            }
            if token_types is not None:
                forward_kwargs["token_types"] = token_types
                forward_kwargs["flow_sigma"] = sigma
            if is_multimodal and image_latents is not None:
                forward_kwargs["image_latents"] = image_latents

            model_output = model(**forward_kwargs)
            loss = model_output.loss

            # Track per-modality loss across micro-batches
            per_mod = getattr(model_output, "per_modality_loss", None)
            if per_mod is not None:
                t = per_mod["text_loss"]
                i = per_mod["image_loss"]
                has_text = (((token_types == 0) | (token_types == 2)) & (labels != -100)).any()
                has_image = (token_types == 1).any()
                if has_text:
                    acc_text_loss += t.detach() if isinstance(t, torch.Tensor) else 0.0
                    acc_text_batches += 1
                if has_image:
                    acc_image_loss += i.detach() if isinstance(i, torch.Tensor) else 0.0
                    acc_image_batches += 1
                    flow_stats = getattr(model_output, "flow_debug_stats", None)
                    if flow_stats:
                        for key, value in flow_stats.items():
                            acc_flow_stats[key] = acc_flow_stats.get(
                                key, torch.tensor(0.0, device=accelerator.device)
                            ) + value.detach().to(accelerator.device)
                        acc_flow_stat_batches += 1
            acc_loss += loss.detach()

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if config.training.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if ema_model is not None and (global_step + 1) >= ema_update_after_step:
                    _update_ema_model(ema_model, model, accelerator, ema_decay_value)

                # 记录梯度范数 (可选)
                if (global_step + 1) % config.experiment.log_grad_norm_every == 0 and accelerator.is_main_process:
                    log_grad_norm(model, accelerator, global_step + 1)

        # *-------*-------*-------*-------*-------*-------*
        # Logging & Saving & Validation
        # *-------*-------*-------*-------*-------*-------*
        if accelerator.sync_gradients:
            global_step += 1
            
            batch_time_m.update(time.time() - end)
            end = time.time()

            # Logging
            if global_step % config.experiment.log_every == 0:
                grad_accum = accelerator.gradient_accumulation_steps
                # Average loss across all micro-batches and all ranks
                avg_loss_per_step = acc_loss / grad_accum
                global_avg_loss = accelerator.reduce(avg_loss_per_step, reduction="mean")

                samples_per_second_per_gpu = (
                        accelerator.gradient_accumulation_steps * config.training.batch_size / batch_time_m.val
                )

                logs = {
                    "t_1": t_1,
                    "step_loss": global_avg_loss.item(),
                    "train_ppl": math.exp(global_avg_loss.item()),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "batch_time": batch_time_m.val,
                }
                if ema_model is not None:
                    logs["ema/decay"] = ema_decay_value
                if is_multimodal and pack_stats is not None:
                    valid_tokens, image_tokens, padding_tokens, packed_len = pack_stats.tolist()
                    logs["pack/valid_tokens"] = valid_tokens
                    logs["pack/image_tokens"] = image_tokens
                    logs["pack/padding_tokens"] = padding_tokens
                    logs["pack/padding_ratio"] = padding_tokens / max(1, config.training.batch_size * packed_len)
                    logs["pack/seq_len"] = packed_len

                # Per-modality loss (accumulated across all micro-batches)
                if is_multimodal:
                    avg_text_loss = acc_text_loss / acc_text_batches.clamp_min(1.0)
                    global_text_loss = accelerator.reduce(avg_text_loss, reduction="mean")
                    logs["train/loss_text"] = global_text_loss.item()
                    logs["train/ppl_text"] = math.exp(min(global_text_loss.item(), 100))

                    avg_image_loss = acc_image_loss / acc_image_batches.clamp_min(1.0)
                    global_image_loss = accelerator.reduce(avg_image_loss, reduction="mean")
                    logs["train/loss_image_flow"] = global_image_loss.item()
                    if acc_flow_stats:
                        flow_stat_count = acc_flow_stat_batches.clamp_min(1.0)
                        global_flow_stats = {}
                        for key, value in acc_flow_stats.items():
                            stat = accelerator.reduce(value / flow_stat_count, reduction="mean")
                            global_flow_stats[key] = stat.item()
                            logs[f"train/{key}"] = stat.item()

                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    msg = (
                        f"Step: {global_step} | "
                        f"Loss: {global_avg_loss.item():0.4f}"
                    )
                    if is_multimodal:
                        msg += f" | Text: {global_text_loss.item():0.4f}"
                        msg += f" | Image: {global_image_loss.item():0.4f}"
                        if acc_flow_stats:
                            msg += (
                                f" | FlowMSE: {global_flow_stats.get('flow/v_mse', 0.0):0.4f}"
                                f" | FlowPredVRMS: {global_flow_stats.get('flow/v_pred_rms', 0.0):0.4f}"
                            )
                    msg += (
                        f" | LR: {lr_scheduler.get_last_lr()[0]:0.6f} | "
                        f"Sec/Iter: {batch_time_m.val:0.4f}"
                    )
                    logger.info(msg)

                batch_time_m.reset()
                data_time_m.reset()

            # Checkpointing
            if global_step % config.experiment.save_every == 0:
                save_checkpoint(model, config, accelerator, global_step)
                _save_ema_state(ema_model, config, accelerator, global_step)
                adapter_model = (
                    ema_model
                    if ema_model is not None and bool(config.training.get("ema_save_adapter", True))
                    else model
                )
                _save_image_flow_adapter(adapter_model, config, accelerator, global_step)
            
            if global_step % config.experiment.save_hfmodel_every == 0:
                save_hf_model(model, tokenizer, config, accelerator, global_step)
                _save_ema_hf_model(ema_model, tokenizer, config, accelerator, global_step)
                
            # Validation
            if global_step % config.experiment.val_every == 0:
                eval_model = (
                    ema_model
                    if ema_model is not None and bool(config.training.get("ema_validate", True))
                    else model
                )
                validate(eval_model, val_dataloader, selfless_sampler, accelerator, global_step, config)
                # if accelerator.is_main_process:
                #     pre_text, label_text = get_text(logits_pred=logits_pred[0], label_ids=label_ids[0], tokenizer=tokenizer)
                #     accelerator.print(f"pre_text: {pre_text}")
                #     accelerator.print(f"label_text: {label_text}")
                
                model.train()
                if ema_model is not None:
                    ema_model.eval()

            # Reset per-step accumulators for the next optimizer step
            acc_loss.zero_()
            acc_text_loss.zero_()
            acc_image_loss.zero_()
            acc_text_batches.zero_()
            acc_image_batches.zero_()
            acc_flow_stats.clear()
            acc_flow_stat_batches.zero_()
            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_hf_model(model, tokenizer, config, accelerator, "final")
    _save_ema_hf_model(ema_model, tokenizer, config, accelerator, "final")
    _save_ema_state(ema_model, config, accelerator, "final")
    adapter_model = (
        ema_model
        if ema_model is not None and bool(config.training.get("ema_save_adapter", True))
        else model
    )
    _save_image_flow_adapter(adapter_model, config, accelerator, "final")
    accelerator.end_training()


@torch.no_grad()
def validate(model, val_dataloader, selfless_sampler, accelerator, global_step, config=None):
    model.eval()  # DeepSpeed requires explicit eval mode for no_grad forward
    ds = _unwrap_omnicorpus_dataset(val_dataloader.dataset)
    is_multimodal = (
        hasattr(ds, "set_epoch")
        or hasattr(ds, "_packs")
        or ds.__class__.__name__ in {"OmniCorpusPackedDataset", "ImageNetFlowCacheDataset"}
    )

    try:
        if is_multimodal:
            _validate_multimodal(model, val_dataloader, accelerator, global_step, config)
        else:
            _validate_text_only(model, val_dataloader, selfless_sampler, accelerator, global_step)
    finally:
        model.train()


@torch.no_grad()
def _validate_text_only(model, val_dataloader, selfless_sampler, accelerator, global_step):
    local_total_loss_selfless = torch.tensor(0.0, device=accelerator.device)
    local_total_loss_ar = torch.tensor(0.0, device=accelerator.device)
    local_total_count = torch.tensor(0.0, device=accelerator.device)

    for batch in val_dataloader:
        text_ids = batch["input_ids"][:, :-1].contiguous()
        current_batch_size = text_ids.size(0)

        _, v_sample = selfless_sampler.sample_v(text_ids)
        B, L = text_ids.shape
        v_sample = v_sample.to(accelerator.device)

        selfless_attention_mask = get_selfless_mask(sigma=v_sample, seq_len=L, device=accelerator.device)
        loss_selfless = model(
            X0_input_ids=text_ids, labels=text_ids,
            attention_mask=selfless_attention_mask,
            calculate_likelihood=True,
        ).loss
        local_total_loss_selfless += loss_selfless.detach() * current_batch_size

        AR_mask = get_selfless_mask(
            sigma=torch.arange(L, device=accelerator.device).unsqueeze(0).expand(B, L),
            seq_len=L, device=accelerator.device
        )
        loss_ar = model(
            X0_input_ids=text_ids, labels=text_ids,
            attention_mask=AR_mask,
            calculate_likelihood=True,
        ).loss
        local_total_loss_ar += loss_ar.detach() * current_batch_size
        local_total_count += current_batch_size

    # Ensure all ranks have the same count before reduce
    # DistributedSampler may give different ranks different numbers of batches.
    # accelerator.reduce on the count handles rank-level variation — each rank
    # participates with its own count, and the sum across ranks gives the total.
    # The model forward calls must be synchronized — each rank must call forward
    # the same number of times. With DistributedSampler using pad/drop_last,
    # the dataloader guarantees equal iterations per rank.
    global_total_count = accelerator.reduce(local_total_count, reduction="sum")
    global_total_loss_selfless = accelerator.reduce(local_total_loss_selfless, reduction="sum")
    avg_loss_selfless = (global_total_loss_selfless / global_total_count).item()
    ppl_selfless = math.exp(avg_loss_selfless)
    global_total_loss_ar = accelerator.reduce(local_total_loss_ar, reduction="sum")
    avg_loss_ar = (global_total_loss_ar / global_total_count).item()
    ppl_ar = math.exp(avg_loss_ar)

    if accelerator.is_main_process:
        logs = {
            "val/loss_selfless": avg_loss_selfless,
            "val/ppl_selfless": ppl_selfless,
            "val/loss_ar": avg_loss_ar,
            "val/ppl_ar": ppl_ar,
        }
        accelerator.log(logs, step=global_step)
        logger.info(
            f"[Validation] Step {global_step + 1} | "
            f"Selfless Loss: {avg_loss_selfless:.4f} (PPL: {ppl_selfless:.2f}) | "
            f"AR Loss: {avg_loss_ar:.4f} (PPL: {ppl_ar:.2f})"
        )

    return avg_loss_selfless, ppl_selfless


@torch.no_grad()
def _load_vae_decoder(config, accelerator):
    global _VAE_CACHE
    if _VAE_CACHE is not None:
        return _VAE_CACHE

    vae_path = Path(config.experiment.get("validation_vae_path", "public/vae/mar-kl16/kl16.ckpt"))
    if not vae_path.exists():
        logger.warning(f"Skipping validation image decode; missing VAE checkpoint: {vae_path}")
        return None

    vae_module_root = Path(
        config.experiment.get(
            "validation_vae_module_root",
            config.experiment.get("validation_mar_root", "/inspire/hdd/global_user/wanjiaxin-253108030048/code/mar"),
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
    vae_dtype_name = str(config.experiment.get("validation_vae_dtype", "fp32")).lower()
    dtype = torch.float16 if vae_dtype_name in {"fp16", "float16", "half"} and accelerator.device.type == "cuda" else torch.float32
    vae = vae.to(device=accelerator.device, dtype=dtype).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    _VAE_CACHE = vae
    return vae


def _image_spans(token_types: torch.Tensor, image_tokens_per_img: int):
    spans = []
    bsz, seq_len = token_types.shape
    for b in range(bsz):
        pos = 0
        while pos < seq_len:
            if token_types[b, pos].item() != 1:
                pos += 1
                continue
            start = pos
            while pos < seq_len and token_types[b, pos].item() == 1:
                pos += 1
            end = pos
            if end - start == image_tokens_per_img:
                spans.append((b, start, end))
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


@torch.no_grad()
def _save_validation_flow_images(
    model,
    output,
    input_ids,
    token_types,
    sigma,
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
    spans = _image_spans(token_types, image_tokens_per_img)

    gather_context = nullcontext()
    flow_head_params = list(unwrapped.image_flow_head.parameters())
    if any(hasattr(param, "ds_id") for param in flow_head_params):
        try:
            import deepspeed
            gather_context = deepspeed.zero.GatheredParameters(
                flow_head_params,
                modifier_rank=0,
            )
        except Exception:
            pass

    with gather_context:
        if not accelerator.is_main_process:
            return
        if not spans:
            logger.warning("Skipping validation image decode; no complete image span in validation batch.")
            return

        vae = _load_vae_decoder(config, accelerator)
        if vae is None:
            return

        sample_count = min(int(config.experiment.get("validation_image_samples", 4)), len(spans))
        flow_temperature = float(config.experiment.get("validation_flow_temperature", 1.0))
        flow_cfg = float(config.experiment.get("validation_flow_cfg", 1.0))
        flow_cfg_schedule = str(config.experiment.get("validation_flow_cfg_schedule", "linear"))
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
        uncond_hidden_states = None
        if flow_cfg != 1.0:
            image_uncond_rows = torch.ones(
                input_ids.shape[0],
                device=accelerator.device,
                dtype=torch.bool,
            )
            attention_mask = get_selfless_mask(
                sigma=sigma,
                seq_len=input_ids.shape[1],
                device=accelerator.device,
                input_ids=input_ids,
                token_types=token_types,
                boi_token_id=int(config.model.boi_token_id),
                image_uncond_rows=image_uncond_rows,
            )
            uncond_output = unwrapped(
                X0_input_ids=input_ids,
                attention_mask=attention_mask,
                token_types=token_types,
                image_latents=image_latents,
                calculate_likelihood=True,
                return_logits=False,
            )
            uncond_hidden_states = uncond_output.last_hidden_state
        pred_latents = []
        probe_x0_latents = {time_value: [] for time_value in probe_times}
        probe_v_mse = {time_value: [] for time_value in probe_times}
        probe_x0_mse = {time_value: [] for time_value in probe_times}
        target_latents = []
        selected_spans = spans[:sample_count]

        def _sequence_mixer_context(
            target: torch.Tensor,
            span_sigma: torch.Tensor,
            local_positions: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            sigma_row = span_sigma.to(device=target.device, dtype=torch.float32).unsqueeze(0)
            positions = local_positions.to(device=target.device, dtype=torch.long).unsqueeze(0)
            return {
                "context_latents": target.unsqueeze(0),
                "context_mask": sigma_row.unsqueeze(1) < sigma_row.unsqueeze(2),
                "query_positions": positions,
                "context_positions": positions,
            }

        def _flat_query_mixer_context(
            target: torch.Tensor,
            span_sigma: torch.Tensor,
            local_positions: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            query_count = target.shape[0]
            sigma_values = span_sigma.to(device=target.device, dtype=torch.float32)
            positions = local_positions.to(device=target.device, dtype=torch.long)
            return {
                "context_latents": target.unsqueeze(0).expand(query_count, -1, -1).contiguous(),
                "context_mask": (sigma_values.unsqueeze(0) < sigma_values.unsqueeze(1)).unsqueeze(1),
                "query_positions": positions,
                "context_positions": positions.unsqueeze(0).expand(query_count, -1).contiguous(),
            }

        for b, start, end in selected_spans:
            local_positions = torch.arange(
                end - start,
                device=accelerator.device,
                dtype=torch.long,
            )
            target = image_latents[b, start:end].to(device=accelerator.device)
            span_sigma = sigma[b, start:end].to(device=accelerator.device, dtype=torch.float32)
            z = unwrapped._prepare_image_flow_condition(
                hidden_states[b, start:end].to(device=accelerator.device),
                local_positions,
            )
            z_uncond = None
            if uncond_hidden_states is not None:
                z_uncond = unwrapped._prepare_image_flow_condition(
                    uncond_hidden_states[b, start:end].to(device=accelerator.device),
                    local_positions,
                )
            pred = unwrapped.sample_image_flow_with_cfg(
                z,
                z_uncond=z_uncond,
                temperature=flow_temperature,
                cfg=flow_cfg,
                solver=flow_solver,
                **_flat_query_mixer_context(target, span_sigma, local_positions),
            )
            target = target.to(dtype=pred.dtype)
            sequence_context = _sequence_mixer_context(target, span_sigma, local_positions)
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
                legacy_strategy = config.experiment.get("validation_single_stream_order_strategy", None)
                strategies = [legacy_strategy] if legacy_strategy else default_strategies
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
        vae_dtype = next(vae.parameters()).dtype
        decoded_pred = vae.decode(raw_pred_latents.to(dtype=vae_dtype) / scaling_factor).float().clamp(-1, 1)
        decoded_probe_x0 = {
            time_value: vae.decode(latents.to(dtype=vae_dtype) / scaling_factor).float().clamp(-1, 1)
            for time_value, latents in raw_probe_x0_latents.items()
        }
        decoded_target = vae.decode(raw_target_latents.to(dtype=vae_dtype) / scaling_factor).float().clamp(-1, 1)

        from torchvision.utils import make_grid, save_image

        image_dir = Path(config.experiment.output_dir) / "validation_flow_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        wandb_images = {}
        save_debug_images = bool(config.experiment.get("validation_save_debug_images", False))
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
                decoded_single_stream = vae.decode(
                    single_stream_latents.to(dtype=vae_dtype) / scaling_factor
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
                            order_grid = make_grid(_heatmap_images(order_map), nrow=sample_count)
                            order_path = image_dir / f"step-{global_step:08d}-single_stream_order_{trace_strategy}.png"
                            save_image(order_grid, order_path)
                            wandb_images[f"{log_prefix}/debug/generation_order"] = order_path
                    if isinstance(step_map, torch.Tensor):
                        logs[f"{log_prefix}/generation_step_max"] = step_map.float().max().item()
                        if save_debug_images:
                            step_grid = make_grid(_heatmap_images(step_map), nrow=sample_count)
                            step_path = image_dir / f"step-{global_step:08d}-single_stream_steps_{trace_strategy}.png"
                            save_image(step_grid, step_path)
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
        accelerator.log(logs, step=global_step)
        _log_wandb_validation_images(accelerator, wandb_images, global_step)
        logger.info(f"Saved validation flow images to {image_dir}")


@torch.no_grad()
def _validate_multimodal(model, val_dataloader, accelerator, global_step, config=None):
    local_weighted_loss = torch.tensor(0.0, device=accelerator.device)
    local_weighted_text = torch.tensor(0.0, device=accelerator.device)
    local_weighted_image = torch.tensor(0.0, device=accelerator.device)
    local_total_tokens = torch.tensor(0.0, device=accelerator.device)
    local_text_tokens = torch.tensor(0.0, device=accelerator.device)
    local_image_tokens = torch.tensor(0.0, device=accelerator.device)
    saved_validation_images = False

    for batch in val_dataloader:
        input_ids = batch["input_ids"].contiguous().to(accelerator.device)
        token_types = batch["token_types"].to(accelerator.device)
        sigma = batch["sigma"].to(accelerator.device)
        labels = batch["labels"].to(accelerator.device)
        image_latents = batch.get("image_latents", None)
        if image_latents is not None:
            image_latents = image_latents.to(accelerator.device)
        B, L = input_ids.shape

        # Count valid (non-ignored) tokens per modality for proper weighting.
        # model.loss is F.cross_entropy(reduction='mean'), so we need to
        # multiply by the number of valid tokens to recover the sum, then
        # divide by total valid tokens across all batches.
        valid_mask = labels != -100
        n_valid = valid_mask.sum().float()
        text_mask = ((token_types == 0) | (token_types == 2)) & valid_mask
        image_mask = token_types == 1

        # Sigma and labels pre-computed by dataloader
        selfless_attention_mask = get_selfless_mask(
            sigma=sigma, seq_len=L, device=accelerator.device
        )
        output = model(
            X0_input_ids=input_ids, labels=labels,
            attention_mask=selfless_attention_mask,
            token_types=token_types,
            image_latents=image_latents,
            flow_sigma=sigma,
            calculate_likelihood=True,
        )
        if not saved_validation_images and image_latents is not None and image_mask.any():
            _save_validation_flow_images(
                model=model,
                output=output,
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                image_latents=image_latents,
                accelerator=accelerator,
                global_step=global_step,
                config=config,
            )
            saved_validation_images = True
        if hasattr(output, "per_modality_loss"):
            n_text = text_mask.sum().float()
            n_image = image_mask.sum().float()
            local_weighted_text += output.per_modality_loss["text_loss"] * n_text
            local_weighted_image += output.per_modality_loss["image_loss"] * n_image
            local_text_tokens += n_text
            local_image_tokens += n_image
        else:
            local_weighted_loss += output.loss.detach() * n_valid
            local_total_tokens += n_valid

    # Reduce across ranks: sum of weighted losses and token counts
    global_total_tokens = accelerator.reduce(local_total_tokens, reduction="sum")
    global_weighted_loss = accelerator.reduce(local_weighted_loss, reduction="sum")

    # Per-modality: always reduce (even if zero) to keep collective-op count equal
    global_weighted_text = accelerator.reduce(local_weighted_text, reduction="sum")
    global_weighted_image = accelerator.reduce(local_weighted_image, reduction="sum")
    global_text_tokens = accelerator.reduce(local_text_tokens, reduction="sum")
    global_image_tokens = accelerator.reduce(local_image_tokens, reduction="sum")

    if global_text_tokens.item() > 0 or global_image_tokens.item() > 0:
        avg_text_for_loss = (
            global_weighted_text / global_text_tokens
            if global_text_tokens.item() > 0
            else torch.tensor(0.0, device=accelerator.device)
        )
        avg_image_for_loss = (
            global_weighted_image / global_image_tokens
            if global_image_tokens.item() > 0
            else torch.tensor(0.0, device=accelerator.device)
        )
        unwrapped = accelerator.unwrap_model(model)
        lambda_text = getattr(unwrapped.config, "lambda_text", 1.0)
        lambda_image = getattr(unwrapped.config, "lambda_image", 0.5)
        avg_loss = (lambda_text * avg_text_for_loss + lambda_image * avg_image_for_loss).item()
    else:
        avg_loss = (global_weighted_loss / global_total_tokens).item()
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    logs = {
        "val/loss": avg_loss,
        "val/ppl": ppl,
    }
    if global_text_tokens.item() > 0:
        avg_text = (global_weighted_text / global_text_tokens).item()
        logs["val/loss_text"] = avg_text
        logs["val/ppl_text"] = math.exp(avg_text) if avg_text < 100 else float("inf")
    if global_image_tokens.item() > 0:
        avg_image = (global_weighted_image / global_image_tokens).item()
        logs["val/loss_image_flow"] = avg_image

    if accelerator.is_main_process:
        accelerator.log(logs, step=global_step)
        msg = f"[Validation] Step {global_step + 1} | Loss: {avg_loss:.4f} (PPL: {ppl:.2f})"
        if "val/loss_text" in logs:
            msg += f" | Text: {logs['val/loss_text']:.4f}"
        if "val/loss_image_flow" in logs:
            msg += f" | ImageFlow: {logs['val/loss_image_flow']:.4f}"
        logger.info(msg)

    return avg_loss, ppl


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error", message="None of the inputs have requires_grad=True")
    main() 
