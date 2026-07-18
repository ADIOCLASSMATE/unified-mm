#!/usr/bin/env python3
"""Train a Qwen3 backbone with the original Show-o discrete image objective.

This entry point intentionally supports only class-conditional text-to-image
training. Image codes live in the same vocabulary and output projection as text
tokens, while image targets use same-position masked-token cross entropy.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from models.modeling_model.modeling_qwen_showo import QwenShowOForCausalLM
from utils.dataset_qwen_showo_imagenet import build_qwen_showo_imagenet_dataloaders
from utils.utils import flatten_omega_conf, get_config
from utils.wsd_schedule import get_wsd_schedule


LOGGER = get_logger(__name__, log_level="INFO")
SPECIAL_TOKENS = ("<|t2i|>", "<|boi|>", "<|eoi|>")


def _disabled(value) -> bool:
    return value is None or str(value).lower() in {"", "none", "null", "false"}


def _model_dtype(mixed_precision: str) -> torch.dtype:
    return torch.bfloat16 if str(mixed_precision).lower() == "bf16" else torch.float32


def _add_showo_special_tokens(tokenizer) -> dict[str, int]:
    tokenizer.add_tokens(
        [token for token in SPECIAL_TOKENS if token not in tokenizer.get_vocab()],
        special_tokens=True,
    )
    token_ids = {
        "t2i_token_id": int(tokenizer.convert_tokens_to_ids("<|t2i|>")),
        "boi_token_id": int(tokenizer.convert_tokens_to_ids("<|boi|>")),
        "eoi_token_id": int(tokenizer.convert_tokens_to_ids("<|eoi|>")),
    }
    if any(value < 0 for value in token_ids.values()):
        raise ValueError(f"failed to register Show-o special tokens: {token_ids}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return token_ids


def _configure_image_vocabulary(
    model: QwenShowOForCausalLM,
    *,
    image_offset: int,
    image_vocab_size: int,
    image_mask_token_id: int,
    image_loss_chunk_size: int,
) -> None:
    total_vocab_size = int(image_mask_token_id) + 1
    if hasattr(model, "configure_image_vocabulary"):
        model.configure_image_vocabulary(
            image_offset=int(image_offset),
            image_vocab_size=int(image_vocab_size),
            image_mask_token_id=int(image_mask_token_id),
            image_loss_chunk_size=int(image_loss_chunk_size),
            resize_embeddings=True,
        )
    else:
        try:
            model.resize_token_embeddings(total_vocab_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(total_vocab_size)
        model.config.image_offset = int(image_offset)
        model.config.image_vocab_size = int(image_vocab_size)
        model.config.image_mask_token_id = int(image_mask_token_id)
        model.config.image_loss_chunk_size = int(image_loss_chunk_size)
        model.config.vocab_size = total_vocab_size
        model.tie_weights()


def _initialize_added_rows(
    model: QwenShowOForCausalLM,
    *,
    original_vocab_size: int,
    special_token_ids: list[int],
    seed: int,
) -> None:
    embeddings = model.get_input_embeddings().weight
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    initializer_range = float(getattr(model.config, "initializer_range", 0.02))
    rows = sorted(
        set(int(value) for value in special_token_ids)
        | set(range(int(original_vocab_size), int(embeddings.shape[0])))
    )
    if not rows:
        return
    initialized = torch.empty(
        (len(rows), embeddings.shape[1]),
        device="cpu",
        dtype=torch.float32,
    ).normal_(mean=0.0, std=initializer_range, generator=generator)
    with torch.no_grad():
        embeddings[torch.tensor(rows, device=embeddings.device)] = initialized.to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
    model.tie_weights()


def load_qwen_showo_model_tokenizer(config, model_path: str | None = None):
    """Load either a base Qwen checkpoint or an already trained Qwen-Show-o checkpoint."""
    source = str(model_path or config.model.model_path)
    tokenizer = AutoTokenizer.from_pretrained(source, fix_mistral_regex=True)
    special_ids = _add_showo_special_tokens(tokenizer)
    model_config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    dtype = _model_dtype(config.training.mixed_precision)

    is_showo_checkpoint = all(
        getattr(model_config, name, None) is not None
        for name in ("image_offset", "image_vocab_size", "image_mask_token_id")
    )
    if is_showo_checkpoint:
        model = QwenShowOForCausalLM.from_pretrained(
            source,
            config=model_config,
            dtype=dtype,
            trust_remote_code=True,
        )
        image_offset = int(model_config.image_offset)
        image_vocab_size = int(model_config.image_vocab_size)
        image_mask_token_id = int(model_config.image_mask_token_id)
        expected_vocab_size = image_mask_token_id + 1
        if model.get_input_embeddings().num_embeddings != expected_vocab_size:
            raise ValueError(
                f"checkpoint vocabulary mismatch: embeddings="
                f"{model.get_input_embeddings().num_embeddings}, expected={expected_vocab_size}"
            )
    else:
        original_vocab_size = int(model_config.vocab_size)
        model = QwenShowOForCausalLM.from_pretrained(
            source,
            config=model_config,
            dtype=dtype,
            trust_remote_code=True,
        )
        image_vocab_size = int(config.model.get("image_vocab_size", 8192))
        image_offset = int(
            config.model.get("image_offset", None)
            or max(int(model_config.vocab_size), len(tokenizer))
        )
        image_mask_token_id = image_offset + image_vocab_size
        _configure_image_vocabulary(
            model,
            image_offset=image_offset,
            image_vocab_size=image_vocab_size,
            image_mask_token_id=image_mask_token_id,
            image_loss_chunk_size=int(config.model.get("image_loss_chunk_size", 1024)),
        )
        _initialize_added_rows(
            model,
            original_vocab_size=original_vocab_size,
            special_token_ids=list(special_ids.values()),
            seed=int(config.training.seed) + 17,
        )

    for key, value in special_ids.items():
        checkpoint_value = getattr(model.config, key, None)
        if checkpoint_value is not None and int(checkpoint_value) != int(value):
            raise ValueError(
                f"{key} mismatch between tokenizer ({value}) and checkpoint ({checkpoint_value})"
            )
        setattr(model.config, key, int(value))
        config.model[key] = int(value)

    config.model.image_offset = int(image_offset)
    config.model.image_vocab_size = int(image_vocab_size)
    config.model.image_mask_token_id = int(image_mask_token_id)
    config.model.vocab_size = int(image_mask_token_id) + 1
    model.config.image_offset = int(image_offset)
    model.config.image_vocab_size = int(image_vocab_size)
    model.config.image_mask_token_id = int(image_mask_token_id)
    model.config.image_tokens_per_img = int(config.model.get("image_tokens_per_img", 256))
    model.config.image_loss_chunk_size = int(config.model.get("image_loss_chunk_size", 1024))
    model.config.qwen_showo_version = 1
    model.config.architectures = ["QwenShowOForCausalLM"]
    model.config.use_cache = False
    model.tie_weights()

    if bool(config.training.get("use_gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, tokenizer


def build_optimizer(model, config):
    params = config.optimizer.params
    base_lr = float(params.get("backbone_learning_rate", params.learning_rate))
    embedding_lr = float(params.get("embedding_learning_rate", params.learning_rate))
    weight_decay = float(params.weight_decay)
    no_decay_terms = (
        "bias",
        "norm.weight",
        "layernorm.weight",
        "layer_norm.weight",
        "embed_tokens.weight",
        "lm_head.weight",
    )
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = (
            embedding_lr
            if "embed_tokens.weight" in name or "lm_head.weight" in name
            else base_lr
        )
        decay = 0.0 if parameter.ndim < 2 or any(term in name for term in no_decay_terms) else weight_decay
        groups.setdefault((learning_rate, decay), []).append(parameter)
    optimizer_groups = [
        {"params": values, "lr": learning_rate, "weight_decay": decay}
        for (learning_rate, decay), values in groups.items()
    ]
    return AdamW(
        optimizer_groups,
        betas=(float(params.beta1), float(params.beta2)),
        eps=float(params.epsilon),
    )


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_attention_mask(batch, model):
    attention_mask = batch.get("attention_mask")
    if (
        attention_mask is not None
        and torch.is_floating_point(attention_mask)
    ):
        attention_mask = attention_mask.to(dtype=next(model.parameters()).dtype)
    return attention_mask


def append_metrics(path: Path, step: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"step": int(step), **metrics}, sort_keys=True) + "\n")


def rotate_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    for path in checkpoints[:-int(keep)]:
        shutil.rmtree(path)


def save_hf_model(model, tokenizer, accelerator, path: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        unwrapped.save_pretrained(
            path,
            state_dict=state_dict,
            safe_serialization=True,
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(path)
    accelerator.wait_for_everyone()


@torch.no_grad()
def validate(model, dataloader, accelerator, max_batches: int) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(3, device=accelerator.device, dtype=torch.float64)
    for batch_index, raw_batch in enumerate(dataloader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch, accelerator.device)
        output = model(
            input_ids=batch["input_ids"],
            token_types=batch.get("token_types"),
            attention_mask=model_attention_mask(batch, model),
            labels=batch["labels"],
            return_logits=False,
        )
        count = output.get("image_token_count", None)
        correct = output.get("image_token_correct", None)
        if count is None:
            count = (batch["labels"] != -100).sum()
        if correct is None:
            correct = torch.zeros((), device=accelerator.device)
        totals[0] += output.loss.detach().double() * count.detach().double()
        totals[1] += correct.detach().double()
        totals[2] += count.detach().double()
    totals = accelerator.reduce(totals, reduction="sum")
    count = totals[2].clamp_min(1.0)
    model.train()
    return {
        "val/image_ce": float((totals[0] / count).item()),
        "val/image_acc": float((totals[1] / count).item()),
        "val/image_tokens": float(totals[2].item()),
    }


def main():
    config = get_config()
    output_dir = Path(config.experiment.output_dir) / str(config.experiment.project)
    config.experiment.output_dir = str(output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batch = int(config.training.batch_size)
    global_batch = int(config.training.total_batch_size)
    denominator = micro_batch * world_size
    if global_batch % denominator:
        raise ValueError(
            f"total_batch_size={global_batch} must be divisible by "
            f"batch_size({micro_batch}) * world_size({world_size})"
        )
    accumulation = global_batch // denominator
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(config.training.mixed_precision),
        log_with="wandb" if bool(config.experiment.get("use_wandb", True)) else None,
        step_scheduler_with_optimizer=False,
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        plugin = accelerator.state.deepspeed_plugin.deepspeed_config
        plugin["train_micro_batch_size_per_gpu"] = micro_batch
        plugin["gradient_accumulation_steps"] = accumulation

    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(int(config.training.seed), device_specific=True)
    model, tokenizer = load_qwen_showo_model_tokenizer(config)
    train_loader, val_loader = build_qwen_showo_imagenet_dataloaders(config, tokenizer)
    optimizer = build_optimizer(model, config)
    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=int(config.lr_scheduler.params.warmup_steps),
        num_decay_steps=int(config.lr_scheduler.params.decay_steps),
        num_training_steps=int(config.training.max_train_steps),
        min_lr_ratio=float(config.lr_scheduler.params.min_lr_scale),
    )
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, output_dir / "config.yaml")
        if bool(config.experiment.get("use_wandb", True)):
            accelerator.init_trackers(
                str(config.experiment.wandb_project),
                config=dict(flatten_omega_conf(config, resolve=True)),
                init_kwargs={
                    "wandb": {
                        "name": str(config.experiment.name),
                        "mode": os.environ.get("WANDB_MODE", "offline"),
                    }
                },
            )

    unwrapped = accelerator.unwrap_model(model)
    LOGGER.info(
        "Qwen-Show-o parameters=%d global_batch=%d accumulation=%d train=%d val=%d "
        "image_offset=%d image_vocab=%d mask_id=%d",
        sum(parameter.numel() for parameter in unwrapped.parameters()),
        global_batch,
        accumulation,
        len(train_loader.dataset),
        len(val_loader.dataset),
        int(unwrapped.config.image_offset),
        int(unwrapped.config.image_vocab_size),
        int(unwrapped.config.image_mask_token_id),
    )

    global_step = 0
    resume = config.experiment.get("resume_from_checkpoint", None)
    if not _disabled(resume):
        resume_path = Path(str(resume))
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        accelerator.load_state(resume_path)
        metadata_path = resume_path / "metadata.json"
        if metadata_path.exists():
            global_step = int(json.loads(metadata_path.read_text()).get("global_step", 0))
        else:
            global_step = int(resume_path.name.rsplit("-", 1)[-1])

    max_steps = int(config.training.max_train_steps)
    progress = tqdm(
        total=max_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )
    metrics_path = output_dir / "metrics.jsonl"
    train_iterator = iter(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        try:
            raw_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            raw_batch = next(train_iterator)
        batch = move_batch(raw_batch, accelerator.device)

        with accelerator.accumulate(model):
            output = model(
                input_ids=batch["input_ids"],
                token_types=batch.get("token_types"),
                attention_mask=model_attention_mask(batch, model),
                labels=batch["labels"],
                return_logits=False,
            )
            accelerator.backward(output.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), float(config.training.max_grad_norm)
                )
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        global_step += 1
        progress.update(1)
        count = output.get("image_token_count", torch.ones((), device=accelerator.device))
        correct = output.get("image_token_correct", torch.zeros((), device=accelerator.device))
        values = torch.stack(
            [
                output.loss.detach().float(),
                correct.detach().float(),
                count.detach().float(),
            ]
        )
        values = accelerator.reduce(values, reduction="sum")
        train_metrics = {
            "train/image_ce": float((values[0] / accelerator.num_processes).item()),
            "train/image_acc": float((values[1] / values[2].clamp_min(1)).item()),
            "train/masked_tokens": float(values[2].item()),
            "train/lr_backbone": float(scheduler.get_last_lr()[0]),
        }
        mask_ratios = batch.get("mask_ratios")
        if mask_ratios is not None:
            ratio_sum = accelerator.reduce(mask_ratios.float().sum(), reduction="sum")
            ratio_count = accelerator.reduce(
                torch.tensor(mask_ratios.numel(), device=accelerator.device),
                reduction="sum",
            )
            train_metrics["train/mask_ratio"] = float(
                (ratio_sum / ratio_count.clamp_min(1)).item()
            )

        if global_step % int(config.experiment.log_every) == 0:
            accelerator.log(train_metrics, step=global_step)
            if accelerator.is_main_process:
                append_metrics(metrics_path, global_step, train_metrics)
            progress.set_postfix(image_ce=f"{train_metrics['train/image_ce']:.3f}")

        if global_step % int(config.experiment.val_every) == 0:
            val_metrics = validate(
                model,
                val_loader,
                accelerator,
                max_batches=int(config.evaluation.get("max_val_batches", -1)),
            )
            accelerator.log(val_metrics, step=global_step)
            if accelerator.is_main_process:
                append_metrics(metrics_path, global_step, val_metrics)
                LOGGER.info("step=%d validation=%s", global_step, val_metrics)

        if global_step % int(config.experiment.save_every) == 0:
            checkpoint = output_dir / f"checkpoint-{global_step}"
            accelerator.save_state(checkpoint)
            if accelerator.is_main_process:
                (checkpoint / "metadata.json").write_text(
                    json.dumps({"global_step": global_step}, indent=2) + "\n"
                )
                rotate_checkpoints(
                    output_dir, int(config.experiment.checkpoints_total_limit)
                )
        if global_step % int(config.experiment.save_hf_every) == 0:
            save_hf_model(
                model, tokenizer, accelerator, output_dir / f"hf_model-{global_step}"
            )

    if bool(config.experiment.get("save_final", True)):
        save_hf_model(model, tokenizer, accelerator, output_dir / "hf_model-final")
    if accelerator.is_main_process:
        append_metrics(metrics_path, global_step, {"status": "complete"})
    accelerator.end_training()


if __name__ == "__main__":
    main()
