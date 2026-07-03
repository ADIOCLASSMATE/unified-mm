import json
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW

from models.logging import set_verbosity_error, set_verbosity_info
from utils.dataset_utils import get_dataloaders
from utils.utils import (
    AverageMeter,
    flatten_omega_conf,
    get_config,
    get_selfless_mask,
    load_model_tokenizer,
    log_grad_norm,
    save_checkpoint,
    save_hf_model,
)
from utils.wsd_schedule import get_wsd_schedule

logger = get_logger(__name__, log_level="INFO")


def _set_epoch_on_loader(loader, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    while dataset is not None:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
            break
        dataset = getattr(dataset, "dataset", None)

    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    batch_sampler = getattr(loader, "batch_sampler", None)
    sampler = getattr(batch_sampler, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _freeze_unused_image_modules(model, logger=None) -> None:
    frozen = []
    for module_name in ("image_flow_head",):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad_(False)
        frozen.append(module_name)

    image_token_embedder = getattr(model, "image_token_embedder", None)
    if image_token_embedder is not None:
        for param in image_token_embedder.parameters():
            param.requires_grad_(False)
        frozen.append("image_token_embedder")

    if logger is not None and frozen:
        logger.info(f"Frozen unused image modules for text-only tuning: {', '.join(frozen)}")


def _build_optimizer(model, config):
    optimizer_config = config.optimizer.params
    no_decay = (
        "bias",
        "layer_norm.weight",
        "layernorm.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "norm.weight",
        "embed_tokens.weight",
        "lm_head.weight",
    )
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    if config.optimizer.name != "adamw":
        raise ValueError(f"Optimizer {config.optimizer.name} not supported")

    return AdamW(
        optimizer_grouped_parameters,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )


def _move_batch_to_device(batch, device):
    return {
        "input_ids": batch["input_ids"].contiguous().to(device),
        "sigma": batch["sigma"].to(device),
        "labels": batch["labels"].to(device),
    }


def main():
    config = get_config()
    total_batch_size_per_gpu = config.training.batch_size
    config.experiment.output_dir = os.path.join(config.experiment.output_dir, config.experiment.project)

    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    print(f"Number of processes: {num_processes}")
    print(f"Total batch size: {config.training.total_batch_size}")
    print(f"Batch size per GPU: {total_batch_size_per_gpu}")
    print(
        "Gradient accumulation steps: "
        f"{(config.training.total_batch_size // config.training.batch_size) // num_processes}"
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=((config.training.total_batch_size // config.training.batch_size) // num_processes),
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        step_scheduler_with_optimizer=config.training.step_scheduler_with_optimizer,
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = (
            accelerator.gradient_accumulation_steps
        )

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

    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    logger.info("Loading tokenizer and model")
    model, tokenizer = load_model_tokenizer(config=config, logger=logger)
    if config.training.get("freeze_unused_image_modules", True):
        _freeze_unused_image_modules(model, logger=logger)
    if config.training.get("use_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    optimizer = _build_optimizer(model, config)
    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        num_decay_steps=config.lr_scheduler.params.decay_steps,
        num_training_steps=config.training.max_train_steps,
        min_lr_ratio=config.lr_scheduler.params.min_lr_scale,
    )

    logger.info("Creating text dataloaders")
    train_dataloader, val_dataloader = get_dataloaders(config, tokenizer)

    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

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
        accelerator.load_state(resume_checkpoint_dir)
        metadata_file = resume_checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            resume_step = metadata.get("global_step", 0)
        global_step = resume_step
        logger.info(f"Resumed at global_step={global_step}")
    else:
        logger.warning("No valid checkpoint found or specified, starting fresh text tuning.")

    total_batch_size = (
        total_batch_size_per_gpu
        * accelerator.num_processes
        * accelerator.gradient_accumulation_steps
    )
    logger.info("***** Running selfless text fine-tuning *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Context length = {config.dataset.preprocessing.max_seq_length}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Text attention order = left-to-right sigma, no label shift")

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    batches_to_skip = 0
    if resume_step > 0:
        batches_to_skip = resume_step * accelerator.gradient_accumulation_steps
        logger.info(f"Resuming from step {resume_step}, skipping {batches_to_skip} batches...")
        train_dataloader = accelerator.skip_first_batches(train_dataloader, batches_to_skip)

    batch_time_m = AverageMeter()
    end = time.time()
    train_iter = iter(train_dataloader)
    epoch = 0
    acc_loss = torch.tensor(0.0, device=accelerator.device)

    model.train()
    while global_step < config.training.max_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            _set_epoch_on_loader(train_dataloader, epoch)
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        batch = _move_batch_to_device(batch, accelerator.device)
        input_ids = batch["input_ids"]
        sigma = batch["sigma"]
        labels = batch["labels"]
        _, seq_len = input_ids.shape

        if global_step == 0 and accelerator.is_main_process and not hasattr(main, "_logged_first_batch"):
            main._logged_first_batch = True
            logger.info(f"Input ids shape: {tuple(input_ids.shape)}")
            logger.info(f"sigma range: [{sigma.min().item()}, {sigma.max().item()}]")
            logger.info(f"labels -100 ratio: {(labels == -100).sum().item() / labels.numel():.3f}")

        attention_mask = get_selfless_mask(sigma=sigma, seq_len=seq_len, device=accelerator.device)
        with accelerator.accumulate(model):
            output = model(
                X0_input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                calculate_likelihood=True,
            )
            loss = output.loss
            acc_loss += loss.detach()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if config.training.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if (global_step + 1) % config.experiment.log_grad_norm_every == 0 and accelerator.is_main_process:
                    log_grad_norm(model, accelerator, global_step + 1)

        if accelerator.sync_gradients:
            global_step += 1
            batch_time_m.update(time.time() - end)
            end = time.time()

            if global_step % config.experiment.log_every == 0:
                avg_loss_per_step = acc_loss / accelerator.gradient_accumulation_steps
                global_avg_loss = accelerator.reduce(avg_loss_per_step, reduction="mean")
                loss_item = global_avg_loss.item()
                logs = {
                    "step_loss": loss_item,
                    "train_ppl": math.exp(min(loss_item, 100)),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "samples/sec/gpu": (
                        accelerator.gradient_accumulation_steps
                        * config.training.batch_size
                        / batch_time_m.val
                    ),
                    "batch_time": batch_time_m.val,
                }
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process:
                    logger.info(
                        f"Step: {global_step} | Loss: {loss_item:0.4f} | "
                        f"PPL: {logs['train_ppl']:0.2f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.8f} | "
                        f"Sec/Iter: {batch_time_m.val:0.4f}"
                    )
                batch_time_m.reset()

            if global_step % config.experiment.save_every == 0:
                save_checkpoint(model, config, accelerator, global_step)
            if global_step % config.experiment.save_hfmodel_every == 0:
                save_hf_model(model, tokenizer, config, accelerator, global_step)
            if global_step % config.experiment.val_every == 0:
                validate(model, val_dataloader, accelerator, global_step)
                model.train()

            acc_loss.zero_()

    accelerator.wait_for_everyone()
    save_hf_model(model, tokenizer, config, accelerator, "final")
    accelerator.end_training()


@torch.no_grad()
def validate(model, val_dataloader, accelerator, global_step):
    model.eval()
    local_loss_sum = torch.tensor(0.0, device=accelerator.device)
    local_token_count = torch.tensor(0.0, device=accelerator.device)

    for batch in val_dataloader:
        batch = _move_batch_to_device(batch, accelerator.device)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        sigma = batch["sigma"]
        _, seq_len = input_ids.shape
        valid_tokens = (labels != -100).sum().float()
        attention_mask = get_selfless_mask(sigma=sigma, seq_len=seq_len, device=accelerator.device)
        output = model(
            X0_input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            calculate_likelihood=True,
        )
        local_loss_sum += output.loss.detach() * valid_tokens
        local_token_count += valid_tokens

    global_loss_sum = accelerator.reduce(local_loss_sum, reduction="sum")
    global_token_count = accelerator.reduce(local_token_count, reduction="sum")
    avg_loss = (global_loss_sum / global_token_count).item()
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    if accelerator.is_main_process:
        accelerator.log({"val/loss": avg_loss, "val/ppl": ppl}, step=global_step)
        logger.info(f"[Validation] Step {global_step + 1} | Loss: {avg_loss:.4f} (PPL: {ppl:.2f})")

    return avg_loss, ppl


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", message=".*Using the model-agnostic default.*")
    main()
