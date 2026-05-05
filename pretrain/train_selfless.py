import os
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
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf
import torch
from torch.optim import AdamW
import torch.nn.functional as F


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from utils.dataset_utils import get_dataloaders
from utils.selfless_utils import SelflessSampler
from utils.wsd_schedule import get_wsd_schedule
from models.logging import set_verbosity_info, set_verbosity_error

from utils.utils import get_config, flatten_omega_conf, get_selfless_mask, get_selfless_ar_mask, load_model_tokenizer, log_grad_norm, AverageMeter, save_checkpoint, save_hf_model

logger = get_logger(__name__, log_level="INFO")


def main():
    #########################
    #      SETUP Config     #
    #########################
    config = get_config()
        
    total_batch_size_per_gpu = config.training.batch_size
    
    config.experiment.output_dir = os.path.join(config.experiment.output_dir, config.experiment.project)
    config.experiment.logging_dir = os.path.join(config.experiment.output_dir, "logs")
    
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
        log_with="tensorboard",
        project_dir=config.experiment.logging_dir,
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

        accelerator.init_trackers(
            config.experiment.name,
            config=log_config,
        )

    # Set training seed
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    #########################
    # MODELS and TOKENIZER  #
    #########################
    logger.info("Loading tokenizer and model")
    model, tokenizer = load_model_tokenizer(config=config, logger=logger)
    
    selfless_sampler = SelflessSampler(mask_token_id=model.config.mask_token_id, config=config)
    
    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params

    # No decay on bias and layernorm
    no_decay = ["bias", "layer_norm.weight", "ln_f.weight", "wte.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

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
        min_lr_ratio=optimizer_config.learning_rate * config.lr_scheduler.params.min_lr_scale
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
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

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
    if resume_step > 0:
        batches_to_skip = resume_step * accelerator.gradient_accumulation_steps
        logger.info(f"Resuming from step {resume_step}, skipping {batches_to_skip} batches...")
        train_dataloader = accelerator.skip_first_batches(train_dataloader, batches_to_skip)

    model.train()

    train_iter = iter(train_dataloader)

    while global_step < config.training.max_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # 一个 epoch 跑完了，重新来
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        
        # *-------*-------*-------*-------*-------*-------*
        # Data Processing
        # *-------*-------*-------*-------*-------*-------*
        is_multimodal = "token_types" in batch
        t_1 = 0.0

        if is_multimodal:
            input_ids = batch["input_ids"].contiguous()  # [B, L] — no shift for selfless
            token_types = batch["token_types"]  # [B, L]
            task_modes = batch["task_mode"]  # list of strings
            B, L = input_ids.shape

            # Per-sample sigma assignment
            from utils.selfless_utils import assign_sigma_multimodal_batch
            v_sample = assign_sigma_multimodal_batch(
                token_types, task_modes, L
            ).to(accelerator.device)

            selfless_attention_mask = get_selfless_mask(
                v_sample=v_sample, seq_len=L, device=accelerator.device
            )

            if global_step == 0 and accelerator.is_main_process:
                logger.info(f"Input ids shape: {input_ids.shape}, multimodal mode")
                logger.info(f"token type counts: text={(token_types==0).sum().item()}, "
                           f"image={(token_types==1).sum().item()}, "
                           f"special={(token_types==2).sum().item()}")

        else:
            # Legacy text-only path
            text_ids = batch["input_ids"][:, :-1].contiguous()
            token_types = None
            B, L = text_ids.shape

            t_sample, v_sample = selfless_sampler.sample_v(text_ids)
            t_1 = t_sample[0, 0].item()

            v_sample = v_sample.to(accelerator.device)
            selfless_attention_mask = get_selfless_mask(
                v_sample=v_sample, seq_len=L, device=accelerator.device
            )
            del t_sample

            if global_step == 0 and accelerator.is_main_process:
                logger.info(f"Input ids shape: {text_ids.shape}, text-only mode")
            input_ids = text_ids

        # *-------*-------*-------*-------*-------*-------*
        # Forward & Backward
        # *-------*-------*-------*-------*-------*-------*
        with accelerator.accumulate(model):
            forward_kwargs = {
                "X0_input_ids": input_ids,
                "labels": input_ids,
                "attention_mask": selfless_attention_mask,
            }
            if token_types is not None:
                forward_kwargs["token_types"] = token_types

            model_output = model(**forward_kwargs)
            loss = model_output.loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if config.training.max_grad_norm:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

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
                avg_loss = accelerator.reduce(loss.detach(), reduction="mean")

                samples_per_second_per_gpu = (
                        accelerator.gradient_accumulation_steps * config.training.batch_size / batch_time_m.val
                )

                logs = {
                    "t_1": t_1,
                    "step_loss": avg_loss.item(),
                    "train_ppl": math.exp(avg_loss.item()),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "batch_time": batch_time_m.val,
                }

                # Per-modality loss logging (multimodal mode)
                per_mod = getattr(model_output, "per_modality_loss", None)
                if per_mod is not None:
                    text_l = accelerator.reduce(per_mod["text_loss"], reduction="mean")
                    image_l = accelerator.reduce(per_mod["image_loss"], reduction="mean")
                    logs["train/loss_text"] = text_l.item()
                    logs["train/loss_image"] = image_l.item()
                    logs["train/ppl_text"] = math.exp(text_l.item()) if text_l.item() < 100 else float("inf")
                    logs["train/ppl_image"] = math.exp(image_l.item()) if image_l.item() < 100 else float("inf")

                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    logger.info(
                        f"Step: {global_step} | "
                        f"T: {t_1} | "
                        f"Loss: {avg_loss.item():0.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} | "
                        f"Sec/Iter: {batch_time_m.val:0.4f}"
                    )

                batch_time_m.reset()
                data_time_m.reset()

            # Checkpointing
            if global_step % config.experiment.save_every == 0:
                save_checkpoint(model, config, accelerator, global_step)
            
            if global_step % config.experiment.save_hfmodel_every == 0:
                save_hf_model(model, tokenizer, config, accelerator, global_step)
                
            # Validation
            if global_step % config.experiment.val_every == 0:
                validate(model, val_dataloader, selfless_sampler, accelerator, global_step)
                # if accelerator.is_main_process:
                #     pre_text, label_text = get_text(logits_pred=logits_pred[0], label_ids=label_ids[0], tokenizer=tokenizer)
                #     accelerator.print(f"pre_text: {pre_text}")
                #     accelerator.print(f"label_text: {label_text}")
                
                model.train() 

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_hf_model(model, tokenizer, config, accelerator, "final")
    accelerator.end_training()


@torch.no_grad()
def validate(model, val_dataloader, selfless_sampler, accelerator, global_step):
    # Check if validation data is multimodal
    is_multimodal = hasattr(val_dataloader.dataset, 'samples') or (
        hasattr(val_dataloader.dataset, 'dataset') and
        hasattr(val_dataloader.dataset.dataset, 'samples')
    )
    # Heuristic: if first batch has token_types, we're in multimodal mode
    test_batch = next(iter(val_dataloader))
    is_multimodal = "token_types" in test_batch

    if is_multimodal:
        _validate_multimodal(model, val_dataloader, test_batch, accelerator, global_step)
    else:
        _validate_text_only(model, val_dataloader, test_batch, selfless_sampler, accelerator, global_step)


@torch.no_grad()
def _validate_text_only(model, val_dataloader, first_batch, selfless_sampler, accelerator, global_step):
    local_total_loss_selfless = torch.tensor(0.0, device=accelerator.device)
    local_total_loss_ar = torch.tensor(0.0, device=accelerator.device)
    local_total_count = torch.tensor(0.0, device=accelerator.device)

    # Process first batch (already fetched)
    for step, batch in enumerate(_prepend_first_batch(first_batch, val_dataloader)):
        text_ids = batch["input_ids"][:, :-1].contiguous()
        current_batch_size = text_ids.size(0)

        _, v_sample = selfless_sampler.sample_v(text_ids)
        B, L = text_ids.shape
        v_sample = v_sample.to(accelerator.device)

        selfless_attention_mask = get_selfless_mask(v_sample=v_sample, seq_len=L, device=accelerator.device)
        loss_selfless = model(
            X0_input_ids=text_ids, labels=text_ids,
            attention_mask=selfless_attention_mask,
        ).loss
        local_total_loss_selfless += loss_selfless.detach() * current_batch_size

        AR_mask = get_selfless_ar_mask(seq_len=L, device=accelerator.device)
        loss_ar = model(
            X0_input_ids=text_ids, labels=text_ids,
            attention_mask=AR_mask,
        ).loss
        local_total_loss_ar += loss_ar.detach() * current_batch_size
        local_total_count += current_batch_size

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
def _validate_multimodal(model, val_dataloader, first_batch, accelerator, global_step):
    from utils.selfless_utils import assign_sigma_multimodal_batch

    local_total_loss = torch.tensor(0.0, device=accelerator.device)
    local_text_loss = torch.tensor(0.0, device=accelerator.device)
    local_image_loss = torch.tensor(0.0, device=accelerator.device)
    local_total_count = torch.tensor(0.0, device=accelerator.device)

    for batch in _prepend_first_batch(first_batch, val_dataloader):
        input_ids = batch["input_ids"].contiguous()
        token_types = batch["token_types"]
        task_modes = batch["task_mode"]
        B, L = input_ids.shape
        current_batch_size = B

        v_sample = assign_sigma_multimodal_batch(token_types, task_modes, L)
        v_sample = v_sample.to(accelerator.device)

        # Selfless sigma validation
        selfless_attention_mask = get_selfless_mask(v_sample=v_sample, seq_len=L, device=accelerator.device)
        output = model(
            X0_input_ids=input_ids, labels=input_ids,
            attention_mask=selfless_attention_mask,
            token_types=token_types,
        )
        local_total_loss += output.loss.detach() * current_batch_size
        if hasattr(output, "per_modality_loss"):
            local_text_loss += output.per_modality_loss["text_loss"] * current_batch_size
            local_image_loss += output.per_modality_loss["image_loss"] * current_batch_size

        local_total_count += current_batch_size

    global_total_count = accelerator.reduce(local_total_count, reduction="sum")
    global_total_loss = accelerator.reduce(local_total_loss, reduction="sum")
    avg_loss = (global_total_loss / global_total_count).item()
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    logs = {
        "val/loss": avg_loss,
        "val/ppl": ppl,
    }
    # Per-modality metrics
    if local_text_loss.item() > 0:
        global_text_loss = accelerator.reduce(local_text_loss, reduction="sum")
        avg_text = (global_text_loss / global_total_count).item()
        logs["val/loss_text"] = avg_text
        logs["val/ppl_text"] = math.exp(avg_text) if avg_text < 100 else float("inf")
    if local_image_loss.item() > 0:
        global_image_loss = accelerator.reduce(local_image_loss, reduction="sum")
        avg_image = (global_image_loss / global_total_count).item()
        logs["val/loss_image"] = avg_image
        logs["val/ppl_image"] = math.exp(avg_image) if avg_image < 100 else float("inf")

    if accelerator.is_main_process:
        accelerator.log(logs, step=global_step)
        msg = f"[Validation] Step {global_step + 1} | Loss: {avg_loss:.4f} (PPL: {ppl:.2f})"
        if "val/loss_text" in logs:
            msg += f" | Text: {logs['val/loss_text']:.4f}"
        if "val/loss_image" in logs:
            msg += f" | Image: {logs['val/loss_image']:.4f}"
        logger.info(msg)

    return avg_loss, ppl


def _prepend_first_batch(first_batch, dataloader):
    """Yield first_batch, then remaining batches from dataloader."""
    yield first_batch
    yield from dataloader


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("error", message="None of the inputs have requires_grad=True")
    main() 