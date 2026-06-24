import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW

from models.logging import set_verbosity_error, set_verbosity_info
from models.modeling_model.modeling_selfless_flow import FlowMatchingHead
from utils.dataset_flow_latent import build_flow_latent_dataloaders
from utils.utils import AverageMeter, flatten_omega_conf, get_config
from utils.wsd_schedule import get_wsd_schedule


logger = get_logger(__name__, log_level="INFO")
_MAR_VAE_CACHE = None


class PureFlowHeadPretrainer(nn.Module):
    def __init__(
        self,
        image_tokens_per_img: int,
        image_latent_dim: int,
        hidden_size: int,
        flow_width: int,
        flow_depth: int,
        flow_time_scale: float = 1000.0,
        flow_sample_method: str = "heun",
        freeze_condition: bool = True,
    ):
        super().__init__()
        self.image_tokens_per_img = image_tokens_per_img
        self.image_latent_dim = image_latent_dim
        self.hidden_size = hidden_size
        self.flow_head = FlowMatchingHead(
            target_channels=image_latent_dim,
            z_channels=hidden_size,
            width=flow_width,
            depth=flow_depth,
            time_scale=flow_time_scale,
            sample_method=flow_sample_method,
        )
        if freeze_condition:
            for param in self.flow_head.cond_embed.parameters():
                param.requires_grad_(False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        bsz, image_tokens, channels = latents.shape
        if image_tokens != self.image_tokens_per_img:
            raise ValueError(f"Expected {self.image_tokens_per_img} image tokens, got {image_tokens}")
        if channels != self.image_latent_dim:
            raise ValueError(f"Expected latent dim {self.image_latent_dim}, got {channels}")
        target = latents.reshape(bsz * image_tokens, channels)
        z = torch.zeros(
            bsz * image_tokens,
            self.hidden_size,
            device=latents.device,
            dtype=self.flow_head.cond_embed.weight.dtype,
        )
        return self.flow_head(target=target, z=z)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_steps: int = 50,
        temperature: float = 1.0,
        sample_method: str | None = None,
    ) -> torch.Tensor:
        device = self.flow_head.input_proj.weight.device
        dtype = self.flow_head.input_proj.weight.dtype
        z = torch.zeros(
            batch_size,
            self.image_tokens_per_img,
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        flat = self.flow_head.sample(
            z.reshape(batch_size * self.image_tokens_per_img, -1),
            num_steps=num_steps,
            temperature=temperature,
            sample_method=sample_method,
        )
        return flat.view(batch_size, self.image_tokens_per_img, self.image_latent_dim)


def _load_mar_vae(config, accelerator):
    global _MAR_VAE_CACHE
    if _MAR_VAE_CACHE is not None:
        return _MAR_VAE_CACHE

    vae_path = Path(config.experiment.get("validation_vae_path", "public/vae/mar-kl16/kl16.ckpt"))
    if not vae_path.exists():
        logger.warning(f"Skipping validation image decode; missing VAE checkpoint: {vae_path}")
        return None

    mar_root = Path(config.experiment.get("validation_mar_root", "/inspire/hdd/global_user/wanjiaxin-253108030048/code/mar"))
    vae_module_path = mar_root / "models" / "vae.py"
    if not vae_module_path.exists():
        logger.warning(f"Skipping validation image decode; missing MAR VAE module: {vae_module_path}")
        return None

    spec = importlib.util.spec_from_file_location("mar_vae", vae_module_path)
    mar_vae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mar_vae)
    vae = mar_vae.AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=str(vae_path))
    dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    vae = vae.to(device=accelerator.device, dtype=dtype).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    _MAR_VAE_CACHE = vae
    return vae


def _tokens_to_latent_grid(tokens: torch.Tensor) -> torch.Tensor:
    bsz, image_tokens, channels = tokens.shape
    side = int(image_tokens ** 0.5)
    if side * side != image_tokens:
        raise ValueError(f"image_tokens_per_img={image_tokens} is not square")
    return tokens.view(bsz, side, side, channels).permute(0, 3, 1, 2).contiguous()


def _predict_velocity(model, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    bsz, image_tokens, channels = x_t.shape
    flat_x = x_t.reshape(bsz * image_tokens, channels)
    dtype = model.flow_head.input_proj.weight.dtype
    z = torch.zeros(
        bsz * image_tokens,
        model.hidden_size,
        device=x_t.device,
        dtype=dtype,
    )
    flat_t = t[:, None].expand(bsz, image_tokens).reshape(-1).float()
    pred = model.flow_head.predict_velocity(flat_x.to(dtype=dtype), flat_t, z)
    return pred.view(bsz, image_tokens, channels)


@torch.no_grad()
def _velocity_denoise_tokens(model, target_tokens: torch.Tensor, denoise_t: float):
    noise = torch.randn_like(target_tokens)
    t = torch.full(
        (target_tokens.shape[0],),
        denoise_t,
        device=target_tokens.device,
        dtype=target_tokens.dtype,
    )
    x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * target_tokens
    pred_velocity = _predict_velocity(model, x_t, t)
    pred_x0 = x_t.float() + (1.0 - t[:, None, None].float()) * pred_velocity
    return x_t.float(), pred_x0


@torch.no_grad()
def validate(model, val_dataloader, accelerator, global_step, config):
    model.eval()
    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_count = torch.tensor(0.0, device=accelerator.device)
    first_latents = None
    max_batches = int(config.experiment.get("validation_batches", 16))

    for batch_idx, batch in enumerate(val_dataloader):
        if batch_idx >= max_batches:
            break
        latents = batch["latents"].to(accelerator.device)
        loss = model(latents)
        count = torch.tensor(latents.shape[0], device=accelerator.device, dtype=torch.float32)
        total_loss += loss.detach() * count
        total_count += count
        if first_latents is None:
            first_latents = latents.detach()

    global_loss = accelerator.reduce(total_loss, reduction="sum")
    global_count = accelerator.reduce(total_count, reduction="sum")
    avg_loss = (global_loss / global_count.clamp_min(1.0)).item()

    if accelerator.is_main_process:
        accelerator.log({"val/loss_flow": avg_loss}, step=global_step)
        logger.info(f"[Validation] Step {global_step} | Flow Loss: {avg_loss:.4f}")

    image_every = int(config.experiment.get("validation_image_every", 0))
    if image_every and global_step % image_every == 0 and first_latents is not None:
        save_validation_images(model, first_latents, accelerator, global_step, config)

    model.train()
    return avg_loss


@torch.no_grad()
def save_validation_images(model, target_tokens, accelerator, global_step, config):
    if not accelerator.is_main_process:
        return
    vae = _load_mar_vae(config, accelerator)
    if vae is None:
        return

    unwrapped = accelerator.unwrap_model(model)
    sample_count = min(int(config.experiment.get("validation_image_samples", 4)), target_tokens.shape[0])
    flow_steps = int(config.experiment.get("validation_flow_steps", 50))
    temperature = float(config.experiment.get("validation_flow_temperature", 1.0))
    sample_method = config.experiment.get(
        "validation_flow_sample_method",
        config.model.get("flow_sample_method", None),
    )
    scaling_factor = float(config.experiment.get("validation_vae_scaling_factor", 0.2325))

    pred_tokens = unwrapped.sample(
        sample_count,
        num_steps=flow_steps,
        temperature=temperature,
        sample_method=sample_method,
    )
    denoise_t = float(config.experiment.get("validation_denoise_t", 0.5))
    noisy_tokens, denoised_tokens = _velocity_denoise_tokens(
        unwrapped,
        target_tokens[:sample_count],
        denoise_t=denoise_t,
    )

    target_grid = _tokens_to_latent_grid(target_tokens[:sample_count].to(dtype=next(vae.parameters()).dtype)) / scaling_factor
    pred_grid = _tokens_to_latent_grid(pred_tokens.to(dtype=next(vae.parameters()).dtype)) / scaling_factor
    noisy_grid = _tokens_to_latent_grid(noisy_tokens.to(dtype=next(vae.parameters()).dtype)) / scaling_factor
    denoised_grid = _tokens_to_latent_grid(denoised_tokens.to(dtype=next(vae.parameters()).dtype)) / scaling_factor

    decoded_target = vae.decode(target_grid).float().clamp(-1, 1)
    decoded_pred = vae.decode(pred_grid).float().clamp(-1, 1)
    decoded_noisy = vae.decode(noisy_grid).float().clamp(-1, 1)
    decoded_denoised = vae.decode(denoised_grid).float().clamp(-1, 1)

    from torchvision.utils import make_grid, save_image

    image_dir = Path(config.experiment.output_dir) / "validation_flow_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    target_img = (decoded_target + 1.0) / 2.0
    pred_img = (decoded_pred + 1.0) / 2.0
    noisy_img = (decoded_noisy + 1.0) / 2.0
    denoised_img = (decoded_denoised + 1.0) / 2.0
    save_image(target_img, image_dir / f"step-{global_step:08d}-target.png")
    save_image(pred_img, image_dir / f"step-{global_step:08d}-pure_flow_pred.png")
    save_image(noisy_img, image_dir / f"step-{global_step:08d}-noisy_t{denoise_t:.2f}.png")
    save_image(denoised_img, image_dir / f"step-{global_step:08d}-velocity_denoised_t{denoise_t:.2f}.png")
    grid = make_grid(torch.stack([target_img, pred_img], dim=1).flatten(0, 1), nrow=2)
    save_image(grid, image_dir / f"step-{global_step:08d}-target_pure_flow_grid.png")
    denoise_grid = make_grid(
        torch.stack([target_img, noisy_img, denoised_img, pred_img], dim=1).flatten(0, 1),
        nrow=4,
    )
    save_image(denoise_grid, image_dir / f"step-{global_step:08d}-target_noisy_denoised_sample_grid.png")
    logger.info(f"Saved pure-flow validation images to {image_dir}")


def save_flow_head(model, accelerator, config, global_step):
    output_dir = Path(config.experiment.output_dir)
    save_dir = output_dir / f"flow_head-{global_step}"
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        flow_state = {
            key.removeprefix("flow_head."): value.cpu()
            for key, value in state_dict.items()
            if key.startswith("flow_head.")
        }
        prefixed_flow_state = {f"flow_head.{key}": value for key, value in flow_state.items()}
        torch.save(flow_state, save_dir / "flow_head.pt")
        torch.save(prefixed_flow_state, save_dir / "flow_head_prefixed.pt")
        torch.save(
            {
                "flow_head": flow_state,
                "model": OmegaConf.to_container(config.model, resolve=True),
                "global_step": global_step,
            },
            save_dir / "pure_flow_pretrainer.pt",
        )
        with (save_dir / "metadata.json").open("w") as f:
            json.dump(
                {
                    "global_step": global_step,
                    "load_into_full_model": "model.load_state_dict(torch.load('flow_head_prefixed.pt'), strict=False)",
                    "image_tokens_per_img": int(config.model.image_tokens_per_img),
                    "image_latent_dim": int(config.model.image_latent_dim),
                    "hidden_size": int(config.model.hidden_size),
                    "flow_width": int(config.model.flow_width),
                    "flow_depth": int(config.model.flow_depth),
                    "flow_time_scale": float(config.model.get("flow_time_scale", 1000.0)),
                    "flow_sample_method": config.model.get("flow_sample_method", "heun"),
                    "pure_flow_condition": "disabled; z is all zeros and cond_embed is frozen during pretraining",
                },
                f,
                indent=2,
            )


def main():
    config = get_config()
    config.experiment.output_dir = os.path.join(config.experiment.output_dir, config.experiment.project)

    total_batch_size_per_gpu = int(config.training.batch_size)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = (int(config.training.total_batch_size) // total_batch_size_per_gpu) // num_processes
    if grad_accum < 1:
        raise ValueError("training.total_batch_size must be >= batch_size * num_processes")

    log_with = config.experiment.get("log_with", "wandb")
    if str(log_with).lower() in {"none", "false", "null"}:
        log_with = None

    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=config.training.mixed_precision,
        log_with=log_with,
        step_scheduler_with_optimizer=config.training.get("step_scheduler_with_optimizer", False),
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = total_batch_size_per_gpu
        accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = accelerator.gradient_accumulation_steps

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

    if config.training.get("enable_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    if accelerator.is_main_process and log_with is not None:
        log_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        log_config.pop("experiment.resume_from_checkpoint", None)
        accelerator.init_trackers(
            config.experiment.wandb_project,
            config=log_config,
            init_kwargs={"wandb": {"name": config.experiment.name}},
        )

    model = PureFlowHeadPretrainer(
        image_tokens_per_img=int(config.model.image_tokens_per_img),
        image_latent_dim=int(config.model.image_latent_dim),
        hidden_size=int(config.model.hidden_size),
        flow_width=int(config.model.flow_width),
        flow_depth=int(config.model.flow_depth),
        flow_time_scale=float(config.model.get("flow_time_scale", 1000.0)),
        flow_sample_method=config.model.get("flow_sample_method", "heun"),
        freeze_condition=bool(config.model.get("freeze_condition", True)),
    )
    pretrained_flow_head = config.model.get("pretrained_flow_head", None)
    if isinstance(pretrained_flow_head, str) and pretrained_flow_head.lower() in {"none", "null", "false", ""}:
        pretrained_flow_head = None
    if pretrained_flow_head:
        state = torch.load(pretrained_flow_head, map_location="cpu")
        if "flow_head" in state:
            state = state["flow_head"]
        if any(key.startswith("flow_head.") for key in state):
            state = {key.removeprefix("flow_head."): value for key, value in state.items()}
        missing, unexpected = model.flow_head.load_state_dict(state, strict=False)
        logger.info(f"Loaded pretrained flow_head: missing={missing}, unexpected={unexpected}")

    optimizer_config = config.optimizer.params
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(optimizer_config.learning_rate),
        betas=(float(optimizer_config.beta1), float(optimizer_config.beta2)),
        weight_decay=float(optimizer_config.weight_decay),
        eps=float(optimizer_config.epsilon),
    )
    lr_scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=int(config.lr_scheduler.params.warmup_steps),
        num_decay_steps=int(config.lr_scheduler.params.decay_steps),
        num_training_steps=int(config.training.max_train_steps),
        min_lr_ratio=float(config.lr_scheduler.params.min_lr_scale),
    )

    train_dataloader, val_dataloader = build_flow_latent_dataloaders(config)
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    global_step = 0
    resume_checkpoint_dir = None
    if config.experiment.get("resume_from_checkpoint", None) is not None:
        candidate = Path(config.experiment.resume_from_checkpoint)
        if candidate.exists():
            resume_checkpoint_dir = candidate
    if resume_checkpoint_dir:
        logger.info(f"Resuming from {resume_checkpoint_dir}")
        accelerator.load_state(resume_checkpoint_dir)
        metadata_file = resume_checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with metadata_file.open() as f:
                global_step = int(json.load(f).get("global_step", 0))

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        OmegaConf.save(config, Path(config.experiment.output_dir) / "config.yaml")

    total_batch_size = total_batch_size_per_gpu * accelerator.num_processes * accelerator.gradient_accumulation_steps
    logger.info("***** Running pure flow-head pretraining *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total image batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Image tokens per image = {config.model.image_tokens_per_img}")
    logger.info(f"  Latent dim = {config.model.image_latent_dim}")

    batch_time_m = AverageMeter()
    end = time.time()
    train_iter = iter(train_dataloader)
    model.train()

    while global_step < int(config.training.max_train_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        latents = batch["latents"].to(accelerator.device)
        with accelerator.accumulate(model):
            loss = model(latents)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                max_grad_norm = float(config.training.get("max_grad_norm", 0.0))
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            global_step += 1
            batch_time_m.update(time.time() - end)
            end = time.time()

            if global_step % int(config.experiment.log_every) == 0:
                global_loss = accelerator.reduce(loss.detach(), reduction="mean")
                logs = {
                    "train/loss_flow": global_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "samples/sec/gpu": accelerator.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val,
                    "batch_time": batch_time_m.val,
                }
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process:
                    logger.info(
                        f"Step: {global_step} | Loss: {global_loss.item():.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.6f} | Sec/Iter: {batch_time_m.val:.4f}"
                    )
                batch_time_m.reset()

            if global_step % int(config.experiment.save_every) == 0:
                save_path = Path(config.experiment.output_dir) / f"checkpoint-{global_step}"
                accelerator.save_state(save_path)
                if accelerator.is_main_process:
                    with (save_path / "metadata.json").open("w") as f:
                        json.dump({"global_step": global_step}, f, indent=2)
                save_flow_head(model, accelerator, config, global_step)

            if global_step % int(config.experiment.val_every) == 0:
                validate(model, val_dataloader, accelerator, global_step, config)

    accelerator.wait_for_everyone()
    save_flow_head(model, accelerator, config, "final")
    accelerator.end_training()


if __name__ == "__main__":
    main()
