import json
import os
from pathlib import Path
import re
import shutil
import sys
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers import AutoConfig, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


##################################################
#              config utils
##################################################
def get_config():
    argv = sys.argv[1:]
    config_path = None
    cleaned_argv = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise ValueError("--config requires a path")
            config_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
            continue
        cleaned_argv.append(arg)
        i += 1

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + cleaned_argv
        cli_conf = OmegaConf.from_cli()
    finally:
        sys.argv = old_argv

    config_path = config_path or cli_conf.get("config")
    if config_path is None:
        raise ValueError("Missing config path. Pass config=path.yaml or --config path.yaml")

    yaml_conf = OmegaConf.load(config_path)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              training utils
##################################################
def load_model_tokenizer(
    config: OmegaConf,
    logger=None,
    model_dtype: torch.dtype = torch.bfloat16,
):
    from models.modeling_model.image_backbone import validate_image_data_layout
    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM

    validate_image_data_layout(config)
    if model_dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError(
            "model_dtype must be torch.bfloat16 or torch.float32, "
            f"got {model_dtype}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_path,
        fix_mistral_regex=True,
    )
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

    if logger is not None:
        logger.info("Using the finalized selfless-flow model.")
        logger.info("Special tokens: %s", tokenizer.special_tokens_map)
        logger.info(
            "BOI token id: %s, EOI token id: %s, IMG_MASK token id: %s",
            config.model.boi_token_id,
            config.model.eoi_token_id,
            config.model.image_mask_token_id,
        )

    multimodal_config_keys = (
        "boi_token_id",
        "eoi_token_id",
        "image_mask_token_id",
        "image_tokens_per_img",
        "image_latent_dim",
        "image_flow_width",
        "image_flow_depth",
        "image_flow_num_sampling_steps",
        "image_flow_batch_mul",
        "image_flow_grad_checkpointing",
        "image_flow_time_scale",
        "image_flow_time_sampling",
        "image_flow_logit_mean",
        "image_flow_logit_std",
        "image_flow_time_eps",
        "image_flow_time_uniform_mix",
        "image_flow_solver",
        "image_input_noise_strength",
        "image_uncond_prob",
        "backbone_attention_output_gate",
    )

    model_config = AutoConfig.from_pretrained(
        config.model.model_path,
        trust_remote_code=True,
    )
    source_has_image_flow = hasattr(model_config, "image_flow_width")
    source_attention_gate = str(
        getattr(model_config, "backbone_attention_output_gate", "none")
    )
    model_config.mask_token_id = config.model.mask_token_id
    model_config.use_flex_attention = config.model.use_flex_attention
    model_config.eos_token_id = tokenizer.eos_token_id
    for key in multimodal_config_keys:
        value = config.model.get(key)
        if value is not None:
            setattr(model_config, key, value)

    if (
        hasattr(tokenizer, "im_end_token_id")
        and tokenizer.im_end_token_id is not None
    ):
        model_config.im_end_token_id = tokenizer.im_end_token_id
    else:
        try:
            im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            model_config.im_end_token_id = im_end_ids[0] if im_end_ids else None
        except Exception:
            model_config.im_end_token_id = None

    if config.training.from_scratch:
        if logger is not None:
            logger.info(
                "Initializing selfless-flow from config: %s",
                config.model.model_path,
            )
        model = Qwen3ForCausalLM(model_config).to(dtype=model_dtype)
    else:
        if logger is not None:
            logger.info(
                "Loading selfless-flow weights from: %s",
                config.model.model_path,
            )
        model = Qwen3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path=config.model.model_path,
            config=model_config,
            dtype=model_dtype,
            trust_remote_code=True,
        )
        if not source_has_image_flow:
            model.reset_image_modules()
        if (
            str(config.model.get("backbone_attention_output_gate", "none"))
            != "none"
            and source_attention_gate == "none"
        ):
            model.reset_backbone_attention_output_gates()

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    image_mask_token_id = getattr(model.config, "image_mask_token_id", None)
    if image_mask_token_id is not None and added_image_mask_token:
        with torch.no_grad():
            embed = model.model.embed_tokens.weight
            mask_token_id = int(model.config.mask_token_id)
            image_mask_token_id = int(image_mask_token_id)
            if (
                0 <= mask_token_id < embed.shape[0]
                and 0 <= image_mask_token_id < embed.shape[0]
                and mask_token_id != image_mask_token_id
            ):
                embed[image_mask_token_id].copy_(embed[mask_token_id])
                if logger is not None:
                    logger.info(
                        f"Initialized newly added image mask token id={image_mask_token_id} "
                        f"from text mask token id={mask_token_id}"
                    )

    if config.training.get("use_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if logger is not None:
            logger.info("Gradient checkpointing enabled")

    return model, tokenizer
    
    
def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


def rotate_checkpoints_for_save(
    output_dir: str | Path,
    checkpoints_total_limit: int,
    *,
    current_checkpoint_name: str,
    milestone_every_steps: int = 0,
) -> list[Path]:
    """Rotate ordinary checkpoints while permanently retaining milestones.

    In distributed saves, a non-main rank may create the destination directory
    before rank 0 scans the output directory.  Excluding the current name makes
    retention deterministic even if that happens or a prior write left an
    incomplete destination behind.  Positive multiples of
    ``milestone_every_steps`` never participate in rotation or consume one of
    the rolling slots.
    """

    limit = int(checkpoints_total_limit)
    if limit < 1:
        raise ValueError(f"checkpoints_total_limit must be positive, got {limit}")
    milestone_every_steps = int(milestone_every_steps)
    if milestone_every_steps < 0:
        raise ValueError(
            "milestone_every_steps must be non-negative, got "
            f"{milestone_every_steps}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def checkpoint_step(path: Path) -> int | None:
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        return int(match.group(1)) if match else None

    def is_milestone(step: int) -> bool:
        return (
            milestone_every_steps > 0
            and step > 0
            and step % milestone_every_steps == 0
        )

    current_match = re.fullmatch(r"checkpoint-(\d+)", current_checkpoint_name)
    if current_match is None:
        raise ValueError(
            "current_checkpoint_name must match checkpoint-<step>, got "
            f"{current_checkpoint_name!r}"
        )
    current_is_milestone = is_milestone(int(current_match.group(1)))

    rotating_checkpoints: list[tuple[int, Path]] = []
    for path in output_path.iterdir():
        step = checkpoint_step(path)
        if (
            not path.is_dir()
            or path.name == current_checkpoint_name
            or step is None
            or is_milestone(step)
        ):
            continue
        rotating_checkpoints.append((step, path))
    checkpoints = [
        path for _, path in sorted(rotating_checkpoints, key=lambda item: item[0])
    ]
    rolling_slots_before_save = limit if current_is_milestone else limit - 1
    num_to_remove = max(0, len(checkpoints) - rolling_slots_before_save)
    removing = checkpoints[:num_to_remove]
    for checkpoint in removing:
        shutil.rmtree(checkpoint)
    return removing


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)
    checkpoint_milestone_every = int(
        config.experiment.get("checkpoint_milestone_every", 0)
    )
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        rotate_checkpoints_for_save(
            output_dir,
            checkpoints_total_limit,
            current_checkpoint_name=save_path.name,
            milestone_every_steps=checkpoint_milestone_every,
        )

    # Hold non-main ranks outside accelerator.save_state until rank 0 has
    # finished retention.  Otherwise they can recreate or populate a directory
    # while rank 0 is deleting old checkpoints.
    accelerator.wait_for_everyone()

    # 这一步保存了：Model, Optimizer, LR Scheduler, Random States
    accelerator.save_state(save_path)

    if accelerator.is_main_process:
        meta_file = save_path / "metadata.json"
        metadata = {
            "global_step": global_step,
            "model_config": OmegaConf.to_container(config.model, resolve=True),
        }
        with open(meta_file, "w+") as f:
            json.dump(metadata, f, indent=4)
      
        
def save_hf_model(model, tokenizer, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    save_path = Path(output_dir) / f"hf_model-{global_step}"

    # 取出模型权重
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        tokenizer.save_pretrained(save_path)
    accelerator.wait_for_everyone()


##################################################
#              misc
##################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_selfless_mask(
    sigma: torch.Tensor,
    seq_len: int,
    device,
    *,
    input_ids: torch.Tensor | None = None,
    token_types: torch.Tensor | None = None,
    boi_token_id: int | None = None,
    image_uncond_rows: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    image_uncond_mask: torch.Tensor | None = None,
) -> torch.Tensor | BlockMask:
    """
    Selfless Attention mask — removes the diagonal (self-attention) from both streams.

    Both content and query streams use strict S_kv < S_q, meaning no position can
    attend to itself. This is the key difference from XLNet's selfish mask (S_kv <= S_q
    for content stream), which allows the diagonal shortcut.

    Args:
        sigma: Permutation sorting values, shape: (batch_size, seq_len)
        seq_len: Sequence length

    Returns:
        A dense disallow mask on NPU, or a BlockMask on CPU/CUDA.
    """

    B = sigma.shape[0]
    if tuple(sigma.shape) != (int(B), int(seq_len)):
        raise ValueError(
            f"sigma must have shape {(int(B), int(seq_len))}, "
            f"got {tuple(sigma.shape)}"
        )
    use_segments = segment_ids is not None
    if use_segments:
        if tuple(segment_ids.shape) != tuple(sigma.shape):
            raise ValueError(
                "segment_ids must align with sigma: "
                f"{tuple(segment_ids.shape)} != {tuple(sigma.shape)}"
            )
        segment_ids = segment_ids.to(device=device, dtype=torch.long)
    use_image_uncond = (
        image_uncond_rows is not None or image_uncond_mask is not None
    )
    if use_image_uncond:
        if token_types is None:
            raise ValueError(
                "token_types are required when image conditioning dropout is used."
            )
        token_types = token_types.to(device=device)
        if image_uncond_mask is not None:
            if not use_segments:
                raise ValueError(
                    "image_uncond_mask requires segment_ids"
                )
            if tuple(image_uncond_mask.shape) != tuple(sigma.shape):
                raise ValueError(
                    "image_uncond_mask must align with sigma: "
                    f"{tuple(image_uncond_mask.shape)} != {tuple(sigma.shape)}"
                )
            image_uncond_mask = image_uncond_mask.to(
                device=device, dtype=torch.bool
            )
            image_span_ids = segment_ids
        else:
            if (
                input_ids is None
                or boi_token_id is None
                or image_uncond_rows is None
            ):
                raise ValueError(
                    "input_ids, boi_token_id, and image_uncond_rows are "
                    "required for row-level image conditioning dropout."
                )
            input_ids = input_ids.to(device=device)
            image_uncond_rows = image_uncond_rows.to(
                device=device, dtype=torch.bool
            )
            image_span_ids = torch.cumsum(
                (input_ids == int(boi_token_id)).to(torch.long),
                dim=1,
            )

    # Build the visibility truth table once. Ascend consumes its dense inverse
    # directly, while CPU/CUDA reference paths retain the BlockMask contract
    # expected by torch flex_attention and the architecture tests.
    S_q = sigma.unsqueeze(-1)    # [B, S, 1]
    S_kv = sigma.unsqueeze(1)    # [B, 1, S]
    allowed = S_kv < S_q         # strict — no diagonal, no self-view
    if use_segments:
        q_seg = segment_ids.unsqueeze(-1)
        kv_seg = segment_ids.unsqueeze(1)
        allowed = allowed & (q_seg >= 0) & (kv_seg >= 0) & (q_seg == kv_seg)
    if use_image_uncond:
        if image_uncond_mask is not None:
            q_is_uncond_image = image_uncond_mask.unsqueeze(-1)
        else:
            q_is_uncond_image = (
                image_uncond_rows.view(B, 1)
                & (token_types == 1)
            ).unsqueeze(-1)
        kv_same_image_span = (
            (token_types.unsqueeze(1) == 1)
            & (image_span_ids.unsqueeze(1) == image_span_ids.unsqueeze(-1))
        )
        allowed = allowed & (~q_is_uncond_image | kv_same_image_span)
    if sigma.device.type == "npu":
        return (~allowed).unsqueeze(1)  # [B, 1, S, S], True = disallow

    def selfless_fn(b, h, q_idx, kv_idx):
        del h
        return allowed[b, q_idx, kv_idx]

    return create_block_mask(
        selfless_fn,
        B=B,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
