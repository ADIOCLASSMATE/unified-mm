import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple, Union
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, or_masks, and_masks
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
    from models.modeling_model.image_backbone import resolve_image_backbone_config
    from models.modeling_model.image_flow_position import (
        resolve_flow_head_position_config,
    )

    backbone = resolve_image_backbone_config(config)
    flow_position, _ = resolve_flow_head_position_config(config)
    if model_dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError(
            "model_dtype must be torch.bfloat16 or torch.float32, "
            f"got {model_dtype}"
        )
    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, fix_mistral_regex=True)
    mask_token = "<|mdm_mask|>"

    # 检查 tokenizer 是否已经有该 token
    if mask_token in tokenizer.get_vocab():
        # 如果存在，获取 id
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    else:
        # 如果不存在，则添加到 tokenizer
        tokenizer.add_special_tokens({"mask_token": f"{mask_token}"})
        # tokenizer.add_tokens([mask_token])
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)

    config.model.mask_token_id = mask_token_id

    # Register BOI/EOI tokens for multimodal (begin/end of image)
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

    # image_offset: where image codebook indices start in the unified input_ids space.
    # Keep the configured value so dual-head and unified-head ablations can share
    # the same pre-tokenized Arrow shards.
    unified_head = getattr(config.model, "unified_head", False)
    config.model.image_offset = getattr(config.model, "image_offset", None) or (len(tokenizer) if unified_head else 200000)

    if logger is not None:
        logger.info('special tokens : \n', tokenizer.special_tokens_map)
        logger.info(f'BOI token id: {config.model.boi_token_id}, '
                    f'EOI token id: {config.model.eoi_token_id}, '
                    f'IMG_MASK token id: {config.model.image_mask_token_id}')
    
    
    project = config.experiment.project
    if "sdar" in project.lower():
        from models.modeling_model.modeling_sdar import SDARForCausalLM
        model_class = SDARForCausalLM
    elif "llada" in project.lower():
        from models.modeling_model.modeling_llada import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "dream" in project.lower():
        from models.modeling_model.modeling_dream import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "mad" in project.lower():
        from models.modeling_model.modeling_mad import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "dam" in project.lower():
        from models.modeling_model.modeling_dam import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "pnts" in project.lower():
        from models.modeling_model.modeling_pnts import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "xlnet" in project.lower():
        from models.modeling_model.modeling_xlnet import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "flow" in project.lower():
        from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "selfless" in project.lower() or "sigma" in project.lower():
        from models.modeling_model.modeling_selfless import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "causal" in project.lower() or "ar" in project.lower():
        from models.modeling_model.modeling_ar import Qwen3ForCausalLM
        model_class = Qwen3ForCausalLM
    elif "omega" in project.lower():
        raise ValueError(
            f"Project name '{project}' contains 'omega'. "
            f"OMEGA has been renamed to Selfless Attention. "
            f"Please rename your project to use 'selfless' instead."
        )
    else:
        raise ValueError
    
    
    multimodal_config_keys = (
        "image_vocab_size", "image_offset", "lambda_image", "lambda_text",
        "boi_token_id", "eoi_token_id", "image_mask_token_id", "unified_head", "image_tokens_per_img",
        "image_latent_dim", "continuous_image_latents",
        "image_generation_head_type", "image_flow_width", "image_flow_depth",
        "image_flow_num_sampling_steps", "image_flow_batch_mul",
        "image_flow_grad_checkpointing", "image_flow_time_scale",
        "image_flow_time_sampling", "image_flow_logit_mean", "image_flow_logit_std",
        "image_flow_time_eps", "image_flow_time_uniform_mix", "image_flow_solver",
        "image_flow_mlp_ratio", "image_flow_head_arch", "image_flow_head_variant", "image_flow_zero_init_gate",
        "image_flow_latent_mixer_heads", "image_flow_latent_mixer_dropout",
        "image_flow_latent_mixer_zero_init_gate",
        "image_flow_position_variant",
        "image_flow_query_position_mode",
        "image_flow_context_position_mode",
        "image_flow_rope_mode",
        "image_flow_rope_axis_dims",
        "image_flow_rope_rotate_value",
        "image_input_noise_strength", "image_input_noise_strength_std",
        "image_input_noise_strength_min", "image_input_noise_strength_max",
        "image_uncond_prob",
        "image_token_embedder_init_mode",
        "image_token_embedder_latent_rms",
        "image_backbone_variant",
    )

    if config.training.from_scratch:
        if logger is not None:
            logger.info(f"Initializing model from scratch (Random Weights) based on config from: {config.model.model_path}")
        # Initialize model
        model_config = AutoConfig.from_pretrained(config.model.model_path, trust_remote_code=True)
        # 更新 model.config
        model_config.mask_token_id = config.model.mask_token_id
        model_config.use_flex_attention = config.model.use_flex_attention
        model_config.eos_token_id = tokenizer.eos_token_id
        # Propagate multimodal config values from YAML to model config
        for key in multimodal_config_keys:
            val = config.model.get(key)
            if val is not None:
                setattr(model_config, key, val)
        # 设置 im_end_token_id
        if hasattr(tokenizer, 'im_end_token_id') and tokenizer.im_end_token_id is not None:
            model_config.im_end_token_id = tokenizer.im_end_token_id
        else:
            # 尝试通过编码获取 <|im_end|> 的 token ID
            try:
                im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
                if len(im_end_ids) > 0:
                    model_config.im_end_token_id = im_end_ids[0]
            except:
                model_config.im_end_token_id = None
        model = model_class(model_config)
        
        model = model.to(dtype=model_dtype)
    else:
        if logger is not None:
            logger.info(f"Loading pretrained model weights from: {config.model.model_path}")
        model_config = AutoConfig.from_pretrained(config.model.model_path, trust_remote_code=True)
        model_config.mask_token_id = config.model.mask_token_id
        model_config.use_flex_attention = config.model.use_flex_attention
        model_config.eos_token_id = tokenizer.eos_token_id
        for key in multimodal_config_keys:
            val = config.model.get(key)
            if val is not None:
                setattr(model_config, key, val)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=config.model.model_path,
            config=model_config,
            dtype=model_dtype,
            trust_remote_code=True
        )

        if (
            "flow" in project.lower()
            or config.model.get("continuous_image_latents", False)
        ):
            if len(tokenizer) > model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
        # Unified head: expand embeddings + lm_head to include image vocab
        elif config.model.get("unified_head", False):
            model.unified_head = True
            model.model.unified_head = True
            model.model.image_embed_tokens = None
            image_offset = model.config.image_offset
            image_vocab_size = model.config.image_vocab_size
            total_vocab = image_offset + image_vocab_size
            # resize_token_embeddings expands both embed_tokens and lm_head (tied)
            model.resize_token_embeddings(total_vocab)
            model.image_lm_head = None
            if logger is not None:
                logger.info(f"Resized embeddings + lm_head to {total_vocab} (unified)")
        else:
            # Dual-head: resize text embedding for any new special tokens
            model.unified_head = False
            model.model.unified_head = False
            if len(tokenizer) > model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))

            # Resize image embedding/lm_head to match config (from_pretrained uses default 16384)
            target_vocab = model.config.image_vocab_size
            current_vocab = model.model.image_embed_tokens.weight.shape[0]
            if current_vocab != target_vocab:
                hidden_dim = model.model.image_embed_tokens.weight.shape[1]
                device = model.model.image_embed_tokens.weight.device
                dtype = model.model.image_embed_tokens.weight.dtype

                new_embed = nn.Embedding(target_vocab, hidden_dim)
                n_copy = min(current_vocab, target_vocab)
                new_embed.weight.data[:n_copy] = model.model.image_embed_tokens.weight.data[:n_copy]
                model.model.image_embed_tokens = new_embed.to(device=device, dtype=dtype)
                model.model.image_vocab_size = target_vocab

                new_head = nn.Linear(hidden_dim, target_vocab, bias=False)
                new_head.weight.data[:n_copy] = model.image_lm_head.weight.data[:n_copy]
                model.image_lm_head = new_head.to(device=device, dtype=dtype)
                model.image_vocab_size = target_vocab

                # Re-tie image pair
                model.image_lm_head.weight = model.model.image_embed_tokens.weight

            if logger is not None:
                logger.info("Dual-head mode: text lm_head tied to embed_tokens, "
                            "image_lm_head tied to image_embed_tokens")
        # 设置 im_end_token_id
        if hasattr(tokenizer, 'im_end_token_id') and tokenizer.im_end_token_id is not None:
            model.config.im_end_token_id = tokenizer.im_end_token_id
        else:
            # 尝试通过编码获取 <|im_end|> 的 token ID
            try:
                im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
                if len(im_end_ids) > 0:
                    model.config.im_end_token_id = im_end_ids[0]
            except:
                model.config.im_end_token_id = None

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

    if hasattr(model, "image_token_embedder"):
        actual_variant = str(model.image_token_embedder.backbone_variant)
        if actual_variant != backbone.variant:
            raise ValueError(
                "image_backbone_variant was not propagated into the loaded model: "
                f"expected={backbone.variant!r}, actual={actual_variant!r}"
            )
    if flow_position is not None and hasattr(model, "image_flow_head"):
        actual_flow_variant = str(model.image_flow_head.net.position_variant)
        if actual_flow_variant != flow_position.variant:
            raise ValueError(
                "image_flow_position_variant was not propagated into the "
                f"loaded model: expected={flow_position.variant!r}, "
                f"actual={actual_flow_variant!r}"
            )
    
    # 启用 Gradient Checkpointing
    if config.training.get("use_gradient_checkpointing", True):
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


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        # 使用 glob 或 listdir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        
        def get_step(name):
            # 尝试从 "checkpoint-1000" 中提取 "1000"
            match = re.search(r"checkpoint-(\d+)", name)
            if match:
                return int(match.group(1))
            return -1 # 无法解析的文件夹排在最前面或被忽略
        
        checkpoints = [c for c in checkpoints if get_step(c) != -1]
        checkpoints = sorted(checkpoints, key=get_step)

        if len(checkpoints) >= checkpoints_total_limit:
            # 删除最旧的，保留最近的 (total_limit - 1) 个，以便腾出位置给新的
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[:num_to_remove]
            
            for rm in removing_checkpoints:
                rm_path = os.path.join(output_dir, rm)
                shutil.rmtree(rm_path)
        
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    # 这一步保存了：Model, Optimizer, LR Scheduler, Random States
    accelerator.save_state(save_path)

    if accelerator.is_main_process:
        meta_file = save_path / "metadata.json"
        metadata = {
            "global_step": global_step,
            "model_config": OmegaConf.to_container(config.model, resolve=True),
        }
        if str(config.experiment.get("ablation_phase", "screen")) == "confirmation":
            metadata["confirmation_provenance"] = {
                "path": str(config.experiment.confirmation_provenance_path),
                "sha256": str(config.experiment.confirmation_provenance_sha256),
                "declaration_sha256": str(
                    config.experiment.confirmation_protocol.declaration_sha256
                ),
            }
        elif str(config.experiment.get("ablation_phase", "")) == "mask_position_q_factor":
            declaration = config.experiment.q_factor_protocol
            metadata["q_factor_provenance"] = {
                "path": str(config.experiment.q_factor_provenance_path),
                "sha256": str(config.experiment.q_factor_provenance_sha256),
                "declaration_sha256": str(declaration.declaration_sha256),
                "study_manifest_sha256": str(declaration.study_manifest_sha256),
                "config_contract_sha256": str(declaration.config_contract_sha256),
                "source_manifest_sha256": str(declaration.source_manifest_sha256),
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
        if str(config.experiment.get("ablation_phase", "screen")) == "confirmation":
            provenance_path = Path(str(config.experiment.confirmation_provenance_path))
            shutil.copy2(provenance_path, save_path / provenance_path.name)
        elif str(config.experiment.get("ablation_phase", "")) == "mask_position_q_factor":
            provenance_path = Path(str(config.experiment.q_factor_provenance_path))
            shutil.copy2(provenance_path, save_path / provenance_path.name)
    tokenizer.save_pretrained(save_path)


def register_reasoning_tokens(tokenizer, model):
    """
    注册 reasoning token，保证 decode 时可见。
    使用 HuggingFace 官方 API，不直接修改内部方法。
    
    Args:
        tokenizer: tokenizer 对象
        model: 预训练模型
    """
    SPECIAL_TOKENS = {
        "start_of_reasoning": "<|Reasoning|>",
        "end_of_reasoning": "<|/Reasoning|>",
        "start_of_response": "<|Response|>",
        "end_of_response": "<|/Response|>",
    }
    special_tokens_list = list(SPECIAL_TOKENS.values())

    print(f"Old tokenizer length: {len(tokenizer)}")

    # 1. 检查哪些 token 没有
    tokens_to_add = [tok for tok in special_tokens_list if tokenizer.convert_tokens_to_ids(tok) == tokenizer.unk_token_id]

    if tokens_to_add:
        num_added = tokenizer.add_tokens(tokens_to_add, special_tokens=False)
        print(f"Added {num_added} new special tokens: {tokens_to_add}")
    else:
        print("All special tokens already exist in tokenizer vocab.")

    # 2. 保存 id
    SPECIAL_TOKEN_IDS = {name: tokenizer.convert_tokens_to_ids(tok) for name, tok in SPECIAL_TOKENS.items()}

    print("Registered special tokens:")
    for name, tid in SPECIAL_TOKEN_IDS.items():
        print(f"  {name}: {tid} -> {tokenizer.convert_ids_to_tokens(tid)}")

    # 3. 检查模型 embedding 大小
    input_emb_size = model.get_input_embeddings().weight.shape[0]
    lm_head_size = model.get_output_embeddings().weight.shape[0] if model.get_output_embeddings() is not None else input_emb_size
    new_vocab_size = len(tokenizer)

    print(f"Embedding size: {input_emb_size}, LM head size: {lm_head_size}, Tokenizer size: {new_vocab_size}")

    if new_vocab_size > input_emb_size or new_vocab_size > lm_head_size:
        model.resize_token_embeddings(new_vocab_size)
        print(f"Resized embeddings to {new_vocab_size}")
    else:
        print("No resize needed, embedding layers are already large enough.")

    # 4. 给 tokenizer 添加属性
    for name, tok in SPECIAL_TOKENS.items():
        setattr(tokenizer, name, tok)
        setattr(tokenizer, f"{name}_id", SPECIAL_TOKEN_IDS[name])

    print(f"Final tokenizer length: {len(tokenizer)}")
    
    
##################################################
#                   loss util
##################################################
def reverse_kl_loss(
    logits_masked: torch.Tensor,        # [B, L, V]
    logits_clean: torch.Tensor,        # [B, L, V]
    loss_mask: torch.Tensor = None,     # [B, L], 1=compute, 0=ignore
    temperature: float = 1.0,
):
    """
    Reverse KL:
        KL(q_masked || p_clean)

    masked logits  : [B, L, V]
    clean logits   : [B, L, V]

    By default:
    - gradients flow only through masked branch
    - clean branch is treated as teacher (detached)
    """

    # --- build distributions ---
    log_q_masked = F.log_softmax(logits_masked / temperature, dim=-1)
    log_p_clean  = F.log_softmax(logits_clean  / temperature, dim=-1)

    # teacher should not receive gradients
    log_p_clean = log_p_clean.detach()

    q_masked = log_q_masked.exp()    # [B, L, V]

    # --- reverse KL: sum_y q(y) [log q(y) - log p(y)] ---
    # shape: [B, L]
    rev_kl = (q_masked * (log_q_masked - log_p_clean)).sum(dim=-1)

    # --- masking ---
    if loss_mask is not None:
        rev_kl = rev_kl * loss_mask
        loss = rev_kl.sum() / (loss_mask.sum() + 1e-8)
    else:
        loss = rev_kl.mean()

    return loss


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


##################################################
#              llada_config
##################################################
from transformers import AutoConfig
import copy

# 不同模型规模的配置参数
MODEL_CONFIG_MAP = {
    16: {
        "n_layers": 2,
        "d_model": 64,
        "n_heads": 4,
        "n_kv_heads": 1,
        "mlp_hidden_size": 128,
        "vocab_size": 126464
    },
    71: {
        "n_layers": 6,
        "d_model": 256,
        "n_heads": 4,
        "n_kv_heads": 1,
        "mlp_hidden_size": 1024,
    },
    1678: {
        "n_layers": 22,
        "d_model": 2048,
        "n_heads": 32,
        "n_kv_heads": 32,
        "mlp_hidden_size": 5632,
    },
    426: {
        "n_layers": 16,
        "d_model": 1024,
        "n_heads": 16,
        "n_kv_heads": 16,
        "mlp_hidden_size": 2048,
    },
    
}

def get_config_by_model_size(model_path: str, model_size_key: str):
    if model_size_key not in MODEL_CONFIG_MAP:
        raise ValueError(f"Unknown model size '{model_size_key}'. Available: {list(MODEL_CONFIG_MAP.keys())}")

    base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config = copy.deepcopy(base_config)

    for key, value in MODEL_CONFIG_MAP[model_size_key].items():
        setattr(config, key, value)

    config.n_kv_heads = config.n_heads

    return config

def get_selfless_mask(
    sigma: torch.Tensor,
    seq_len: int,
    device,
    *,
    input_ids: torch.Tensor | None = None,
    token_types: torch.Tensor | None = None,
    boi_token_id: int | None = None,
    image_uncond_rows: torch.Tensor | None = None,
) -> BlockMask:
    """
    Selfless Attention mask — removes the diagonal (self-attention) from both streams.

    Both content and query streams use strict S_kv < S_q, meaning no position can
    attend to itself. This is the key difference from XLNet's selfish mask (S_kv <= S_q
    for content stream), which allows the diagonal shortcut.

    Args:
        sigma: Permutation sorting values, shape: (batch_size, seq_len)
        seq_len: Sequence length

    Returns:
        BlockMask for selfless attention.
    """

    B = sigma.shape[0]
    use_image_uncond = image_uncond_rows is not None
    if use_image_uncond:
        if input_ids is None or token_types is None or boi_token_id is None:
            raise ValueError(
                "input_ids, token_types, and boi_token_id are required when image_uncond_rows is provided."
            )
        input_ids = input_ids.to(device=device)
        token_types = token_types.to(device=device)
        image_uncond_rows = image_uncond_rows.to(device=device, dtype=torch.bool)
        image_span_ids = torch.cumsum(
            (input_ids == int(boi_token_id)).to(torch.long),
            dim=1,
        )

    def selfless_fn(b, h, q_idx, kv_idx):
        S_q = sigma[b, q_idx]
        S_kv = sigma[b, kv_idx]
        allowed = S_kv < S_q  # strict — no diagonal, no self-view
        if not use_image_uncond:
            return allowed

        q_is_uncond_image = image_uncond_rows[b] & (token_types[b, q_idx] == 1)
        kv_same_image_span = (
            (token_types[b, kv_idx] == 1)
            & (image_span_ids[b, kv_idx] == image_span_ids[b, q_idx])
        )
        return allowed & (~q_is_uncond_image | kv_same_image_span)

    return create_block_mask(selfless_fn, B=B, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
