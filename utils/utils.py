import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple, Union
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, or_masks, and_masks
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


##################################################
#              config utils
##################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
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
def load_model_tokenizer(config: OmegaConf, logger=None):
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

    if boi_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([boi_token, eoi_token], special_tokens=True)

    config.model.boi_token_id = tokenizer.convert_tokens_to_ids(boi_token)
    config.model.eoi_token_id = tokenizer.convert_tokens_to_ids(eoi_token)

    # image_offset: where image codebook indices start in the unified input_ids space
    # For unified_head: offset = len(tokenizer) so image tokens sit right after text vocab
    # For dual_head: use a large safe offset (200000)
    unified_head = getattr(config.model, "unified_head", False)
    if unified_head:
        config.model.image_offset = len(tokenizer)
    else:
        config.model.image_offset = getattr(config.model, "image_offset", 200000)

    if logger is not None:
        logger.info('special tokens : \n', tokenizer.special_tokens_map)
        logger.info(f'BOI token id: {config.model.boi_token_id}, '
                    f'EOI token id: {config.model.eoi_token_id}')
    
    
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
    
    
    if config.training.from_scratch:
        if logger is not None:
            logger.info(f"Initializing model from scratch (Random Weights) based on config from: {config.model.model_path}")
        # Initialize model
        model_config = AutoConfig.from_pretrained(config.model.model_path, trust_remote_code=True)
        # 更新 model.config
        model_config.mask_token_id = config.model.mask_token_id
        model_config.use_flex_attention = config.model.use_flex_attention
        model_config.eos_token_id = tokenizer.eos_token_id
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
        
        model = model.to(dtype=torch.bfloat16)
    else:
        if logger is not None:
            logger.info(f"Loading pretrained model weights from: {config.model.model_path}")
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=config.model.model_path,
            dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        model.config.mask_token_id = config.model.mask_token_id
        model.config.use_flex_attention = config.model.use_flex_attention
        model.config.eos_token_id = tokenizer.eos_token_id
        # Propagate multimodal config values from YAML to model config
        for key in ("image_vocab_size", "image_offset", "lambda_image",
                     "boi_token_id", "eoi_token_id", "unified_head"):
            val = config.model.get(key)
            if val is not None:
                setattr(model.config, key, val)

        # Unified head: expand lm_head to include image vocab after text vocab
        if config.model.get("unified_head", False):
            model.unified_head = True
            image_offset = model.config.image_offset
            image_vocab_size = model.config.image_vocab_size
            total_vocab = image_offset + image_vocab_size
            old_weight = model.lm_head.weight.data  # [151936, hidden_dim]
            hidden_dim = old_weight.shape[1]

            new_lm_head = torch.nn.Linear(hidden_dim, total_vocab, bias=False)
            new_lm_head.weight.data[:old_weight.shape[0]] = old_weight
            # Random init image portion
            new_lm_head.weight.data[old_weight.shape[0]:] = old_weight.mean() + \
                torch.randn(total_vocab - old_weight.shape[0], hidden_dim) * old_weight.std()
            new_lm_head = new_lm_head.to(dtype=old_weight.dtype, device=old_weight.device)
            model.lm_head = new_lm_head
            model.image_lm_head = None
            if logger is not None:
                logger.info(f"Expanded lm_head: {old_weight.shape[0]} -> {total_vocab}")
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
        with open(meta_file, "w+") as f:
            json.dump({
                "global_step": global_step,
                "model_config": config.model.to_dict() if hasattr(config.model, "to_dict") else {}
            }, f, indent=4)
      
        
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




def get_AR_attention_mask(seq_len: int, B=None, device="cuda") -> BlockMask:
    """
    获取 AR attention mask，用于 causal attention。
    
    Args:
        v_sample: 序列采样的转移速率，速率较大的不能atten到速率较小的位置, shape: (batch_size, seq_len+1)
        v_sample的第一个值必须是1.1，表示任何token都要能atten到这个位置，防止attention score全为0
        seq_len: 序列长度
        
    Returns:
        BlockMask 对象，表示 attention mask。
    """

    def causal_mask(b, h, q_idx, kv_idx):
        return kv_idx <= q_idx # True表示要atten的部分
        
    return create_block_mask(causal_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

def get_full_attention_mask(seq_len: int, B=None, device="cuda") -> BlockMask:
    
    def full_mask(b, h, q_idx, kv_idx):
        return torch.tensor(True, device=device)
        
    return create_block_mask(full_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

def get_AR_attention_mask_4eage(seq_len: int, device=None) -> torch.Tensor:
    """
    返回标准 causal attention mask:
    shape: (1, seq_len, seq_len)
      mask[q, k] = 0    if k <= q
      mask[q, k] = -inf if k > q
    """
    # causal mask: upper-triangular part is True (should be masked)
    # torch.triu: diagonal offset = 1 means strictly k > q
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    # Convert True → -inf, False → 0
    attention_mask = torch.zeros(seq_len, seq_len, device=device)
    attention_mask.masked_fill_(causal_mask, float("-inf"))

    # Add batch dimension (optional)
    return attention_mask.unsqueeze(0)



def get_selfless_mask(v_sample: torch.Tensor, seq_len: int, device) -> BlockMask:
    """
    Selfless Attention mask — removes the diagonal (self-attention) from both streams.

    Both content and query streams use strict v_kv > v_q, meaning no position can
    attend to itself. This is the key difference from XLNet's selfish mask (v_kv >= v_q
    for content stream), which allows the diagonal shortcut.

    Args:
        v_sample: Permutation sorting values, shape: (batch_size, seq_len)
        seq_len: Sequence length

    Returns:
        BlockMask for selfless attention.
    """

    B = v_sample.shape[0]
    def selfless_fn(b, h, q_idx, kv_idx):
        v_q = v_sample[b, q_idx]
        v_kv = v_sample[b, kv_idx]
        return v_kv > v_q  # strict — no diagonal, no self-view

    return create_block_mask(selfless_fn, B=B, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def get_selfless_ar_mask(seq_len: int, B=None, device="cuda") -> BlockMask:
    """
    Selfless AR attention mask — strict causal with no self-attention.
    mask[q, k] = 0    if k < q  (strict: diagonal excluded)
    mask[q, k] = -inf if k >= q
    """
    def causal_mask(b, h, q_idx, kv_idx):
        return kv_idx < q_idx  # strict — no diagonal

    return create_block_mask(causal_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def get_xlnet_mask(v_sample: torch.Tensor, seq_len: int, device) -> BlockMask:
    """
    获取 xlnet attention mask。
    
    Args:
        v_sample: 序列采样的转移速率，速率较大的不能atten到速率较小的位置, shape: (batch_size, seq_len+1)
        v_sample的第一个值必须是1.1，表示任何token都要能atten到这个位置，防止attention score全为0
        seq_len: 序列长度
        
    Returns:
        BlockMask 对象，表示 attention mask。
    """

    B = v_sample.shape[0]
    def query_attention_mask(b, h, q_idx, kv_idx):
        v_q = v_sample[b, q_idx] # 当前q的采样速率
        v_kv = v_sample[b, kv_idx] # 当前kv的采样速率
        return v_kv > v_q # True表示要atten的部分
    
    def kv_attention_mask(b, h, q_idx, kv_idx):
        v_q = v_sample[b, q_idx] # 当前q的采样速率
        v_kv = v_sample[b, kv_idx] # 当前kv的采样速率
        return v_kv >= v_q # True表示要atten的部分
    
    # def prompt_mask(b, h, q_idx, kv_idx):
    #     v_kv = v_sample[b, kv_idx] # 当前kv的采样速率
    #     return v_kv > 1.0 # True表示为prompt位置，必须atten
    
    # def prompt_causal_mask(b, h, q_idx, kv_idx):
    #     return q_idx > kv_idx
    
    # prompt_combined_mask = and_masks(prompt_mask, prompt_causal_mask)
    # combined_mask = or_masks(selfless_fn, prompt_combined_mask)
        
    return create_block_mask(query_attention_mask, B=B, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device), create_block_mask(kv_attention_mask, B=B, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

def get_xlnet_mask_ar(seq_len: int, device) -> BlockMask:
    """
    获取 xlnet attention mask。
    
    Args:
        v_sample: 序列采样的转移速率，速率较大的不能atten到速率较小的位置, shape: (batch_size, seq_len+1)
        v_sample的第一个值必须是1.1，表示任何token都要能atten到这个位置，防止attention score全为0
        seq_len: 序列长度
        
    Returns:
        BlockMask 对象，表示 attention mask。
    """

    def query_attention_mask(b, h, q_idx, kv_idx):

        return kv_idx < q_idx # True表示要atten的部分
    
    def kv_attention_mask(b, h, q_idx, kv_idx):

        return kv_idx <= q_idx # True表示要atten的部分
        
    return create_block_mask(query_attention_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device), create_block_mask(kv_attention_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
