"""Controlled Qwen3/KL16 Show-o2-style ablation, with one optional SigLIP branch.

This is a full-image rectified-flow model, not the retired masked-token target.
Text scores use either shifted content states or same-position text queries.
Image velocities always use content states and refresh the entire model at
every ODE call, including in the text two-stream ablation.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.checkpoint import checkpoint

# Import the project's CPU/Ascend compatibility setup before HF vision modules.
from .modeling_selfless_flow import (
    Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel, Qwen3Attention,
    Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, compiled_flex_attention,
    _to_bool_atten_mask, build_row_col_position_ids,
)
from .modeling_single_stream_text_ar import _materialize_allowed_mask
from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.modeling_outputs import CausalLMOutputWithPast


class Showo2UnifiedConfig(Qwen3Config):
    model_type = "showo2_unified_qwen3"


AutoConfig.register(Showo2UnifiedConfig.model_type, Showo2UnifiedConfig)


def image_block_mask(input_ids, token_types, boi_token_id):
    # Reuse BOI as the time carrier to retain B's 512-position image layout.
    return token_types.eq(1) | input_ids.eq(int(boi_token_id))


def omni_allowed_mask(input_ids, token_types, *, boi_token_id, segment_ids=None,
                      image_uncond_rows=None, image_uncond_mask=None):
    """True means visible: causal text/block order, bidirectional inside images."""
    b, length = input_ids.shape
    valid = token_types.ne(3)
    if segment_ids is None:
        segment_ids = torch.zeros_like(input_ids)
    valid = valid & segment_ids.ge(0)
    block = image_block_mask(input_ids, token_types, boi_token_id)
    starts = block & ~F.pad(block[:, :-1], (1, 0), value=False)
    block_ids = starts.long().cumsum(-1)
    same_image = (block.unsqueeze(2) & block.unsqueeze(1)
                  & block_ids.unsqueeze(2).eq(block_ids.unsqueeze(1)))
    pos = torch.arange(length, device=input_ids.device)
    allowed = (pos[None, :] <= pos[:, None]).unsqueeze(0) | same_image
    allowed = (allowed & valid.unsqueeze(1) & valid.unsqueeze(2)
               & segment_ids.unsqueeze(1).eq(segment_ids.unsqueeze(2)))
    dropped_queries = torch.zeros_like(block)
    if image_uncond_rows is not None:
        dropped_queries |= block & image_uncond_rows[:, None]
    if image_uncond_mask is not None:
        # A span's BOI must lose conditioning together with its latent tokens.
        dropped = image_uncond_mask.bool()
        dropped_queries |= block & (dropped | F.pad(dropped[:, 1:], (0, 1)))
    allowed &= ~(dropped_queries.unsqueeze(2) & ~block.unsqueeze(1))
    return allowed


def attention_from_allowed(allowed):
    if allowed.device.type == "npu":
        return (~allowed).unsqueeze(1)
    b, q, k = allowed.shape
    def mask_mod(batch, head, qi, ki):
        return allowed[batch, qi, ki]
    return create_block_mask(mask_mod, B=b, H=None, Q_LEN=q, KV_LEN=k,
                             device=allowed.device)


def text_query_allowed_mask(content_allowed, token_types):
    """Text queries read earlier content only; image queries are unused.

    Intersect with the S2 content mask to retain packed-document isolation,
    padding and CFG. Earlier image content is fully bidirectional within its
    own block, so a caption query can still observe the entire preceding image.
    """
    length = token_types.shape[1]
    positions = torch.arange(length, device=token_types.device)
    earlier = positions[None, :] < positions[:, None]
    text = token_types.eq(0) | token_types.eq(2)
    return content_allowed & earlier[None] & text.unsqueeze(-1)


def _prepared_mask(mask):
    return _to_bool_atten_mask(mask) if isinstance(mask, torch.Tensor) else mask


def _checkpoint(module, *args, enabled):
    if enabled and torch.is_grad_enabled():
        return checkpoint(module, *args, use_reentrant=False)
    return module(*args)


def _zero_parameters(module, reference):
    zero = reference.reshape(-1)[0].float() * 0.0
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.numel():
            zero = zero + parameter.reshape(-1)[0].float() * 0.0
    return zero


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):
        freq = torch.exp(-math.log(10000) * torch.arange(128, device=t.device).float() / 128)
        phase = t.float().unsqueeze(-1) * freq
        return self.mlp(torch.cat([phase.cos(), phase.sin()], -1).to(self.mlp[0].weight.dtype))


class SiglipAttention(nn.Module):
    """Official SigLIP parameter layout with the repository's NPU fused attention."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x, mask):
        b, n, d = x.shape
        q, k, v = [p(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj)]
        y = compiled_flex_attention(q, k, v, mask, (d // self.heads) ** -0.5, False)
        return self.out_proj(y.transpose(1, 2).reshape(b, n, d))


class SiglipBlock(nn.Module):
    def __init__(self, width, intermediate, heads):
        super().__init__()
        self.self_attn = SiglipAttention(width, heads)
        self.layer_norm1 = nn.LayerNorm(width, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(width, intermediate)
        self.mlp.fc2 = nn.Linear(intermediate, width)

    def forward(self, x, mask):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp.fc2(F.gelu(self.mlp.fc1(self.layer_norm2(x)), approximate="tanh"))


class SiglipSemanticEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = int(getattr(config, "s2_semantic_width", 1152))
        self.position_embedding = nn.Embedding(729, width)
        self.layers = nn.ModuleList([
            SiglipBlock(width, int(getattr(config, "s2_semantic_intermediate", 4304)),
                        int(getattr(config, "s2_semantic_heads", 16)))
            for _ in range(int(getattr(config, "s2_semantic_depth", 26)))])
        self.gradient_checkpointing = True

    def forward(self, x):
        side = math.isqrt(x.shape[1])
        if side * side != x.shape[1]:
            raise ValueError("SigLIP latent grid must be square")
        pos = self.position_embedding.weight.float().reshape(27, 27, -1).permute(2, 0, 1)[None]
        pos = F.interpolate(pos, (side, side), mode="bicubic", align_corners=False)
        x = x + pos.flatten(2).transpose(1, 2).to(x.dtype)
        mask = _prepared_mask(attention_from_allowed(torch.ones(
            x.shape[0], x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)))
        for layer in self.layers:
            x = _checkpoint(layer, x, mask, enabled=self.training and self.gradient_checkpointing)
        return x

    def load_official_weights(self, path):
        from safetensors import safe_open
        source = Path(path) / "model.safetensors"
        loaded = []
        with safe_open(str(source), framework="pt", device="cpu") as f:
            for name, parameter in self.named_parameters():
                key = ("vision_model.embeddings.position_embedding.weight" if name == "position_embedding.weight"
                       else "vision_model.encoder." + name)
                tensor = f.get_tensor(key)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"SigLIP shape mismatch: {key}: {tensor.shape} != {parameter.shape}")
                with torch.no_grad():
                    parameter.copy_(tensor)
                loaded.append({"key": key, "shape": list(tensor.shape)})
        return {"source": str(source), "bytes": source.stat().st_size,
                "layers": len(self.layers), "loaded_tensors": loaded,
                "rgb_stem_loaded": False, "text_encoder_loaded": False}


class Showo2Backbone(Qwen3Model):
    supported_training_objectives = frozenset({"showo2_full_image_flow"})

    def __init__(self, config):
        super().__init__(config)
        self.image_time_embedder = TimeEmbedding(config.hidden_size)
        self.semantic_encoder = None
        # Adding the semantic branch must not alter shared head/special-token RNG.
        with torch.random.fork_rng(devices=[]):
            if bool(getattr(config, "s2_use_siglip", False)):
                width = int(getattr(config, "s2_semantic_width", 1152))
                self.semantic_input_proj = nn.Linear(config.image_latent_dim, width)
                self.semantic_encoder = SiglipSemanticEncoder(config)
                self.semantic_encoder.gradient_checkpointing = bool(
                    getattr(config, "s2_semantic_gradient_checkpointing", True))
                self.image_fusion = nn.Sequential(
                    Qwen3RMSNorm(width + config.hidden_size, eps=config.rms_norm_eps),
                    nn.Linear(width + config.hidden_size, config.hidden_size), nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size))

    @property
    def text_two_stream(self):
        return self.config.dual_stream_attention_contract == "showo2_text_two_stream"

    def _needs_query_stream(self, *, calculate_likelihood, **kwargs):
        return self.text_two_stream and bool(calculate_likelihood)

    def _build_xt_inputs_embeds(self, input_ids, token_types, image_spans_present=None):
        # S2 supplies precomputed content embeddings instead of input IDs.
        # Every text query starts from the existing learned text-mask token;
        # neither target IDs nor image features enter the query residual path.
        return self.embed_tokens(torch.full_like(token_types, self.config.mask_token_id, dtype=torch.long))

    def _finalize_stream_hidden(self, X0_hidden_states, XT_hidden_states, *, use_query_stream, **kwargs):
        if use_query_stream:
            text = kwargs["token_types"].eq(0) | kwargs["token_types"].eq(2)
            return self.norm(torch.where(text.unsqueeze(-1), XT_hidden_states, X0_hidden_states))
        return self.norm(X0_hidden_states)

    def forward(self, X0_input_ids=None, attention_mask=None, token_types=None,
                image_latents=None, image_span_table=None, image_latent_mask=None,
                calculate_likelihood=False, _text_segment_ids=None,
                s2_time=None, s2_image_uncond_rows=None, s2_image_uncond_mask=None,
                image_latents_are_noisy=False, **kwargs):
        if X0_input_ids is None:
            raise ValueError("Show-o2 backbone requires token IDs")
        if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
            raise ValueError("Show-o2 ablation uses full refresh; cached Selfless decoding is incompatible")
        ids = X0_input_ids
        if token_types is None:
            token_types = torch.zeros_like(ids)
            if attention_mask is not None and _text_segment_ids is None:
                original = _materialize_allowed_mask(attention_mask, batch_size=ids.shape[0],
                    query_length=ids.shape[1], device=ids.device)[:, :, :ids.shape[1]]
                connected = original | original.transpose(1, 2)
                # Recover packed segment boundaries and padding for native text scoring.
                valid = connected.any(-1)
                valid[:, 0] |= ids.shape[1] == 1
                token_types = torch.where(valid, token_types, 3)
                adjacent = connected.diagonal(offset=1, dim1=1, dim2=2)
                _text_segment_ids = F.pad((~adjacent).long(), (1, 0)).cumsum(-1)
        allowed = omni_allowed_mask(ids, token_types, boi_token_id=self.config.boi_token_id,
            segment_ids=_text_segment_ids, image_uncond_rows=s2_image_uncond_rows,
            image_uncond_mask=s2_image_uncond_mask)
        mask = attention_from_allowed(allowed)
        embeds = self.embed_tokens(ids)
        has_images = image_latents is not None and (image_span_table is None or image_span_table.shape[0] > 0)
        if has_images:
            latents = image_latents.to(embeds.dtype)
            if self.training and not image_latents_are_noisy:
                latents = self._maybe_add_image_input_noise(latents)
            projected = self.image_token_embedder.z_proj(latents)
            if self.semantic_encoder is not None:
                if image_span_table is None:
                    # Generation/scoring can supply a dense mask; extract spans outside training.
                    raise ValueError("SigLIP forward requires image_span_table")
                rows = image_span_table[:, 0].long()
                starts = image_span_table[:, 2].long()
                indices = starts[:, None] + torch.arange(self.config.image_tokens_per_img, device=ids.device)
                semantic = self.semantic_encoder(self.semantic_input_proj(latents[rows[:, None], indices]))
                fused = self.image_fusion(torch.cat([semantic, projected[rows[:, None], indices]], -1))
                projected = projected.clone()
                projected[rows[:, None], indices] = fused
            embeds = torch.where(token_types.eq(1).unsqueeze(-1), projected, embeds)
        if s2_time is None:
            s2_time = torch.ones(ids.shape[0], device=ids.device)
        time = self.image_time_embedder(s2_time * float(getattr(self.config, "image_flow_time_scale", 1000)))
        embeds = embeds + ids.eq(int(self.config.boi_token_id)).unsqueeze(-1) * time[:, None]
        kwargs.pop("_text_ar_mode", None)
        kwargs.pop("X0_inputs_embeds", None)
        kwargs.pop("content_attention_mask", None)
        query_mode = self.text_two_stream and calculate_likelihood
        query_mask = (attention_from_allowed(text_query_allowed_mask(allowed, token_types))
                      if query_mode else mask)
        output = super().forward(X0_inputs_embeds=embeds, attention_mask=query_mask,
            content_attention_mask=mask if query_mode else None,
            token_types=token_types, calculate_likelihood=query_mode, _text_ar_mode=False, **kwargs)
        if calculate_likelihood and not self.text_two_stream:
            raw = output.last_hidden_state
            aligned = F.pad(raw[:, :-1], (0, 0, 1, 0))
            output.last_hidden_state = torch.where(token_types.eq(1).unsqueeze(-1), raw, aligned)
        return output


class ModulatedFlowBlock(nn.Module):
    """Show-o2's six-way image-only AdaLN, QK norm/GQA, and gated SiLU MLP."""
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = Qwen3Attention(config, index)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size))

    def forward(self, x, time, image_positions, mask, rope):
        shifts = self.adaLN_modulation(time).chunk(6, -1)
        region = image_positions.unsqueeze(-1)
        sa, ca, ga, sm, cm, gm = [s[:, None] for s in shifts]
        def modulate(norm, shift, scale):
            normalized = norm(x)
            return torch.where(region, (normalized.float() * (1 + scale.float()) + shift.float()).to(x.dtype), normalized)
        h = modulate(self.input_layernorm, sa, ca)
        attn = self.self_attn(h, None, position_embeddings=rope, attention_mask=mask)[0]
        x = x + attn * torch.where(region, ga, 1)
        h = modulate(self.post_attention_layernorm, sm, cm)
        return x + self.mlp(h) * torch.where(region, gm, 1)


class Showo2FlowHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = int(config.image_flow_width)
        head_dim = int(getattr(config, "s2_flow_head_dim", 64))
        heads = width // head_dim
        head_config = Qwen3Config(hidden_size=width,
            intermediate_size=int(getattr(config, "s2_flow_intermediate", 1472)),
            num_attention_heads=heads, num_key_value_heads=max(1, heads // 4),
            head_dim=head_dim, attention_bias=False, rms_norm_eps=config.rms_norm_eps,
            rope_parameters=getattr(config, "rope_parameters", None),
            max_position_embeddings=config.max_position_embeddings)
        self.input_proj = nn.Linear(config.hidden_size, width)
        self.time_embedder = TimeEmbedding(width)
        self.layers = nn.ModuleList([ModulatedFlowBlock(head_config, i) for i in range(config.image_flow_depth)])
        self.rotary_emb = Qwen3RotaryEmbedding(head_config)
        self.norm = Qwen3RMSNorm(width, eps=config.rms_norm_eps)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output_proj = nn.Linear(width, config.image_latent_dim)
        self.gradient_checkpointing = bool(getattr(config, "image_flow_grad_checkpointing", True))
        self.time_scale = float(getattr(config, "image_flow_time_scale", 1000))
        self.last_forward_stats = {}

    def zero_output_modulation(self):
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden, times, image_positions, mask, position_ids):
        x = self.input_proj(hidden)
        time = self.time_embedder(times * self.time_scale)
        rope = self.rotary_emb(x, position_ids)
        mask = _prepared_mask(mask)
        for layer in self.layers:
            x = _checkpoint(layer, x, time, image_positions, mask, rope,
                            enabled=self.training and self.gradient_checkpointing)
        shift, scale = self.final_modulation(time).chunk(2, -1)
        x = (self.norm(x).float() * (1 + scale[:, None].float()) + shift[:, None].float()).to(x.dtype)
        return self.output_proj(x)


def span_table_from_spans(spans, device):
    return torch.tensor([(row, start - 1, start, end) for row, start, end in spans],
                        dtype=torch.long, device=device).reshape(-1, 4)


class Showo2UnifiedForCausalLM(Qwen3ForCausalLM):
    config_class = Showo2UnifiedConfig
    architecture_variant = "showo2_unified"
    text_prediction_offset = 1
    text_hidden_alignment = "target_position_contains_previous_content_hidden"

    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        if getattr(config, "training_objective", None) != "showo2_full_image_flow":
            raise ValueError("Show-o2 requires its dedicated full-image objective")
        if config.dual_stream_attention_contract not in {"showo2_omni_attention", "showo2_text_two_stream"}:
            raise ValueError("Unsupported S2 backbone attention contract")
        self.model = Showo2Backbone(config)
        if self.model.text_two_stream:
            self.text_prediction_offset = 0
            self.text_hidden_alignment = "target_position_contains_text_query_hidden"
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_flow_head = Showo2FlowHead(config)
        self.image_latent_dim = int(config.image_latent_dim)
        self.image_flow_batch_mul = int(config.image_flow_batch_mul)
        self.vocab_size = config.vocab_size
        self.training_objective = config.training_objective
        self.flow_condition_contract = "backbone_noisy_image_hidden"
        self.lambda_text = float(config.lambda_text)
        self.lambda_image = float(config.lambda_image)
        self.semantic_initialization_report = None
        with torch.random.fork_rng(devices=[]):
            self.post_init()
        self.reset_image_modules()

    def reset_image_modules(self):
        # Module-scoped seeds keep every shared tensor identical across the two arms.
        seed = int(getattr(self.config, "s2_initialization_seed", 42))
        shared = [self.model.image_token_embedder, self.model.image_time_embedder, self.image_flow_head]
        for index, module in enumerate(shared):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 1000 + index)
                module.apply(self._init_weights)
        self.image_flow_head.zero_output_modulation()
        if self.model.semantic_encoder is not None:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 2000)
                self.model.semantic_input_proj.apply(self._init_weights)
                self.model.image_fusion.apply(self._init_weights)

    def initialize_semantic_weights(self, path):
        if self.model.semantic_encoder is not None:
            self.semantic_initialization_report = self.model.semantic_encoder.load_official_weights(path)
        return self.semantic_initialization_report

    def _sample_times(self, count, device):
        mode = getattr(self.config, "image_flow_time_sampling", "logit_normal")
        if mode == "uniform":
            times = torch.rand(count, device=device)
        elif mode == "logit_normal":
            times = (torch.randn(count, device=device) * float(self.config.image_flow_logit_std)
                     + float(self.config.image_flow_logit_mean)).sigmoid()
            mix = float(self.config.image_flow_time_uniform_mix)
            if mix > 0:
                times = torch.where(torch.rand(count, device=device) < mix,
                                    torch.rand(count, device=device), times)
        else:
            raise ValueError(f"Unsupported Show-o2 time sampler: {mode}")
        eps = float(self.config.image_flow_time_eps)
        return times.clamp(eps, 1 - eps)

    def predict_velocity(self, input_ids, token_types, noisy_latents, times, *,
                         segment_ids=None, image_span_table=None, position_ids=None,
                         image_uncond_rows=None, image_uncond_mask=None):
        """No clean labels enter this function, including the semantic branch."""
        if position_ids is None:
            position_ids = build_row_col_position_ids(token_types, self.config.image_tokens_per_img)
        hidden = self.model(X0_input_ids=input_ids, token_types=token_types,
            image_latents=noisy_latents, image_span_table=image_span_table,
            _text_segment_ids=segment_ids, s2_time=times, image_latents_are_noisy=True,
            s2_image_uncond_rows=image_uncond_rows, s2_image_uncond_mask=image_uncond_mask,
            position_ids=position_ids).last_hidden_state
        allowed = omni_allowed_mask(input_ids, token_types, boi_token_id=self.config.boi_token_id,
            segment_ids=segment_ids, image_uncond_rows=image_uncond_rows,
            image_uncond_mask=image_uncond_mask)
        return self.image_flow_head(hidden, times,
            image_block_mask(input_ids, token_types, self.config.boi_token_id),
            attention_from_allowed(allowed), position_ids)

    def forward(self, X0_input_ids=None, attention_mask=None, labels=None, token_types=None,
                image_latents=None, image_span_table=None, image_loss_mask=None,
                compute_text_loss=True, compute_image_loss=False, _text_segment_ids=None,
                s2_image_uncond_rows=None, s2_image_uncond_mask=None,
                return_logits=True, return_per_modality_loss_graph=False,
                calculate_likelihood=False, position_ids=None, s2_flow_state=None, **kwargs):
        ids = X0_input_ids
        types = torch.zeros_like(ids) if token_types is None else token_types
        hidden = None
        zero = self.lm_head.weight.reshape(-1)[0].float() * 0
        text_loss = zero
        image_loss = zero
        text_count = torch.zeros((), dtype=torch.long, device=ids.device)
        image_count = torch.zeros_like(text_count)
        if compute_text_loss or labels is None:
            output = self.model(X0_input_ids=ids, attention_mask=attention_mask,
                token_types=token_types, image_latents=image_latents, image_span_table=image_span_table,
                _text_segment_ids=_text_segment_ids, calculate_likelihood=True,
                s2_image_uncond_rows=s2_image_uncond_rows, s2_image_uncond_mask=s2_image_uncond_mask,
                position_ids=position_ids)
            hidden = output.last_hidden_state
            if labels is not None and compute_text_loss:
                valid = (types.eq(0) | types.eq(2)) & labels.ne(-100)
                valid[:, 0] = False
                if _text_segment_ids is not None:
                    valid[:, 1:] &= _text_segment_ids[:, 1:].eq(_text_segment_ids[:, :-1])
                text_count = valid.sum()
                targets, features = labels[valid], hidden[valid]
                # Bound vocabulary-logit memory, preserving one global token denominator.
                for start in range(0, targets.shape[0], 256):
                    text_loss = text_loss + F.cross_entropy(self.lm_head(features[start:start+256]).float(),
                        targets[start:start+256], reduction="sum") / text_count.clamp_min(1)
        if labels is not None and compute_image_loss:
            if image_latents is None or image_span_table is None or image_span_table.shape[0] != ids.shape[0]:
                raise ValueError("Full-image training requires exactly one image span per row")
            clean = image_latents.float()
            loss_mask = types.eq(1) if image_loss_mask is None else image_loss_mask.bool()
            image_count = loss_mask.sum()
            repeats = self.image_flow_batch_mul
            if s2_flow_state is None:
                times = self._sample_times(repeats * ids.shape[0], ids.device).view(repeats, ids.shape[0])
                noise = torch.randn((repeats,) + tuple(clean.shape), device=ids.device, dtype=torch.float32)
                xt = (1 - times[:, :, None, None]) * noise + times[:, :, None, None] * clean
                target = clean.unsqueeze(0) - noise
                if self.training and self.model.image_input_noise_strength > 0:
                    xt = xt + torch.randn_like(xt) * self.model.image_input_noise_strength
            else:
                times, xt, target = (s2_flow_state[k] for k in ("times", "x_t", "velocity_target"))
                if times.shape != (repeats, ids.shape[0]):
                    raise ValueError("Supplied flow state must retain every Monte Carlo sample")
            # Sample all MC randomness before batching, preserving the draw order.
            # Every draw still refreshes the complete noisy-image model. Only the
            # batch dimension changes; the loss keeps its original denominator.
            chunk = int(getattr(self.config, "s2_mc_batch_size", 1))
            if chunk not in (1, 2, 4) or repeats % chunk:
                raise ValueError("S2 MC batch size must be 1, 2 or 4 and divide the draw count")
            def repeat_rows(value, dim=0):
                if value is None:
                    return None
                factors = [1] * value.ndim
                factors[dim] = chunk
                return value.repeat(*factors)
            table = image_span_table.repeat(chunk, 1).clone()
            table[:, 0] += torch.arange(chunk, device=ids.device).repeat_interleave(
                image_span_table.shape[0]) * ids.shape[0]
            positions = position_ids
            if positions is None:
                positions = build_row_col_position_ids(types, self.config.image_tokens_per_img)
            prediction_args = (repeat_rows(ids), repeat_rows(types), repeat_rows(_text_segment_ids),
                table, repeat_rows(positions, 1 if positions.ndim == 3 else 0),
                repeat_rows(s2_image_uncond_rows), repeat_rows(s2_image_uncond_mask))
            def velocity(current, t, batch_ids, batch_types, segments, spans, pos, rows, mask):
                return self.predict_velocity(batch_ids, batch_types, current, t,
                    segment_ids=segments, image_span_table=spans, position_ids=pos,
                    image_uncond_rows=rows, image_uncond_mask=mask)
            for start in range(0, repeats, chunk):
                pred = _checkpoint(velocity, xt[start:start+chunk].flatten(0, 1),
                    times[start:start+chunk].flatten(), *prediction_args,
                    enabled=self.training and bool(getattr(self.config, "s2_full_prediction_checkpointing", True)))
                mse = (pred.float().reshape_as(target[start:start+chunk]) - target[start:start+chunk]).square().mean(-1)
                image_loss = image_loss + (mse * loss_mask[None]).sum() / (image_count.clamp_min(1) * repeats)
            self.image_flow_head.last_forward_stats = {
                "flow_mc_repeats": repeats, "whole_image_time": 1,
                "flow_mc_batch_size": chunk, "prediction_batch_rows": chunk * ids.shape[0],
                "backbone_calls": repeats // chunk, "flow_head_calls": repeats // chunk}
        if labels is not None:
            # ZeRO-2 sees a gradient for all trainable parameters on every task.
            skipped = _zero_parameters(self.image_flow_head, zero) if not compute_image_loss else zero
            if image_latents is None or (image_span_table is not None and image_span_table.shape[0] == 0):
                skipped = skipped + _zero_parameters(self.model.image_token_embedder, zero)
                if self.model.semantic_encoder is not None:
                    for module in (self.model.semantic_encoder, self.model.semantic_input_proj, self.model.image_fusion):
                        skipped = skipped + _zero_parameters(module, zero)
            loss = self.lambda_text * text_loss + self.lambda_image * image_loss + skipped
        else:
            loss = None
        logits = self.lm_head(hidden) if hidden is not None and return_logits and labels is None else None
        result = CausalLMOutputWithPast(loss=loss, logits=logits)
        if hidden is not None:
            result["last_hidden_state"] = hidden
        result["per_modality_loss"] = {"text_loss": text_loss.detach(), "image_loss": image_loss.detach()}
        result["per_modality_count"] = {"text_tokens": text_count, "image_tokens": image_count}
        if return_per_modality_loss_graph:
            result["per_modality_loss_graph"] = {"text_loss": text_loss, "image_loss": image_loss}
        return result

    @torch.no_grad()
    def generate_image(self, input_ids, token_types, sigma, spans, *, segment_ids=None,
                       image_latent_dim=None, initial_image_latents=None,
                       initial_image_latent_mask=None, initial_noise_bank=None,
                       flow_temperature=1., flow_cfg=3.5, flow_cfg_schedule="constant",
                       flow_solver=None, flow_num_steps=None, parallel_rate=1,
                       order_strategy="spatial_halton", use_cache=True,
                       return_trace=False, debug_finite=False, **kwargs):
        if flow_cfg_schedule != "constant":
            raise ValueError("The S2 comparison fixes a constant CFG schedule")
        solver = flow_solver or self.config.image_flow_solver
        steps = int(flow_num_steps or self.config.image_flow_num_sampling_steps)
        if solver not in {"heun", "euler"} or steps < 1:
            raise ValueError("S2 generation requires heun/euler and positive steps")
        dim = int(image_latent_dim or self.image_latent_dim)
        if dim != self.image_latent_dim:
            raise ValueError("Generation latent dimensions must match training")
        shape = (*input_ids.shape, dim)
        state = (torch.zeros(shape, device=input_ids.device) if initial_image_latents is None
                 else initial_image_latents.to(device=input_ids.device, dtype=torch.float32).clone())
        table = span_table_from_spans(spans, input_ids.device)
        if table.shape[0] != input_ids.shape[0]:
            raise ValueError("S2 generation requires one image per batch row")
        rows = table[:, 0]
        indices = table[:, 2, None] + torch.arange(self.config.image_tokens_per_img, device=input_ids.device)
        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        target_mask[rows[:, None], indices] = True
        if initial_noise_bank is None:
            noise = torch.randn(shape, device=input_ids.device)
        elif tuple(initial_noise_bank.shape) == shape:
            noise = initial_noise_bank.to(device=input_ids.device, dtype=torch.float32)
        elif tuple(initial_noise_bank.shape) == (table.shape[0], self.config.image_tokens_per_img, dim):
            noise = torch.zeros_like(state)
            noise[rows[:, None], indices] = initial_noise_bank.to(noise)
        else:
            raise ValueError("Initial noise must be dense sequence or per-image latent grids")
        state = torch.where(target_mask[..., None], noise * float(flow_temperature), state)
        uncond = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        calls = 0
        def velocity(x, time):
            nonlocal calls
            t = torch.full((input_ids.shape[0],), float(time), device=x.device)
            conditional = self.predict_velocity(input_ids, token_types, x, t,
                segment_ids=segment_ids, image_span_table=table)
            calls += 1
            if float(flow_cfg) != 1.0:
                unconditional = self.predict_velocity(input_ids, token_types, x, t,
                    segment_ids=segment_ids, image_span_table=table, image_uncond_rows=uncond)
                calls += 1
                conditional = unconditional + float(flow_cfg) * (conditional - unconditional)
            return conditional.float() * target_mask[..., None]
        dt = 1. / steps
        for index in range(steps):
            first = velocity(state, index * dt)
            candidate = state + dt * first
            if solver == "heun":
                second = velocity(candidate, (index + 1) * dt)
                candidate = state + (dt / 2) * (first + second)
            state = candidate
            if debug_finite and not torch.isfinite(state).all():
                raise FloatingPointError(f"Nonfinite S2 latent at ODE step {index}")
        trace = {"generation_mode": "showo2_full_image_flow", "order_strategy": "whole_image",
                 "solver": solver, "steps": steps, "ode_function_evals": steps * (2 if solver == "heun" else 1),
                 "backbone_calls": calls, "flow_head_calls": calls,
                 "semantic_encoder_calls": calls if self.model.semantic_encoder is not None else 0,
                 "backbone_kv_cache_enabled": False, "cfg": float(flow_cfg)}
        side = math.isqrt(int(self.config.image_tokens_per_img))
        generated = state[rows[:, None], indices].reshape(table.shape[0], side, side, dim).permute(0, 3, 1, 2)
        return (generated, trace) if return_trace else generated

    @torch.no_grad()
    def generate_text(self, input_ids, *, max_new_tokens, token_types=None, sigma=None,
                      segment_ids=None, image_latents=None, image_latent_mask=None,
                      temperature=0., eos_token_id=None, use_cache=True, return_trace=False):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        prompt_types = torch.zeros_like(input_ids) if token_types is None else token_types
        lengths = prompt_types.ne(3).sum(-1)
        if bool(lengths.eq(0).any()):
            raise ValueError("Text generation requires a nonempty prompt")
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        width = input_ids.shape[1]
        capacity = width + int(max_new_tokens)
        ids = F.pad(input_ids, (0, int(max_new_tokens)), value=int(self.config.pad_token_id or 0))
        types = F.pad(prompt_types, (0, int(max_new_tokens)), value=0)
        prompt_segments = torch.zeros_like(input_ids) if segment_ids is None else segment_ids
        last_segment = prompt_segments[batch, lengths - 1]
        segments = last_segment[:, None].expand(-1, capacity).clone()
        segments[:, :width] = prompt_segments
        latents = torch.zeros(*ids.shape, self.image_latent_dim, device=ids.device,
                              dtype=self.image_flow_head.output_proj.weight.dtype)
        if image_latents is not None:
            latents[:, :width] = image_latents.to(latents)
        elif bool(prompt_types.eq(1).any()):
            raise ValueError("Image-conditioned text generation requires observed image latents")
        prepared = dict(ids=ids, types=types, segments=segments, latents=latents,
                        prompt_lengths=lengths, prompt_width=width)
        ids, types, segments = (prepared[k] for k in ("ids", "types", "segments"))
        lengths = prepared["prompt_lengths"]
        batch = torch.arange(ids.shape[0], device=ids.device)
        stop_ids = self._generation_stop_token_ids(eos_token_id)
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        image_spans = []
        for row, row_types in enumerate(types[:, :prepared["prompt_width"]].cpu().tolist()):
            start = None
            for pos, kind in enumerate(row_types + [3]):
                if kind == 1 and start is None:
                    start = pos
                elif kind != 1 and start is not None:
                    image_spans.append((row, start, pos))
                    start = None
        table = span_table_from_spans(image_spans, ids.device) if image_spans else None
        generated = 0
        for step in range(int(max_new_tokens)):
            target_positions = lengths + step
            query_mode = self.model.text_two_stream
            width = int(target_positions.max()) + int(query_mode)
            current_types = types[:, :width].clone()
            valid = torch.arange(width, device=ids.device)[None] < (target_positions[:, None] + int(query_mode))
            current_types = torch.where(valid, current_types, 3)
            if query_mode:
                ids[batch, target_positions] = int(self.config.mask_token_id)
                current_types[batch, target_positions] = 0
                segments[batch, target_positions] = last_segment
            hidden = self.model(X0_input_ids=ids[:, :width], token_types=current_types,
                _text_segment_ids=segments[:, :width], image_span_table=table,
                image_latents=prepared["latents"][:, :width] if image_spans else None,
                calculate_likelihood=query_mode).last_hidden_state
            next_token = self._sample_token(
                self.lm_head(hidden[batch, target_positions - self.text_prediction_offset]), float(temperature))
            if stop_ids:
                next_token = torch.where(finished, stop_ids[0], next_token)
            ids[batch, target_positions] = next_token
            types[batch, target_positions] = 0
            segments[batch, target_positions] = last_segment
            generated += 1
            if stop_ids:
                finished |= self._matches_stop_token(next_token, stop_ids)
                if bool(finished.all()):
                    break
        result = ids[:, :int(lengths.max()) + generated]
        trace = {"generation_mode": "showo2_text_ar", "backbone_kv_cache_enabled": False,
                 "text_prediction_offset": self.text_prediction_offset,
                 "text_hidden_alignment": self.text_hidden_alignment,
                 "generated_tokens": generated, "backbone_calls": generated}
        return (result, trace) if return_trace else result
