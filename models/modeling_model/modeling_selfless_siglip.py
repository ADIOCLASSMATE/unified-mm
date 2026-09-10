"""B + a SigLIP-initialized latent content encoder with sigma-causal visibility.

Only X0 image inputs are fused. XT stays the original learned mask query, and
B's backbone, XT/X0 flow conditions, random order and RF loss remain intact.
Every semantic layer inherits X0 visibility, restricted to the same image.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, Qwen3Config

from .modeling_selfless_flow import Qwen3ForCausalLM, Qwen3Model, Qwen3RMSNorm, compiled_flex_attention
from .modeling_selfless_cache import SelflessStaticCache
from .modeling_showo2_unified import SiglipSemanticEncoder, attention_from_allowed, _prepared_mask, _zero_parameters


class SelflessSiglipConfig(Qwen3Config):
    model_type = "selfless_flow_siglip"


AutoConfig.register(SelflessSiglipConfig.model_type, SelflessSiglipConfig)


def _allowed_at(mask, rows, queries, keys):
    """Sample native mask edges without inferring order from physical positions."""
    if isinstance(mask, tuple):
        safe, valid = mask
        return ~safe[rows, 0, queries, keys] & valid[rows, 0, queries, 0]
    if isinstance(mask, torch.Tensor):
        dense = mask[:, 0] if mask.ndim == 4 else mask
        blocked = dense if dense.dtype == torch.bool else dense < 0
        return ~blocked[rows, queries, keys]
    if isinstance(mask, BlockMask):
        return mask.mask_mod(rows, torch.zeros((), device=rows.device, dtype=torch.long), queries, keys)
    raise TypeError(f"Unsupported semantic visibility source: {type(mask).__name__}")


class SemanticCacheState:
    def __init__(self, batch, capacity, depth, device):
        self.kv = SelflessStaticCache(SimpleNamespace(num_hidden_layers=depth), capacity)
        self.span_start = torch.full((batch, capacity), -1, device=device, dtype=torch.long)
        self.valid = torch.zeros((batch, capacity), device=device, dtype=torch.bool)

    def register_spans(self, ids, positions, boi_id, image_tokens):
        # BOI is present in the original prefill even when image slots are absent.
        for row, query in ids.eq(boi_id).nonzero().detach().cpu().tolist():
            start = int(positions[row, query]) + 1
            end = start + image_tokens
            if end > self.span_start.shape[1]:
                raise ValueError("SigLIP image span exceeds static cache capacity")
            self.span_start[row, start:end] = start

    def fork(self):
        other = object.__new__(type(self))
        other.kv = self.kv.fork()
        other.span_start, other.valid = self.span_start.clone(), self.valid.clone()
        return other

    def repeat_batch_(self, repeats):
        self.kv.repeat_batch_(repeats)
        self.span_start = self.span_start.repeat(repeats, 1)
        self.valid = self.valid.repeat(repeats, 1)

    def bytes(self):
        tensors = [self.span_start, self.valid]
        for layer in self.kv.layers:
            if layer.is_initialized:
                tensors += [layer.keys, layer.values]
        return sum(t.numel() * t.element_size() for t in tensors)


class SiglipBackboneCache(SelflessStaticCache):
    def fork(self):
        result = super().fork()
        copy_semantic_cache(self, result, repeats=1)
        return result

    def repeat_batch_(self, repeats):
        super().repeat_batch_(repeats)
        state = getattr(self, "semantic_state", None)
        if state is not None:
            state.repeat_batch_(repeats)
        return self


def copy_semantic_cache(source, destination, *, repeats):
    state = getattr(source, "semantic_state", None)
    if state is not None:
        destination.__class__ = SiglipBackboneCache
        destination.semantic_state = state.fork()
        if repeats != 1:
            destination.semantic_state.repeat_batch_(repeats)


class SigmaSiglipEncoder(SiglipSemanticEncoder):
    def __init__(self, config):
        vision_config = SimpleNamespace(**{
            f"s2_semantic_{key}": getattr(config, f"b_siglip_{key}", default)
            for key, default in [("width", 1152), ("intermediate", 4304), ("heads", 16), ("depth", 26)]
        })
        super().__init__(vision_config)
        self.image_tokens = int(config.image_tokens_per_img)
        self.gradient_checkpointing = bool(getattr(config, "b_siglip_gradient_checkpointing", True))

    def forward(self, x, local_positions, allowed, *, cache=None, cache_positions=None, cache_write_mask=None):
        side = math.isqrt(self.image_tokens)
        position = self.position_embedding.weight.float().reshape(27, 27, -1).permute(2, 0, 1)[None]
        position = F.interpolate(position, (side, side), mode="bicubic", align_corners=False)
        position = position.flatten(2).transpose(1, 2)[0]
        x = x + position[local_positions.clamp(0, self.image_tokens - 1)].to(x.dtype)
        mask = _prepared_mask(attention_from_allowed(allowed))
        for index, layer in enumerate(self.layers):
            if cache is None:
                if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                    x = checkpoint(layer, x, mask, use_reentrant=False)
                else:
                    x = layer(x, mask)
                continue
            if self.training or torch.is_grad_enabled():
                raise ValueError("Semantic KV caching is inference-only")
            normalized = layer.layer_norm1(x)
            batch, length, width = x.shape
            attention = layer.self_attn
            heads, dim = attention.heads, width // attention.heads
            q, k, v = [projection(normalized).view(batch, length, heads, dim).transpose(1, 2)
                       for projection in (attention.q_proj, attention.k_proj, attention.v_proj)]
            k, v = cache.update(k, v, index, {"cache_position": cache_positions, "cache_write_mask": cache_write_mask})
            out = compiled_flex_attention(q, k, v, mask, dim ** -.5, False)
            x = x + attention.out_proj(out.transpose(1, 2).reshape(batch, length, width))
            x = x + layer.mlp.fc2(F.gelu(layer.mlp.fc1(layer.layer_norm2(x)), approximate="tanh"))
        return x


class SiglipQwen3Model(Qwen3Model):
    def _init_weights(self, module):
        if getattr(module, "_b_siglip_module", False):
            with torch.random.fork_rng(devices=[]):
                return super()._init_weights(module)
        return super()._init_weights(module)

    def _build_x0_inputs_embeds(self, input_ids, token_types, image_latents, image_latent_mask,
            image_spans_present=None, image_latents_are_noisy=False, debug_finite=False,
            debug_label="", image_context=None):
        context = image_context or {}
        has_images = image_spans_present
        if has_images is None:
            has_images = token_types is not None and bool(token_types.eq(1).any())
        if image_latents is not None:
            image_latents = image_latents.to(device=input_ids.device, dtype=self.image_token_embedder.weight_dtype)
            if has_images and not image_latents_are_noisy:
                image_latents = self._maybe_add_image_input_noise(image_latents)
                image_latents_are_noisy = True
        result = super()._build_x0_inputs_embeds(input_ids, token_types, image_latents,
            image_latent_mask, image_spans_present, image_latents_are_noisy, debug_finite, debug_label)
        if not hasattr(self, "semantic_encoder"):
            return result
        batch, length = input_ids.shape
        cache = context.get("past_key_values")
        state = None
        if cache is not None:
            if not isinstance(cache, SelflessStaticCache):
                raise TypeError("B + SigLIP requires the native physical-position static cache")
            if self.training or torch.is_grad_enabled():
                raise ValueError("Semantic KV caching is inference-only")
            cache.__class__ = SiglipBackboneCache
            if not hasattr(cache, "semantic_state"):
                cache.semantic_state = SemanticCacheState(batch, cache.get_max_cache_shape(),
                    len(self.semantic_encoder.layers), input_ids.device)
            state = cache.semantic_state
            positions = context.get("cache_position")
            if positions is None:
                raise ValueError("SigLIP cached forward requires physical cache positions")
            if positions.ndim == 1:
                positions = positions[None].expand(batch, -1)
            state.register_spans(input_ids, positions, int(self.config.boi_token_id), self.config.image_tokens_per_img)
        else:
            positions = torch.arange(length, device=input_ids.device)[None].expand(batch, -1)
        observed = torch.zeros_like(input_ids, dtype=torch.bool) if token_types is None else token_types.eq(1)
        if image_latent_mask is not None:
            observed &= image_latent_mask.bool()
        count = int(observed.sum(-1).max())
        if count == 0:
            if self.training:
                for module in (self.semantic_encoder, self.semantic_input_proj, self.image_fusion):
                    result = result + _zero_parameters(module, result).to(result.dtype)
            return result
        # Each row keeps its spatial image positions; padding contributes zero.
        local_query = torch.arange(length, device=input_ids.device)[None].expand(batch, -1)
        selected = torch.where(observed, local_query, length).topk(count, largest=False, sorted=True).values
        valid = selected.ne(length)
        selected = selected.clamp_max(length - 1)
        physical = positions.gather(1, selected)
        row = torch.arange(batch, device=input_ids.device)[:, None]
        source_mask = context.get("content_attention_mask")
        if source_mask is None:
            source_mask = context.get("attention_mask")
        if state is None:
            is_image = token_types.eq(1)
            starts = is_image & ~F.pad(is_image[:, :-1], (1, 0), value=False)
            all_starts = torch.where(starts, local_query, torch.full_like(local_query, -1)).cummax(-1).values
            spans = all_starts.gather(1, selected)
            local_positions = physical - spans
            allowed = _allowed_at(source_mask, row[:, :, None], selected[:, :, None], selected[:, None, :])
            allowed &= valid[:, :, None] & valid[:, None, :] & spans[:, :, None].eq(spans[:, None, :])
            cache_kwargs = {}
        else:
            spans = state.span_start.gather(1, physical)
            if bool((valid & spans.lt(0)).any()):
                raise ValueError("Cached image content lacks a BOI-defined semantic span")
            local_positions = physical - spans
            if context.get("cache_read_only", False):
                raise ValueError("Read-only image queries must retain B's mask embedding")
            write = valid.clone()
            prefix = context.get("cache_write_prefix")
            if prefix is not None:
                write &= selected.lt(prefix)
            if context.get("cache_write_mask") is not None:
                write &= context["cache_write_mask"].gather(1, selected)
            # Slot zero is a BOI/text slot, never an observed image token.
            physical = torch.where(valid, physical, 0)
            state.valid.scatter_(1, physical, write | state.valid.gather(1, physical))
            keys = torch.arange(state.valid.shape[1], device=input_ids.device)[None, None, :]
            allowed = _allowed_at(source_mask, row[:, :, None], selected[:, :, None], keys)
            allowed &= valid[:, :, None] & state.valid[:, None, :] & spans[:, :, None].eq(state.span_start[:, None, :])
            cache_kwargs = dict(cache=state.kv, cache_positions=physical, cache_write_mask=write)
        semantic = self.semantic_encoder(self.semantic_input_proj(image_latents[row, selected]),
            local_positions, allowed, **cache_kwargs)
        projected = result[row, selected]
        fused = self.image_fusion(torch.cat([semantic.to(projected.dtype), projected], -1))
        delta = (fused - projected) * valid[..., None]
        result = result.scatter_add(1, selected[..., None].expand_as(delta), delta)
        self.semantic_forward_calls += 1
        if state is not None:
            self.semantic_cache_peak_bytes = max(self.semantic_cache_peak_bytes, state.bytes())
        return result


class SelflessSiglipForCausalLM(Qwen3ForCausalLM):
    config_class = SelflessSiglipConfig
    backbone_model_class = SiglipQwen3Model

    def __init__(self, config):
        expected = {"architecture_variant": "selfless_siglip", "training_objective": "selfless_dual_stream",
            "dual_stream_attention_contract": "xlnet_content_diagonal", "flow_head_attention_contract": "xlnet_content_diagonal",
            "flow_condition_contract": "backbone_xt_query_backbone_x0_content",
            "b_siglip_visibility_contract": "same_image_native_x0_sigma_causal"}
        for field, value in expected.items():
            if getattr(config, field, value) != value:
                raise ValueError(f"B + SigLIP requires {field}={value}")
            setattr(config, field, value)
        super().__init__(config)
        # Append extra modules after all B modules: preserve B's construction RNG.
        with torch.random.fork_rng(devices=[]):
            width = int(getattr(config, "b_siglip_width", 1152))
            self.model.semantic_input_proj = nn.Linear(config.image_latent_dim, width)
            self.model.semantic_encoder = SigmaSiglipEncoder(config)
            self.model.image_fusion = nn.Sequential(Qwen3RMSNorm(width + config.hidden_size, eps=config.rms_norm_eps),
                nn.Linear(width + config.hidden_size, config.hidden_size), nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size))
            for module in (self.model.semantic_encoder, self.model.semantic_input_proj, self.model.image_fusion):
                for child in module.modules():
                    child._b_siglip_module = True
                module.apply(self._init_weights)
        self.reset_semantic_input_modules()
        self.model.semantic_forward_calls = 0
        self.model.semantic_cache_peak_bytes = 0
        self.semantic_initialization_report = None

    def _init_weights(self, module):
        if getattr(module, "_b_siglip_module", False):
            with torch.random.fork_rng(devices=[]):
                return super()._init_weights(module)
        return super()._init_weights(module)

    def reset_semantic_input_modules(self):
        # Same new-input/fusion initialization as the S2-dual-siglip control.
        with torch.random.fork_rng(devices=[]):
            # These modules are initialized on CPU (or meta during HF loading).
            # torch.manual_seed also seeds the NPU, which a CPU RNG fork would
            # not restore and would change B's subsequent training noise.
            torch.random.default_generator.manual_seed(
                int(getattr(self.config, "b_siglip_initialization_seed", 42)) + 2000)
            self.model.semantic_input_proj.apply(super()._init_weights)
            self.model.image_fusion.apply(super()._init_weights)

    def initialize_semantic_weights(self, path):
        self.reset_semantic_input_modules()
        self.semantic_initialization_report = self.model.semantic_encoder.load_official_weights(path)
        return self.semantic_initialization_report

    def _semantic_generation(self, fn, *args, **kwargs):
        self.model.semantic_forward_calls = 0
        self.model.semantic_cache_peak_bytes = 0
        result = fn(*args, **kwargs)
        if isinstance(result, tuple) and isinstance(result[1], dict):
            result[1].update(semantic_visibility="same_image_native_x0_sigma_causal",
                semantic_forward_calls=self.model.semantic_forward_calls,
                semantic_cache_peak_bytes=self.model.semantic_cache_peak_bytes)
        return result

    def generate_image(self, *args, **kwargs):
        return self._semantic_generation(super().generate_image, *args, **kwargs)

    def generate_text(self, *args, **kwargs):
        return self._semantic_generation(super().generate_text, *args, **kwargs)
