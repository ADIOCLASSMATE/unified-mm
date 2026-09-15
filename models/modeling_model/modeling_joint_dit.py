"""B's dual-stream backbone with fixed conditions for a full-image DiT.

An image is one sigma block: content sees the whole clean image, while mask
queries cannot see any of their own image's content. Only the single-stream
DiT receives noisy latents and flow time. Text retains B's same-position loss.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .modeling_selfless_flow import Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel
from .modeling_showo2_unified import (
    Showo2FlowHead, attention_from_allowed, _prepared_mask, _checkpoint,
)
from .image_flow_loss_positionwise import PositionwiseFlowLoss


class JointDiTConfig(Qwen3Config):
    model_type = "selfless_joint_dit"


AutoConfig.register(JointDiTConfig.model_type, JointDiTConfig)


def joint_image_sigma(token_types, sigma=None):
    """Tie all positions in each contiguous image; never sample a reveal order.

    Missing sigma uses B's EOI-before-image convention. Supplied sigma retains
    the caller's text/segment ordering but collapses every image to its first
    position's rank. Each image must occupy a contiguous physical block.
    """
    image = token_types.eq(1)
    positions = torch.arange(token_types.shape[1], device=token_types.device)[None].expand_as(token_types)
    previous_image = F.pad(image[:, :-1], (1, 0), value=False)
    starts = torch.where(image & ~previous_image, positions, 0).cummax(-1).values
    if sigma is None:
        values = positions.float()
        values = torch.where(~image & previous_image, starts.float(), values)
        image_values = starts.float() + 1
    else:
        values = sigma.to(device=token_types.device, dtype=torch.float32)
        image_values = values.gather(1, starts)
    return torch.where(image, image_values, values)


def joint_backbone_masks(input_ids, token_types, sigma=None, *, segment_ids=None,
                         image_uncond_rows=None, image_uncond_mask=None, boi_token_id=None):
    from utils.utils import get_selfless_mask
    sigma = joint_image_sigma(token_types, sigma)
    if segment_ids is None:
        segment_ids = torch.where(token_types.ne(3), 0, -1)
    else:
        segment_ids = torch.where(token_types.ne(3), segment_ids, -1)
    args = dict(sigma=sigma, seq_len=input_ids.shape[1], device=input_ids.device,
                input_ids=input_ids, token_types=token_types, boi_token_id=boi_token_id,
                segment_ids=segment_ids, image_uncond_rows=image_uncond_rows,
                image_uncond_mask=image_uncond_mask)
    return get_selfless_mask(**args), get_selfless_mask(**args, include_diagonal=True)


class JointDiT(Showo2FlowHead):
    """256 noisy tokens, full self-attention, fixed per-position conditions."""
    def __init__(self, config):
        # Reuse S2's parameter-matched Transformer blocks, with a separate
        # latent stem because the backbone never sees x_t or t in this arm.
        head_config = JointDiTConfig(**{k: v for k, v in config.to_dict().items() if k != "model_type"})
        head_config.s2_flow_head_dim = int(getattr(config, "joint_dit_head_dim", 64))
        head_config.s2_flow_intermediate = int(getattr(config, "joint_dit_intermediate", 1472))
        super().__init__(head_config)
        self.latent_proj = nn.Linear(config.image_latent_dim, config.image_flow_width)
        self.image_tokens = int(config.image_tokens_per_img)

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.zero_output_modulation()

    @staticmethod
    def position_contract():
        return {"schema": "joint_dit_position_v1", "architecture": "single_stream_dit",
                "image_qk_rotary": "row_col_2d", "cross_token_attention": True,
                "uses_clean_target_content": False, "shared_time_per_image": True}

    @staticmethod
    def cache_contract():
        return {"schema": "joint_dit_fixed_condition_v1", "content_cache": False,
                "backbone_condition_cached": True, "refresh_backbone_each_velocity": False}

    def forward(self, x_t, times, condition):
        if x_t.ndim != 3 or x_t.shape[:2] != condition.shape[:2] or x_t.shape[1] != self.image_tokens:
            raise ValueError("Joint DiT requires aligned, complete [batch, image_tokens, channels] grids")
        if times.shape != (x_t.shape[0],):
            raise ValueError("Joint DiT requires one shared time per image")
        dtype = self.input_proj.weight.dtype
        x = self.latent_proj(x_t.to(dtype)) + self.input_proj(condition.to(dtype))
        time = self.time_embedder(times * self.time_scale)
        side = math.isqrt(self.image_tokens)
        local = torch.arange(self.image_tokens, device=x.device)
        position_ids = torch.stack((local // side, local % side))[:, None].expand(-1, x.shape[0], -1)
        rope = self.rotary_emb(x, position_ids)
        image_positions = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        mask = _prepared_mask(attention_from_allowed(torch.ones(
            x.shape[0], self.image_tokens, self.image_tokens, device=x.device, dtype=torch.bool)))
        for layer in self.layers:
            x = _checkpoint(layer, x, time, image_positions, mask, rope,
                            enabled=self.training and self.gradient_checkpointing)
        shift, scale = self.final_modulation(time).chunk(2, -1)
        x = (self.norm(x).float() * (1 + scale[:, None].float()) + shift[:, None].float()).to(x.dtype)
        return self.output_proj(x)


class JointDiTFlowLoss(PositionwiseFlowLoss):
    """Whole-image RF objective; B's outer forward supplies the four MC draws."""
    def __init__(self, config):
        nn.Module.__init__(self)
        self.net = JointDiT(config)
        self.in_channels = int(config.image_latent_dim)
        self.image_tokens_per_img = int(config.image_tokens_per_img)
        self.num_sampling_steps = int(config.image_flow_num_sampling_steps)
        self.time_sampling = str(getattr(config, "image_flow_time_sampling", "logit_normal"))
        self.logit_mean = float(getattr(config, "image_flow_logit_mean", 0))
        self.logit_std = float(getattr(config, "image_flow_logit_std", 1))
        self.time_eps = float(getattr(config, "image_flow_time_eps", 1e-5))
        self.uniform_mix = float(getattr(config, "image_flow_time_uniform_mix", .1))
        self.solver = str(getattr(config, "image_flow_solver", "euler"))
        if self.solver != "euler" or self.num_sampling_steps < 1:
            raise ValueError("Joint DiT uses Euler: one head evaluation per sampling step")
        if not 0 <= self.uniform_mix <= 1 or not 0 <= self.time_eps < .5:
            raise ValueError("Invalid joint flow time distribution")
        self.last_forward_stats = {}
        self.collect_guidance_diagnostics = False

    def velocity(self, x_t, t, z, **unused):
        return self.net(x_t, t, z)

    def forward(self, target, z, mask=None, sigma=None, image_positions=None,
                context_latents=None, record_stats=True, **unused):
        # Clean same-image content and sigma are deliberately absent from DiT.
        del sigma, image_positions, context_latents, unused
        target = target.float()
        t = self._sample_times((target.shape[0],), target.device)
        noise = torch.randn_like(target)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        prediction = self.velocity(x_t, t, z)
        token_loss = (prediction.float() - (target - noise)).square().mean(-1)
        weights = torch.ones_like(token_loss) if mask is None else mask.to(token_loss)
        loss = (token_loss * weights).sum() / weights.sum().clamp_min(1)
        self.last_forward_stats = ({
            "flow/loss": loss.detach(), "flow/v_mse": loss.detach(),
            "flow/t_mean": t.detach().mean(), "flow/t_min": t.detach().min(),
            "flow/t_max": t.detach().max(),
            "flow/v_pred_rms": prediction.detach().float().square().mean().sqrt(),
            "flow/nonfinite_count": (~torch.isfinite(prediction)).sum().float(),
        } if record_stats else {})
        return loss

    @torch.no_grad()
    def sample(self, z, temperature=1., cfg=1., cfg_schedule="constant", solver=None,
               num_steps=None, initial_noise=None, return_trace=False, debug_finite=False, **unused):
        steps = self.num_sampling_steps if num_steps is None else int(num_steps)
        if (solver or self.solver) != "euler" or steps < 1:
            raise ValueError("Joint DiT requires Euler and positive steps")
        paired = float(cfg) != 1.
        if paired and z.shape[0] % 2:
            raise ValueError("CFG conditions must contain paired conditional/unconditional rows")
        batch = z.shape[0] // (2 if paired else 1)
        shape = (batch, self.image_tokens_per_img, self.in_channels)
        if initial_noise is not None and tuple(initial_noise.shape) != shape:
            raise ValueError(f"initial_noise must have shape {shape}")
        x = (torch.randn(shape, device=z.device, dtype=torch.float32) if initial_noise is None
             else initial_noise.to(device=z.device, dtype=torch.float32).clone()) * float(temperature)
        for step in range(steps):
            t = torch.full((batch,), step / steps, device=z.device, dtype=torch.float32)
            cfg_now = self._scheduled_cfg(float(cfg), cfg_schedule, step / steps)
            x = x + self._guided_velocity(x, t, z, cfg_now).float() / steps
            if debug_finite and not torch.isfinite(x).all():
                raise FloatingPointError(f"Nonfinite joint DiT state at step {step}")
        output = x.to(self.net.input_proj.weight.dtype)
        return (output, {"solver": "euler", "steps": steps, "flow_head_calls": steps}) if return_trace else output


class JointDiTForCausalLM(Qwen3ForCausalLM):
    config_class = JointDiTConfig
    architecture_variant = "selfless_joint_dit"

    def __init__(self, config):
        required = {"training_objective": "selfless_dual_stream",
                    "dual_stream_attention_contract": "xlnet_content_diagonal",
                    "flow_head_attention_contract": "joint_bidirectional",
                    "flow_condition_contract": "backbone_xt_fixed",
                    "training_image_sigma_order": "joint"}
        for field, expected in required.items():
            if getattr(config, field, None) != expected:
                raise ValueError(f"Joint DiT requires {field}={expected}")
        if float(getattr(config, "image_input_noise_strength", 0)) != 0:
            raise ValueError("Joint DiT backbone requires clean image context")
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.image_latent_dim = config.image_latent_dim
        self.image_flow_batch_mul = int(config.image_flow_batch_mul)
        self.training_objective = config.training_objective
        self.flow_condition_contract = config.flow_condition_contract
        self.lambda_text = float(config.lambda_text)
        self.lambda_image = float(config.lambda_image)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_flow_condition_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.image_flow_head = JointDiTFlowLoss(config)
        self.post_init()
        self.reset_image_modules()

    def forward(self, X0_input_ids=None, *, token_types=None, flow_sigma=None,
                joint_image_uncond_rows=None, joint_image_uncond_mask=None, **kwargs):
        if token_types is not None:
            flow_sigma = joint_image_sigma(token_types, flow_sigma)
            query, content = joint_backbone_masks(X0_input_ids, token_types, flow_sigma,
                segment_ids=kwargs.get("_text_segment_ids"), boi_token_id=self.config.boi_token_id,
                image_uncond_rows=joint_image_uncond_rows, image_uncond_mask=joint_image_uncond_mask)
            kwargs.update(attention_mask=query, content_attention_mask=content)
        return super().forward(X0_input_ids=X0_input_ids, token_types=token_types,
                               flow_sigma=flow_sigma, **kwargs)

    @torch.no_grad()
    def generate_image(self, input_ids, token_types, sigma, spans, *, segment_ids=None,
                       image_latent_dim=None, initial_image_latents=None,
                       initial_image_latent_mask=None, initial_noise_bank=None,
                       flow_temperature=1., flow_cfg=3.5, flow_cfg_schedule="constant",
                       flow_solver=None, flow_num_steps=None, parallel_rate=1,
                       order_strategy="joint", use_cache=False, return_trace=False,
                       debug_finite=False, **kwargs):
        del parallel_rate, order_strategy, use_cache
        if kwargs:
            raise TypeError(f"Unsupported joint image generation arguments: {sorted(kwargs)}")
        dim = self.image_latent_dim
        if image_latent_dim not in (None, dim):
            raise ValueError("Generation latent dimension differs from training")
        if len(spans) != input_ids.shape[0] or sorted(row for row, _, _ in spans) != list(range(input_ids.shape[0])):
            raise ValueError("Joint generation requires exactly one target image per batch row")
        for row, start, end in spans:
            if not 0 <= start < end <= input_ids.shape[1] or end - start != self.config.image_tokens_per_img:
                raise ValueError("Each target span must contain the complete image grid")
            if not token_types[row, start:end].eq(1).all():
                raise ValueError("Target spans must refer to image tokens")
        table = torch.tensor(spans, device=input_ids.device, dtype=torch.long)
        rows = table[:, 0]
        indices = table[:, 1, None] + torch.arange(self.config.image_tokens_per_img, device=input_ids.device)
        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        target_mask[rows[:, None], indices] = True
        shape = (*input_ids.shape, dim)
        latents = (torch.zeros(shape, device=input_ids.device) if initial_image_latents is None
                   else initial_image_latents.to(input_ids.device).clone())
        if tuple(latents.shape) != shape:
            raise ValueError("Initial image latents must align with the full sequence")
        latent_mask = token_types.eq(1) & ~target_mask
        if initial_image_latent_mask is not None:
            if (latent_mask & ~initial_image_latent_mask.to(latent_mask)).any():
                raise ValueError("All context image tokens must be observed")
        elif initial_image_latents is None and latent_mask.any():
            raise ValueError("Clean latents are required for context images")
        # Target content slots are masks even if a caller supplied target x0.
        latents = torch.where(target_mask[..., None], 0., latents)
        sigma = joint_image_sigma(token_types, sigma)
        segments = (torch.where(token_types.ne(3), 0, -1) if segment_ids is None else segment_ids)
        paired = float(flow_cfg) != 1.
        def repeat(value):
            return torch.cat((value, value), 0) if paired else value
        uncond = torch.zeros_like(token_types, dtype=torch.bool)
        if paired:
            uncond = torch.cat((uncond, target_mask), 0)
        query, content = joint_backbone_masks(repeat(input_ids), repeat(token_types), repeat(sigma),
            segment_ids=repeat(segments), boi_token_id=self.config.boi_token_id,
            image_uncond_mask=uncond)
        hidden = self.model(X0_input_ids=repeat(input_ids), token_types=repeat(token_types),
            attention_mask=query, content_attention_mask=content, image_latents=repeat(latents),
            image_latent_mask=repeat(latent_mask), calculate_likelihood=True, use_cache=False).last_hidden_state
        selected = hidden[rows[:, None], indices]
        if paired:
            selected = torch.cat((selected, hidden[(rows + input_ids.shape[0])[:, None], indices]), 0)
        condition = self._prepare_image_flow_condition(selected)
        noise = None if initial_noise_bank is None else initial_noise_bank.to(input_ids.device)
        if noise is not None and tuple(noise.shape) == shape:
            noise = noise[rows[:, None], indices]
        generated, head_trace = self.image_flow_head.sample(condition, temperature=flow_temperature,
            cfg=flow_cfg, cfg_schedule=flow_cfg_schedule, solver=flow_solver,
            num_steps=flow_num_steps, initial_noise=noise, return_trace=True, debug_finite=debug_finite)
        side = math.isqrt(self.config.image_tokens_per_img)
        output = generated.reshape(len(spans), side, side, dim).permute(0, 3, 1, 2)
        trace = {**head_trace, "generation_mode": "joint_dit_full_image_flow", "order_strategy": "joint",
                 "backbone_calls": 1, "backbone_kv_cache_enabled": False,
                 "backbone_condition_cached": True, "backbone_streams": 2, "flow_head_streams": 1,
                 "cfg_batched": paired, "cfg": float(flow_cfg),
                 "ode_function_evals": head_trace["steps"], "shared_time_per_image": True}
        return (output, trace) if return_trace else output

    def generate_text(self, input_ids, *, token_types=None, sigma=None, **kwargs):
        if token_types is not None:
            sigma = joint_image_sigma(token_types, sigma)
        return super().generate_text(input_ids, token_types=token_types, sigma=sigma, **kwargs)

    def _build_generation_cache_mask(self, *, key_sigma, key_valid, key_is_target_image,
            query_sigma, query_valid, query_positions, content_query_mask,
            image_uncond_rows=None, content_self_diagonal=None):
        # Text generation prefills each clean context image bidirectionally.
        # A newly predicted text token remains a strict mask query.
        del key_is_target_image, query_positions, content_self_diagonal
        if image_uncond_rows is not None:
            raise ValueError("Joint DiT CFG belongs to full-image generation")
        allowed = ((key_sigma[:, None] < query_sigma[..., None])
                   | (content_query_mask[..., None] & (key_sigma[:, None] <= query_sigma[..., None])))
        return attention_from_allowed(allowed & query_valid[..., None] & key_valid[:, None])

    def _build_generation_attention_mask(self, *, input_ids, token_types, sigma,
            segment_ids, content_query_mask, image_uncond_rows=None, content_self_diagonal=None):
        del input_ids, image_uncond_rows, content_self_diagonal
        sigma = joint_image_sigma(token_types, sigma)
        allowed = ((sigma[:, None] < sigma[..., None])
                   | (content_query_mask[..., None] & (sigma[:, None] <= sigma[..., None])))
        valid = token_types.ne(3) & segment_ids.ge(0)
        return attention_from_allowed(allowed & valid[..., None] & valid[:, None]
                                      & segment_ids[..., None].eq(segment_ids[:, None]))
