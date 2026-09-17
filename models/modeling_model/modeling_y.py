"""Y: bidirectional visible image content and independent conditional flow tokens."""
from __future__ import annotations

import math

import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .image_flow_loss_positionwise import PositionwiseFlowLoss
from .modeling_joint_dit import JointDiTForCausalLM, joint_image_sigma
from .modeling_selfless_flow import Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel
from .modeling_showo2_unified import attention_from_allowed


class YConfig(Qwen3Config):
    model_type = "selfless_y"


AutoConfig.register(YConfig.model_type, YConfig)


def cosine_reveal_counts(tokens, rounds):
    if not 1 <= rounds <= tokens:
        raise ValueError("Y requires 1 <= reveal_steps <= image_tokens")
    remaining, counts = tokens, []
    for step in range(rounds):
        following = (0 if step == rounds - 1 else max(1, min(remaining - 1,
                     math.floor(tokens * math.cos(math.pi / 2 * (step + 1) / rounds)))))
        counts.append(remaining - following)
        remaining = following
    return counts


def y_backbone_allowed(input_ids, token_types, sigma, visible, target, *,
                       segment_ids=None, uncond_mask=None, boi_token_id=None):
    """Separate content validity from query validity; -1 is never a rank mask."""
    image = token_types.eq(1)
    sigma = joint_image_sigma(token_types, sigma)
    valid = token_types.ne(3)
    if segment_ids is None:
        segment_ids = torch.where(valid, 0, -1)
    valid = valid & segment_ids.ge(0)
    kv_valid = valid & (~image | visible)
    same_segment = segment_ids[:, :, None].eq(segment_ids[:, None, :])
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None].expand_as(input_ids)
    previous_image = torch.nn.functional.pad(image[:, :-1], (1, 0), value=False)
    starts = torch.where(image & ~previous_image, positions, 0).cummax(-1).values
    same_image = image[:, :, None] & image[:, None, :] & starts[:, :, None].eq(starts[:, None, :])
    content = sigma[:, None, :] <= sigma[:, :, None]
    query = (sigma[:, None, :] < sigma[:, :, None]) | (target[:, :, None] & same_image)
    permitted = valid[:, :, None] & kv_valid[:, None, :] & same_segment
    content = content & permitted
    query = query & permitted
    if uncond_mask is not None:
        if boi_token_id is None:
            raise ValueError("Y CFG requires a BOI token")
        boi = input_ids.eq(boi_token_id)
        span = boi.long().cumsum(-1)
        same_span = span[:, :, None].eq(span[:, None, :]) & same_segment
        # BOI is a safe unconditional anchor, including the empty-V state.
        # Isolate it too, otherwise visible image content could relay prompt text.
        dropped_boi = boi & (same_span & uncond_mask[:, None, :]).any(-1)
        dropped_queries = uncond_mask | dropped_boi
        uncond_allowed = same_span & (image | boi)[:, None, :] & permitted
        query = torch.where(uncond_mask[:, :, None], uncond_allowed, query)
        content = torch.where(dropped_queries[:, :, None], uncond_allowed, content)
        content = content & (~dropped_boi[:, :, None] | boi[:, None, :])
    # Invalid content slots contain mask embeddings, never clean targets. Their
    # private self edge is numerical padding; no effective row can read them.
    diagonal = torch.eye(input_ids.shape[1], device=input_ids.device, dtype=torch.bool)[None]
    content = content | (diagonal & ~kv_valid[:, :, None])
    return query, content


class YFlowLoss(PositionwiseFlowLoss):
    """Independent token RF with equal image weight, even for different |U|."""

    def forward(self, target, z, mask=None, record_stats=True, **unused):
        del unused
        weights = torch.ones_like(target[..., 0], dtype=torch.bool) if mask is None else mask.bool()
        if not weights.any():
            zero = z.float().sum() * 0.
            for parameter in self.parameters():
                zero = zero + parameter.reshape(-1)[0].float() * 0.
            self.last_forward_stats = {}
            return zero
        counts = weights.sum(-1)
        indices = weights.nonzero(as_tuple=True)
        clean = target[indices].float()
        condition = z[indices]
        times = self._sample_times(clean.shape[:-1], clean.device)
        noise = torch.randn_like(clean)
        noisy = (1 - times[:, None]) * noise + times[:, None] * clean
        prediction = self.velocity(noisy, times, condition)
        errors = (prediction.float() - (clean - noise)).square().mean(-1)
        per_image = torch.zeros(target.shape[0], device=target.device).scatter_add(0, indices[0], errors)
        active = counts.gt(0)
        loss = (per_image / counts.clamp_min(1) * active).sum() / active.sum().clamp_min(1)
        self.last_forward_stats = ({"flow/loss": loss.detach(), "flow/v_mse": loss.detach(),
            "flow/unknown_fraction": weights.float().mean().detach(),
            "flow/t_mean": times.detach().mean(),
            "flow/nonfinite_count": (~torch.isfinite(prediction)).sum().float()} if record_stats else {})
        return loss


class YForCausalLM(JointDiTForCausalLM):
    """Reuse Z's full-image I2T/text behavior, replace T2I attention and head."""
    config_class = YConfig
    architecture_variant = "selfless_y"

    def __init__(self, config):
        required = dict(training_objective="selfless_dual_stream",
                        dual_stream_attention_contract="xlnet_content_diagonal",
                        flow_head_attention_contract="not_applicable",
                        flow_condition_contract="backbone_xt_fixed",
                        training_image_sigma_order="joint")
        for key, expected in required.items():
            if getattr(config, key, None) != expected:
                raise ValueError(f"Y requires {key}={expected}")
        config.y_empty_visible_prob = float(getattr(config, "y_empty_visible_prob", .1))
        config.y_reveal_steps = int(getattr(config, "y_reveal_steps", 8))
        if not 0 <= config.y_empty_visible_prob <= 1:
            raise ValueError("Y empty-visible probability must be in [0,1]")
        cosine_reveal_counts(config.image_tokens_per_img, config.y_reveal_steps)
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.image_latent_dim = config.image_latent_dim
        self.image_flow_batch_mul = int(config.image_flow_batch_mul)
        self.training_objective = config.training_objective
        self.flow_condition_contract = config.flow_condition_contract
        self.lambda_text, self.lambda_image = float(config.lambda_text), float(config.lambda_image)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_flow_condition_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.image_flow_head = YFlowLoss(target_channels=config.image_latent_dim,
            z_channels=config.hidden_size, width=config.image_flow_width, depth=config.image_flow_depth,
            num_sampling_steps=config.image_flow_num_sampling_steps,
            grad_checkpointing=getattr(config, "image_flow_grad_checkpointing", False),
            time_scale=getattr(config, "image_flow_time_scale", 1000.),
            time_sampling=getattr(config, "image_flow_time_sampling", "logit_normal"),
            logit_mean=getattr(config, "image_flow_logit_mean", 0.),
            logit_std=getattr(config, "image_flow_logit_std", 1.),
            time_eps=getattr(config, "image_flow_time_eps", 1e-5),
            uniform_mix=getattr(config, "image_flow_time_uniform_mix", .1),
            solver=getattr(config, "image_flow_solver", "heun"),
            image_tokens_per_img=config.image_tokens_per_img)
        self.post_init()
        self.reset_backbone_attention_output_gates()
        self.reset_image_modules()

    def _sample_visible(self, types, target, spans):
        visible = types.eq(1) & ~target
        if spans is None or spans.shape[0] == 0:
            return visible
        count, n = spans.shape[0], self.config.image_tokens_per_img
        progress = torch.rand(count, device=types.device)
        unknown = torch.floor(n * torch.cos(math.pi / 2 * progress)).long().clamp(1, n)
        empty = torch.rand(count, device=types.device) < self.config.y_empty_visible_prob
        unknown = torch.where(empty, n, unknown)
        ranks = torch.rand(count, n, device=types.device).argsort(-1).argsort(-1)
        rows = spans[:, 0, None]
        positions = spans[:, 2, None] + torch.arange(n, device=types.device)
        visible[rows, positions] |= (ranks >= unknown[:, None]) & target[rows, positions]
        return visible

    def forward(self, X0_input_ids=None, *, token_types=None, flow_sigma=None,
                y_visible_mask=None, y_image_uncond_rows=None, y_image_uncond_mask=None, **kwargs):
        if token_types is not None:
            image = token_types.eq(1)
            target = kwargs.get("image_loss_mask")
            if target is None or kwargs.get("compute_image_loss") is False:
                target = torch.zeros_like(image)
            else:
                target = target.to(device=image.device, dtype=torch.bool) & image
            if not target.any():
                visible = image
            elif y_visible_mask is None:
                visible = self._sample_visible(token_types, target, kwargs.get("image_span_table"))
            else:
                if y_visible_mask.shape != image.shape:
                    raise ValueError("Y visible mask must align with the full sequence")
                visible = (y_visible_mask.to(image.device).bool() & target) | (image & ~target)
            unknown = target & ~visible
            if kwargs.get("compute_image_loss") and not unknown.any():
                raise ValueError("Y T2I requires at least one unknown target")
            uncond = y_image_uncond_mask
            if uncond is None and y_image_uncond_rows is not None:
                uncond = target & y_image_uncond_rows[:, None]
            if uncond is not None:
                uncond = uncond & target
            query, content = y_backbone_allowed(X0_input_ids, token_types, flow_sigma, visible, target,
                segment_ids=kwargs.get("_text_segment_ids"), uncond_mask=uncond,
                boi_token_id=self.config.boi_token_id)
            kwargs.update(attention_mask=attention_from_allowed(query),
                          content_attention_mask=attention_from_allowed(content),
                          image_latent_mask=visible, image_loss_mask=unknown)
            flow_sigma = joint_image_sigma(token_types, flow_sigma)
        # B's objective owns loss, RF4 and clean/noisy input separation; skip
        # Z.forward, which would overwrite Y's masks with a tied full-image mask.
        return Qwen3ForCausalLM.forward(self, X0_input_ids=X0_input_ids, token_types=token_types,
                                       flow_sigma=flow_sigma, **kwargs)

    @torch.no_grad()
    def generate_image(self, input_ids, token_types, sigma, spans, *, segment_ids=None,
                       image_latent_dim=None, initial_image_latents=None,
                       initial_image_latent_mask=None, initial_noise_bank=None,
                       flow_temperature=1., flow_cfg=3.5, flow_cfg_schedule="constant",
                       flow_solver=None, flow_num_steps=None, parallel_rate=1,
                       order_strategy="random", use_cache=False, return_trace=False,
                       debug_finite=False, reveal_steps=None, reveal_order=None, **kwargs):
        del parallel_rate, use_cache
        if kwargs:
            raise TypeError(f"Unsupported Y generation arguments: {sorted(kwargs)}")
        if order_strategy not in {"random", "cosine", "joint", "sequential"}:
            raise ValueError("Y supports random/cosine or sequential position orders")
        batch, length = input_ids.shape
        n, dim = self.config.image_tokens_per_img, self.image_latent_dim
        rounds = self.config.y_reveal_steps if reveal_steps is None else int(reveal_steps)
        counts = cosine_reveal_counts(n, rounds)
        if flow_num_steps is not None and int(flow_num_steps) < 1:
            raise ValueError("Y flow_num_steps must be positive")
        if image_latent_dim not in (None, dim):
            raise ValueError("Y latent dimension differs from checkpoint")
        if len(spans) != batch or sorted(row for row, _, _ in spans) != list(range(batch)):
            raise ValueError("Y generation requires one complete target image per row")
        for row, start, end in spans:
            if not 0 <= start < end <= length or end - start != n or not token_types[row, start:end].eq(1).all():
                raise ValueError("Y target spans must contain a complete image grid")
        table = torch.tensor(sorted(spans), device=input_ids.device, dtype=torch.long)
        rows = table[:, 0, None]
        positions = table[:, 1, None] + torch.arange(n, device=input_ids.device)
        target = torch.zeros_like(input_ids, dtype=torch.bool)
        target[rows, positions] = True
        visible = token_types.eq(1) & ~target
        shape = (batch, length, dim)
        latents = (torch.zeros(shape, device=input_ids.device) if initial_image_latents is None else
                   initial_image_latents.to(input_ids.device).clone())
        if tuple(latents.shape) != shape:
            raise ValueError("Y initial image latents must align with the sequence")
        if visible.any() and initial_image_latents is None:
            raise ValueError("Y requires clean latents for context images")
        if initial_image_latent_mask is not None and (visible & ~initial_image_latent_mask.to(visible)).any():
            raise ValueError("All context image tokens must be observed")
        latents.masked_fill_(target[..., None], 0.)
        if reveal_order is None:
            order = (torch.arange(n, device=input_ids.device)[None].expand(batch, -1) if order_strategy == "sequential"
                     else torch.rand(batch, n, device=input_ids.device).argsort(-1))
        else:
            order = reveal_order.to(device=input_ids.device, dtype=torch.long)
            if order.shape != (batch, n) or not torch.equal(order.sort(-1).values,
                    torch.arange(n, device=input_ids.device)[None].expand(batch, -1)):
                raise ValueError("Y reveal order must be a permutation per image")
        noise = (torch.randn(batch, n, dim, device=input_ids.device) if initial_noise_bank is None
                 else initial_noise_bank.to(input_ids.device))
        if tuple(noise.shape) == shape:
            noise = noise[rows, positions]
        if tuple(noise.shape) != (batch, n, dim):
            raise ValueError("Y noise must align with the target grids or full sequence")
        paired = float(flow_cfg) != 1.
        def repeat(value):
            return torch.cat((value, value), 0) if paired else value
        sigma = joint_image_sigma(token_types, sigma)
        segments = torch.where(token_types.ne(3), 0, -1) if segment_ids is None else segment_ids
        uncond = torch.cat((torch.zeros_like(target), target), 0) if paired else None
        offset = 0
        for added in counts:
            query, content = y_backbone_allowed(repeat(input_ids), repeat(token_types), repeat(sigma),
                repeat(visible), repeat(target), segment_ids=repeat(segments), uncond_mask=uncond,
                boi_token_id=self.config.boi_token_id)
            hidden = self.model(X0_input_ids=repeat(input_ids), token_types=repeat(token_types),
                attention_mask=attention_from_allowed(query), content_attention_mask=attention_from_allowed(content),
                image_latents=repeat(latents), image_latent_mask=repeat(visible),
                calculate_likelihood=True, use_cache=False).last_hidden_state
            selected = order[:, offset:offset + added]
            indices = positions.gather(1, selected)
            condition = hidden[rows, indices]
            if paired:
                condition = torch.cat((condition, hidden[rows + batch, indices]), 0)
            condition = self._prepare_image_flow_condition(condition)
            generated = self.image_flow_head.sample(condition, temperature=flow_temperature,
                cfg=flow_cfg, cfg_schedule=flow_cfg_schedule, solver=flow_solver,
                num_steps=flow_num_steps, initial_noise=noise.gather(1, selected[..., None].expand(-1, -1, dim)),
                debug_finite=debug_finite)
            latents[rows, indices] = generated.to(latents.dtype)
            visible[rows, indices] = True
            offset += added
        side = math.isqrt(n)
        output = latents[rows, positions].reshape(batch, side, side, dim).permute(0, 3, 1, 2)
        steps = int(flow_num_steps or self.image_flow_head.num_sampling_steps)
        solver = flow_solver or self.image_flow_head.solver
        evaluations = steps * (2 if solver == "heun" else 1)
        trace = dict(architecture_variant=self.architecture_variant, generation_mode="y_masked_token_flow",
            order_strategy=order_strategy, reveal_steps=rounds, reveal_counts=counts,
            reveal_order=order.cpu().tolist(), solver=solver, steps=steps,
            flow_solver=solver, flow_num_steps=steps, backbone_calls=rounds,
            flow_head_calls=rounds * evaluations, flow_head_token_evals=batch * n * evaluations * (2 if paired else 1),
            backbone_kv_cache_enabled=False, backbone_condition_cached=True, backbone_streams=2,
            flow_head_streams=1, shared_time_per_image=False, cfg_batched=paired, cfg=float(flow_cfg),
            flow_head_architecture="positionwise_adaln_mlp", visible_tokens_preserved=True)
        return (output, trace) if return_trace else output
