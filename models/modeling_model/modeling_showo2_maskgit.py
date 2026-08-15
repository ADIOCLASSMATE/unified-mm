"""Isolated Show-o2-style single-stream MaskGIT architecture ablation.

The production selfless model remains untouched.  This module deliberately
uses one Qwen stream, ordinary autoregressive text attention (including the
diagonal), bidirectional attention inside each image span, MaskGIT training,
and the independent position-wise flow head.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from utils.showo2_maskgit import (
    build_showo2_position_ids,
    get_showo2_attention_mask,
    sample_maskgit_training_masks,
)

from .image_flow_loss_positionwise import PositionwiseFlowLoss
from .modeling_positionwise_flow import PositionwiseFlowQwen3ForCausalLM
from .modeling_selfless_flow import (
    Qwen3Model,
    Qwen3PreTrainedModel,
    _debug_require_finite_tensor,
    _to_bool_atten_mask,
)


class ShowO2Qwen3Model(Qwen3Model):
    """Single Qwen stream with AR-text/full-image-block attention."""

    def forward(
        self,
        X0_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        X0_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        calculate_likelihood: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        del position_ids, calculate_likelihood
        if (X0_input_ids is None) ^ (X0_inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if use_cache or past_key_values is not None:
            raise ValueError(
                "Show-o2 MaskGIT uses full-sequence bidirectional image blocks "
                "and does not support incremental backbone KV caching."
            )

        debug_finite = bool(kwargs.get("debug_finite_backbone", False))
        debug_label = str(kwargs.get("debug_backbone_label", ""))
        token_types = kwargs.get("token_types", None)
        image_span_table = kwargs.pop("image_span_table", None)
        kwargs.pop("image_local_positions", None)
        image_spans_present = (
            image_span_table.shape[0] > 0
            if image_span_table is not None
            else None
        )

        if X0_inputs_embeds is None:
            X0_inputs_embeds = self._build_x0_inputs_embeds(
                input_ids=X0_input_ids,
                token_types=token_types,
                image_latents=kwargs.get("image_latents", None),
                image_latent_mask=kwargs.get("image_latent_mask", None),
                image_spans_present=image_spans_present,
                image_latents_are_noisy=bool(
                    kwargs.get("image_latents_are_noisy", False)
                ),
                debug_finite=debug_finite,
                debug_label=debug_label,
            )
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            "input_embeddings.showo2_single_stream",
            X0_inputs_embeds,
        )

        batch_size, seq_len = X0_inputs_embeds.shape[:2]
        device = X0_inputs_embeds.device
        if token_types is None:
            token_types = torch.zeros(
                batch_size,
                seq_len,
                device=device,
                dtype=torch.long,
            )
        else:
            token_types = token_types.to(device=device, dtype=torch.long)
        attention_mask_contract = kwargs.pop("attention_mask_contract", None)
        if attention_mask_contract == "showo2" and attention_mask is not None:
            showo2_attention_mask = attention_mask
        else:
            showo2_attention_mask = get_showo2_attention_mask(
                token_types,
                segment_ids=kwargs.get("segment_ids", None),
                image_uncond_rows=kwargs.get("image_uncond_rows", None),
                image_uncond_mask=kwargs.get("image_uncond_mask", None),
            )
        if isinstance(showo2_attention_mask, torch.Tensor):
            # Prepare the all-masked-row guard once, then reuse it in every
            # decoder layer.  The attention kernel itself remains the compact
            # GQA npu_fusion_attention path shared with production.
            showo2_attention_mask = _to_bool_atten_mask(showo2_attention_mask)

        if cache_position is None:
            cache_position = torch.arange(seq_len, device=device)
        physical_position_ids = build_showo2_position_ids(
            token_types,
            kwargs.get("segment_ids", None),
        )
        hidden_states = X0_inputs_embeds
        position_embeddings = self.rotary_emb(
            hidden_states,
            physical_position_ids,
        )
        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            hidden_states, unused_second_stream = decoder_layer(
                hidden_states,
                None,
                showo2_attention_mask,
                position_ids=physical_position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if unused_second_stream is not None:
                raise AssertionError("Show-o2 ablation unexpectedly created a second stream")
            _debug_require_finite_tensor(
                debug_finite,
                debug_label,
                f"layers.{layer_idx}.showo2_output",
                hidden_states,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )


class ShowO2MaskGITQwen3ForCausalLM(PositionwiseFlowQwen3ForCausalLM):
    """Single-stream Qwen + MaskGIT + position-wise rectified flow."""

    architecture_variant = "showo2_maskgit"

    def __init__(self, config):
        # Build this architecture directly: do not transiently allocate either
        # the production contextual head or the selfless two-stream backbone.
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = ShowO2Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.image_latent_dim = int(getattr(config, "image_latent_dim", 4))
        self.image_flow_batch_mul = int(
            getattr(config, "image_flow_batch_mul", 1)
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.image_flow_condition_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
        )
        self.image_flow_head = PositionwiseFlowLoss(
            target_channels=self.image_latent_dim,
            z_channels=config.hidden_size,
            width=getattr(config, "image_flow_width", 1280),
            depth=getattr(config, "image_flow_depth", 8),
            num_sampling_steps=getattr(
                config,
                "image_flow_num_sampling_steps",
                "10",
            ),
            grad_checkpointing=getattr(
                config,
                "image_flow_grad_checkpointing",
                False,
            ),
            time_scale=getattr(config, "image_flow_time_scale", 1000.0),
            time_sampling=getattr(
                config,
                "image_flow_time_sampling",
                "logit_normal",
            ),
            logit_mean=getattr(config, "image_flow_logit_mean", 0.0),
            logit_std=getattr(config, "image_flow_logit_std", 1.0),
            time_eps=getattr(config, "image_flow_time_eps", 1.0e-4),
            uniform_mix=getattr(
                config,
                "image_flow_time_uniform_mix",
                0.1,
            ),
            solver=getattr(config, "image_flow_solver", "heun"),
            image_tokens_per_img=getattr(config, "image_tokens_per_img", 256),
        )

        self.post_init()
        self.reset_backbone_attention_output_gates()
        self.reset_image_modules()

    def forward(
        self,
        X0_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_latents: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep=0,
        calculate_likelihood: bool = False,
        **kwargs,
    ):
        token_types = kwargs.get("token_types", None)
        image_span_table = kwargs.get("image_span_table", None)
        if (
            labels is not None
            and token_types is not None
            and image_span_table is not None
            and "image_latent_mask" not in kwargs
        ):
            validation_ratio = None
            if not self.training:
                validation_ratio = float(
                    getattr(
                        self.config,
                        "maskgit_validation_mask_ratio",
                        0.5,
                    )
                )
            visible, loss_mask, _ = sample_maskgit_training_masks(
                token_types,
                image_span_table,
                image_tokens_per_img=int(self.config.image_tokens_per_img),
                validation_mask_ratio=validation_ratio,
                flow_sigma=kwargs.get("flow_sigma", None),
            )
            kwargs["image_latent_mask"] = visible
            kwargs["image_loss_mask"] = loss_mask

        return super().forward(
            X0_input_ids=X0_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            image_latents=image_latents,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            calculate_likelihood=calculate_likelihood,
            **kwargs,
        )

    @torch.no_grad()
    def sample_image_latents_single_stream(
        self,
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        sigma: torch.Tensor,
        spans: list[tuple[int, int, int]],
        image_latent_dim: int | None = None,
        initial_image_latents: torch.Tensor | None = None,
        initial_image_latent_mask: torch.Tensor | None = None,
        initial_noise_bank: torch.Tensor | None = None,
        flow_temperature: float = 1.0,
        flow_cfg: float = 1.0,
        flow_cfg_schedule: str = "constant",
        flow_solver: str | None = None,
        flow_num_steps: int | None = None,
        parallel_rate: int = 1,
        order_strategy: str = "maskgit",
        use_backbone_cache: bool = True,
        return_trace: bool = False,
        debug_finite: bool = False,
        _debug_max_generation_steps: int | None = None,
    ):
        del sigma
        if not spans:
            return (None, {}) if return_trace else None
        if int(parallel_rate) != 1:
            raise ValueError(
                "MaskGIT owns its cosine reveal schedule; parallel_rate must remain 1."
            )
        order_strategy = str(order_strategy or "maskgit").lower()
        if order_strategy not in {"maskgit", "condition_norm", "hidden_norm"}:
            raise ValueError(
                f"Show-o2 ablation requires order_strategy='maskgit', got {order_strategy!r}."
            )

        device = input_ids.device
        num_images = len(spans)
        image_tokens = int(self.config.image_tokens_per_img)
        latent_dim = int(image_latent_dim or self.image_latent_dim)
        side = math.isqrt(image_tokens)
        if side * side != image_tokens:
            raise ValueError("image_tokens_per_img must be a square number")
        if any(end - start != image_tokens for _, start, end in spans):
            raise ValueError(
                "every generated image span must match image_tokens_per_img"
            )

        selected_input_ids = torch.stack(
            [input_ids[row] for row, _, _ in spans]
        ).to(device=device)
        selected_token_types = torch.stack(
            [token_types[row] for row, _, _ in spans]
        ).to(device=device)
        starts = torch.tensor(
            [start for _, start, _ in spans],
            device=device,
            dtype=torch.long,
        )
        image_rows = torch.arange(num_images, device=device, dtype=torch.long)
        offsets = torch.arange(image_tokens, device=device, dtype=torch.long)
        seq_positions = starts.unsqueeze(1) + offsets.unsqueeze(0)
        flow_dtype = self.image_flow_head.net.final_layer.linear.weight.dtype
        work_latents = torch.zeros(
            num_images,
            selected_input_ids.shape[1],
            latent_dim,
            device=device,
            dtype=flow_dtype,
        )
        visible_mask = torch.zeros_like(selected_token_types, dtype=torch.bool)
        if initial_image_latents is not None:
            if initial_image_latents.shape[:2] != input_ids.shape:
                raise ValueError(
                    "initial_image_latents must align with input_ids"
                )
            work_latents.copy_(
                torch.stack(
                    [initial_image_latents[row] for row, _, _ in spans]
                ).to(device=device, dtype=flow_dtype)
            )
        if initial_image_latent_mask is not None:
            if tuple(initial_image_latent_mask.shape) != tuple(input_ids.shape):
                raise ValueError(
                    "initial_image_latent_mask must align with input_ids"
                )
            if initial_image_latents is None and bool(
                initial_image_latent_mask.any().item()
            ):
                raise ValueError(
                    "initial_image_latents are required for visible initial tokens"
                )
            visible_mask = torch.stack(
                [initial_image_latent_mask[row] for row, _, _ in spans]
            ).to(device=device, dtype=torch.bool)
        if bool(
            visible_mask[image_rows.unsqueeze(1), seq_positions].any().item()
        ):
            raise ValueError(
                "partial target-span refinement is not part of the MaskGIT ablation"
            )

        selected_noise = None
        if initial_noise_bank is not None:
            expected = (num_images, image_tokens, latent_dim)
            if tuple(initial_noise_bank.shape) != expected:
                raise ValueError(
                    f"initial_noise_bank must have shape {expected}, got "
                    f"{tuple(initial_noise_bank.shape)}"
                )
            selected_noise = initial_noise_bank.to(
                device=device,
                dtype=torch.float32,
            )

        generated = torch.zeros(
            num_images,
            image_tokens,
            latent_dim,
            device=device,
            dtype=flow_dtype,
        )
        generation_order = torch.zeros(
            num_images,
            image_tokens,
            device=device,
            dtype=torch.long,
        )
        generation_step = torch.zeros_like(generation_order)
        generation_score = torch.zeros(
            num_images,
            image_tokens,
            device=device,
            dtype=torch.float32,
        )
        reveal_fraction = torch.zeros_like(generation_score)
        filled = torch.zeros_like(generation_order, dtype=torch.bool)
        span_table = torch.stack(
            [
                image_rows,
                image_rows,
                starts,
                starts + image_tokens,
            ],
            dim=1,
        )
        use_cfg = float(flow_cfg) != 1.0
        uncond_rows = (
            torch.ones(num_images, device=device, dtype=torch.bool)
            if use_cfg
            else None
        )
        maskgit_steps = min(
            image_tokens,
            max(1, int(getattr(self.config, "maskgit_generation_steps", 18))),
        )
        committed = 0

        for step_idx in range(1, maskgit_steps + 1):
            hidden = self.model(
                X0_input_ids=selected_input_ids,
                token_types=selected_token_types,
                image_latents=work_latents,
                image_latent_mask=visible_mask,
                image_span_table=span_table,
                calculate_likelihood=False,
                debug_finite_backbone=debug_finite,
                debug_backbone_label=f"showo2_maskgit_cond_step={step_idx}",
            ).last_hidden_state
            image_hidden = hidden[image_rows.unsqueeze(1), seq_positions]
            all_conditions = self._prepare_image_flow_condition(image_hidden)
            scores = all_conditions.detach().float().pow(2).mean(dim=-1).sqrt()
            scores = scores.masked_fill(filled, -torch.inf)

            progress = step_idx / maskgit_steps
            target_committed = min(
                image_tokens,
                max(
                    committed + 1,
                    math.ceil(
                        image_tokens
                        * (1.0 - math.cos(progress * math.pi / 2.0))
                    ),
                ),
            )
            fill_count = target_committed - committed
            selected_scores, local_positions = torch.topk(
                scores,
                k=fill_count,
                dim=1,
            )
            flat_rows = image_rows.unsqueeze(1).expand(-1, fill_count).reshape(-1)
            flat_local = local_positions.reshape(-1)
            z = all_conditions[flat_rows, flat_local]

            z_uncond = None
            if use_cfg:
                uncond_hidden = self.model(
                    X0_input_ids=selected_input_ids,
                    token_types=selected_token_types,
                    image_latents=work_latents,
                    image_latent_mask=visible_mask,
                    image_span_table=span_table,
                    image_uncond_rows=uncond_rows,
                    calculate_likelihood=False,
                    debug_finite_backbone=debug_finite,
                    debug_backbone_label=(
                        f"showo2_maskgit_uncond_step={step_idx}"
                    ),
                ).last_hidden_state
                uncond_image_hidden = uncond_hidden[
                    image_rows.unsqueeze(1),
                    seq_positions,
                ]
                all_uncond_conditions = self._prepare_image_flow_condition(
                    uncond_image_hidden
                )
                z_uncond = all_uncond_conditions[flat_rows, flat_local]

            initial_noise = None
            if selected_noise is not None:
                initial_noise = selected_noise[flat_rows, flat_local]
            pred = self.sample_image_flow_with_cfg(
                z,
                z_uncond=z_uncond,
                temperature=flow_temperature,
                cfg=flow_cfg,
                cfg_schedule=flow_cfg_schedule,
                solver=flow_solver,
                num_steps=flow_num_steps,
                initial_noise=initial_noise,
                debug_finite=debug_finite,
                debug_label=f"showo2_maskgit_flow_step={step_idx}",
            ).reshape(num_images, fill_count, latent_dim)

            selected_seq_positions = starts.unsqueeze(1) + local_positions
            work_latents[
                image_rows.unsqueeze(1),
                selected_seq_positions,
            ] = pred
            visible_mask[
                image_rows.unsqueeze(1),
                selected_seq_positions,
            ] = True
            generated[image_rows.unsqueeze(1), local_positions] = pred
            filled[image_rows.unsqueeze(1), local_positions] = True
            generation_order[image_rows.unsqueeze(1), local_positions] = (
                torch.arange(
                    committed + 1,
                    target_committed + 1,
                    device=device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(num_images, -1)
            )
            generation_step[image_rows.unsqueeze(1), local_positions] = step_idx
            generation_score[image_rows.unsqueeze(1), local_positions] = (
                selected_scores
            )
            reveal_fraction[image_rows.unsqueeze(1), local_positions] = (
                target_committed / image_tokens
            )
            committed = target_committed
            if (
                _debug_max_generation_steps is not None
                and step_idx >= int(_debug_max_generation_steps)
            ):
                break

        generated = generated.view(
            num_images,
            side,
            side,
            latent_dim,
        ).permute(0, 3, 1, 2)
        if return_trace:
            trace = {
                "order_strategy": "maskgit",
                "flow_cfg_schedule": str(flow_cfg_schedule),
                "generation_order": generation_order.view(
                    num_images,
                    side,
                    side,
                ),
                "generation_step": generation_step.view(
                    num_images,
                    side,
                    side,
                ),
                "generation_score": generation_score.view(
                    num_images,
                    side,
                    side,
                ),
                "reveal_fraction": reveal_fraction.view(
                    num_images,
                    side,
                    side,
                ),
                "flow_head_architecture": "positionwise_adaln_mlp",
                "flow_head_consumes_prior_latents": False,
                "backbone_architecture": "showo2_single_stream_maskgit",
                "backbone_kv_cache_enabled": False,
                "backbone_kv_cache_fallback_reason": (
                    "bidirectional image blocks require full-sequence MaskGIT rounds"
                    if use_backbone_cache
                    else None
                ),
                "backbone_kv_cache_context_tokens": 0,
                "backbone_kv_cache_tokens_committed": 0,
                "backbone_kv_cache_peak_bytes": 0,
                "flow_content_cache_peak_bytes_per_sample": 0,
                "flow_cfg_content_cache_divergence_by_layer": None,
                "maskgit_generation_steps": maskgit_steps,
            }
            return generated, trace
        return generated


__all__ = ["ShowO2MaskGITQwen3ForCausalLM", "ShowO2Qwen3Model"]
