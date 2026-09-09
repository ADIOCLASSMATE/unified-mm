"""Generation-only implementation for ablation D (Dynamic-``x_t``).

Every velocity evaluation rebuilds the XT query from the current ``x_t`` and
``t`` while reading a fixed X0 KV cache. The fused serialized backbone forward
also exposes the previous generated token's X0 content hidden; D uses that
hidden exclusively for the flow-content AdaLN cache update.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import create_block_mask


class DynamicXtGenerationMixin:
    """D-only image generation behavior.

    The mixin is intentionally separate from both the baseline model and the
    Dynamic-XT training model.  It only supplies the dynamic ODE condition
    evaluator and generation trace; generic decoding orchestration is reused
    through ``super().generate_image``.
    """

    _supports_paired_backbone_cfg = False

    def _order_probe_content_conditions(self, *, pending_hidden, previous_conditions):
        del previous_conditions
        if pending_hidden is None:
            raise RuntimeError("Dynamic-XT confidence probe lost pending X0 content")
        self._dynamic_xt_content_condition_commits += 1
        return self._prepare_image_flow_condition(pending_hidden)

    def sample_image_flow_with_cfg(self, z, *args, **kwargs):
        """Replace D's retired static-query content condition with fused X0."""

        pending_positions = kwargs.get("pending_context_positions")
        if pending_positions is not None:
            span_starts = getattr(
                self,
                "_dynamic_xt_generation_span_starts",
                None,
            )
            if span_starts is None:
                raise RuntimeError(
                    "Dynamic-XT pending flow content is missing span metadata"
                )
            batch_size = int(span_starts.numel())
            pending_positions = pending_positions.to(
                device=span_starts.device,
                dtype=torch.long,
            ).reshape(-1)
            paired = bool(kwargs.get("latent_mixer_cache_is_paired", False))
            expected_rows = batch_size * (2 if paired else 1)
            if int(pending_positions.numel()) != expected_rows:
                raise ValueError(
                    "Dynamic-XT pending flow positions have the wrong batch: "
                    f"{pending_positions.numel()} != {expected_rows}"
                )
            local_positions = pending_positions[:batch_size]
            if paired and not torch.equal(
                local_positions,
                pending_positions[batch_size:],
            ):
                raise ValueError(
                    "Dynamic-XT conditional/unconditional pending positions "
                    "must be identical"
                )
            physical_positions = span_starts + local_positions
            conditional_hidden = self.model.dynamic_xt_generation_x0_hidden(
                "conditional",
                physical_positions,
            )
            content_conditions = self._prepare_image_flow_condition(
                conditional_hidden
            )
            if paired:
                unconditional_hidden = (
                    self.model.dynamic_xt_generation_x0_hidden(
                        "unconditional",
                        physical_positions,
                    )
                )
                content_conditions = torch.cat(
                    [
                        content_conditions,
                        self._prepare_image_flow_condition(
                            unconditional_hidden
                        ),
                    ],
                    dim=0,
                )
            kwargs["pending_context_conditions"] = content_conditions
            self._dynamic_xt_content_condition_commits += 1
        return super().sample_image_flow_with_cfg(z, *args, **kwargs)

    def _make_backbone_flow_condition_evaluator(self, **state):
        sample_indices = state["sample_indices"]
        seq_positions = state["seq_positions"]
        selected_input_ids = state["selected_input_ids"]
        selected_token_types = state["selected_token_types"]
        current_sigma = state["current_sigma"]
        work_latents = state["work_latents"]
        use_cfg = bool(state["use_flow_cfg"])
        cache_enabled = bool(state["backbone_cache_enabled"])
        debug_finite = bool(state["debug_finite"])
        generation_step = int(state["generation_step"])
        batch_size = int(sample_indices.numel())
        multiple_queries = seq_positions.ndim == 2
        query_indices = seq_positions if multiple_queries else seq_positions.unsqueeze(1)
        query_count = query_indices.shape[1]
        device = selected_input_ids.device
        self._dynamic_xt_generation_span_starts = state["span_starts"]

        def cache_mask(image_uncond: bool):
            query_sigma = (torch.gather(current_sigma, 1, query_indices) if multiple_queries else
                           current_sigma[sample_indices, seq_positions].unsqueeze(1))
            allowed = state["backbone_key_valid"].unsqueeze(1) & (
                state["backbone_key_sigma"].unsqueeze(1)
                < query_sigma.unsqueeze(-1)
            )
            if image_uncond:
                allowed &= state["backbone_key_is_image"].unsqueeze(1)
            if device.type == "npu":
                return (~allowed).unsqueeze(1)

            def mask_mod(b, h, q_idx, kv_idx):
                del h
                return allowed[b, q_idx, kv_idx]

            return create_block_mask(
                mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=query_count,
                KV_LEN=int(state["backbone_max_cache_len"]),
                device=device,
            )

        def evaluate_branch(x_t, t, *, image_uncond: bool):
            aligned_x_t = x_t if multiple_queries else x_t.unsqueeze(1)
            aligned_t = t if multiple_queries else t.unsqueeze(1)
            dynamic_query_mask = torch.ones(
                batch_size,
                query_count,
                device=device,
                dtype=torch.bool,
            )
            if cache_enabled:
                expected = torch.arange(batch_size, device=device)
                if not torch.equal(sample_indices, expected):
                    raise RuntimeError(
                        "Dynamic-XT cached flow evaluation requires one query per row"
                    )
                position_ids = torch.gather(
                    state["full_position_ids"],
                    dim=2,
                    index=query_indices.unsqueeze(0).expand(2, -1, -1),
                )
                query_latents = (torch.gather(work_latents, 1,
                    query_indices.unsqueeze(-1).expand(-1, -1, self.image_latent_dim))
                    if multiple_queries else work_latents[sample_indices, seq_positions].unsqueeze(1))
                hidden = self.model(
                    X0_input_ids=torch.gather(
                        selected_input_ids,
                        1,
                        query_indices,
                    ),
                    attention_mask=cache_mask(image_uncond),
                    position_ids=position_ids,
                    past_key_values=(
                        state["backbone_uncond_cache"]
                        if image_uncond
                        else state["backbone_cond_cache"]
                    ),
                    use_cache=False,
                    cache_position=query_indices,
                    cache_read_only=True,
                    token_types=torch.gather(
                        selected_token_types,
                        1,
                        query_indices,
                    ),
                    image_latents=query_latents,
                    image_latent_mask=torch.zeros_like(dynamic_query_mask),
                    calculate_likelihood=True,
                    xt_flow_latents=aligned_x_t,
                    xt_flow_times=aligned_t,
                    xt_flow_query_mask=dynamic_query_mask,
                    debug_finite_backbone=debug_finite,
                    debug_backbone_label=(
                        f"dynamic_xt_{'uncond' if image_uncond else 'cond'}_"
                        f"flow_eval_generation_step={generation_step}"
                    ),
                ).last_hidden_state
                if not multiple_queries:
                    hidden = hidden[:, 0]
            else:
                if multiple_queries:
                    raise ValueError("Dynamic-XT candidate probes require a read-only backbone cache")
                full_x_t = torch.zeros(
                    *selected_input_ids.shape,
                    self.image_latent_dim,
                    device=device,
                    dtype=x_t.dtype,
                )
                full_t = torch.zeros_like(current_sigma, dtype=torch.float32)
                full_dynamic_query_mask = torch.zeros_like(
                    selected_token_types,
                    dtype=torch.bool,
                )
                full_x_t[sample_indices, seq_positions] = x_t
                full_t[sample_indices, seq_positions] = t
                full_dynamic_query_mask[sample_indices, seq_positions] = True
                image_latent_mask = state["base_image_latent_mask"].clone()
                offsets = torch.arange(
                    int(state["image_tokens_per_img"]),
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                span_indices = state["span_starts"].unsqueeze(1) + offsets
                span_rows = torch.arange(
                    state["span_starts"].shape[0],
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(1)
                image_latent_mask[span_rows, span_indices] = state["filled"]
                content_query_mask = selected_token_types.ne(3) & (
                    selected_token_types.ne(1) | image_latent_mask
                )
                strict_query_mask = self._build_generation_attention_mask(
                    input_ids=selected_input_ids,
                    token_types=selected_token_types,
                    sigma=current_sigma,
                    # Generation has one logical sample per row and therefore
                    # does not need the packed-training segment layout.
                    segment_ids=torch.where(
                        selected_token_types.ne(3),
                        torch.zeros_like(
                            selected_token_types,
                            dtype=torch.long,
                        ),
                        torch.full_like(
                            selected_token_types,
                            -1,
                            dtype=torch.long,
                        ),
                    ),
                    content_query_mask=content_query_mask,
                    image_uncond_rows=(
                        torch.ones(batch_size, device=device, dtype=torch.bool)
                        if image_uncond
                        else None
                    ),
                    content_self_diagonal=False,
                )
                content_attention_mask = (
                    state["uncond_attention_mask"]
                    if image_uncond
                    else state["attention_mask"]
                )
                hidden_all = self.model(
                    X0_input_ids=selected_input_ids,
                    attention_mask=strict_query_mask,
                    content_attention_mask=content_attention_mask,
                    token_types=selected_token_types,
                    image_latents=work_latents,
                    image_latent_mask=image_latent_mask,
                    calculate_likelihood=True,
                    xt_flow_latents=full_x_t,
                    xt_flow_times=full_t,
                    xt_flow_query_mask=full_dynamic_query_mask,
                    debug_finite_backbone=debug_finite,
                ).last_hidden_state
                hidden = hidden_all[sample_indices, seq_positions]
            branch = "unconditional" if image_uncond else "conditional"
            self._dynamic_xt_eval_counts[branch] += 1
            return self._prepare_image_flow_condition(hidden)

        def evaluator(x_t, t):
            conditional = evaluate_branch(x_t, t, image_uncond=False)
            if not use_cfg:
                return conditional
            unconditional = evaluate_branch(x_t, t, image_uncond=True)
            return torch.cat([conditional, unconditional], dim=0)

        return evaluator

    @torch.no_grad()
    def generate_image(self, *args, **kwargs):
        self._dynamic_xt_eval_counts = {"conditional": 0, "unconditional": 0}
        self._dynamic_xt_content_condition_commits = 0
        self._dynamic_xt_generation_span_starts = None
        self.model._dynamic_xt_validate_generation_capture = bool(
            kwargs.get("debug_finite", False)
        )
        self.model.clear_dynamic_xt_generation_x0_captures()
        try:
            result = super().generate_image(*args, **kwargs)
        finally:
            self.model.clear_dynamic_xt_generation_x0_captures()
            self.model._dynamic_xt_validate_generation_capture = False
            self._dynamic_xt_generation_span_starts = None
        if not bool(kwargs.get("return_trace", False)) or result is None:
            return result
        generated, trace = result
        trace.update(
            {
                "backbone_condition_mode": "dynamic_xt",
                "dynamic_xt_base_attention_contract": (
                    self.dynamic_xt_attention_contract
                ),
                "dynamic_xt_training_query_batch_mul": (
                    self.dynamic_xt_flow_batch_mul
                ),
                "dynamic_xt_conditional_velocity_evaluations": int(
                    self._dynamic_xt_eval_counts["conditional"]
                ),
                "dynamic_xt_unconditional_velocity_evaluations": int(
                    self._dynamic_xt_eval_counts["unconditional"]
                ),
                "dynamic_xt_query_cache_policy": "read_only_x0_kv",
                "dynamic_xt_flow_query_condition": "backbone_xt_hidden",
                "dynamic_xt_flow_content_condition": "backbone_x0_hidden",
                "dynamic_xt_flow_condition_contract": (
                    self.dynamic_xt_flow_condition_contract
                ),
                "dynamic_xt_flow_content_condition_commits": int(
                    self._dynamic_xt_content_condition_commits
                ),
                "dynamic_xt_parameter_count": self.dynamic_xt_parameter_count(),
            }
        )
        return generated, trace


__all__ = ["DynamicXtGenerationMixin"]
