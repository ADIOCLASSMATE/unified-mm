"""Dynamic-XT successor backbone for Selfless-Flow.

Dynamic-XT keeps the clean X0 stream and strict selfless mask, but replaces
image XT mask queries with ``embed(x_t) + time_embed(t)``. Training uses one
rectified-flow state per image and therefore one backbone execution per step.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .image_flow_loss import FlowLoss, TimestepEmbedder
from .modeling_selfless_flow import (
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    _debug_require_finite_tensor,
    _normal_init_fp32_,
    _to_bool_atten_mask,
    build_row_col_position_ids,
)


class SelflessFlowDynamicXtConfig(Qwen3Config):
    """Checkpoint-identifying config for the isolated Dynamic-XT model."""

    model_type = "selfless_flow_dynamic_xt"


AutoConfig.register(SelflessFlowDynamicXtConfig.model_type, SelflessFlowDynamicXtConfig)


class DynamicXtQwen3Model(Qwen3Model):
    """Selfless two-stream Qwen with a flow-state-dependent XT image query."""

    def _initialize_weights(
        self,
        module,
        is_remote_code: bool = False,
    ) -> None:
        """Initialize Dynamic-XT parameters without shifting shared RNG.

        Transformers also invokes this hook for base-checkpoint keys created
        on a meta device. Initializing the complete timestep module at its
        root guarantees materialized weights and zero biases on that path.
        """

        time_embedder = getattr(
            self,
            "backbone_flow_time_embedder",
            None,
        )
        if time_embedder is not None and any(
            module is child for child in time_embedder.modules()
        ):
            if module is time_embedder:
                with torch.random.fork_rng(devices=[]):
                    self._reset_backbone_flow_time_embedder_impl()
                for child in time_embedder.modules():
                    child._is_hf_initialized = True
            return
        super()._initialize_weights(module, is_remote_code)

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # Do not advance the global RNG for ablation-only parameters: with the
        # same seed, all parameters shared with the static model must retain
        # identical initialization and RF sampling must start at the same RNG
        # state.  Initialization inside the fork still follows the repository
        # convention below.
        with torch.random.fork_rng(devices=[]):
            self.backbone_flow_time_embedder = TimestepEmbedder(
                config.hidden_size
            )
            self._reset_backbone_flow_time_embedder_impl()

    def reset_backbone_flow_time_embedder(self) -> None:
        with torch.random.fork_rng(devices=[]):
            self._reset_backbone_flow_time_embedder_impl()

    def _reset_backbone_flow_time_embedder_impl(self) -> None:
        std = float(getattr(self.config, "initializer_range", 0.02))
        for module in self.backbone_flow_time_embedder.mlp:
            if isinstance(module, nn.Linear):
                _normal_init_fp32_(module.weight, mean=0.0, std=std)
                nn.init.zeros_(module.bias)

    def dynamic_xt_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.backbone_flow_time_embedder.parameters()
        )

    def _build_dynamic_xt_inputs_embeds(
        self,
        *,
        input_ids: torch.LongTensor,
        token_types: torch.Tensor | None,
        image_spans_present: bool | None,
        xt_flow_latents: torch.Tensor | None,
        xt_flow_times: torch.Tensor | None,
        xt_flow_query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self._build_xt_inputs_embeds(
            input_ids=input_ids,
            token_types=token_types,
            image_spans_present=image_spans_present,
        )
        supplied = (
            xt_flow_latents is not None,
            xt_flow_times is not None,
            xt_flow_query_mask is not None,
        )
        if not any(supplied):
            return inputs_embeds
        if not all(supplied):
            raise ValueError(
                "xt_flow_latents, xt_flow_times, and xt_flow_query_mask must be "
                "provided together"
            )
        if token_types is None:
            raise ValueError("Dynamic-XT image queries require token_types")
        if tuple(xt_flow_latents.shape[:2]) != tuple(input_ids.shape):
            raise ValueError("xt_flow_latents must align with input_ids")
        if xt_flow_latents.shape[-1] != self.image_latent_dim:
            raise ValueError("xt_flow_latents has the wrong latent dimension")
        if tuple(xt_flow_times.shape) != tuple(input_ids.shape):
            raise ValueError("xt_flow_times must align with input_ids")
        if tuple(xt_flow_query_mask.shape) != tuple(input_ids.shape):
            raise ValueError("xt_flow_query_mask must align with input_ids")

        device = inputs_embeds.device
        query_mask = (
            xt_flow_query_mask.to(device=device, dtype=torch.bool)
            & token_types.to(device=device).eq(1)
        )
        projected_state = self.image_token_embedder(
            xt_flow_latents.to(
                device=device,
                dtype=self.image_token_embedder.weight_dtype,
            )
        )
        scaled_times = xt_flow_times.to(device=device, dtype=torch.float32)
        scaled_times = scaled_times * float(
            getattr(self.config, "image_flow_time_scale", 1000.0)
        )
        time_condition = self.backbone_flow_time_embedder(
            scaled_times.reshape(-1)
        ).view(*scaled_times.shape, -1)
        dynamic_queries = (projected_state + time_condition).to(
            dtype=inputs_embeds.dtype
        )
        return torch.where(
            query_mask.unsqueeze(-1),
            dynamic_queries,
            inputs_embeds,
        )

    def forward(
        self,
        X0_input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        X0_inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        calculate_likelihood: bool | None = None,
        XT_input_ids: torch.LongTensor | None = None,
        xt_flow_latents: torch.Tensor | None = None,
        xt_flow_times: torch.Tensor | None = None,
        xt_flow_query_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (X0_input_ids is None) == (X0_inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of X0_input_ids or X0_inputs_embeds"
            )
        if attention_mask is None:
            raise ValueError("Dynamic-XT requires the strict selfless attention mask")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = _to_bool_atten_mask(attention_mask)

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
            "input_embeddings.dynamic_xt.x0",
            X0_inputs_embeds,
        )

        needs_xt = bool(self.training or calculate_likelihood)
        if needs_xt:
            xt_ids = X0_input_ids if X0_input_ids is not None else XT_input_ids
            if xt_ids is None:
                raise ValueError(
                    "XT_input_ids are required when X0_inputs_embeds are precomputed"
                )
            XT_inputs_embeds = self._build_dynamic_xt_inputs_embeds(
                input_ids=xt_ids,
                token_types=token_types,
                image_spans_present=image_spans_present,
                xt_flow_latents=xt_flow_latents,
                xt_flow_times=xt_flow_times,
                xt_flow_query_mask=xt_flow_query_mask,
            )
        else:
            XT_inputs_embeds = None

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + X0_inputs_embeds.shape[1],
                device=X0_inputs_embeds.device,
            )
        if position_ids is None:
            if token_types is not None:
                position_ids = build_row_col_position_ids(
                    token_types.to(device=X0_inputs_embeds.device),
                    self.image_token_embedder.image_tokens_per_img,
                )
            else:
                position_ids = cache_position.unsqueeze(0)

        X0_hidden_states = X0_inputs_embeds
        XT_hidden_states = XT_inputs_embeds
        position_embeddings = self.rotary_emb(X0_hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            X0_hidden_states, XT_hidden_states = decoder_layer(
                X0_hidden_states,
                XT_hidden_states,
                attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            _debug_require_finite_tensor(
                debug_finite,
                debug_label,
                f"layers.{layer_idx}.dynamic_xt.output_x0",
                X0_hidden_states,
            )
            if XT_hidden_states is not None:
                _debug_require_finite_tensor(
                    debug_finite,
                    debug_label,
                    f"layers.{layer_idx}.dynamic_xt.output_xt",
                    XT_hidden_states,
                )

        hidden_states = XT_hidden_states if needs_xt else X0_hidden_states
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class DynamicXtQwen3ForCausalLM(Qwen3ForCausalLM):
    """Dynamic-XT model with the unchanged contextual dual-stream flow head."""

    config_class = SelflessFlowDynamicXtConfig
    model_type = "selfless_flow_dynamic_xt"
    _supports_paired_backbone_cfg = False

    def __init__(self, config: Qwen3Config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = DynamicXtQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.image_latent_dim = int(getattr(config, "image_latent_dim", 4))
        self.image_flow_batch_mul = int(
            getattr(config, "image_flow_batch_mul", 1)
        )
        if self.image_flow_batch_mul != 1:
            raise ValueError(
                "Dynamic-XT single-state training requires "
                "image_flow_batch_mul=1"
            )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.image_flow_condition_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=True
        )
        self.image_flow_head = FlowLoss(
            target_channels=self.image_latent_dim,
            z_channels=config.hidden_size,
            width=getattr(config, "image_flow_width", 1280),
            depth=getattr(config, "image_flow_depth", 8),
            num_sampling_steps=str(
                getattr(config, "image_flow_num_sampling_steps", "10")
            ),
            grad_checkpointing=getattr(
                config, "image_flow_grad_checkpointing", False
            ),
            time_scale=getattr(config, "image_flow_time_scale", 1000.0),
            time_sampling=getattr(
                config, "image_flow_time_sampling", "logit_normal"
            ),
            logit_mean=getattr(config, "image_flow_logit_mean", 0.0),
            logit_std=getattr(config, "image_flow_logit_std", 1.0),
            time_eps=getattr(config, "image_flow_time_eps", 1.0e-4),
            uniform_mix=getattr(
                config, "image_flow_time_uniform_mix", 0.1
            ),
            solver=getattr(config, "image_flow_solver", "heun"),
            image_tokens_per_img=getattr(config, "image_tokens_per_img", 256),
        )
        self._dynamic_xt_eval_counts = {"conditional": 0, "unconditional": 0}

        self.post_init()
        self.reset_backbone_attention_output_gates()
        self.reset_image_modules()

    def reset_image_modules(self) -> None:
        super().reset_image_modules()
        self.model.reset_backbone_flow_time_embedder()

    def dynamic_xt_parameter_count(self) -> int:
        return self.model.dynamic_xt_parameter_count()

    def _image_training_layout(
        self,
        *,
        hidden_device: torch.device,
        image_latents: torch.Tensor,
        image_span_table: torch.Tensor,
        image_local_positions: torch.Tensor | None,
        flow_sigma: torch.Tensor | None,
        image_loss_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        table = image_span_table.to(device=hidden_device, dtype=torch.long)
        if table.ndim != 2 or table.shape[1] < 4 or table.shape[0] == 0:
            raise ValueError("image_span_table must have shape [num_images, >=4]")
        image_tokens = int(self.config.image_tokens_per_img)
        rows = table[:, 0]
        starts = table[:, 2]
        offsets = torch.arange(
            image_tokens, device=hidden_device, dtype=torch.long
        ).unsqueeze(0)
        token_indices = starts.unsqueeze(1) + offsets
        targets = image_latents.to(hidden_device)[
            rows.unsqueeze(1), token_indices
        ]
        if flow_sigma is None:
            sigmas = offsets.expand(table.shape[0], -1).float()
        else:
            sigma = flow_sigma.to(device=hidden_device, dtype=torch.float32)
            sigmas = sigma[rows.unsqueeze(1), token_indices]
        if image_local_positions is None:
            positions = offsets.expand(table.shape[0], -1)
        else:
            local = image_local_positions.to(device=hidden_device, dtype=torch.long)
            positions = local[rows.unsqueeze(1), token_indices]
        loss_mask = None
        if image_loss_mask is not None:
            full_mask = image_loss_mask.to(device=hidden_device, dtype=torch.bool)
            loss_mask = full_mask[rows.unsqueeze(1), token_indices]
        return {
            "table": table,
            "rows": rows,
            "token_indices": token_indices,
            "targets": targets,
            "sigmas": sigmas,
            "positions": positions,
            "loss_mask": loss_mask,
        }

    def forward(
        self,
        X0_input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        image_latents: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep=0,
        calculate_likelihood: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        token_types = kwargs.get("token_types", None)
        image_span_table = kwargs.get("image_span_table", None)
        if labels is None or token_types is None:
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
        if inputs_embeds is not None:
            raise ValueError("Dynamic-XT image training expects X0_input_ids")
        if image_latents is None or image_span_table is None:
            raise ValueError(
                "Dynamic-XT image training requires image_latents and image_span_table"
            )

        model_kwargs = dict(kwargs)
        return_per_modality_loss_graph = bool(
            model_kwargs.pop("return_per_modality_loss_graph", False)
        )
        image_loss_mask = model_kwargs.pop("image_loss_mask", None)
        record_flow_stats = bool(model_kwargs.pop("record_flow_stats", True))
        image_local_positions = model_kwargs.get("image_local_positions", None)
        flow_sigma = model_kwargs.get("flow_sigma", None)
        layout = self._image_training_layout(
            hidden_device=X0_input_ids.device,
            image_latents=image_latents,
            image_span_table=image_span_table,
            image_local_positions=image_local_positions,
            flow_sigma=flow_sigma,
            image_loss_mask=image_loss_mask,
        )
        targets = layout["targets"]
        # Preserve static RNG ordering: X0 input noise is sampled first.  The
        # complete RF state is still sampled before any backbone evaluation.
        context_image_latents = self._shared_noisy_image_latents(
            image_latents,
            token_types,
        )
        training_state = self.image_flow_head.sample_training_state(targets)
        x0_inputs_embeds = self.model._build_x0_inputs_embeds(
            input_ids=X0_input_ids,
            token_types=token_types,
            image_latents=context_image_latents,
            image_latent_mask=model_kwargs.get("image_latent_mask", None),
            image_spans_present=True,
            image_latents_are_noisy=context_image_latents is not image_latents,
            debug_finite=bool(model_kwargs.get("debug_finite_backbone", False)),
            debug_label=str(model_kwargs.get("debug_backbone_label", "")),
        )

        rows = layout["rows"]
        token_indices = layout["token_indices"]
        batch_size, seq_len = X0_input_ids.shape
        aligned_x_t = torch.zeros(
            batch_size,
            seq_len,
            self.image_latent_dim,
            device=X0_input_ids.device,
            dtype=training_state.x_t.dtype,
        )
        aligned_t = torch.zeros(
            batch_size,
            seq_len,
            device=X0_input_ids.device,
            dtype=torch.float32,
        )
        query_mask = torch.zeros(
            batch_size,
            seq_len,
            device=X0_input_ids.device,
            dtype=torch.bool,
        )
        aligned_x_t[rows.unsqueeze(1), token_indices] = training_state.x_t
        aligned_t[rows.unsqueeze(1), token_indices] = training_state.t
        local_loss_mask = layout["loss_mask"]
        if local_loss_mask is None:
            query_mask[rows.unsqueeze(1), token_indices] = True
        else:
            query_mask[rows.unsqueeze(1), token_indices] = local_loss_mask

        hidden_states = self.model(
            X0_inputs_embeds=x0_inputs_embeds,
            XT_input_ids=X0_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            calculate_likelihood=True,
            xt_flow_latents=aligned_x_t,
            xt_flow_times=aligned_t,
            xt_flow_query_mask=query_mask,
            **model_kwargs,
        ).last_hidden_state
        selected_hidden = torch.index_select(hidden_states, 0, rows)
        gather_index = token_indices.unsqueeze(-1).expand(
            -1, -1, hidden_states.shape[-1]
        )
        image_hidden = torch.gather(selected_hidden, 1, gather_index)
        image_conditions = self._prepare_image_flow_condition(image_hidden)
        context_for_loss = (
            image_latents
            if context_image_latents is None
            else context_image_latents
        ).to(X0_input_ids.device)
        image_context = context_for_loss[
            rows.unsqueeze(1), token_indices
        ]
        sigmas = layout["sigmas"]
        positions = layout["positions"]
        loss_mask = layout["loss_mask"]
        image_loss = self.image_flow_head(
            target=targets,
            z=image_conditions,
            mask=loss_mask,
            sigma=sigmas,
            image_positions=positions,
            context_latents=image_context,
            record_stats=record_flow_stats,
            training_state=training_state,
        )
        # Keep the training output contract identical to the static model.
        # Dynamic-XT is image-only, but the trainer aggregates both modalities
        # on every rank and therefore still requires an explicit zero text loss
        # and the corresponding token counts.
        labels_on_device = labels.to(hidden_states.device)
        token_types_on_device = token_types.to(hidden_states.device)
        valid_text_mask = (
            ((token_types_on_device == 0) | (token_types_on_device == 2))
            & (labels_on_device != -100)
        )
        text_loss = hidden_states.sum() * 0.0
        text_token_count = valid_text_mask.sum()
        image_token_count = (
            loss_mask.sum()
            if loss_mask is not None
            else torch.tensor(
                targets.shape[0] * targets.shape[1],
                device=hidden_states.device,
                dtype=torch.long,
            )
        )
        output = CausalLMOutputWithPast(
            loss=image_loss,
            logits=None,
            past_key_values=None,
        )
        output["last_hidden_state"] = hidden_states
        output["per_modality_loss"] = {
            "text_loss": text_loss.detach(),
            "image_loss": image_loss.detach(),
        }
        if return_per_modality_loss_graph:
            output["per_modality_loss_graph"] = {
                "text_loss": text_loss,
                "image_loss": image_loss,
            }
        output["per_modality_count"] = {
            "text_tokens": text_token_count.detach(),
            "image_tokens": image_token_count.detach(),
        }
        output["flow_debug_stats"] = {
            key: value.detach()
            for key, value in self.image_flow_head.last_forward_stats.items()
            if isinstance(value, torch.Tensor)
        }
        output["dynamic_xt_parameter_count"] = self.dynamic_xt_parameter_count()
        return output

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
        device = selected_input_ids.device

        def cache_mask(image_uncond: bool):
            query_sigma = current_sigma[
                sample_indices, seq_positions
            ].unsqueeze(1)
            allowed = (
                state["backbone_key_valid"].unsqueeze(1)
                & (
                    state["backbone_key_sigma"].unsqueeze(1)
                    < query_sigma.unsqueeze(-1)
                )
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
                Q_LEN=1,
                KV_LEN=int(state["backbone_max_cache_len"]),
                device=device,
            )

        def evaluate_branch(x_t, t, *, image_uncond: bool):
            aligned_x_t = x_t.unsqueeze(1)
            aligned_t = t.unsqueeze(1)
            query_mask = torch.ones(
                batch_size, 1, device=device, dtype=torch.bool
            )
            if cache_enabled:
                expected = torch.arange(batch_size, device=device)
                if not torch.equal(sample_indices, expected):
                    raise RuntimeError(
                        "Dynamic-XT cached flow evaluation requires one query per row"
                    )
                query_indices = seq_positions.unsqueeze(1)
                position_ids = torch.gather(
                    state["full_position_ids"],
                    dim=2,
                    index=query_indices.unsqueeze(0).expand(2, -1, -1),
                )
                query_latents = work_latents[
                    sample_indices, seq_positions
                ].unsqueeze(1)
                hidden = self.model(
                    X0_input_ids=torch.gather(
                        selected_input_ids, 1, query_indices
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
                        selected_token_types, 1, query_indices
                    ),
                    image_latents=query_latents,
                    image_latent_mask=torch.zeros_like(query_mask),
                    calculate_likelihood=True,
                    xt_flow_latents=aligned_x_t,
                    xt_flow_times=aligned_t,
                    xt_flow_query_mask=query_mask,
                    debug_finite_backbone=debug_finite,
                    debug_backbone_label=(
                        f"dynamic_xt_{'uncond' if image_uncond else 'cond'}_"
                        f"flow_eval_generation_step={generation_step}"
                    ),
                ).last_hidden_state[:, 0]
            else:
                full_x_t = torch.zeros(
                    *selected_input_ids.shape,
                    self.image_latent_dim,
                    device=device,
                    dtype=x_t.dtype,
                )
                full_t = torch.zeros_like(current_sigma, dtype=torch.float32)
                full_query_mask = torch.zeros_like(
                    selected_token_types, dtype=torch.bool
                )
                full_x_t[sample_indices, seq_positions] = x_t
                full_t[sample_indices, seq_positions] = t
                full_query_mask[sample_indices, seq_positions] = True
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
                hidden_all = self.model(
                    X0_input_ids=selected_input_ids,
                    attention_mask=(
                        state["uncond_attention_mask"]
                        if image_uncond
                        else state["attention_mask"]
                    ),
                    token_types=selected_token_types,
                    image_latents=work_latents,
                    image_latent_mask=image_latent_mask,
                    calculate_likelihood=True,
                    xt_flow_latents=full_x_t,
                    xt_flow_times=full_t,
                    xt_flow_query_mask=full_query_mask,
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
        result = super().generate_image(*args, **kwargs)
        if not bool(kwargs.get("return_trace", False)) or result is None:
            return result
        generated, trace = result
        trace.update(
            {
                "backbone_condition_mode": "dynamic_xt",
                "dynamic_xt_conditional_velocity_evaluations": int(
                    self._dynamic_xt_eval_counts["conditional"]
                ),
                "dynamic_xt_unconditional_velocity_evaluations": int(
                    self._dynamic_xt_eval_counts["unconditional"]
                ),
                "dynamic_xt_query_cache_policy": "read_only_x0_kv",
                "dynamic_xt_parameter_count": self.dynamic_xt_parameter_count(),
            }
        )
        return generated, trace


__all__ = [
    "DynamicXtQwen3ForCausalLM",
    "DynamicXtQwen3Model",
    "SelflessFlowDynamicXtConfig",
]
