"""Ablation D: Dynamic-``x_t`` queries on top of baseline B.

The B X0/content stream is evaluated once with the XLNet-style content
diagonal. Four rectified-flow states share its hidden states as their flow
content AdaLN conditions, while their XT query streams receive
``embed(x_t) + time_embed(t)`` and exclusively condition flow query AdaLN.
This preserves B's ``image_flow_batch_mul=4`` estimator without executing four
complete backbones or allowing the current X0 token into its velocity query.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .image_flow_loss import FlowLoss, TimestepEmbedder
from .modeling_selfless_flow import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    _debug_require_finite_tensor,
    _normal_init_fp32_,
    _to_bool_atten_mask,
    build_row_col_position_ids,
    compiled_flex_attention,
    rotate_half,
)
from .modeling_selfless_flow_dynamic_xt_generation import (
    DynamicXtGenerationMixin,
)

DYNAMIC_XT_ARCHITECTURE = "dynamic_xt"
DYNAMIC_XT_ATTENTION_CONTRACT = "xlnet_content_diagonal"
DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT = "xlnet_content_diagonal"
DYNAMIC_XT_FLOW_CONDITION_CONTRACT = "backbone_xt_query_backbone_x0_content"
DYNAMIC_XT_FLOW_BATCH_MUL = 4
DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING = True


class DynamicXtFlowLoss(FlowLoss):
    """D-only flow loss with distinct XT-query and X0-content conditions.

    ``FlowLoss`` intentionally keeps the historical A/B behavior where an
    omitted content condition falls back to the query condition. D must not
    use that fallback: its query AdaLN is conditioned by the Dynamic-XT stream,
    while its content AdaLN is conditioned by the shared X0 stream. Keeping
    this override in the D module preserves the public A/B/F flow-head API and
    their exact execution path.
    """

    def _training_context(self, *args, **kwargs):
        context = super()._training_context(*args, **kwargs)
        context_conditions = self.__dict__.get(
            "_dynamic_xt_training_context_conditions"
        )
        if context_conditions is None:
            raise RuntimeError(
                "Dynamic-XT training requires an explicit X0 content condition"
            )
        expected = context["context_latents"].shape[:2]
        if tuple(context_conditions.shape[:2]) != tuple(expected):
            raise ValueError(
                "Dynamic-XT X0 content conditions must align with flow content: "
                f"{tuple(context_conditions.shape[:2])} != {tuple(expected)}"
            )
        context["context_conditions"] = context_conditions
        return context

    def forward(
        self,
        target,
        z,
        mask=None,
        sigma=None,
        image_positions=None,
        context_latents=None,
        context_mask=None,
        content_attention_mask=None,
        context_conditions=None,
        record_stats: bool = True,
        training_state=None,
    ):
        if context_conditions is None:
            raise ValueError(
                "Dynamic-XT flow training requires X0 context_conditions"
            )
        key = "_dynamic_xt_training_context_conditions"
        if key in self.__dict__:
            raise RuntimeError("Dynamic-XT flow loss is not reentrant")
        self.__dict__[key] = context_conditions
        try:
            return super().forward(
                target=target,
                z=z,
                mask=mask,
                sigma=sigma,
                image_positions=image_positions,
                context_latents=context_latents,
                context_mask=context_mask,
                content_attention_mask=content_attention_mask,
                record_stats=record_stats,
                training_state=training_state,
            )
        finally:
            self.__dict__.pop(key, None)


class SelflessFlowDynamicXtConfig(Qwen3Config):
    """Checkpoint-identifying configuration for ablation D."""

    model_type = "selfless_flow_dynamic_xt"


AutoConfig.register(SelflessFlowDynamicXtConfig.model_type, SelflessFlowDynamicXtConfig)


def _repeat_prepared_attention_mask(attention_mask, repeats: int):
    if isinstance(attention_mask, tuple):
        if len(attention_mask) != 2:
            raise ValueError(
                "prepared attention_mask must contain (safe_mask, valid_rows)"
            )
        return tuple(
            value.repeat(repeats, *([1] * (value.ndim - 1)))
            for value in attention_mask
        )
    if isinstance(attention_mask, torch.Tensor):
        return attention_mask.repeat(
            repeats,
            *([1] * (attention_mask.ndim - 1)),
        )
    raise TypeError(
        "Dynamic-XT multi-state NPU attention requires a dense or prepared "
        f"attention mask, got {type(attention_mask).__name__}"
    )


def _apply_dynamic_rotary_pos_emb(X0_q, XT_q, key, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_batch_mul = int(XT_q.shape[0]) // int(X0_q.shape[0])
    query_cos = cos.repeat(query_batch_mul, 1, 1, 1)
    query_sin = sin.repeat(query_batch_mul, 1, 1, 1)
    return (
        (X0_q * cos) + (rotate_half(X0_q) * sin),
        (XT_q * query_cos) + (rotate_half(XT_q) * query_sin),
        (key * cos) + (rotate_half(key) * sin),
    )


class DynamicXtQwen3Attention(Qwen3Attention):
    """D-only attention supporting R query streams over one B content stream."""

    def forward(
        self,
        X0_hidden_states: torch.Tensor,
        XT_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask,
        content_attention_mask=None,
        past_key_values: Cache | None = None,
        cache_read_only: bool = False,
        cache_write_mask: torch.BoolTensor | None = None,
        cache_write_prefix: int | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        if (
            XT_hidden_states is None
            or XT_hidden_states.shape[0] == X0_hidden_states.shape[0]
        ):
            return super().forward(
                X0_hidden_states=X0_hidden_states,
                XT_hidden_states=XT_hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                content_attention_mask=content_attention_mask,
                past_key_values=past_key_values,
                cache_read_only=cache_read_only,
                cache_write_mask=cache_write_mask,
                cache_write_prefix=cache_write_prefix,
                cache_position=cache_position,
                **kwargs,
            )

        content_batch = int(X0_hidden_states.shape[0])
        query_batch = int(XT_hidden_states.shape[0])
        if query_batch % content_batch:
            raise ValueError(
                "Dynamic-XT query batch must be an integer multiple of the "
                f"content batch: query={query_batch}, content={content_batch}"
            )
        query_batch_mul = query_batch // content_batch
        if past_key_values is not None:
            raise ValueError(
                "Dynamic-XT multi-state training does not support a KV cache"
            )

        debug_finite = bool(kwargs.pop("debug_finite_backbone", False))
        debug_label = str(kwargs.pop("debug_backbone_label", ""))
        record_gate_stats = bool(
            kwargs.pop("record_backbone_gate_stats", False)
        )
        detailed_gate_stats = (
            str(kwargs.pop("backbone_gate_stats_level", "summary"))
            .strip()
            .lower()
            == "detailed"
        )
        self.last_gate_stats.clear()
        prefix = f"layers.{self.layer_idx}.self_attn"
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            f"{prefix}.input_x0",
            X0_hidden_states,
        )
        x0_shape = X0_hidden_states.shape[:-1]
        xt_shape = XT_hidden_states.shape[:-1]
        x0_hidden_shape = (*x0_shape, -1, self.head_dim)
        xt_hidden_shape = (*xt_shape, -1, self.head_dim)

        XT_query_states = self.q_norm(
            self.q_proj(XT_hidden_states).view(xt_hidden_shape)
        ).transpose(1, 2)
        X0_query_states = self.q_norm(
            self.q_proj(X0_hidden_states).view(x0_hidden_shape)
        ).transpose(1, 2)
        X0_key_states = self.k_norm(
            self.k_proj(X0_hidden_states).view(x0_hidden_shape)
        ).transpose(1, 2)
        X0_value_states = self.v_proj(X0_hidden_states).view(
            x0_hidden_shape
        ).transpose(1, 2)
        cos, sin = position_embeddings
        X0_query_states, XT_query_states, X0_key_states = (
            _apply_dynamic_rotary_pos_emb(
                X0_query_states,
                XT_query_states,
                X0_key_states,
                cos,
                sin,
            )
        )
        enable_gqa = (
            self.config.num_attention_heads
            != self.config.num_key_value_heads
        )
        x0_attention_mask = (
            attention_mask
            if content_attention_mask is None
            else content_attention_mask
        )
        X0_attn_output = compiled_flex_attention(
            X0_query_states,
            X0_key_states,
            X0_value_states,
            x0_attention_mask,
            self.scaling,
            enable_gqa,
        )
        if XT_query_states.device.type == "npu":
            XT_attn_output = compiled_flex_attention(
                XT_query_states,
                X0_key_states.repeat(query_batch_mul, 1, 1, 1),
                X0_value_states.repeat(query_batch_mul, 1, 1, 1),
                _repeat_prepared_attention_mask(
                    attention_mask,
                    query_batch_mul,
                ),
                self.scaling,
                enable_gqa,
            )
        else:
            # CPU FlexAttention uses an opaque BlockMask.  Chunk only the query
            # attention for unit tests; the formal NPU path is one fused call.
            XT_attn_output = torch.cat(
                [
                    compiled_flex_attention(
                        query_chunk,
                        X0_key_states,
                        X0_value_states,
                        attention_mask,
                        self.scaling,
                        enable_gqa,
                    )
                    for query_chunk in XT_query_states.chunk(
                        query_batch_mul,
                        dim=0,
                    )
                ],
                dim=0,
            )

        query_token_types = kwargs.get("token_types", None)
        query_flow_sigma = kwargs.get("flow_sigma", None)
        if query_token_types is not None:
            query_token_types = query_token_types.repeat(query_batch_mul, 1)
        if query_flow_sigma is not None:
            query_flow_sigma = query_flow_sigma.repeat(query_batch_mul, 1)
        X0_attn_output = self._apply_attention_output_gate(
            X0_attn_output,
            X0_hidden_states,
            stream="x0",
            token_types=kwargs.get("token_types", None),
            flow_sigma=kwargs.get("flow_sigma", None),
            record_stats=record_gate_stats,
            detailed_stats=detailed_gate_stats,
        )
        XT_attn_output = self._apply_attention_output_gate(
            XT_attn_output,
            XT_hidden_states,
            stream="xt",
            token_types=query_token_types,
            flow_sigma=query_flow_sigma,
            record_stats=record_gate_stats,
            detailed_stats=detailed_gate_stats,
        )
        X0_attn_output = X0_attn_output.transpose(1, 2).reshape(
            *x0_shape,
            -1,
        ).contiguous()
        XT_attn_output = XT_attn_output.transpose(1, 2).reshape(
            *xt_shape,
            -1,
        ).contiguous()
        X0_attn_output = self.o_proj(X0_attn_output)
        XT_attn_output = self.o_proj(XT_attn_output)
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            f"{prefix}.dynamic_xt.output_x0",
            X0_attn_output,
        )
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            f"{prefix}.dynamic_xt.output_xt",
            XT_attn_output,
        )
        return X0_attn_output, XT_attn_output, None


class DynamicXtQwen3Model(Qwen3Model):
    """Baseline-B backbone with flow-state-dependent image XT queries."""

    def _initialize_weights(
        self,
        module,
        is_remote_code: bool = False,
    ) -> None:
        """Initialize the D-only time module without shifting shared RNG."""

        time_embedder = getattr(
            self,
            "backbone_flow_time_embedder",
            None,
        )
        if time_embedder is not None and any(
            module is child for child in time_embedder.modules()
        ):
            if module is time_embedder and not getattr(
                module, "_is_hf_initialized", False
            ):
                with torch.random.fork_rng(devices=[]):
                    self._reset_backbone_flow_time_embedder_impl(
                        only_uninitialized=True
                    )
                for child in time_embedder.modules():
                    child._is_hf_initialized = True
            return
        super()._initialize_weights(module, is_remote_code)

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # Reuse every baseline-B parameter object and replace behavior only.
        # The D attention subclass adds no parameters or buffers.
        for layer in self.layers:
            layer.self_attn.__class__ = DynamicXtQwen3Attention
        # The extra module must not alter initialization of any B parameter.
        with torch.random.fork_rng(devices=[]):
            self.backbone_flow_time_embedder = TimestepEmbedder(
                config.hidden_size
            )
            self._reset_backbone_flow_time_embedder_impl()
        self.last_dynamic_xt_checkpointed_layers = 0
        self._dynamic_xt_generation_x0_captures = {}

    def clear_dynamic_xt_generation_x0_captures(self) -> None:
        """Drop transient X0 rows retained by D's serialized decoder."""

        self._dynamic_xt_generation_x0_captures.clear()

    def _capture_dynamic_xt_generation_x0(
        self,
        output: BaseModelOutputWithPast,
        *,
        token_types: torch.Tensor | None,
        image_latent_mask: torch.Tensor | None,
        cache_position: torch.Tensor | None,
        debug_label: str,
    ) -> None:
        """Retain the fused forward's generated X0 row for flow-content AdaLN."""

        if debug_label.startswith("conditional_"):
            branch = "conditional"
        elif debug_label.startswith("unconditional_"):
            branch = "unconditional"
        else:
            return
        if token_types is None or image_latent_mask is None:
            raise RuntimeError(
                "Dynamic-XT generation capture requires token types and an "
                "image-latent visibility mask"
            )
        hidden = output.last_hidden_state
        token_types = token_types.to(device=hidden.device)
        image_latent_mask = image_latent_mask.to(
            device=hidden.device,
            dtype=torch.bool,
        )
        visible_image = token_types.eq(1) & image_latent_mask
        batch_size, sequence_length = visible_image.shape
        if cache_position is None:
            physical_positions = torch.arange(
                sequence_length,
                device=hidden.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            physical_positions = cache_position.to(
                device=hidden.device,
                dtype=torch.long,
            )
            if physical_positions.ndim == 1:
                physical_positions = physical_positions.unsqueeze(0).expand(
                    batch_size,
                    -1,
                )
        if tuple(physical_positions.shape) != (batch_size, sequence_length):
            raise ValueError(
                "Dynamic-XT generation cache positions must align with X0 rows: "
                f"{tuple(physical_positions.shape)} != "
                f"{(batch_size, sequence_length)}"
            )
        self._dynamic_xt_generation_x0_captures[branch] = {
            "hidden": hidden,
            "physical_positions": physical_positions,
            "visible_image": visible_image,
        }

    def dynamic_xt_generation_x0_hidden(
        self,
        branch: str,
        physical_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Select one just-committed X0 content hidden per generation row."""

        capture = self._dynamic_xt_generation_x0_captures.get(str(branch))
        if capture is None:
            raise RuntimeError(
                f"missing Dynamic-XT {branch} X0 generation capture"
            )
        requested = physical_positions.to(
            device=capture["hidden"].device,
            dtype=torch.long,
        )
        matches = capture["physical_positions"].eq(
            requested.unsqueeze(1)
        ) & capture["visible_image"]
        if int(requested.shape[0]) != int(capture["hidden"].shape[0]):
            raise ValueError(
                "Dynamic-XT pending X0 positions must match the captured batch"
            )
        if getattr(self, "_dynamic_xt_validate_generation_capture", False):
            match_counts = matches.sum(dim=1)
            if not bool(match_counts.eq(1).all()):
                raise RuntimeError(
                    "Dynamic-XT fused generation forward did not expose exactly "
                    "one visible X0 row for each pending flow-content token: "
                    f"match_counts={match_counts.detach().cpu().tolist()}"
                )
        indices = matches.to(dtype=torch.long).argmax(dim=1)
        return capture["hidden"][
            torch.arange(requested.shape[0], device=requested.device),
            indices,
        ]

    def reset_backbone_flow_time_embedder(self) -> None:
        with torch.random.fork_rng(devices=[]):
            self._reset_backbone_flow_time_embedder_impl()

    def _reset_backbone_flow_time_embedder_impl(
        self, *, only_uninitialized: bool = False
    ) -> None:
        std = float(getattr(self.config, "initializer_range", 0.02))
        for module in self.backbone_flow_time_embedder.mlp:
            if isinstance(module, nn.Linear):
                # HF marks loaded parameters, including partially loaded
                # modules. The FP32 temporary + copy_ helper bypasses HF's
                # guarded nn.init calls, so check the destination explicitly.
                if not only_uninitialized or not getattr(
                    module.weight, "_is_hf_initialized", False
                ):
                    _normal_init_fp32_(module.weight, mean=0.0, std=std)
                if module.bias is not None and (
                    not only_uninitialized
                    or not getattr(module.bias, "_is_hf_initialized", False)
                ):
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
        static_queries = self._build_xt_inputs_embeds(
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
            return static_queries
        if not all(supplied):
            raise ValueError(
                "xt_flow_latents, xt_flow_times, and xt_flow_query_mask must "
                "be provided together"
            )
        if token_types is None:
            raise ValueError("Dynamic-XT image queries require token_types")

        content_batch, sequence_length = input_ids.shape
        query_batch = int(xt_flow_latents.shape[0])
        if query_batch % content_batch:
            raise ValueError(
                "Dynamic-XT query batch must be an integer multiple of the "
                f"content batch: query={query_batch}, content={content_batch}"
            )
        query_batch_mul = query_batch // content_batch
        expected_prefix = (query_batch, sequence_length)
        if tuple(xt_flow_latents.shape[:2]) != expected_prefix:
            raise ValueError(
                "xt_flow_latents must have shape [R*B,L,D] aligned with input_ids"
            )
        if int(xt_flow_latents.shape[-1]) != self.image_latent_dim:
            raise ValueError("xt_flow_latents has the wrong latent dimension")
        if tuple(xt_flow_times.shape) != expected_prefix:
            raise ValueError("xt_flow_times must align with Dynamic-XT queries")
        if tuple(xt_flow_query_mask.shape) != expected_prefix:
            raise ValueError(
                "xt_flow_query_mask must align with Dynamic-XT queries"
            )

        repeated_queries = static_queries.repeat(query_batch_mul, 1, 1)
        repeated_token_types = token_types.to(input_ids.device).repeat(
            query_batch_mul,
            1,
        )
        query_mask = (
            xt_flow_query_mask.to(device=input_ids.device, dtype=torch.bool)
            & repeated_token_types.eq(1)
        )
        projected_state = self.image_token_embedder(
            xt_flow_latents.to(
                device=input_ids.device,
                dtype=self.image_token_embedder.weight_dtype,
            )
        )
        scaled_times = xt_flow_times.to(
            device=input_ids.device,
            dtype=torch.float32,
        ) * float(getattr(self.config, "image_flow_time_scale", 1000.0))
        time_condition = self.backbone_flow_time_embedder(
            scaled_times.reshape(-1)
        ).view(*scaled_times.shape, -1)
        dynamic_queries = (projected_state + time_condition).to(
            dtype=repeated_queries.dtype
        )
        return torch.where(
            query_mask.unsqueeze(-1),
            dynamic_queries,
            repeated_queries,
        )

    def forward(
        self,
        X0_input_ids: torch.LongTensor | None = None,
        attention_mask=None,
        content_attention_mask=None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        X0_inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        calculate_likelihood: bool | None = None,
        xt_flow_latents: torch.Tensor | None = None,
        xt_flow_times: torch.Tensor | None = None,
        xt_flow_query_mask: torch.Tensor | None = None,
        return_x0_hidden_state: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        # The ordinary B path (ClimbMix and I2T) must never inherit D's
        # activation-checkpointing cost. Only the four-query T2I path below is
        # selectively checkpointed.
        self.last_dynamic_xt_checkpointed_layers = 0
        supplied = (
            xt_flow_latents is not None,
            xt_flow_times is not None,
            xt_flow_query_mask is not None,
        )
        if not any(supplied):
            output = super().forward(
                X0_input_ids=X0_input_ids,
                attention_mask=attention_mask,
                content_attention_mask=content_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                X0_inputs_embeds=X0_inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                calculate_likelihood=calculate_likelihood,
                **kwargs,
            )
            self._capture_dynamic_xt_generation_x0(
                output,
                token_types=kwargs.get("token_types", None),
                image_latent_mask=kwargs.get("image_latent_mask", None),
                cache_position=cache_position,
                debug_label=str(kwargs.get("debug_backbone_label", "")),
            )
            return output
        if not all(supplied):
            raise ValueError(
                "xt_flow_latents, xt_flow_times, and xt_flow_query_mask must "
                "be provided together"
            )
        if X0_input_ids is None or X0_inputs_embeds is not None:
            raise ValueError(
                "Dynamic-XT queries require X0_input_ids and do not accept "
                "precomputed X0_inputs_embeds"
            )
        if attention_mask is None:
            raise ValueError("Dynamic-XT requires a strict query attention mask")
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = _to_bool_atten_mask(attention_mask)
        if isinstance(content_attention_mask, torch.Tensor):
            content_attention_mask = _to_bool_atten_mask(
                content_attention_mask
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
        XT_inputs_embeds = self._build_dynamic_xt_inputs_embeds(
            input_ids=X0_input_ids,
            token_types=token_types,
            image_spans_present=image_spans_present,
            xt_flow_latents=xt_flow_latents,
            xt_flow_times=xt_flow_times,
            xt_flow_query_mask=xt_flow_query_mask,
        )
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            "input_embeddings.dynamic_xt.x0",
            X0_inputs_embeds,
        )
        _debug_require_finite_tensor(
            debug_finite,
            debug_label,
            "input_embeddings.dynamic_xt.xt",
            XT_inputs_embeds,
        )

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
        position_embeddings = self.rotary_emb(
            X0_hidden_states,
            position_ids,
        )
        checkpoint_dynamic_layers = bool(
            self.training
            and torch.is_grad_enabled()
            and getattr(
                self.config,
                "dynamic_xt_t2i_gradient_checkpointing",
                False,
            )
        )
        for layer_idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            layer_kwargs = {
                "content_attention_mask": content_attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
                **kwargs,
            }
            if checkpoint_dynamic_layers:
                X0_hidden_states, XT_hidden_states = checkpoint(
                    partial(decoder_layer.__call__, **layer_kwargs),
                    X0_hidden_states,
                    XT_hidden_states,
                    attention_mask,
                    use_reentrant=False,
                )
                self.last_dynamic_xt_checkpointed_layers += 1
            else:
                X0_hidden_states, XT_hidden_states = decoder_layer(
                    X0_hidden_states,
                    XT_hidden_states,
                    attention_mask,
                    **layer_kwargs,
                )
            _debug_require_finite_tensor(
                debug_finite,
                debug_label,
                f"layers.{layer_idx}.dynamic_xt.output_x0",
                X0_hidden_states,
            )
            _debug_require_finite_tensor(
                debug_finite,
                debug_label,
                f"layers.{layer_idx}.dynamic_xt.output_xt",
                XT_hidden_states,
            )
        output = BaseModelOutputWithPast(
            last_hidden_state=self.norm(XT_hidden_states),
            past_key_values=past_key_values if use_cache else None,
        )
        # D training consumes both final streams. Keep the standard return
        # field bound to XT so existing non-D model interfaces remain intact.
        if return_x0_hidden_state:
            output["x0_last_hidden_state"] = self.norm(X0_hidden_states)
        return output


class DynamicXtQwen3ForCausalLM(
    DynamicXtGenerationMixin,
    Qwen3ForCausalLM,
):
    """Unified ablation D with one B content stream and four XT states."""

    config_class = SelflessFlowDynamicXtConfig
    model_type = SelflessFlowDynamicXtConfig.model_type
    architecture_variant = DYNAMIC_XT_ARCHITECTURE
    dynamic_xt_attention_contract = DYNAMIC_XT_ATTENTION_CONTRACT
    dynamic_xt_flow_condition_contract = DYNAMIC_XT_FLOW_CONDITION_CONTRACT
    dynamic_xt_flow_batch_mul = DYNAMIC_XT_FLOW_BATCH_MUL
    backbone_model_class = DynamicXtQwen3Model

    def _initialize_weights(
        self,
        module,
        is_remote_code: bool = False,
    ) -> None:
        """Keep D-only initialization outside B's shared RNG sequence."""

        dynamic_model = getattr(self, "model", None)
        time_embedder = getattr(
            dynamic_model,
            "backbone_flow_time_embedder",
            None,
        )
        if time_embedder is not None and any(
            module is child for child in time_embedder.modules()
        ):
            dynamic_model._initialize_weights(module, is_remote_code)
            return
        super()._initialize_weights(module, is_remote_code)

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # This subclass adds no parameters or state-dict keys. It only removes
        # D's invalid query-condition fallback during image training.
        self.image_flow_head.__class__ = DynamicXtFlowLoss
        # Missing metadata is interpreted as the new mandatory D contract so
        # pretrained Qwen configs can still initialize D directly. There is no
        # runtime switch back to the retired shared-condition implementation.
        configured_flow_condition_contract = str(
            getattr(
                config,
                "dynamic_xt_flow_condition_contract",
                DYNAMIC_XT_FLOW_CONDITION_CONTRACT,
            )
        ).strip().lower()
        if configured_flow_condition_contract != DYNAMIC_XT_FLOW_CONDITION_CONTRACT:
            raise ValueError(
                "Dynamic-XT supports only flow query=backbone XT and flow "
                "content=backbone X0 conditions, got "
                f"{configured_flow_condition_contract!r}"
            )
        config.dynamic_xt_flow_condition_contract = (
            DYNAMIC_XT_FLOW_CONDITION_CONTRACT
        )
        architecture = str(
            getattr(config, "architecture_variant", DYNAMIC_XT_ARCHITECTURE)
        ).strip().lower()
        if architecture != DYNAMIC_XT_ARCHITECTURE:
            raise ValueError(
                "Dynamic-XT requires architecture_variant='dynamic_xt', got "
                f"{architecture!r}"
            )
        objective = str(
            getattr(config, "training_objective", "selfless_dual_stream")
        ).strip().lower()
        if objective != "selfless_dual_stream":
            raise ValueError(
                "Dynamic-XT requires training_objective='selfless_dual_stream'"
            )
        attention_contract = str(
            getattr(config, "dual_stream_attention_contract", "")
        ).strip().lower()
        if attention_contract != DYNAMIC_XT_ATTENTION_CONTRACT:
            raise ValueError(
                "Ablation D is defined on baseline B and requires "
                "dual_stream_attention_contract='xlnet_content_diagonal', got "
                f"{attention_contract!r}"
            )
        flow_head_attention_contract = str(
            getattr(config, "flow_head_attention_contract", "")
        ).strip().lower()
        if (
            flow_head_attention_contract
            != DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT
        ):
            raise ValueError(
                "Ablation D is defined on corrected baseline B and requires "
                "flow_head_attention_contract="
                f"'{DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT}', got "
                f"{flow_head_attention_contract!r}"
            )
        if self.image_flow_batch_mul != DYNAMIC_XT_FLOW_BATCH_MUL:
            raise ValueError(
                "Ablation D must preserve baseline B's image_flow_batch_mul=4, "
                f"got {self.image_flow_batch_mul}"
            )

    def reset_image_modules(self) -> None:
        super().reset_image_modules()
        self.model.reset_backbone_flow_time_embedder()

    def _zero_image_module_loss(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Keep D's time embedder in every distributed microbatch graph."""

        zero = super()._zero_image_module_loss(hidden_states)
        for parameter in self.model.backbone_flow_time_embedder.parameters():
            if parameter.requires_grad and parameter.numel() > 0:
                zero = zero + parameter.reshape(-1)[0].float() * 0.0
        return zero

    def dynamic_xt_parameter_count(self) -> int:
        return self.model.dynamic_xt_parameter_count()

    def _image_training_layout(
        self,
        *,
        device: torch.device,
        image_latents: torch.Tensor,
        image_span_table: torch.Tensor,
        image_local_positions: torch.Tensor | None,
        flow_sigma: torch.Tensor | None,
        image_loss_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor | None]:
        table = image_span_table.to(device=device, dtype=torch.long)
        if table.ndim != 2 or table.shape[1] < 4 or table.shape[0] == 0:
            raise ValueError("image_span_table must have shape [num_images, >=4]")
        image_tokens = int(self.config.image_tokens_per_img)
        rows = table[:, 0]
        starts = table[:, 2]
        offsets = torch.arange(
            image_tokens,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        token_indices = starts.unsqueeze(1) + offsets
        targets = image_latents.to(device)[rows.unsqueeze(1), token_indices]
        if flow_sigma is None:
            sigmas = offsets.expand(table.shape[0], -1).float()
        else:
            sigma = flow_sigma.to(device=device, dtype=torch.float32)
            sigmas = sigma[rows.unsqueeze(1), token_indices]
        if image_local_positions is None:
            positions = offsets.expand(table.shape[0], -1)
        else:
            local = image_local_positions.to(device=device, dtype=torch.long)
            positions = local[rows.unsqueeze(1), token_indices]
        loss_mask = None
        if image_loss_mask is not None:
            full_mask = image_loss_mask.to(device=device, dtype=torch.bool)
            loss_mask = full_mask[rows.unsqueeze(1), token_indices]
        return {
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
        compute_image_loss_arg = kwargs.get("compute_image_loss", None)
        compute_image_loss = (
            bool(compute_image_loss_arg)
            if compute_image_loss_arg is not None
            else bool(
                image_span_table is not None
                and image_span_table.shape[0] > 0
            )
        )
        if labels is None or token_types is None or not compute_image_loss:
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
        if X0_input_ids is None or inputs_embeds is not None:
            raise ValueError("Dynamic-XT image training requires X0_input_ids")
        if image_latents is None or image_span_table is None:
            raise ValueError(
                "Dynamic-XT image training requires image_latents and "
                "image_span_table"
            )
        if past_key_values is not None or bool(use_cache):
            raise ValueError("Dynamic-XT multi-state training does not use a KV cache")

        model_kwargs = dict(kwargs)
        return_per_modality_loss_graph = bool(
            model_kwargs.pop("return_per_modality_loss_graph", False)
        )
        compute_text_loss = bool(model_kwargs.pop("compute_text_loss", True))
        model_kwargs.pop("compute_image_loss", None)
        image_loss_mask = model_kwargs.pop("image_loss_mask", None)
        record_flow_stats = bool(model_kwargs.pop("record_flow_stats", True))
        image_local_positions = model_kwargs.get("image_local_positions", None)
        flow_sigma = model_kwargs.get("flow_sigma", None)
        layout = self._image_training_layout(
            device=X0_input_ids.device,
            image_latents=image_latents,
            image_span_table=image_span_table,
            image_local_positions=image_local_positions,
            flow_sigma=flow_sigma,
            image_loss_mask=image_loss_mask,
        )

        # Match B's stochastic contract: one shared noisy X0 context, then four
        # independent RF states.  Only the XT stream is expanded fourfold.
        context_image_latents = self._shared_noisy_image_latents(
            image_latents,
            token_types,
        )
        repeats = int(self.image_flow_batch_mul)
        targets = layout["targets"]
        repeated_targets = targets.repeat(repeats, 1, 1)
        training_state = self.image_flow_head.sample_training_state(
            repeated_targets
        )

        rows = layout["rows"]
        token_indices = layout["token_indices"]
        batch_size, sequence_length = X0_input_ids.shape
        query_batch = repeats * batch_size
        aligned_x_t = torch.zeros(
            query_batch,
            sequence_length,
            self.image_latent_dim,
            device=X0_input_ids.device,
            dtype=training_state.x_t.dtype,
        )
        aligned_t = torch.zeros(
            query_batch,
            sequence_length,
            device=X0_input_ids.device,
            dtype=torch.float32,
        )
        query_mask = torch.zeros(
            query_batch,
            sequence_length,
            device=X0_input_ids.device,
            dtype=torch.bool,
        )
        expanded_rows = torch.cat(
            [rows + repeat_idx * batch_size for repeat_idx in range(repeats)]
        )
        expanded_token_indices = token_indices.repeat(repeats, 1)
        aligned_x_t[
            expanded_rows.unsqueeze(1),
            expanded_token_indices,
        ] = training_state.x_t
        aligned_t[
            expanded_rows.unsqueeze(1),
            expanded_token_indices,
        ] = training_state.t
        repeated_loss_mask = (
            layout["loss_mask"].repeat(repeats, 1)
            if layout["loss_mask"] is not None
            else None
        )
        query_mask[
            expanded_rows.unsqueeze(1),
            expanded_token_indices,
        ] = (
            repeated_loss_mask
            if repeated_loss_mask is not None
            else torch.ones_like(training_state.t, dtype=torch.bool)
        )

        if context_image_latents is not image_latents:
            model_kwargs["image_latents_are_noisy"] = True
        backbone_output = self.model(
            X0_input_ids=X0_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            calculate_likelihood=True,
            image_latents=context_image_latents,
            xt_flow_latents=aligned_x_t,
            xt_flow_times=aligned_t,
            xt_flow_query_mask=query_mask,
            return_x0_hidden_state=True,
            **model_kwargs,
        )
        hidden_states = backbone_output.last_hidden_state
        x0_hidden_states = backbone_output["x0_last_hidden_state"]
        if int(hidden_states.shape[0]) != query_batch:
            raise RuntimeError(
                "Dynamic-XT backbone returned the wrong query batch: "
                f"{hidden_states.shape[0]} != {query_batch}"
            )

        selected_hidden = torch.index_select(hidden_states, 0, expanded_rows)
        gather_index = expanded_token_indices.unsqueeze(-1).expand(
            -1,
            -1,
            hidden_states.shape[-1],
        )
        image_hidden = torch.gather(selected_hidden, 1, gather_index)
        image_conditions = self._prepare_image_flow_condition(image_hidden)
        x0_gather_index = token_indices.unsqueeze(-1).expand(
            -1,
            -1,
            x0_hidden_states.shape[-1],
        )
        x0_image_hidden = torch.gather(
            torch.index_select(x0_hidden_states, 0, rows),
            1,
            x0_gather_index,
        )
        # X0 is computed once for the B-sized content batch. Its condition is
        # then repeated in the same repeat-major order as the four RF states.
        content_conditions = self._prepare_image_flow_condition(
            x0_image_hidden
        ).repeat(repeats, 1, 1)
        context_for_loss = (
            image_latents
            if context_image_latents is None
            else context_image_latents
        ).to(X0_input_ids.device)
        image_context = context_for_loss[
            rows.unsqueeze(1),
            token_indices,
        ].repeat(repeats, 1, 1)
        repeated_sigmas = layout["sigmas"].repeat(repeats, 1)
        repeated_positions = layout["positions"].repeat(repeats, 1)
        image_loss = self.image_flow_head(
            target=repeated_targets,
            z=image_conditions,
            mask=repeated_loss_mask,
            sigma=repeated_sigmas,
            image_positions=repeated_positions,
            context_latents=image_context,
            context_conditions=content_conditions,
            record_stats=record_flow_stats,
            training_state=training_state,
        )

        # Query repeats are independent across positions, so their text hidden
        # states are identical.  Use the first repeat to preserve B's text count
        # while avoiding a second backbone pass in joint validation.
        original_hidden = hidden_states[:batch_size]
        labels_on_device = labels.to(original_hidden.device)
        token_types_on_device = token_types.to(original_hidden.device)
        valid_text_mask = (
            ((token_types_on_device == 0) | (token_types_on_device == 2))
            & labels_on_device.ne(-100)
            & compute_text_loss
        )
        text_token_count = valid_text_mask.sum()
        text_loss = original_hidden.sum() * 0.0
        if compute_text_loss and self.lambda_text > 0.0:
            text_logits = self.lm_head(original_hidden[valid_text_mask])
            text_loss = F.cross_entropy(
                text_logits,
                labels_on_device[valid_text_mask],
                reduction="sum",
            ) / text_token_count.clamp_min(1).to(text_logits.dtype)
        image_token_count = (
            repeated_loss_mask.sum()
            if repeated_loss_mask is not None
            else torch.tensor(
                repeated_targets.shape[0] * repeated_targets.shape[1],
                device=original_hidden.device,
                dtype=torch.long,
            )
        )
        loss = self.lambda_text * text_loss + self.lambda_image * image_loss
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=None,
        )
        output["last_hidden_state"] = original_hidden
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
        gate_stats = self.model.backbone_attention_gate_stats()
        if gate_stats:
            output["backbone_gate_stats"] = gate_stats
        output["dynamic_xt_parameter_count"] = self.dynamic_xt_parameter_count()
        output["dynamic_xt_query_batch_mul"] = repeats
        output["dynamic_xt_content_batch_size"] = batch_size
        output["dynamic_xt_query_batch_size"] = query_batch
        output["dynamic_xt_checkpointed_layers"] = int(
            self.model.last_dynamic_xt_checkpointed_layers
        )
        return output

__all__ = [
    "DYNAMIC_XT_ARCHITECTURE",
    "DYNAMIC_XT_ATTENTION_CONTRACT",
    "DYNAMIC_XT_FLOW_BATCH_MUL",
    "DYNAMIC_XT_FLOW_CONDITION_CONTRACT",
    "DYNAMIC_XT_FLOW_HEAD_ATTENTION_CONTRACT",
    "DYNAMIC_XT_T2I_GRADIENT_CHECKPOINTING",
    "DynamicXtFlowLoss",
    "DynamicXtQwen3ForCausalLM",
    "DynamicXtQwen3Model",
    "SelflessFlowDynamicXtConfig",
]
