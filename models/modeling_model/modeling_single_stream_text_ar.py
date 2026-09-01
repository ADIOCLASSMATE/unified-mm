"""Text-only next-token ablation for the Unified-MM Selfless baseline.

This model changes only the text prediction contract:

* text likelihood uses one physical-order causal X0 stream;
* the hidden returned at target position ``j`` is the X0 hidden from source
  position ``j - 1`` so existing train/validation/evaluation callers keep their
  target-aligned interface;
* image-flow training stays unchanged, while image generation uses the same
  strict-sigma cached single stream as ablation A.

The class intentionally adds no parameters.  With the same config payload and
seed, every parameter shared with the baseline has the same name, shape,
construction order, and initial value.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from .modeling_selfless_flow import (
    Qwen3ForCausalLM,
    Qwen3Model,
    SelflessStaticCache,
)


class SingleStreamTextARConfig(Qwen3Config):
    """Checkpoint identity for text-AR + unchanged image Selfless flow."""

    model_type = "selfless_flow_single_stream_text_ar"


AutoConfig.register(SingleStreamTextARConfig.model_type, SingleStreamTextARConfig)


def _materialize_allowed_mask(
    attention_mask,
    *,
    batch_size: int,
    query_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Materialize an incoming Selfless mask as ``True == allowed``."""

    if isinstance(attention_mask, torch.Tensor):
        dense = attention_mask.to(device=device)
        if dense.ndim == 4:
            if dense.shape[1] != 1:
                raise ValueError(
                    "single-stream text AR expects a shared-head attention mask"
                )
            dense = dense[:, 0]
        elif dense.ndim != 3:
            raise ValueError(
                "attention_mask must have shape [B,Q,KV] or [B,1,Q,KV], "
                f"got {tuple(dense.shape)}"
            )
        if dense.shape[0] != batch_size or dense.shape[1] != query_length:
            raise ValueError(
                "attention_mask does not align with the text query batch: "
                f"{tuple(dense.shape)} vs B={batch_size}, Q={query_length}"
            )
        disallowed = dense if dense.dtype == torch.bool else dense < 0
        return ~disallowed

    if not isinstance(attention_mask, BlockMask):
        raise TypeError(
            "single-stream text AR requires a dense mask or BlockMask, got "
            f"{type(attention_mask).__name__}"
        )
    key_length = int(attention_mask.shape[-1])
    batch = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
    query = torch.arange(query_length, device=device).view(1, query_length, 1)
    key = torch.arange(key_length, device=device).view(1, 1, key_length)
    head = torch.zeros((), device=device, dtype=torch.long)
    return attention_mask.mask_mod(batch, head, query, key).to(torch.bool)


def _physical_causal_mask(
    attention_mask,
    *,
    input_ids: torch.LongTensor | None,
    token_types: torch.Tensor | None,
    segment_ids: torch.Tensor | None,
    cache_position: torch.LongTensor | None,
) -> torch.Tensor | BlockMask:
    """Convert sigma visibility into physical-order causal visibility.

    Packed rows use explicit segment IDs.  Unpacked multimodal rows are one
    physical sequence, so CFG/image-conditioning dropout in the incoming
    Selfless mask cannot accidentally remove causal text-AR edges.  Pure-text
    callers without token metadata retain reciprocal-connectivity recovery.
    Rectangular cached-prefix masks use absolute ``cache_position`` values.
    """

    reference = input_ids if input_ids is not None else token_types
    if reference is None:
        raise ValueError("text AR attention requires input_ids or token_types")
    if reference.ndim != 2:
        raise ValueError("text AR inputs must be rank two")
    batch_size, query_length = map(int, reference.shape)
    device = reference.device
    original_allowed = _materialize_allowed_mask(
        attention_mask,
        batch_size=batch_size,
        query_length=query_length,
        device=device,
    )
    key_length = int(original_allowed.shape[-1])

    if cache_position is None:
        query_positions = torch.arange(
            query_length, device=device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
    else:
        query_positions = cache_position.to(device=device, dtype=torch.long)
        if query_positions.ndim == 1:
            query_positions = query_positions.unsqueeze(0).expand(
                batch_size, -1
            )
        if tuple(query_positions.shape) != (batch_size, query_length):
            raise ValueError(
                "cache_position must align with [B,Q] for text AR: "
                f"{tuple(query_positions.shape)} != {(batch_size, query_length)}"
            )

    key_positions = torch.arange(
        key_length, device=device, dtype=torch.long
    ).view(1, 1, key_length)
    physical_order = key_positions <= query_positions.unsqueeze(-1)

    if query_length == key_length and cache_position is None:
        connected = original_allowed | original_allowed.transpose(1, 2)
        if segment_ids is not None:
            if tuple(segment_ids.shape) != (batch_size, query_length):
                raise ValueError("segment_ids must align with text AR inputs")
            segments = segment_ids.to(device=device, dtype=torch.long)
            valid = segments.ge(0)
            same_segment = (
                segments.unsqueeze(-1).eq(segments.unsqueeze(1))
                & valid.unsqueeze(-1)
                & valid.unsqueeze(1)
            )
        elif token_types is not None:
            if tuple(token_types.shape) != (batch_size, query_length):
                raise ValueError("token_types must align with text AR inputs")
            valid = token_types.to(device=device).ne(3)
            same_segment = valid.unsqueeze(-1) & valid.unsqueeze(1)
        else:
            valid = connected.any(dim=-1)
            if query_length == 1:
                valid = torch.ones_like(valid)
            same_segment = connected | (
                torch.eye(query_length, device=device, dtype=torch.bool)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                & valid.unsqueeze(-1)
            )
        allowed = (
            physical_order
            & same_segment
            & valid.unsqueeze(-1)
            & valid.unsqueeze(1)
        )
    else:
        key_valid = original_allowed.any(dim=1)
        query_valid = original_allowed.any(dim=-1)
        in_range = query_positions.lt(key_length)
        safe_query_positions = query_positions.clamp(min=0, max=key_length - 1)
        query_valid = query_valid | (
            torch.gather(key_valid, 1, safe_query_positions) & in_range
        )
        if token_types is not None:
            query_valid = query_valid & token_types.to(device=device).ne(3)
        key_valid = key_valid.clone()
        key_valid.scatter_(
            1,
            safe_query_positions,
            query_valid & in_range,
        )
        allowed = (
            physical_order
            & query_valid.unsqueeze(-1)
            & key_valid.unsqueeze(1)
        )

    if device.type == "npu":
        return (~allowed).unsqueeze(1)

    def causal_mask_mod(batch, head, query_index, key_index):
        del head
        return allowed[batch, query_index, key_index]

    return create_block_mask(
        causal_mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=device,
    )


def _causal_generation_mask(
    *,
    key_valid: torch.Tensor,
    key_segment_ids: torch.Tensor,
    query_valid: torch.Tensor,
    query_segment_ids: torch.Tensor,
    query_positions: torch.Tensor,
):
    """Physical causal mask for C full or rectangular cached decoding."""

    device = query_positions.device
    key_positions = torch.arange(
        key_valid.shape[1],
        device=device,
        dtype=torch.long,
    ).view(1, 1, -1)
    allowed = (
        query_valid.unsqueeze(-1)
        & key_valid.unsqueeze(1)
        & query_segment_ids.unsqueeze(-1).eq(
            key_segment_ids.unsqueeze(1)
        )
        & query_segment_ids.unsqueeze(-1).ge(0)
        & key_positions.le(query_positions.unsqueeze(-1))
    )
    if device.type == "npu":
        return (~allowed).unsqueeze(1)

    def mask_mod(batch, head, query_index, key_index):
        del head
        return allowed[batch, query_index, key_index]

    return create_block_mask(
        mask_mod,
        B=query_positions.shape[0],
        H=None,
        Q_LEN=query_positions.shape[1],
        KV_LEN=key_valid.shape[1],
        device=device,
    )


class SingleStreamTextARQwen3Model(Qwen3Model):
    """Qwen backbone with single-stream AR only on text objective calls."""

    _CACHE_LAST_HIDDEN = "_single_stream_text_ar_last_hidden"

    def _resolve_text_ar_mode(
        self,
        *,
        requested: bool | None,
        calculate_likelihood: bool | None,
        require_image_query_stream: bool | None,
        token_types: torch.Tensor | None,
        image_latent_mask: torch.Tensor | None,
    ) -> bool:
        del token_types, image_latent_mask
        if requested is not None:
            return bool(requested)
        if require_image_query_stream is not None:
            return not bool(require_image_query_stream)
        # Direct model.model likelihood calls are text scorers.  Dynamic image
        # generation calls use calculate_likelihood=False and stay on baseline.
        return bool(calculate_likelihood)

    def _needs_query_stream(
        self,
        *,
        calculate_likelihood: bool | None,
        require_image_query_stream: bool | None,
        text_ar_mode: bool,
        image_spans_present: bool | None,
    ) -> bool:
        if text_ar_mode:
            return False
        return super()._needs_query_stream(
            calculate_likelihood=calculate_likelihood,
            require_image_query_stream=require_image_query_stream,
            text_ar_mode=text_ar_mode,
            image_spans_present=image_spans_present,
        )

    def _prepare_stream_attention_masks(
        self,
        attention_mask,
        content_attention_mask,
        *,
        text_ar_mode: bool,
        input_ids: torch.LongTensor | None,
        token_types: torch.Tensor | None,
        text_segment_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor | None,
    ):
        del past_key_values
        if not text_ar_mode:
            return attention_mask, content_attention_mask
        return (
            _physical_causal_mask(
                attention_mask,
                input_ids=input_ids,
                token_types=token_types,
                segment_ids=text_segment_ids,
                cache_position=cache_position,
            ),
            None,
        )

    def _finalize_stream_hidden(
        self,
        X0_hidden_states: torch.Tensor,
        XT_hidden_states: torch.Tensor | None,
        *,
        use_query_stream: bool,
        text_ar_mode: bool,
        token_types: torch.Tensor | None,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor | None,
    ) -> torch.Tensor:
        del cache_position
        if not text_ar_mode:
            return super()._finalize_stream_hidden(
                X0_hidden_states,
                XT_hidden_states,
                use_query_stream=use_query_stream,
                text_ar_mode=text_ar_mode,
                token_types=token_types,
                past_key_values=past_key_values,
                cache_position=None,
            )
        if use_query_stream or XT_hidden_states is not None:
            raise RuntimeError("text AR must not construct or return the XT stream")

        content_hidden = self.norm(X0_hidden_states)
        aligned_hidden = torch.zeros_like(content_hidden)
        if content_hidden.shape[1] > 1:
            aligned_hidden[:, 1:] = content_hidden[:, :-1]

        cached_previous = (
            getattr(past_key_values, self._CACHE_LAST_HIDDEN, None)
            if past_key_values is not None
            else None
        )
        if cached_previous is not None:
            cached_previous = cached_previous.to(
                device=content_hidden.device,
                dtype=content_hidden.dtype,
            )
            if cached_previous.shape[0] == 1 and content_hidden.shape[0] > 1:
                cached_previous = cached_previous.expand(
                    content_hidden.shape[0], -1
                )
            if tuple(cached_previous.shape) != (
                content_hidden.shape[0],
                content_hidden.shape[-1],
            ):
                raise ValueError(
                    "cached text-AR source hidden has incompatible shape: "
                    f"{tuple(cached_previous.shape)}"
                )
            aligned_hidden[:, 0] = cached_previous

        if token_types is not None:
            target_is_text = token_types.to(content_hidden.device).eq(0) | (
                token_types.to(content_hidden.device).eq(2)
            )
            aligned_hidden = torch.where(
                target_is_text.unsqueeze(-1),
                aligned_hidden,
                content_hidden,
            )

        if past_key_values is not None and content_hidden.shape[1] > 0:
            setattr(
                past_key_values,
                self._CACHE_LAST_HIDDEN,
                content_hidden[:, -1].detach(),
            )
        return aligned_hidden


class SingleStreamTextARQwen3ForCausalLM(Qwen3ForCausalLM):
    """Next-token text AR plus ablation-A cached image generation."""

    config_class = SingleStreamTextARConfig
    model_type = SingleStreamTextARConfig.model_type
    architecture_variant = "single_stream_text_ar"
    backbone_model_class = SingleStreamTextARQwen3Model
    text_prediction_offset = 1
    text_hidden_alignment = "target_position_contains_previous_content_hidden"

    def _generation_attention_contract(self) -> str:
        """C keeps the image stream exactly equal to ablation A."""

        contract = super()._generation_attention_contract()
        if contract != "selfless_strict":
            raise ValueError(
                "single_stream_text_ar image generation must use "
                f"selfless_strict, got {contract!r}"
            )
        return contract

    @torch.no_grad()
    def generate_image(self, *args, **kwargs):
        """Generate images with C's unchanged ablation-A image stream."""

        return super().generate_image(*args, **kwargs)

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        token_types: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        image_latents: torch.Tensor | None = None,
        image_latent_mask: torch.Tensor | None = None,
        temperature: float = 0.0,
        eos_token_id: int | tuple[int, ...] | list[int] | None = None,
        use_cache: bool = True,
        return_trace: bool = False,
    ):
        """Generate C text with conventional next-token causal KV caching."""

        state = self._prepare_text_generation_state(
            input_ids=input_ids,
            token_types=token_types,
            sigma=sigma,
            segment_ids=segment_ids,
            image_latents=image_latents,
            image_latent_mask=image_latent_mask,
            max_new_tokens=int(max_new_tokens),
        )
        ids = state["ids"]
        types = state["types"]
        segments = state["segments"]
        latents = state["latents"]
        latent_mask = state["latent_mask"]
        content_mask = state["content_mask"]
        prompt_lengths = state["prompt_lengths"]
        position_ids = state["position_ids"]
        batch_indices = state["batch_indices"]
        prompt_width = int(state["prompt_width"])
        capacity = int(state["capacity"])
        batch_size = ids.shape[0]
        stop_token_ids = self._generation_stop_token_ids(eos_token_id)
        finished_token_id = stop_token_ids[0] if stop_token_ids else None

        if int(max_new_tokens) == 0:
            output = ids[:, :prompt_width]
            if not return_trace:
                return output
            return output, {
                "generation_mode": "single_stream_text_ar",
                "attention_contract": "physical_causal",
                "backbone_kv_cache_enabled": bool(use_cache),
                "generated_tokens": 0,
            }

        cache = None
        prediction_hidden = None
        key_valid = torch.zeros(
            batch_size,
            capacity,
            device=ids.device,
            dtype=torch.bool,
        )
        key_segment_ids = torch.full(
            (batch_size, capacity),
            -1,
            device=ids.device,
            dtype=torch.long,
        )

        if use_cache:
            context_indices = torch.arange(
                prompt_width,
                device=ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
            context_valid = context_indices < prompt_lengths.unsqueeze(1)
            key_valid[:, :prompt_width] = context_valid
            key_segment_ids[:, :prompt_width] = segments[:, :prompt_width]
            attention_mask = _causal_generation_mask(
                key_valid=key_valid,
                key_segment_ids=key_segment_ids,
                query_valid=context_valid,
                query_segment_ids=segments[:, :prompt_width],
                query_positions=context_indices,
            )
            cache = SelflessStaticCache(
                config=self.model.config,
                max_cache_len=capacity,
            )
            prompt_hidden = self.model(
                X0_input_ids=ids[:, :prompt_width],
                attention_mask=attention_mask,
                position_ids=position_ids[:, :, :prompt_width],
                past_key_values=cache,
                use_cache=True,
                cache_position=context_indices,
                cache_write_mask=context_valid,
                token_types=types[:, :prompt_width],
                image_latents=latents[:, :prompt_width],
                image_latent_mask=latent_mask[:, :prompt_width],
                calculate_likelihood=False,
                _text_ar_mode=False,
            ).last_hidden_state
            prediction_hidden = prompt_hidden[
                batch_indices,
                prompt_lengths - 1,
            ]

        finished = torch.zeros(
            batch_size,
            device=ids.device,
            dtype=torch.bool,
        )
        generated_steps = 0

        for step in range(int(max_new_tokens)):
            target_positions = prompt_lengths + step
            if not use_cache:
                source_positions = target_positions - 1
                current_width = int(source_positions.max().item()) + 1
                full_positions = torch.arange(
                    current_width,
                    device=ids.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(batch_size, -1)
                source_valid = content_mask[:, :current_width]
                attention_mask = _causal_generation_mask(
                    key_valid=source_valid,
                    key_segment_ids=segments[:, :current_width],
                    query_valid=source_valid,
                    query_segment_ids=segments[:, :current_width],
                    query_positions=full_positions,
                )
                full_hidden = self.model(
                    X0_input_ids=ids[:, :current_width],
                    attention_mask=attention_mask,
                    position_ids=position_ids[:, :, :current_width],
                    token_types=types[:, :current_width],
                    image_latents=latents[:, :current_width],
                    image_latent_mask=latent_mask[:, :current_width],
                    calculate_likelihood=False,
                    _text_ar_mode=False,
                ).last_hidden_state
                prediction_hidden = full_hidden[
                    batch_indices,
                    source_positions,
                ]

            next_token = self._sample_token(
                self.lm_head(prediction_hidden),
                float(temperature),
            )
            if finished_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, finished_token_id),
                    next_token,
                )
            ids[batch_indices, target_positions] = next_token
            content_mask[batch_indices, target_positions] = True
            generated_steps = step + 1

            if stop_token_ids:
                finished |= self._matches_stop_token(
                    next_token,
                    stop_token_ids,
                )
                if bool(finished.all()):
                    break
            if use_cache and step + 1 < int(max_new_tokens):
                query_indices = target_positions.unsqueeze(1)
                query_segments = torch.gather(
                    segments,
                    1,
                    query_indices,
                )
                key_valid.scatter_(
                    1,
                    query_indices,
                    torch.ones_like(query_indices, dtype=torch.bool),
                )
                key_segment_ids.scatter_(
                    1,
                    query_indices,
                    query_segments,
                )
                attention_mask = _causal_generation_mask(
                    key_valid=key_valid,
                    key_segment_ids=key_segment_ids,
                    query_valid=torch.ones_like(
                        query_indices,
                        dtype=torch.bool,
                    ),
                    query_segment_ids=query_segments,
                    query_positions=query_indices,
                )
                prediction_hidden = self.model(
                    X0_input_ids=torch.gather(ids, 1, query_indices),
                    attention_mask=attention_mask,
                    position_ids=torch.gather(
                        position_ids,
                        2,
                        query_indices.unsqueeze(0).expand(2, -1, -1),
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=query_indices,
                    token_types=torch.gather(types, 1, query_indices),
                    calculate_likelihood=False,
                    _text_ar_mode=False,
                ).last_hidden_state[:, 0]

        output_width = int((prompt_lengths + generated_steps).max().item())
        output = ids[:, :output_width]
        if not return_trace:
            return output
        return output, {
            "generation_mode": "single_stream_text_ar",
            "attention_contract": "physical_causal",
            "backbone_kv_cache_enabled": bool(use_cache),
            "generated_tokens": int(generated_steps),
        }

    def forward(self, *args, **kwargs):
        """Keep joint validation compatible without mixing stream contracts.

        Unified validation requests text and image losses together.  A single
        backbone pass cannot be both physical-causal text AR and bitwise the
        baseline image query stream, so the isolated ablation performs two
        source-specific passes and merges their already weighted losses.  The
        image pass runs first to preserve baseline-a RNG ordering when this
        cold path is invoked while the module is in training mode.
        """

        compute_text_loss = bool(kwargs.get("compute_text_loss", True))
        compute_image_loss_arg = kwargs.get("compute_image_loss", None)
        image_span_table = kwargs.get("image_span_table", None)
        compute_image_loss = (
            bool(compute_image_loss_arg)
            if compute_image_loss_arg is not None
            else bool(
                image_span_table is not None
                and image_span_table.shape[0] > 0
            )
        )
        labels = kwargs.get("labels", args[5] if len(args) > 5 else None)
        if labels is None and compute_image_loss:
            # With no loss to merge, an image-bearing call is a conditioning
            # or generation probe.  Keep it on baseline-a's query stream.
            image_kwargs = dict(kwargs)
            image_kwargs["compute_text_loss"] = False
            image_kwargs["compute_image_loss"] = True
            return super().forward(*args, **image_kwargs)
        if not (compute_text_loss and compute_image_loss):
            return super().forward(*args, **kwargs)

        image_kwargs = dict(kwargs)
        image_kwargs["compute_text_loss"] = False
        image_kwargs["compute_image_loss"] = True
        image_output = super().forward(*args, **image_kwargs)
        image_flow_stats = dict(self.image_flow_head.last_forward_stats)

        text_kwargs = dict(kwargs)
        text_kwargs["compute_text_loss"] = True
        text_kwargs["compute_image_loss"] = False
        text_output = super().forward(*args, **text_kwargs)
        self.image_flow_head.last_forward_stats = image_flow_stats

        image_output["loss"] = text_output.loss + image_output.loss
        image_output["per_modality_loss"] = {
            "text_loss": text_output.per_modality_loss["text_loss"],
            "image_loss": image_output.per_modality_loss["image_loss"],
        }
        image_output["per_modality_count"] = {
            "text_tokens": text_output.per_modality_count["text_tokens"],
            "image_tokens": image_output.per_modality_count["image_tokens"],
        }
        text_loss_graph = getattr(
            text_output, "per_modality_loss_graph", None
        )
        image_loss_graph = getattr(
            image_output, "per_modality_loss_graph", None
        )
        if text_loss_graph is not None or image_loss_graph is not None:
            if text_loss_graph is None or image_loss_graph is None:
                raise RuntimeError(
                    "joint text/image validation produced asymmetric loss graphs"
                )
            image_output["per_modality_loss_graph"] = {
                "text_loss": text_loss_graph["text_loss"],
                "image_loss": image_loss_graph["image_loss"],
            }
        return image_output

    def _prepare_backbone_forward_kwargs(
        self,
        model_kwargs: dict,
        *,
        compute_text_loss: bool,
        compute_image_loss: bool,
    ) -> dict:
        if compute_text_loss and compute_image_loss:
            raise RuntimeError(
                "joint text/image loss must be split by the outer text-AR model"
            )
        prepared = dict(model_kwargs)
        prepared["_text_ar_mode"] = bool(compute_text_loss)
        prepared["_require_image_query_stream"] = bool(compute_image_loss)
        return prepared


__all__ = [
    "SingleStreamTextARConfig",
    "SingleStreamTextARQwen3ForCausalLM",
    "SingleStreamTextARQwen3Model",
]
