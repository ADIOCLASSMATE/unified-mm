"""Cache-first single-stream generation shared by ablations A and B.

This module contains inference only. Training and likelihood forward paths stay
in modeling_selfless_flow.py. The A/B behavior switch is exactly one attention
contract: strict sigma ordering for A, or content-only physical self edges for
B.
"""

from __future__ import annotations

import math
import types

import torch
from torch.nn.attention.flex_attention import create_block_mask
from transformers.cache_utils import Cache

from .image_position_utils import build_row_col_position_ids


class _SelflessStaticCacheLayer:
    def __init__(self, max_cache_len: int):
        self.max_cache_len = int(max_cache_len)
        self.is_initialized = False
        self.keys = None
        self.values = None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self.keys = torch.zeros(
            key_states.shape[0],
            key_states.shape[1],
            self.max_cache_len,
            key_states.shape[-1],
            device=key_states.device,
            dtype=key_states.dtype,
        )
        self.values = torch.zeros(
            value_states.shape[0],
            value_states.shape[1],
            self.max_cache_len,
            value_states.shape[-1],
            device=value_states.device,
            dtype=value_states.dtype,
        )
        self.is_initialized = True


class SelflessStaticCache(Cache):
    """Static K/V cache that writes batch-specific original sequence slots."""

    def __init__(self, config, max_cache_len: int):
        super().__init__(
            layers=[
                _SelflessStaticCacheLayer(max_cache_len)
                for _ in range(int(config.num_hidden_layers))
            ]
        )
        self._max_cache_len = int(max_cache_len)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return 0

    def get_max_cache_shape(self) -> int:
        return self._max_cache_len

    def fork(self) -> "SelflessStaticCache":
        """Clone an initialized cache for an independent branch."""

        cloned = SelflessStaticCache(
            config=types.SimpleNamespace(num_hidden_layers=len(self.layers)),
            max_cache_len=self._max_cache_len,
        )
        for source, target in zip(self.layers, cloned.layers, strict=True):
            if not source.is_initialized:
                continue
            target.keys = source.keys.clone()
            target.values = source.values.clone()
            target.is_initialized = True
        return cloned

    def repeat_batch_(self, repeats: int) -> "SelflessStaticCache":
        """Repeat cache rows in branch-major order without another model pass."""

        repeats = int(repeats)
        if repeats <= 0:
            raise ValueError(f"repeats must be positive, got {repeats}")
        if repeats == 1:
            return self
        for layer in self.layers:
            if not layer.is_initialized:
                continue
            repeat_shape = (repeats, *([1] * (layer.keys.ndim - 1)))
            layer.keys = layer.keys.repeat(repeat_shape)
            layer.values = layer.values.repeat(repeat_shape)
        return self

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_kwargs = args[0] if args else kwargs
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            raise ValueError(
                "SelflessStaticCache requires explicit cache_position."
            )
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states, value_states)
        cache_position = cache_position.to(
            device=key_states.device,
            dtype=torch.long,
        )
        if cache_position.ndim == 1:
            cache_position = cache_position.unsqueeze(0).expand(
                key_states.shape[0], -1
            )
        expected_shape = (key_states.shape[0], key_states.shape[-2])
        if tuple(cache_position.shape) != expected_shape:
            raise ValueError(
                "cache_position must align with [batch, query_length]: "
                f"{tuple(cache_position.shape)} != {expected_shape}"
            )
        cache_write_prefix = cache_kwargs.get("cache_write_prefix")
        if cache_write_prefix is not None:
            cache_write_prefix = int(cache_write_prefix)
            if not 0 < cache_write_prefix <= key_states.shape[-2]:
                raise ValueError(
                    "cache_write_prefix must be in [1, query_length], got "
                    f"{cache_write_prefix} for {key_states.shape[-2]} queries"
                )
            cache_position = cache_position[:, :cache_write_prefix]
            key_states = key_states[:, :, :cache_write_prefix]
            value_states = value_states[:, :, :cache_write_prefix]
            expected_shape = (
                key_states.shape[0],
                key_states.shape[-2],
            )
        key_indices = cache_position[:, None, :, None].expand(
            -1,
            key_states.shape[1],
            -1,
            key_states.shape[-1],
        )
        value_indices = cache_position[:, None, :, None].expand(
            -1,
            value_states.shape[1],
            -1,
            value_states.shape[-1],
        )
        cache_write_mask = cache_kwargs.get("cache_write_mask")
        if cache_write_mask is not None:
            cache_write_mask = cache_write_mask.to(
                device=key_states.device,
                dtype=torch.bool,
            )
            if cache_write_prefix is not None:
                cache_write_mask = cache_write_mask[:, :cache_write_prefix]
            if tuple(cache_write_mask.shape) != expected_shape:
                raise ValueError(
                    "cache_write_mask must align with cache_position: "
                    f"{tuple(cache_write_mask.shape)} != {expected_shape}"
                )
            key_states = torch.where(
                cache_write_mask[:, None, :, None],
                key_states,
                layer.keys.gather(2, key_indices),
            )
            value_states = torch.where(
                cache_write_mask[:, None, :, None],
                value_states,
                layer.values.gather(2, value_indices),
            )
        layer.keys.scatter_(2, key_indices, key_states)
        layer.values.scatter_(2, value_indices, value_states)
        return layer.keys, layer.values



class SelflessGenerationMixin:
    # Static A/B/C flow conditions can evaluate CFG branches in one batched
    # backbone call. Dynamic-XT overrides this because its ODE evaluator owns
    # two independently queried read-only caches.
    _supports_paired_backbone_cfg = True

    def _generation_attention_contract(self) -> str:
        """Return the only A/B generation switch.

        Ablation A uses strict sigma ordering for every query. Ablation B adds
        the physical self edge only to queries whose input is real content.
        """

        contract = str(
            getattr(
                self.config,
                "dual_stream_attention_contract",
                "selfless_strict",
            )
        ).strip().lower()
        if contract not in {
            "selfless_strict",
            "xlnet_content_diagonal",
        }:
            raise ValueError(
                "dual_stream_attention_contract must be selfless_strict or "
                f"xlnet_content_diagonal, got {contract!r}"
            )
        return contract

    def _build_generation_attention_mask(
        self,
        *,
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        sigma: torch.Tensor,
        segment_ids: torch.Tensor,
        content_query_mask: torch.Tensor,
        image_uncond_rows: torch.Tensor | None = None,
        content_self_diagonal: bool | None = None,
    ):
        """Build the full-sequence single-stream mask used by A or B."""

        from utils.utils import get_selfless_mask

        if content_self_diagonal is None:
            content_self_diagonal = (
                self._generation_attention_contract()
                == "xlnet_content_diagonal"
            )
        return get_selfless_mask(
            sigma=sigma,
            seq_len=input_ids.shape[1],
            device=input_ids.device,
            input_ids=input_ids if image_uncond_rows is not None else None,
            token_types=token_types if image_uncond_rows is not None else None,
            boi_token_id=(
                int(self.config.boi_token_id)
                if image_uncond_rows is not None
                else None
            ),
            image_uncond_rows=image_uncond_rows,
            segment_ids=segment_ids,
            diagonal_query_mask=(
                content_query_mask if content_self_diagonal else None
            ),
        )

    def _build_generation_cache_mask(
        self,
        *,
        key_sigma: torch.Tensor,
        key_valid: torch.Tensor,
        key_is_target_image: torch.Tensor,
        query_sigma: torch.Tensor,
        query_valid: torch.Tensor,
        query_positions: torch.Tensor,
        content_query_mask: torch.Tensor,
        image_uncond_rows: torch.Tensor | None = None,
        content_self_diagonal: bool | None = None,
    ):
        """Build a rectangular mask over original sequence cache slots."""

        device = query_sigma.device
        query_sigma = query_sigma.to(device=device, dtype=torch.float32)
        query_valid = query_valid.to(device=device, dtype=torch.bool)
        query_positions = query_positions.to(device=device, dtype=torch.long)
        content_query_mask = content_query_mask.to(
            device=device,
            dtype=torch.bool,
        )
        expected = tuple(query_sigma.shape)
        for name, value in (
            ("query_valid", query_valid),
            ("query_positions", query_positions),
            ("content_query_mask", content_query_mask),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must align with query_sigma: "
                    f"{tuple(value.shape)} != {expected}"
                )

        allowed = key_sigma.unsqueeze(1) < query_sigma.unsqueeze(-1)
        if content_self_diagonal is None:
            content_self_diagonal = (
                self._generation_attention_contract()
                == "xlnet_content_diagonal"
            )
        if content_self_diagonal:
            key_positions = torch.arange(
                key_sigma.shape[1],
                device=device,
                dtype=torch.long,
            ).view(1, 1, -1)
            allowed = allowed | (
                content_query_mask.unsqueeze(-1)
                & query_positions.unsqueeze(-1).eq(key_positions)
            )
        allowed = (
            allowed
            & query_valid.unsqueeze(-1)
            & key_valid.unsqueeze(1)
        )
        if image_uncond_rows is not None:
            image_uncond_rows = image_uncond_rows.to(
                device=device,
                dtype=torch.bool,
            )
            if tuple(image_uncond_rows.shape) != (query_sigma.shape[0],):
                raise ValueError(
                    "image_uncond_rows must have shape [batch], got "
                    f"{tuple(image_uncond_rows.shape)}"
                )
            allowed = allowed & (
                ~image_uncond_rows[:, None, None]
                | key_is_target_image.unsqueeze(1)
            )

        if device.type == "npu":
            return (~allowed).unsqueeze(1)

        def mask_mod(batch, head, query_index, key_index):
            del head
            return allowed[batch, query_index, key_index]

        return create_block_mask(
            mask_mod,
            B=query_sigma.shape[0],
            H=None,
            Q_LEN=query_sigma.shape[1],
            KV_LEN=key_sigma.shape[1],
            device=device,
        )

    @staticmethod
    def _halton_image_order(
        image_tokens_per_img: int,
        side: int,
        device: torch.device,
    ) -> torch.Tensor:
        def halton(index: int, base: int) -> float:
            value = 0.0
            scale = 1.0 / float(base)
            while index > 0:
                value += (index % base) * scale
                index //= base
                scale /= float(base)
            return value

        seen: set[int] = set()
        order: list[int] = []
        index = 1
        while (
            len(order) < image_tokens_per_img
            and index < image_tokens_per_img * 32
        ):
            row = min(side - 1, int(halton(index, 2) * side))
            col = min(side - 1, int(halton(index, 3) * side))
            position = row * side + col
            if position not in seen:
                seen.add(position)
                order.append(position)
            index += 1
        order.extend(
            position
            for position in range(image_tokens_per_img)
            if position not in seen
        )
        return torch.tensor(order, device=device, dtype=torch.long)

    def _image_generation_orders(
        self,
        *,
        strategy: str,
        original_sigma: torch.Tensor,
        span_starts: torch.Tensor,
        image_tokens_per_img: int,
        side: int,
    ) -> tuple[str, torch.Tensor, bool]:
        """Resolve one fixed token order per image before decoding starts."""

        strategy = str(strategy or "spatial_halton").strip().lower()
        aliases = {
            "raster": "sequential",
            "row_major": "sequential",
            "prefix": "sequential",
            "halton": "spatial_halton",
            "uniform": "spatial_uniform",
            "causal_sigma": "sigma",
            "sigma_replay": "sigma",
        }
        strategy = aliases.get(strategy, strategy)
        device = original_sigma.device
        token_count = int(image_tokens_per_img)

        if strategy == "sequential":
            base = torch.arange(token_count, device=device, dtype=torch.long)
            orders = base.unsqueeze(0).expand(original_sigma.shape[0], -1)
        elif strategy == "spatial_halton":
            base = self._halton_image_order(token_count, side, device)
            orders = base.unsqueeze(0).expand(original_sigma.shape[0], -1)
        elif strategy == "spatial_uniform":
            rows, cols = torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            )
            center = (side - 1) / 2.0
            ring = torch.maximum(
                (rows.float() - center).abs(),
                (cols.float() - center).abs(),
            )
            checker = (rows % 2) * 2 + (cols % 2)
            base = torch.argsort((ring * 4.0 + checker.float()).flatten())
            orders = base.unsqueeze(0).expand(original_sigma.shape[0], -1)
        elif strategy == "random":
            orders = torch.stack(
                [
                    torch.randperm(token_count, device=device)
                    for _ in range(original_sigma.shape[0])
                ]
            )
        elif strategy == "sigma":
            offsets = torch.arange(
                token_count,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            span_sigma = torch.gather(
                original_sigma,
                1,
                span_starts.unsqueeze(1) + offsets,
            )
            orders = torch.argsort(span_sigma, dim=1)
        else:
            raise ValueError(
                "order_strategy must be one of sequential, spatial_halton, "
                f"spatial_uniform, random, or sigma; got {strategy!r}"
            )
        return strategy, orders.clone(), strategy == "sigma"

    @staticmethod
    def _generation_debug_check(
        enabled: bool,
        name: str,
        tensor: torch.Tensor,
        step: int | None = None,
    ) -> None:
        if not enabled or bool(torch.isfinite(tensor).all()):
            return
        raise FloatingPointError(
            "non-finite tensor during image generation: "
            f"component={name!r}, generation_step={step}, "
            f"shape={tuple(tensor.shape)}"
        )

    @torch.no_grad()
    def generate_image(
        self,
        *,
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        sigma: torch.Tensor,
        spans: list[tuple[int, int, int]],
        segment_ids: torch.Tensor | None = None,
        image_latent_dim: int | None = None,
        initial_image_latents: torch.Tensor | None = None,
        initial_image_latent_mask: torch.Tensor | None = None,
        initial_noise_bank: torch.Tensor | None = None,
        flow_temperature: float = 1.0,
        flow_cfg: float = 3.5,
        flow_cfg_schedule: str = "constant",
        flow_solver: str | None = None,
        flow_num_steps: int | None = None,
        parallel_rate: int = 1,
        order_strategy: str = "spatial_halton",
        use_cache: bool = True,
        return_trace: bool = False,
        debug_finite: bool = False,
        _debug_max_generation_steps: int | None = None,
    ):
        """Generate image latents with one single-stream decoder.

        Cache is the production path. Setting use_cache=False runs the same
        state machine with full-sequence recomputation and exists only as a
        numerical reference.
        """

        if not spans:
            return (None, {}) if return_trace else None
        if input_ids.ndim != 2 or token_types.shape != input_ids.shape:
            raise ValueError("input_ids and token_types must be aligned [B,L]")
        if sigma.shape != input_ids.shape:
            raise ValueError("sigma must align with input_ids")
        if int(parallel_rate) != 1:
            raise ValueError(
                "cache-first generation is serialized; parallel_rate must be 1"
            )
        flow_temperature = float(flow_temperature)
        if not math.isfinite(flow_temperature) or flow_temperature < 0.0:
            raise ValueError(
                "flow_temperature must be finite and non-negative, got "
                f"{flow_temperature}"
            )
        flow_cfg = float(flow_cfg)
        if not math.isfinite(flow_cfg):
            raise ValueError(f"flow_cfg must be finite, got {flow_cfg}")

        device = input_ids.device
        attention_contract = self._generation_attention_contract()
        content_self_diagonal = (
            attention_contract == "xlnet_content_diagonal"
        )
        batch_size, sequence_length = input_ids.shape
        image_latent_dim = int(image_latent_dim or self.image_latent_dim)
        image_tokens_per_img = int(
            getattr(self.config, "image_tokens_per_img", 256)
        )
        side = int(math.isqrt(image_tokens_per_img))
        if side * side != image_tokens_per_img:
            raise ValueError(
                f"image_tokens_per_img={image_tokens_per_img} is not square"
            )

        row_indices = torch.tensor(
            [int(batch) for batch, _, _ in spans],
            device=device,
            dtype=torch.long,
        )
        if bool((row_indices < 0).any()) or bool((row_indices >= batch_size).any()):
            raise ValueError("span batch index is outside input_ids")
        selected_input_ids = input_ids.index_select(0, row_indices)
        selected_token_types = token_types.index_select(0, row_indices).to(
            device=device
        )
        selected_sigma = sigma.index_select(0, row_indices).to(
            device=device,
            dtype=torch.float32,
        )
        selected_batch = len(spans)

        if segment_ids is None:
            selected_segment_ids = torch.where(
                selected_token_types.ne(3),
                torch.zeros_like(selected_token_types, dtype=torch.long),
                torch.full_like(selected_token_types, -1, dtype=torch.long),
            )
        else:
            if segment_ids.shape != input_ids.shape:
                raise ValueError("segment_ids must align with input_ids")
            selected_segment_ids = segment_ids.index_select(
                0, row_indices
            ).to(device=device, dtype=torch.long)

        span_starts = torch.tensor(
            [int(start) for _, start, _ in spans],
            device=device,
            dtype=torch.long,
        )
        span_ends = torch.tensor(
            [int(end) for _, _, end in spans],
            device=device,
            dtype=torch.long,
        )
        if not bool(
            span_ends.sub(span_starts).eq(image_tokens_per_img).all()
        ):
            raise ValueError(
                "every generated image span must contain exactly "
                f"{image_tokens_per_img} tokens"
            )
        offsets = torch.arange(
            image_tokens_per_img,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        span_positions = span_starts.unsqueeze(1) + offsets
        if bool((span_positions < 0).any()) or bool(
            (span_positions >= sequence_length).any()
        ):
            raise ValueError("generated image span is outside the sequence")
        span_types = torch.gather(
            selected_token_types,
            1,
            span_positions,
        )
        if not bool(span_types.eq(1).all()):
            raise ValueError("generated spans must contain only image tokens")

        target_segment_ids = torch.gather(
            selected_segment_ids,
            1,
            span_starts.unsqueeze(1),
        ).squeeze(1)
        span_segment_ids = torch.gather(
            selected_segment_ids,
            1,
            span_positions,
        )
        if bool((target_segment_ids < 0).any()) or not bool(
            span_segment_ids.eq(target_segment_ids.unsqueeze(1)).all()
        ):
            raise ValueError(
                "each generated image must belong to one non-padding segment"
            )
        target_segment_members = selected_segment_ids.eq(
            target_segment_ids.unsqueeze(1)
        )
        image_tokens_in_target = (
            selected_token_types.eq(1) & target_segment_members
        ).sum(dim=1)
        if not bool(image_tokens_in_target.eq(image_tokens_per_img).all()):
            raise ValueError(
                "cache-first generation currently requires exactly one image "
                "span in each target segment"
            )

        latent_dtype = self.image_flow_head.net.final_layer.linear.weight.dtype
        work_latents = torch.zeros(
            selected_batch,
            sequence_length,
            image_latent_dim,
            device=device,
            dtype=latent_dtype,
        )
        if initial_image_latents is not None:
            expected = (*input_ids.shape, image_latent_dim)
            if tuple(initial_image_latents.shape) != expected:
                raise ValueError(
                    "initial_image_latents must have shape "
                    f"{expected}, got {tuple(initial_image_latents.shape)}"
                )
            work_latents.copy_(
                initial_image_latents.index_select(0, row_indices).to(
                    device=device,
                    dtype=latent_dtype,
                )
            )
        base_image_latent_mask = torch.zeros(
            selected_batch,
            sequence_length,
            device=device,
            dtype=torch.bool,
        )
        if initial_image_latent_mask is not None:
            if initial_image_latent_mask.shape != input_ids.shape:
                raise ValueError(
                    "initial_image_latent_mask must align with input_ids"
                )
            if initial_image_latents is None and bool(
                initial_image_latent_mask.any()
            ):
                raise ValueError(
                    "initial_image_latents are required for visible image tokens"
                )
            base_image_latent_mask.copy_(
                initial_image_latent_mask.index_select(0, row_indices).to(
                    device=device,
                    dtype=torch.bool,
                )
            )
        if bool(
            torch.gather(
                base_image_latent_mask,
                1,
                span_positions,
            ).any()
        ):
            raise ValueError(
                "generated image spans must start fully masked"
            )

        selected_initial_noise = None
        if initial_noise_bank is not None:
            expected = (
                selected_batch,
                image_tokens_per_img,
                image_latent_dim,
            )
            if (
                not isinstance(initial_noise_bank, torch.Tensor)
                or tuple(initial_noise_bank.shape) != expected
                or not initial_noise_bank.is_floating_point()
            ):
                raise ValueError(
                    f"initial_noise_bank must be a floating tensor of shape {expected}"
                )
            if not bool(
                torch.isfinite(
                    initial_noise_bank.to(torch.float32) * flow_temperature
                ).all()
            ):
                raise FloatingPointError(
                    "initial_noise_bank is non-finite after temperature scaling"
                )
            selected_initial_noise = initial_noise_bank.to(
                device=device,
                dtype=torch.float32,
            )

        (
            order_strategy,
            generation_orders,
            replay_original_sigma,
        ) = self._image_generation_orders(
            strategy=order_strategy,
            original_sigma=selected_sigma,
            span_starts=span_starts,
            image_tokens_per_img=image_tokens_per_img,
            side=side,
        )

        original_span_sigma = torch.gather(
            selected_sigma,
            1,
            span_positions,
        )
        context_cutoff = original_span_sigma.min(dim=1).values
        context_member = (
            target_segment_members
            & selected_token_types.ne(1)
            & selected_token_types.ne(3)
            & selected_sigma.lt(context_cutoff.unsqueeze(1))
        )
        context_counts = context_member.sum(dim=1)
        if bool(context_counts.eq(0).any()):
            raise ValueError(
                "image generation requires at least one visible context token "
                "per target segment"
            )

        current_sigma = selected_sigma.clone()
        next_image_sigma = context_cutoff.clone()
        if not replay_original_sigma:
            context_max = selected_sigma.masked_fill(
                ~context_member,
                -torch.inf,
            ).max(dim=1).values
            next_image_sigma = torch.maximum(
                context_cutoff,
                context_max + 1.0,
            )
            current_sigma.scatter_(
                1,
                span_positions,
                (next_image_sigma + image_tokens_per_img)
                .unsqueeze(1)
                .expand_as(span_positions),
            )

        generated = torch.zeros(
            selected_batch,
            image_tokens_per_img,
            image_latent_dim,
            device=device,
            dtype=latent_dtype,
        )
        filled = torch.zeros(
            selected_batch,
            image_tokens_per_img,
            device=device,
            dtype=torch.bool,
        )
        trace_order = (
            torch.zeros(
                selected_batch,
                image_tokens_per_img,
                device=device,
                dtype=torch.long,
            )
            if return_trace
            else None
        )
        batch_indices = torch.arange(
            selected_batch,
            device=device,
            dtype=torch.long,
        )
        full_position_ids = build_row_col_position_ids(
            selected_token_types,
            image_tokens_per_img,
        )
        use_flow_cfg = flow_cfg != 1.0
        if use_flow_cfg and getattr(self.config, "boi_token_id", None) is None:
            raise ValueError("flow_cfg != 1 requires config.boi_token_id")
        pair_backbone_cfg = bool(
            use_cache
            and use_flow_cfg
            and self._supports_paired_backbone_cfg
        )
        unconditional_rows = (
            torch.ones(selected_batch, device=device, dtype=torch.bool)
            if use_flow_cfg
            else None
        )
        paired_cfg_rows = (
            torch.cat(
                [torch.zeros_like(unconditional_rows), unconditional_rows]
            )
            if pair_backbone_cfg
            else None
        )

        flow_cache = self.image_flow_head.empty_latent_mixer_cache(
            batch_size=selected_batch * (2 if use_flow_cfg else 1),
            capacity=image_tokens_per_img,
        )
        conditional_cache = None
        unconditional_cache = None
        key_sigma = None
        key_valid = None
        key_is_target_image = None
        context_length = int(context_counts.max().item())
        backbone_cache_peak_bytes = 0

        def gather_position_ids(indices: torch.Tensor) -> torch.Tensor:
            return torch.gather(
                full_position_ids,
                dim=2,
                index=indices.unsqueeze(0).expand(2, -1, -1),
            )

        if use_cache:
            physical_positions = torch.arange(
                sequence_length,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0).expand(selected_batch, -1)
            context_indices = torch.where(
                context_member,
                physical_positions,
                sequence_length,
            ).topk(
                context_length,
                dim=1,
                largest=False,
                sorted=True,
            ).values
            context_valid = context_indices.ne(sequence_length)
            context_indices.clamp_max_(sequence_length - 1)

            key_sigma = torch.full(
                (selected_batch, sequence_length),
                torch.inf,
                device=device,
                dtype=torch.float32,
            )
            key_valid = torch.zeros(
                selected_batch,
                sequence_length,
                device=device,
                dtype=torch.bool,
            )
            key_is_target_image = torch.zeros_like(key_valid)
            gathered_context_sigma = torch.gather(
                selected_sigma,
                1,
                context_indices,
            )
            key_sigma.scatter_(
                1,
                context_indices,
                gathered_context_sigma.masked_fill(
                    ~context_valid,
                    torch.inf,
                ),
            )
            key_valid.scatter_(1, context_indices, context_valid)
            context_mask = self._build_generation_cache_mask(
                key_sigma=key_sigma,
                key_valid=key_valid,
                key_is_target_image=key_is_target_image,
                query_sigma=gathered_context_sigma.masked_fill(
                    ~context_valid,
                    torch.inf,
                ),
                query_valid=context_valid,
                query_positions=context_indices,
                content_query_mask=context_valid,
                image_uncond_rows=None,
                content_self_diagonal=content_self_diagonal,
            )
            conditional_cache = SelflessStaticCache(
                config=self.model.config,
                max_cache_len=sequence_length,
            )
            context_latent_indices = context_indices.unsqueeze(-1).expand(
                -1,
                -1,
                image_latent_dim,
            )
            self.model(
                X0_input_ids=torch.gather(
                    selected_input_ids,
                    1,
                    context_indices,
                ),
                attention_mask=context_mask,
                position_ids=gather_position_ids(context_indices),
                past_key_values=conditional_cache,
                use_cache=True,
                cache_position=context_indices,
                cache_write_mask=context_valid,
                token_types=torch.gather(
                    selected_token_types,
                    1,
                    context_indices,
                )
                .to(dtype=torch.long)
                .masked_fill(~context_valid, 3),
                image_latents=torch.gather(
                    work_latents,
                    1,
                    context_latent_indices,
                ),
                image_latent_mask=torch.gather(
                    base_image_latent_mask,
                    1,
                    context_indices,
                ),
                image_reveal_sigma=gathered_context_sigma,
                calculate_likelihood=False,
                debug_finite_backbone=debug_finite,
                debug_backbone_label="generation_cache_prefill",
            )
            if use_flow_cfg:
                if pair_backbone_cfg:
                    conditional_cache.repeat_batch_(2)
                else:
                    unconditional_cache = conditional_cache.fork()
            cache_dtype = next(self.model.parameters()).dtype
            backbone_cache_peak_bytes = (
                selected_batch
                * int(self.config.num_hidden_layers)
                * int(self.config.num_key_value_heads)
                * sequence_length
                * int(
                    getattr(
                        self.config,
                        "head_dim",
                        self.config.hidden_size
                        // self.config.num_attention_heads,
                    )
                )
                * cache_dtype.itemsize
                * 2
                * (2 if use_flow_cfg else 1)
            )

        pending_local_positions: torch.Tensor | None = None
        debug_conditional_hidden: list[torch.Tensor] = []
        debug_unconditional_hidden: list[torch.Tensor] = []
        completed_steps = 0
        mask_only_query = torch.zeros(
            selected_batch,
            1,
            device=device,
            dtype=torch.bool,
        )
        content_then_mask_queries = torch.tensor(
            [True, False],
            device=device,
            dtype=torch.bool,
        ).unsqueeze(0).expand(selected_batch, -1)

        def cached_query(
            current_local_positions: torch.Tensor,
            *,
            cache: SelflessStaticCache,
            image_uncond_rows: torch.Tensor | None,
            label: str,
        ) -> torch.Tensor:
            current_positions = span_starts + current_local_positions
            current_indices = current_positions.unsqueeze(1)
            current_query_sigma = current_sigma[
                batch_indices,
                current_positions,
            ].unsqueeze(1)
            if pending_local_positions is None:
                query_indices = current_indices
                query_sigma = current_query_sigma
                content_queries = mask_only_query
                cache_read_only = True
            else:
                pending_positions = span_starts + pending_local_positions
                pending_indices = pending_positions.unsqueeze(1)
                query_indices = torch.cat(
                    [pending_indices, current_indices],
                    dim=1,
                )
                query_sigma = torch.cat(
                    [
                        current_sigma[
                            batch_indices,
                            pending_positions,
                        ].unsqueeze(1),
                        current_query_sigma,
                    ],
                    dim=1,
                )
                content_queries = content_then_mask_queries
                cache_read_only = False

            query_valid = torch.ones_like(content_queries)
            branch_repeats = 1
            if image_uncond_rows is not None:
                if image_uncond_rows.ndim != 1 or (
                    image_uncond_rows.shape[0] % selected_batch
                ):
                    raise ValueError(
                        "image_uncond_rows must contain whole generation batches"
                    )
                branch_repeats = image_uncond_rows.shape[0] // selected_batch

            def repeat_rows(value: torch.Tensor) -> torch.Tensor:
                return value.repeat(
                    (branch_repeats, *([1] * (value.ndim - 1)))
                )

            key_sigma_for_mask = key_sigma
            key_valid_for_mask = key_valid
            key_is_image_for_mask = key_is_target_image
            if branch_repeats > 1:
                query_indices = repeat_rows(query_indices)
                query_sigma = repeat_rows(query_sigma)
                content_queries = repeat_rows(content_queries)
                query_valid = repeat_rows(query_valid)
                key_sigma_for_mask = repeat_rows(key_sigma)
                key_valid_for_mask = repeat_rows(key_valid)
                key_is_image_for_mask = repeat_rows(key_is_target_image)

            attention_mask = self._build_generation_cache_mask(
                key_sigma=key_sigma_for_mask,
                key_valid=key_valid_for_mask,
                key_is_target_image=key_is_image_for_mask,
                query_sigma=query_sigma,
                query_valid=query_valid,
                query_positions=query_indices,
                content_query_mask=content_queries,
                image_uncond_rows=image_uncond_rows,
                content_self_diagonal=content_self_diagonal,
            )
            latent_indices = query_indices.unsqueeze(-1).expand(
                -1,
                -1,
                image_latent_dim,
            )
            query_input_ids = torch.gather(
                selected_input_ids,
                1,
                query_indices[:selected_batch],
            )
            query_token_types = torch.gather(
                selected_token_types,
                1,
                query_indices[:selected_batch],
            )
            query_latents = torch.gather(
                work_latents,
                1,
                latent_indices[:selected_batch],
            )
            query_position_ids = gather_position_ids(
                query_indices[:selected_batch]
            )
            if branch_repeats > 1:
                query_input_ids = repeat_rows(query_input_ids)
                query_token_types = repeat_rows(query_token_types)
                query_latents = repeat_rows(query_latents)
                query_position_ids = query_position_ids.repeat(
                    1,
                    branch_repeats,
                    1,
                )

            hidden = self.model(
                X0_input_ids=query_input_ids,
                attention_mask=attention_mask,
                position_ids=query_position_ids,
                past_key_values=cache,
                use_cache=not cache_read_only,
                cache_position=query_indices,
                cache_read_only=cache_read_only,
                cache_write_prefix=(None if cache_read_only else 1),
                token_types=query_token_types,
                image_latents=query_latents,
                image_latent_mask=content_queries,
                image_reveal_sigma=query_sigma,
                calculate_likelihood=False,
                debug_finite_backbone=debug_finite,
                debug_backbone_label=label,
            ).last_hidden_state
            return hidden[:, -1]

        for step_index in range(image_tokens_per_img):
            if (
                _debug_max_generation_steps is not None
                and step_index >= int(_debug_max_generation_steps)
            ):
                break

            current_local_positions = generation_orders[:, step_index]
            current_positions = span_starts + current_local_positions
            attention_mask = None
            uncond_attention_mask = None

            if use_cache:
                if pending_local_positions is not None:
                    pending_positions = span_starts + pending_local_positions
                    pending_indices = pending_positions.unsqueeze(1)
                    key_sigma.scatter_(
                        1,
                        pending_indices,
                        current_sigma[
                            batch_indices,
                            pending_positions,
                        ].unsqueeze(1),
                    )
                    key_valid.scatter_(
                        1,
                        pending_indices,
                        torch.ones_like(pending_indices, dtype=torch.bool),
                    )
                    key_is_target_image.scatter_(
                        1,
                        pending_indices,
                        torch.ones_like(pending_indices, dtype=torch.bool),
                    )
                if pair_backbone_cfg:
                    paired_hidden = cached_query(
                        current_local_positions,
                        cache=conditional_cache,
                        image_uncond_rows=paired_cfg_rows,
                        label=f"paired_cfg_cache_step={step_index + 1}",
                    )
                    conditional_hidden, unconditional_hidden = (
                        paired_hidden.split(selected_batch, dim=0)
                    )
                else:
                    conditional_hidden = cached_query(
                        current_local_positions,
                        cache=conditional_cache,
                        image_uncond_rows=None,
                        label=f"conditional_cache_step={step_index + 1}",
                    )
                    unconditional_hidden = (
                        cached_query(
                            current_local_positions,
                            cache=unconditional_cache,
                            image_uncond_rows=unconditional_rows,
                            label=f"unconditional_cache_step={step_index + 1}",
                        )
                        if use_flow_cfg
                        else None
                    )
            else:
                image_latent_mask = base_image_latent_mask.clone()
                image_latent_mask[
                    batch_indices.unsqueeze(1),
                    span_positions,
                ] = filled
                content_query_mask = (
                    selected_token_types.ne(3)
                    & (
                        selected_token_types.ne(1)
                        | image_latent_mask
                    )
                )
                attention_mask = self._build_generation_attention_mask(
                    input_ids=selected_input_ids,
                    token_types=selected_token_types,
                    sigma=current_sigma,
                    segment_ids=selected_segment_ids,
                    content_query_mask=content_query_mask,
                    content_self_diagonal=content_self_diagonal,
                )
                full_hidden = self.model(
                    X0_input_ids=selected_input_ids,
                    attention_mask=attention_mask,
                    token_types=selected_token_types,
                    image_latents=work_latents,
                    image_latent_mask=image_latent_mask,
                    image_reveal_sigma=current_sigma,
                    calculate_likelihood=False,
                    debug_finite_backbone=debug_finite,
                    debug_backbone_label=(
                        f"conditional_full_step={step_index + 1}"
                    ),
                ).last_hidden_state
                conditional_hidden = full_hidden[
                    batch_indices,
                    current_positions,
                ]
                unconditional_hidden = None
                if use_flow_cfg:
                    uncond_attention_mask = (
                        self._build_generation_attention_mask(
                            input_ids=selected_input_ids,
                            token_types=selected_token_types,
                            sigma=current_sigma,
                            segment_ids=selected_segment_ids,
                            content_query_mask=content_query_mask,
                            image_uncond_rows=unconditional_rows,
                            content_self_diagonal=content_self_diagonal,
                        )
                    )
                    uncond_full_hidden = self.model(
                        X0_input_ids=selected_input_ids,
                        attention_mask=uncond_attention_mask,
                        token_types=selected_token_types,
                        image_latents=work_latents,
                        image_latent_mask=image_latent_mask,
                        image_reveal_sigma=current_sigma,
                        calculate_likelihood=False,
                        debug_finite_backbone=debug_finite,
                        debug_backbone_label=(
                            f"unconditional_full_step={step_index + 1}"
                        ),
                    ).last_hidden_state
                    unconditional_hidden = uncond_full_hidden[
                        batch_indices,
                        current_positions,
                    ]

            self._generation_debug_check(
                debug_finite,
                "conditional_backbone_hidden",
                conditional_hidden,
                step_index + 1,
            )
            if use_flow_cfg:
                self._generation_debug_check(
                    debug_finite,
                    "unconditional_backbone_hidden",
                    unconditional_hidden,
                    step_index + 1,
                )
            if return_trace and _debug_max_generation_steps is not None:
                debug_conditional_hidden.append(
                    conditional_hidden.detach().float().cpu()
                )
                if unconditional_hidden is not None:
                    debug_unconditional_hidden.append(
                        unconditional_hidden.detach().float().cpu()
                    )

            condition = self._prepare_image_flow_condition(
                conditional_hidden
            )
            unconditional_condition = (
                self._prepare_image_flow_condition(unconditional_hidden)
                if unconditional_hidden is not None
                else None
            )
            flow_context = {
                "query_positions": current_local_positions,
                "latent_mixer_cache": flow_cache,
                "latent_mixer_cache_is_paired": use_flow_cfg,
                "context_prepared": True,
                "initial_noise_prevalidated": True,
            }
            condition_evaluator = self._make_backbone_flow_condition_evaluator(
                selected_input_ids=selected_input_ids,
                selected_token_types=selected_token_types,
                current_sigma=current_sigma,
                work_latents=work_latents,
                base_image_latent_mask=base_image_latent_mask,
                filled=filled,
                span_starts=span_starts,
                sample_indices=batch_indices,
                seq_positions=current_positions,
                local_positions=current_local_positions,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                use_flow_cfg=use_flow_cfg,
                backbone_cache_enabled=bool(use_cache),
                backbone_cond_cache=conditional_cache,
                backbone_uncond_cache=unconditional_cache,
                backbone_key_sigma=key_sigma,
                backbone_key_valid=key_valid,
                backbone_key_is_image=key_is_target_image,
                backbone_max_cache_len=sequence_length,
                full_position_ids=full_position_ids,
                image_tokens_per_img=image_tokens_per_img,
                debug_finite=debug_finite,
                generation_step=step_index + 1,
            )
            prediction = self.sample_image_flow_with_cfg(
                condition,
                z_uncond=unconditional_condition,
                temperature=flow_temperature,
                cfg=flow_cfg,
                cfg_schedule=flow_cfg_schedule,
                solver=flow_solver,
                num_steps=flow_num_steps,
                initial_noise=(
                    selected_initial_noise[
                        batch_indices,
                        current_local_positions,
                    ]
                    if selected_initial_noise is not None
                    else None
                ),
                condition_evaluator=condition_evaluator,
                debug_finite=debug_finite,
                debug_label=f"image_generation_step={step_index + 1}",
                **flow_context,
            ).to(dtype=latent_dtype)
            self._generation_debug_check(
                debug_finite,
                "flow_prediction",
                prediction,
                step_index + 1,
            )

            work_latents[
                batch_indices,
                current_positions,
            ] = prediction
            generated[
                batch_indices,
                current_local_positions,
            ] = prediction
            filled[
                batch_indices,
                current_local_positions,
            ] = True
            if not replay_original_sigma:
                current_sigma[
                    batch_indices,
                    current_positions,
                ] = next_image_sigma
                next_image_sigma += 1.0

            completed_steps = step_index + 1
            if trace_order is not None:
                trace_order[
                    batch_indices,
                    current_local_positions,
                ] = completed_steps

            cached_latents = prediction
            cached_conditions = condition
            cached_positions = current_local_positions
            if use_flow_cfg:
                cached_latents = torch.cat(
                    [cached_latents, cached_latents],
                    dim=0,
                )
                cached_conditions = torch.cat(
                    [cached_conditions, unconditional_condition],
                    dim=0,
                )
                cached_positions = torch.cat(
                    [cached_positions, cached_positions],
                    dim=0,
                )
            flow_cache = self.image_flow_head.append_latent_mixer_cache(
                flow_cache,
                context_latents=cached_latents,
                context_conditions=cached_conditions,
                context_positions=cached_positions,
            )
            pending_local_positions = current_local_positions

        generated = generated.view(
            selected_batch,
            side,
            side,
            image_latent_dim,
        ).permute(0, 3, 1, 2)
        self._generation_debug_check(
            debug_finite,
            "generated_image_latents",
            generated,
        )
        if not return_trace:
            return generated

        flow_cache_peak_bytes = 0
        cfg_cache_divergence = None
        if isinstance(flow_cache, dict) and "layers" in flow_cache:
            flow_cache_peak_bytes = (
                sum(
                    layer[name].numel() * layer[name].element_size()
                    for layer in flow_cache["layers"]
                    for name in ("k", "v")
                )
                // selected_batch
            )
            if use_flow_cfg and flow_cache["layers"]:
                cfg_cache_divergence = [
                    float(
                        (
                            layer["k"][:selected_batch].detach().float()
                            - layer["k"][selected_batch:].detach().float()
                        )
                        .pow(2)
                        .mean()
                        .sqrt()
                        .item()
                    )
                    for layer in flow_cache["layers"]
                ]

        generation_order = trace_order.view(selected_batch, side, side)
        generation_score = torch.where(
            trace_order > 0,
            image_tokens_per_img - trace_order + 1,
            0,
        ).to(torch.float32).view(selected_batch, side, side)
        reveal_fraction = torch.where(
            trace_order > 0,
            (trace_order.to(torch.float32) - 1.0)
            / float(max(image_tokens_per_img - 1, 1)),
            0.0,
        ).view(selected_batch, side, side)
        trace = {
            "generation_mode": "single_stream",
            "attention_contract": attention_contract,
            "single_stream_attention_contract": attention_contract,
            "single_stream_content_self_diagonal": (
                attention_contract == "xlnet_content_diagonal"
            ),
            "order_strategy": order_strategy,
            "generation_order": generation_order,
            "generation_step": generation_order,
            "generation_score": generation_score,
            "reveal_fraction": reveal_fraction,
            "flow_cfg": float(flow_cfg),
            "flow_cfg_schedule": str(flow_cfg_schedule),
            "flow_head_architecture": "dynamic_dual_stream_pure_2d",
            "segment_isolation_enabled": True,
            "backbone_kv_cache_enabled": bool(use_cache),
            "backbone_cfg_batched": pair_backbone_cfg,
            "backbone_kv_cache_context_tokens": context_length if use_cache else 0,
            "backbone_kv_cache_tokens_committed": (
                max(completed_steps - 1, 0) if use_cache else 0
            ),
            "backbone_kv_cache_peak_bytes": int(backbone_cache_peak_bytes),
            "flow_content_cache_peak_bytes_per_sample": int(
                flow_cache_peak_bytes
            ),
            "flow_cfg_content_cache_divergence_by_layer": (
                cfg_cache_divergence
            ),
            "debug_conditional_backbone_hidden": (
                torch.stack(debug_conditional_hidden)
                if debug_conditional_hidden
                else None
            ),
            "debug_unconditional_backbone_hidden": (
                torch.stack(debug_unconditional_hidden)
                if debug_unconditional_hidden
                else None
            ),
        }
        return generated, trace

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Sample one token without a needless softmax on the greedy path."""

        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError(
                f"temperature must be finite and non-negative, got {temperature}"
            )
        if temperature < 1e-6:
            return logits.argmax(dim=-1)
        return torch.multinomial(
            torch.softmax(logits / temperature, dim=-1),
            1,
        ).squeeze(-1)

    def _generation_stop_token_ids(
        self,
        eos_token_id: int | tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        value = (
            getattr(self.config, "eos_token_id", None)
            if eos_token_id is None
            else eos_token_id
        )
        if value is None:
            return ()
        values = value if isinstance(value, (tuple, list)) else (value,)
        return tuple(dict.fromkeys(int(token_id) for token_id in values))

    @staticmethod
    def _matches_stop_token(
        token_ids: torch.Tensor,
        stop_token_ids: tuple[int, ...],
    ) -> torch.Tensor:
        matches = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in stop_token_ids:
            matches |= token_ids.eq(token_id)
        return matches

    def _prepare_text_generation_state(
        self,
        *,
        input_ids: torch.Tensor,
        token_types: torch.Tensor | None,
        sigma: torch.Tensor | None,
        segment_ids: torch.Tensor | None,
        image_latents: torch.Tensor | None,
        image_latent_mask: torch.Tensor | None,
        max_new_tokens: int,
    ) -> dict[str, torch.Tensor | int]:
        """Validate and allocate a left-aligned text/caption generation batch."""

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,L]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        device = input_ids.device
        batch_size, prompt_width = input_ids.shape
        if token_types is None:
            token_types = torch.zeros_like(input_ids, dtype=torch.uint8)
        else:
            token_types = token_types.to(device=device)
            if token_types.shape != input_ids.shape:
                raise ValueError("token_types must align with input_ids")

        prompt_valid = token_types.ne(3)
        prompt_lengths = prompt_valid.sum(dim=1)
        if bool(prompt_lengths.eq(0).any()):
            raise ValueError("every prompt must contain at least one token")
        positions = torch.arange(prompt_width, device=device).unsqueeze(0)
        expected_valid = positions < prompt_lengths.unsqueeze(1)
        if not torch.equal(prompt_valid, expected_valid):
            raise ValueError(
                "text generation requires left-aligned prompts with only "
                "right padding"
            )

        if sigma is None:
            sigma = positions.expand(batch_size, -1).to(torch.float32)
        else:
            sigma = sigma.to(device=device, dtype=torch.float32)
            if sigma.shape != input_ids.shape:
                raise ValueError("sigma must align with input_ids")
        sigma = sigma.masked_fill(~prompt_valid, torch.inf)

        if segment_ids is None:
            segment_ids = torch.where(
                prompt_valid,
                torch.zeros_like(input_ids, dtype=torch.long),
                torch.full_like(input_ids, -1, dtype=torch.long),
            )
        else:
            segment_ids = segment_ids.to(device=device, dtype=torch.long)
            if segment_ids.shape != input_ids.shape:
                raise ValueError("segment_ids must align with input_ids")
            if not bool(segment_ids[prompt_valid].ge(0).all()):
                raise ValueError("valid prompt tokens need non-negative segment IDs")

        image_latent_dim = int(self.image_latent_dim)
        if image_latents is None:
            if bool(token_types.eq(1).any()):
                raise ValueError(
                    "image_latents are required for image-bearing caption prompts"
                )
            image_latents = torch.zeros(
                batch_size,
                prompt_width,
                image_latent_dim,
                device=device,
                dtype=self.image_flow_head.net.final_layer.linear.weight.dtype,
            )
        else:
            expected = (batch_size, prompt_width, image_latent_dim)
            if tuple(image_latents.shape) != expected:
                raise ValueError(
                    f"image_latents must have shape {expected}, got "
                    f"{tuple(image_latents.shape)}"
                )
            image_latents = image_latents.to(
                device=device,
                dtype=self.image_flow_head.net.final_layer.linear.weight.dtype,
            )
        if image_latent_mask is None:
            image_latent_mask = token_types.eq(1)
        else:
            image_latent_mask = image_latent_mask.to(
                device=device,
                dtype=torch.bool,
            )
            if image_latent_mask.shape != input_ids.shape:
                raise ValueError("image_latent_mask must align with input_ids")
        if not bool(
            image_latent_mask[token_types.eq(1) & prompt_valid].all()
        ):
            raise ValueError("all image tokens in a caption prompt must be visible")

        capacity = prompt_width + int(max_new_tokens)
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        ids = torch.full(
            (batch_size, capacity),
            int(pad_token_id),
            device=device,
            dtype=torch.long,
        )
        types = torch.full(
            (batch_size, capacity),
            3,
            device=device,
            dtype=torch.uint8,
        )
        sigmas = torch.full(
            (batch_size, capacity),
            torch.inf,
            device=device,
            dtype=torch.float32,
        )
        segments = torch.full(
            (batch_size, capacity),
            -1,
            device=device,
            dtype=torch.long,
        )
        latents = torch.zeros(
            batch_size,
            capacity,
            image_latent_dim,
            device=device,
            dtype=image_latents.dtype,
        )
        latent_mask = torch.zeros(
            batch_size,
            capacity,
            device=device,
            dtype=torch.bool,
        )
        content_mask = torch.zeros_like(latent_mask)

        ids[:, :prompt_width] = input_ids
        types[:, :prompt_width] = token_types
        sigmas[:, :prompt_width] = sigma
        segments[:, :prompt_width] = segment_ids
        latents[:, :prompt_width] = image_latents
        latent_mask[:, :prompt_width] = image_latent_mask
        content_mask[:, :prompt_width] = prompt_valid

        batch_indices = torch.arange(
            batch_size,
            device=device,
            dtype=torch.long,
        )
        last_prompt_positions = prompt_lengths - 1
        target_segments = segment_ids[
            batch_indices,
            last_prompt_positions,
        ]
        finite_sigma = sigma.masked_fill(~prompt_valid, -torch.inf)
        next_sigma = finite_sigma.max(dim=1).values + 1.0
        generation_offsets = torch.arange(
            int(max_new_tokens),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        target_positions = prompt_lengths.unsqueeze(1) + generation_offsets
        target_rows = batch_indices.unsqueeze(1)
        ids[target_rows, target_positions] = int(self.config.mask_token_id)
        types[target_rows, target_positions] = 0
        sigmas[target_rows, target_positions] = (
            next_sigma.unsqueeze(1) + generation_offsets
        )
        segments[target_rows, target_positions] = target_segments.unsqueeze(1)

        position_ids = build_row_col_position_ids(
            types,
            int(getattr(self.config, "image_tokens_per_img", 256)),
        )
        return {
            "ids": ids,
            "types": types,
            "sigma": sigmas,
            "segments": segments,
            "latents": latents,
            "latent_mask": latent_mask,
            "content_mask": content_mask,
            "prompt_lengths": prompt_lengths,
            "prompt_width": int(prompt_width),
            "capacity": int(capacity),
            "position_ids": position_ids,
            "batch_indices": batch_indices,
        }

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
        """Generate pure text or captions with A/B sigma attention.

        The current position is always a mask query. Previously sampled tokens
        are content. Therefore A and B differ only through
        _generation_attention_contract.
        """

        state = self._prepare_text_generation_state(
            input_ids=input_ids,
            token_types=token_types,
            sigma=sigma,
            segment_ids=segment_ids,
            image_latents=image_latents,
            image_latent_mask=image_latent_mask,
            max_new_tokens=int(max_new_tokens),
        )
        attention_contract = self._generation_attention_contract()
        content_self_diagonal = (
            attention_contract == "xlnet_content_diagonal"
        )
        if int(max_new_tokens) == 0:
            output = state["ids"][:, : state["prompt_width"]]
            if return_trace:
                return output, {
                    "generation_mode": "single_stream_text",
                    "attention_contract": attention_contract,
                    "backbone_kv_cache_enabled": bool(use_cache),
                    "generated_tokens": 0,
                }
            return output

        ids = state["ids"]
        types = state["types"]
        sigmas = state["sigma"]
        segments = state["segments"]
        latents = state["latents"]
        latent_mask = state["latent_mask"]
        content_mask = state["content_mask"]
        prompt_lengths = state["prompt_lengths"]
        position_ids = state["position_ids"]
        batch_indices = state["batch_indices"]
        batch_size = ids.shape[0]
        capacity = int(state["capacity"])
        stop_token_ids = self._generation_stop_token_ids(eos_token_id)
        finished_token_id = stop_token_ids[0] if stop_token_ids else None

        key_sigma = torch.full(
            (batch_size, capacity),
            torch.inf,
            device=ids.device,
            dtype=torch.float32,
        )
        key_valid = torch.zeros(
            batch_size,
            capacity,
            device=ids.device,
            dtype=torch.bool,
        )
        key_is_target_image = torch.zeros_like(key_valid)
        cache = None

        if use_cache:
            prompt_width = int(state["prompt_width"])
            context_indices = torch.arange(
                prompt_width,
                device=ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
            context_valid = context_indices < prompt_lengths.unsqueeze(1)
            key_sigma[:, :prompt_width] = sigmas[:, :prompt_width]
            key_valid[:, :prompt_width] = context_valid
            context_attention = self._build_generation_cache_mask(
                key_sigma=key_sigma,
                key_valid=key_valid,
                key_is_target_image=key_is_target_image,
                query_sigma=sigmas[:, :prompt_width],
                query_valid=context_valid,
                query_positions=context_indices,
                content_query_mask=context_valid,
                image_uncond_rows=None,
                content_self_diagonal=content_self_diagonal,
            )
            cache = SelflessStaticCache(
                config=self.model.config,
                max_cache_len=capacity,
            )
            self.model(
                X0_input_ids=ids[:, :prompt_width],
                attention_mask=context_attention,
                position_ids=position_ids[:, :, :prompt_width],
                past_key_values=cache,
                use_cache=True,
                cache_position=context_indices,
                cache_write_mask=context_valid,
                token_types=types[:, :prompt_width],
                image_latents=latents[:, :prompt_width],
                image_latent_mask=latent_mask[:, :prompt_width],
                image_reveal_sigma=sigmas[:, :prompt_width],
                calculate_likelihood=False,
            )

        finished = torch.zeros(
            batch_size,
            device=ids.device,
            dtype=torch.bool,
        )
        pending_positions = None
        generated_steps = 0
        mask_only_query = torch.zeros(
            batch_size,
            1,
            device=ids.device,
            dtype=torch.bool,
        )
        content_then_mask_queries = torch.tensor(
            [True, False],
            device=ids.device,
            dtype=torch.bool,
        ).unsqueeze(0).expand(batch_size, -1)

        for step in range(int(max_new_tokens)):
            query_positions = prompt_lengths + step
            if use_cache:
                if pending_positions is not None:
                    pending_indices = pending_positions.unsqueeze(1)
                    key_sigma.scatter_(
                        1,
                        pending_indices,
                        sigmas[batch_indices, pending_positions].unsqueeze(1),
                    )
                    key_valid.scatter_(
                        1,
                        pending_indices,
                        torch.ones_like(pending_indices, dtype=torch.bool),
                    )
                    current_indices = query_positions.unsqueeze(1)
                    forward_indices = torch.cat(
                        [pending_indices, current_indices],
                        dim=1,
                    )
                    content_queries = content_then_mask_queries
                    cache_read_only = False
                else:
                    forward_indices = query_positions.unsqueeze(1)
                    content_queries = mask_only_query
                    cache_read_only = True

                query_sigma = torch.gather(
                    sigmas,
                    1,
                    forward_indices,
                )
                attention_mask = self._build_generation_cache_mask(
                    key_sigma=key_sigma,
                    key_valid=key_valid,
                    key_is_target_image=key_is_target_image,
                    query_sigma=query_sigma,
                    query_valid=torch.ones_like(content_queries),
                    query_positions=forward_indices,
                    content_query_mask=content_queries,
                    image_uncond_rows=None,
                    content_self_diagonal=content_self_diagonal,
                )
                hidden = self.model(
                    X0_input_ids=torch.gather(ids, 1, forward_indices),
                    attention_mask=attention_mask,
                    position_ids=torch.gather(
                        position_ids,
                        2,
                        forward_indices.unsqueeze(0).expand(2, -1, -1),
                    ),
                    past_key_values=cache,
                    use_cache=not cache_read_only,
                    cache_position=forward_indices,
                    cache_read_only=cache_read_only,
                    cache_write_prefix=(None if cache_read_only else 1),
                    token_types=torch.gather(types, 1, forward_indices),
                    image_latents=torch.gather(
                        latents,
                        1,
                        forward_indices.unsqueeze(-1).expand(
                            -1,
                            -1,
                            latents.shape[-1],
                        ),
                    ),
                    image_latent_mask=content_queries,
                    image_reveal_sigma=query_sigma,
                    calculate_likelihood=False,
                ).last_hidden_state[:, -1]
            else:
                current_width = int(query_positions.max().item()) + 1
                attention_mask = self._build_generation_attention_mask(
                    input_ids=ids[:, :current_width],
                    token_types=types[:, :current_width],
                    sigma=sigmas[:, :current_width],
                    segment_ids=segments[:, :current_width],
                    content_query_mask=content_mask[:, :current_width],
                    content_self_diagonal=content_self_diagonal,
                )
                full_hidden = self.model(
                    X0_input_ids=ids[:, :current_width],
                    attention_mask=attention_mask,
                    position_ids=position_ids[:, :, :current_width],
                    token_types=types[:, :current_width],
                    image_latents=latents[:, :current_width],
                    image_latent_mask=latent_mask[:, :current_width],
                    image_reveal_sigma=sigmas[:, :current_width],
                    calculate_likelihood=False,
                ).last_hidden_state
                hidden = full_hidden[batch_indices, query_positions]

            next_token = self._sample_token(
                self.lm_head(hidden),
                float(temperature),
            )
            if finished_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, finished_token_id),
                    next_token,
                )
            ids[batch_indices, query_positions] = next_token
            content_mask[batch_indices, query_positions] = True
            pending_positions = query_positions
            generated_steps = step + 1

            if stop_token_ids:
                finished |= self._matches_stop_token(
                    next_token,
                    stop_token_ids,
                )
                if bool(finished.all()):
                    break

        output_width = int((prompt_lengths + generated_steps).max().item())
        output = ids[:, :output_width]
        if not return_trace:
            return output
        return output, {
            "generation_mode": "single_stream_text",
            "attention_contract": attention_contract,
            "content_self_diagonal": content_self_diagonal,
            "backbone_kv_cache_enabled": bool(use_cache),
            "generated_tokens": int(generated_steps),
        }

    @torch.no_grad()
    def generate(
        self,
        task: str,
        /,
        **kwargs,
    ):
        """Unified cache-first generation entry for every evaluation task.

        ``t2i`` dispatches to image-flow generation. ``i2t`` and ``text``
        share text-token generation; the concrete model class owns the text
        attention contract, so ablation C is selected without evaluator-side
        branching.
        """

        task = str(task).strip().lower()
        if task == "t2i":
            return self.generate_image(**kwargs)
        if task in {"i2t", "text"}:
            return self.generate_text(**kwargs)
        raise ValueError(
            f"task must be one of t2i, i2t, or text; got {task!r}"
        )



__all__ = ["SelflessGenerationMixin", "SelflessStaticCache"]
