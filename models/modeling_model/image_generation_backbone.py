"""Cached image queries shared by static and dynamic flow generation.

The context keeps references to sequence tensors updated in place by decoding.
Pending positions are supplied on each call because confidence probes may
commit the previous content token before ordinary decoding resumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .modeling_selfless_cache import SelflessStaticCache


@dataclass(frozen=True)
class ImageBackboneQuery:
    model: Any
    span_starts: torch.Tensor
    current_sigma: torch.Tensor
    batch_indices: torch.Tensor
    mask_only_query: torch.Tensor
    content_then_mask_queries: torch.Tensor
    selected_batch: int
    key_sigma: torch.Tensor | None
    key_valid: torch.Tensor | None
    key_is_target_image: torch.Tensor | None
    content_self_diagonal: bool
    image_latent_dim: int
    selected_input_ids: torch.Tensor
    selected_token_types: torch.Tensor
    work_latents: torch.Tensor
    full_position_ids: torch.Tensor
    use_x0_content_condition: bool
    confidence_order: bool
    debug_finite: bool

    def __call__(
        self,
        current_local_positions: torch.Tensor,
        *,
        pending_local_positions: torch.Tensor | None,
        cache: SelflessStaticCache,
        image_uncond_rows: torch.Tensor | None,
        label: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        model = self.model
        span_starts = self.span_starts
        current_sigma = self.current_sigma
        batch_indices = self.batch_indices
        mask_only_query = self.mask_only_query
        content_then_mask_queries = self.content_then_mask_queries
        selected_batch = self.selected_batch
        key_sigma = self.key_sigma
        key_valid = self.key_valid
        key_is_target_image = self.key_is_target_image
        content_self_diagonal = self.content_self_diagonal
        image_latent_dim = self.image_latent_dim
        selected_input_ids = self.selected_input_ids
        selected_token_types = self.selected_token_types
        work_latents = self.work_latents
        full_position_ids = self.full_position_ids
        use_x0_content_condition = self.use_x0_content_condition
        confidence_order = self.confidence_order
        debug_finite = self.debug_finite

        def gather_position_ids(indices: torch.Tensor) -> torch.Tensor:
            return torch.gather(
                full_position_ids,
                dim=2,
                index=indices.unsqueeze(0).expand(2, -1, -1),
            )

        multiple_queries = current_local_positions.ndim == 2
        current_indices = (span_starts[:, None] + current_local_positions
                           if multiple_queries else (span_starts + current_local_positions).unsqueeze(1))
        query_count = current_indices.shape[1]
        current_query_sigma = torch.gather(current_sigma, 1, current_indices)
        if pending_local_positions is None:
            query_indices = current_indices
            query_sigma = current_query_sigma
            content_queries = mask_only_query.expand(-1, query_count)
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
            content_queries = (content_then_mask_queries if query_count == 1 else
                               torch.cat([torch.ones_like(mask_only_query), mask_only_query.expand(-1, query_count)], dim=1))
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

        attention_mask = model._build_generation_cache_mask(
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

        hidden = model.model(
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
        pending_x0_hidden = (
            hidden[:, 0]
            if (use_x0_content_condition or confidence_order)
            and pending_local_positions is not None
            else None
        )
        return (hidden[:, -query_count:] if multiple_queries else hidden[:, -1]), pending_x0_hidden
