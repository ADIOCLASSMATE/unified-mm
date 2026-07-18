"""Qwen3 backbone with the original Show-o unified-vocabulary objective.

The vocabulary layout is::

    [ text and multimodal special tokens ]
    [ image_offset, image_offset + image_vocab_size )
    [ image_mask_token_id ]

``image_mask_token_id`` is always the final vocabulary row.  Unlike the older
dual-head ablation, image codes use the same input embedding table and the same
LM head as text and special tokens.

The class intentionally does not expand a base Qwen vocabulary in ``__init__``.
This lets ``from_pretrained`` load the original Qwen embedding without a shape
mismatch.  Call :meth:`configure_image_vocabulary` after loading the base
checkpoint.  A saved Qwen-Show-o checkpoint already has the expanded
``config.vocab_size`` and loads normally.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Optional, Union

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM as HFQwen3ForCausalLM,
)


TensorOrInt = Union[int, torch.Tensor]


def official_showo_ranking_temperature(
    base_temperature: float,
    step: int,
    timesteps: int,
) -> float:
    """Return Show-o's cumulative Gumbel-ranking temperature for one step.

    The reference sampler mutates ``temperature *= (1 - ratio)`` at every
    iteration.  Recomputing ``base_temperature * (1 - ratio)`` instead is not
    equivalent: the official schedule is the cumulative product over all
    completed ratios.
    """

    if int(timesteps) <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    if int(step) < 0 or int(step) >= int(timesteps):
        raise ValueError(
            f"step must be in [0, {int(timesteps)}), got {step}"
        )
    temperature = float(base_temperature)
    for index in range(int(step) + 1):
        ratio = float(index + 1) / float(timesteps)
        temperature *= 1.0 - ratio
    return temperature


class QwenShowOForCausalLM(HFQwen3ForCausalLM):
    """A Qwen3 implementation of Show-o's unified token prediction model.

    Image labels are predicted at the same sequence position, as in Show-o's
    discrete denoising objective.  Optional text labels retain next-token
    (shifted) causal-LM semantics.  Both losses use the complete unified
    vocabulary softmax.
    """

    def __init__(self, config):
        # Do not resize here.  A base-Qwen checkpoint must first load with its
        # original embedding shape; configure_image_vocabulary() expands it.
        super().__init__(config)
        self.config.unified_architecture = "qwen_showo"
        if not hasattr(self.config, "image_vocab_size"):
            self.config.image_vocab_size = 8192
        if not hasattr(self.config, "image_loss_chunk_size"):
            self.config.image_loss_chunk_size = 1024
        if not hasattr(self.config, "lambda_image"):
            self.config.lambda_image = 1.0
        if not hasattr(self.config, "lambda_text"):
            self.config.lambda_text = 0.0

    @property
    def image_vocab_size(self) -> int:
        return int(getattr(self.config, "image_vocab_size", 8192))

    @property
    def image_offset(self) -> Optional[int]:
        value = getattr(self.config, "image_offset", None)
        return None if value is None else int(value)

    @property
    def image_mask_token_id(self) -> Optional[int]:
        value = getattr(self.config, "image_mask_token_id", None)
        return None if value is None else int(value)

    def configure_image_vocabulary(
        self,
        image_offset: Optional[int] = None,
        image_vocab_size: Optional[int] = None,
        *,
        image_mask_token_id: Optional[int] = None,
        image_loss_chunk_size: Optional[int] = None,
        resize_embeddings: bool = True,
        mean_resizing: bool = False,
    ) -> int:
        """Expand a loaded base model into the final Show-o vocabulary.

        Args:
            image_offset: First unified token id occupied by an image code.
                It should equal the tokenizer length *after* adding textual
                multimodal special tokens such as BOI and EOI.  If omitted,
                an existing ``config.image_offset`` is used, otherwise the
                model's current embedding size is used.
            image_vocab_size: Number of MAGVIT codes.  If omitted, use the
                configured value (8192 by default).
            image_mask_token_id: Optional explicit final mask id.  It must
                equal ``image_offset + image_vocab_size``.
            image_loss_chunk_size: Optional full-softmax CE chunk size.
            resize_embeddings: Resize the input embedding and LM head to the
                final vocabulary size.  This should be true after loading a
                base Qwen checkpoint and may be false for config-only setup.
            mean_resizing: Forwarded to Hugging Face's embedding resize.

        Returns:
            The final image mask token id.  The resulting vocabulary size is
            exactly ``image_mask_token_id + 1``.
        """

        current_vocab = int(self.get_input_embeddings().num_embeddings)
        configured_offset = self.image_offset
        if image_offset is None:
            image_offset = configured_offset if configured_offset is not None else current_vocab
        image_offset = int(image_offset)
        image_vocab_size = (
            self.image_vocab_size
            if image_vocab_size is None
            else int(image_vocab_size)
        )
        if image_offset < 0:
            raise ValueError(f"image_offset must be non-negative, got {image_offset}")
        if image_vocab_size <= 0:
            raise ValueError(
                f"image_vocab_size must be positive, got {image_vocab_size}"
            )

        expected_mask_token_id = image_offset + image_vocab_size
        if image_mask_token_id is None:
            image_mask_token_id = expected_mask_token_id
        image_mask_token_id = int(image_mask_token_id)
        if image_mask_token_id != expected_mask_token_id:
            raise ValueError(
                "the independent image mask token must immediately follow all "
                f"image codes: expected {expected_mask_token_id}, got "
                f"{image_mask_token_id}"
            )
        total_vocab_size = image_mask_token_id + 1
        # A smaller current vocabulary is the expected base-Qwen path.  A
        # current vocabulary equal to total_vocab_size is the checkpoint path.
        # Any other overlap would silently reinterpret a text row as an image.
        if current_vocab > image_offset and current_vocab != total_vocab_size:
            raise ValueError(
                "image_offset overlaps the current text vocabulary: "
                f"image_offset={image_offset}, current_vocab={current_vocab}. "
                "Pass an offset at or above the current tokenizer/model size."
            )

        self.config.image_offset = image_offset
        self.config.image_vocab_size = image_vocab_size
        self.config.image_mask_token_id = image_mask_token_id
        if image_loss_chunk_size is not None:
            image_loss_chunk_size = int(image_loss_chunk_size)
            if image_loss_chunk_size <= 0:
                raise ValueError(
                    "image_loss_chunk_size must be positive, got "
                    f"{image_loss_chunk_size}"
                )
            self.config.image_loss_chunk_size = image_loss_chunk_size
        if bool(resize_embeddings) and current_vocab != total_vocab_size:
            self.resize_token_embeddings(
                total_vocab_size,
                mean_resizing=bool(mean_resizing),
            )
        # resize_token_embeddings updates config.vocab_size and preserves the
        # configured tying policy; apply it again for checkpoint callers.
        if bool(resize_embeddings):
            self.config.vocab_size = total_vocab_size
            self.vocab_size = total_vocab_size
            self.tie_weights()
        return image_mask_token_id

    def _image_layout(self) -> tuple[int, int, int]:
        image_offset = self.image_offset
        image_mask_token_id = self.image_mask_token_id
        if image_offset is None or image_mask_token_id is None:
            raise RuntimeError(
                "The image vocabulary is not configured. Load the base Qwen "
                "checkpoint first, then call configure_image_vocabulary()."
            )
        expected_mask_id = image_offset + self.image_vocab_size
        if image_mask_token_id != expected_mask_id:
            raise RuntimeError(
                "Invalid unified vocabulary layout: expected "
                f"image_mask_token_id={expected_mask_id}, got "
                f"{image_mask_token_id}."
            )
        actual_vocab = int(self.get_input_embeddings().num_embeddings)
        if actual_vocab != image_mask_token_id + 1:
            raise RuntimeError(
                "The independent image mask token must be the final vocabulary "
                f"row, but vocab_size={actual_vocab} and "
                f"image_mask_token_id={image_mask_token_id}. Call "
                "configure_image_vocabulary() after loading the base model."
            )
        if int(self.get_output_embeddings().out_features) != actual_vocab:
            raise RuntimeError("input embedding and unified LM head sizes differ")
        return image_offset, image_offset + self.image_vocab_size, image_mask_token_id

    def _normalize_image_ids(
        self,
        input_ids: torch.LongTensor,
        image_positions: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        """Accept raw cache codes for convenience, but embed unified ids."""

        if image_positions is None or not bool(image_positions.any()):
            return input_ids
        image_offset, image_end, image_mask_token_id = self._image_layout()
        positions = image_positions.to(device=input_ids.device, dtype=torch.bool)
        if positions.shape != input_ids.shape:
            raise ValueError(
                "image positions and input_ids must have the same shape, got "
                f"{tuple(positions.shape)} and {tuple(input_ids.shape)}"
            )
        values = input_ids[positions]
        is_mask = values == image_mask_token_id
        is_raw = (values >= 0) & (values < self.image_vocab_size)
        is_unified = (values >= image_offset) & (values < image_end)
        valid = is_mask | is_raw | is_unified
        if not bool(valid.all()):
            bad = values[~valid]
            raise ValueError(
                "image positions must contain raw MAGVIT codes, offset image "
                f"ids, or image_mask_token_id; invalid range "
                f"[{int(bad.min())}, {int(bad.max())}]"
            )
        if not bool(is_raw.any()):
            return input_ids
        normalized = input_ids.clone()
        normalized_values = values.clone()
        normalized_values[is_raw] += image_offset
        normalized[positions] = normalized_values
        return normalized

    def _normalize_image_targets(self, targets: torch.LongTensor) -> torch.LongTensor:
        image_offset, image_end, _ = self._image_layout()
        targets = targets.long()
        is_raw = (targets >= 0) & (targets < self.image_vocab_size)
        is_unified = (targets >= image_offset) & (targets < image_end)
        if not bool((is_raw | is_unified).all()):
            bad = targets[~(is_raw | is_unified)]
            raise ValueError(
                "supervised image labels must be raw MAGVIT codes or unified "
                f"image ids; invalid range [{int(bad.min())}, {int(bad.max())}]"
            )
        return torch.where(is_raw, targets + image_offset, targets)

    def image_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return logits over raw image codes ``[0, image_vocab_size)``.

        This is a slice of the unified LM head, not a separate image head.
        It is intended for MaskGIT sampling; training uses the complete unified
        vocabulary through :meth:`_chunked_unified_ce`.
        """

        image_offset, image_end, _ = self._image_layout()
        weight = self.get_output_embeddings().weight[image_offset:image_end]
        return F.linear(hidden_states.to(dtype=weight.dtype), weight)

    def unified_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the complete shared text/special/image vocabulary head."""

        return self.lm_head(hidden_states.to(dtype=self.lm_head.weight.dtype))

    def _chunked_unified_ce(
        self,
        hidden_states: torch.Tensor,
        targets: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-vocabulary CE sum and correct count with bounded logits.

        During training each CE chunk is activation-checkpointed so backward
        recomputes at most ``image_loss_chunk_size × vocab_size`` logits rather
        than retaining logits for every masked image position.
        """

        count = int(targets.numel())
        if count == 0:
            zero_loss = (
                hidden_states.sum() * 0.0
                + self.get_output_embeddings().weight.reshape(-1)[:1].sum() * 0.0
            )
            return zero_loss, torch.zeros(
                (), device=hidden_states.device, dtype=torch.long
            )

        chunk_size = int(getattr(self.config, "image_loss_chunk_size", 1024))
        if chunk_size <= 0:
            raise ValueError(
                f"image_loss_chunk_size must be positive, got {chunk_size}"
            )
        loss_sum = hidden_states.new_zeros(())
        correct = torch.zeros((), device=hidden_states.device, dtype=torch.long)

        for start in range(0, count, chunk_size):
            hidden_chunk = hidden_states[start : start + chunk_size]
            target_chunk = targets[start : start + chunk_size]

            def calculate_ce_and_correct(
                hidden: torch.Tensor,
                target: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                logits = self.unified_logits(hidden)
                chunk_correct = (logits.detach().argmax(dim=-1) == target).sum()
                chunk_ce = F.cross_entropy(
                    logits,
                    target,
                    reduction="sum",
                )
                return chunk_ce, chunk_correct

            should_checkpoint = (
                self.training
                and torch.is_grad_enabled()
                and hidden_chunk.requires_grad
            )
            if should_checkpoint:
                chunk_loss, chunk_correct = checkpoint(
                    calculate_ce_and_correct,
                    hidden_chunk,
                    target_chunk,
                    use_reentrant=False,
                )
            else:
                logits = self.unified_logits(hidden_chunk)
                chunk_loss = F.cross_entropy(
                    logits,
                    target_chunk,
                    reduction="sum",
                )
                chunk_correct = (
                    logits.detach().argmax(dim=-1) == target_chunk
                ).sum()
            loss_sum = loss_sum + chunk_loss
            correct = correct + chunk_correct

        return loss_sum, correct

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: TensorOrInt = 0,
        token_types: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        X0_input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Run Qwen with an optional arbitrary 4D additive omni mask.

        ``return_logits`` defaults to ``False`` to avoid materializing a
        ``[batch, sequence, ~160K]`` tensor.  ``last_hidden_state`` is always
        included in the returned ModelOutput.
        """

        if X0_input_ids is not None:
            if input_ids is not None:
                raise ValueError("pass only one of input_ids and X0_input_ids")
            input_ids = X0_input_ids
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")

        image_positions = None
        if token_types is not None:
            if input_ids is not None and token_types.shape != input_ids.shape:
                raise ValueError("token_types and input_ids must have the same shape")
            image_positions = token_types.to(
                device=input_ids.device if input_ids is not None else inputs_embeds.device
            ) == 1
        if input_ids is not None and image_positions is not None:
            input_ids = self._normalize_image_ids(input_ids, image_positions)

        model_kwargs = dict(kwargs)
        if output_hidden_states is not None:
            model_kwargs["output_hidden_states"] = output_hidden_states
        if output_attentions is not None:
            model_kwargs["output_attentions"] = output_attentions
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **model_kwargs,
        )
        hidden_states = outputs.last_hidden_state

        logits = None
        if return_logits:
            if isinstance(logits_to_keep, int):
                indices = (
                    slice(-logits_to_keep, None)
                    if logits_to_keep > 0
                    else slice(None)
                )
            else:
                indices = logits_to_keep
            logits = self.unified_logits(hidden_states[:, indices, :])

        loss = None
        image_loss = hidden_states.sum() * 0.0
        text_loss = hidden_states.sum() * 0.0
        image_correct = torch.zeros(
            (), device=hidden_states.device, dtype=torch.long
        )
        image_count = torch.zeros(
            (), device=hidden_states.device, dtype=torch.long
        )
        text_count = torch.zeros(
            (), device=hidden_states.device, dtype=torch.long
        )

        if labels is not None:
            labels = labels.to(device=hidden_states.device, dtype=torch.long)
            if labels.shape[:2] != hidden_states.shape[:2]:
                raise ValueError(
                    "labels and hidden states must share batch/sequence shape"
                )

            if image_positions is None:
                image_positions = torch.zeros_like(labels, dtype=torch.bool)
            else:
                image_positions = image_positions.to(hidden_states.device)

            image_valid = image_positions & (labels != -100)
            image_hidden = hidden_states[image_valid]
            image_targets = self._normalize_image_targets(labels[image_valid])
            image_loss_sum, image_correct = self._chunked_unified_ce(
                image_hidden,
                image_targets,
            )
            image_count = image_valid.sum()
            image_loss = image_loss_sum / image_count.clamp_min(1)

            # Text/special targets are causal next-token labels.  This branch
            # is normally empty for the requested T2I-only training.
            shifted_labels = labels[:, 1:]
            shifted_image_positions = image_positions[:, 1:]
            text_valid = (shifted_labels != -100) & ~shifted_image_positions
            text_hidden = hidden_states[:, :-1][text_valid]
            text_targets = shifted_labels[text_valid]
            text_loss_sum, _ = self._chunked_unified_ce(
                text_hidden,
                text_targets,
            )
            text_count = text_valid.sum()
            text_loss = text_loss_sum / text_count.clamp_min(1)

            loss = (
                float(getattr(self.config, "lambda_image", 1.0)) * image_loss
                + float(getattr(self.config, "lambda_text", 0.0)) * text_loss
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output["last_hidden_state"] = hidden_states
        if labels is not None:
            output["per_modality_loss"] = {
                "image_loss": image_loss.detach(),
                "text_loss": text_loss.detach(),
            }
            output["image_loss"] = image_loss.detach()
            output["text_loss"] = text_loss.detach()
            output["image_token_correct"] = image_correct.detach()
            output["image_token_count"] = image_count.detach()
            output["text_token_count"] = text_count.detach()
        return output

    @staticmethod
    def _values_at_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=1)
        if counts.numel() == 0 or not bool((counts == counts[0]).all()):
            raise ValueError(
                "every sample must contain the same number of image positions"
            )
        count = int(counts[0].item())
        trailing_shape = tensor.shape[2:]
        return tensor[mask].reshape(tensor.shape[0], count, *trailing_shape)

    @staticmethod
    def _scatter_at_mask(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        expected = int(mask.sum().item())
        if values.numel() != expected:
            raise ValueError(
                f"cannot scatter {values.numel()} values into {expected} positions"
            )
        tensor[mask] = values.reshape(-1)

    @staticmethod
    def _per_sample_generators(
        input_ids: torch.Tensor,
        generator: Optional[torch.Generator],
        generators: Optional[Sequence[torch.Generator]],
        sample_seeds: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Generator], Optional[list[torch.Generator]]]:
        batch_size = input_ids.shape[0]
        if generators is not None:
            if len(generators) != batch_size:
                raise ValueError(
                    f"expected {batch_size} generators, got {len(generators)}"
                )
            return generator, list(generators)
        if sample_seeds is None:
            return generator, None
        seeds = torch.as_tensor(sample_seeds).reshape(-1)
        if seeds.numel() != batch_size:
            raise ValueError(
                f"expected {batch_size} sample_seeds, got {seeds.numel()}"
            )
        per_sample = []
        generator_device = input_ids.device
        for seed in seeds.tolist():
            sample_generator = torch.Generator(device=generator_device)
            sample_generator.manual_seed(int(seed))
            per_sample.append(sample_generator)
        return generator, per_sample

    @staticmethod
    def _multinomial_per_sample(
        probabilities: torch.Tensor,
        generator: Optional[torch.Generator],
        generators: Optional[Sequence[torch.Generator]],
    ) -> torch.LongTensor:
        batch_size, num_tokens, vocab_size = probabilities.shape
        if generators is None:
            return torch.multinomial(
                probabilities.reshape(-1, vocab_size),
                1,
                generator=generator,
            ).reshape(batch_size, num_tokens)
        sampled = []
        for row, row_generator in zip(probabilities, generators):
            sampled.append(
                torch.multinomial(
                    row,
                    1,
                    generator=row_generator,
                ).squeeze(-1)
            )
        return torch.stack(sampled, dim=0)

    @staticmethod
    def _uniform_per_sample(
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator],
        generators: Optional[Sequence[torch.Generator]],
    ) -> torch.Tensor:
        if generators is None:
            return torch.rand(
                shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        return torch.stack(
            [
                torch.rand(
                    shape[1],
                    device=device,
                    dtype=dtype,
                    generator=row_generator,
                )
                for row_generator in generators
            ],
            dim=0,
        )

    @staticmethod
    def _mask_ratio(
        ratio: torch.Tensor,
        mask_schedule: Union[str, Callable[[torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        if callable(mask_schedule):
            result = mask_schedule(ratio)
            return torch.as_tensor(result, device=ratio.device, dtype=ratio.dtype)
        schedule = str(mask_schedule).lower()
        if schedule == "cosine":
            return torch.cos(ratio * math.pi * 0.5)
        if schedule == "linear":
            return 1.0 - ratio
        raise ValueError(f"unsupported mask_schedule={mask_schedule!r}")

    @torch.no_grad()
    def generate_image_tokens_maskgit(
        self,
        input_ids: torch.LongTensor,
        token_types: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        uncond_input_ids: Optional[torch.LongTensor] = None,
        uncond_token_types: Optional[torch.Tensor] = None,
        uncond_attention_mask: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.Tensor] = None,
        timesteps: int = 18,
        guidance_scale: float = 0.0,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None,
        mask_schedule: Union[
            str, Callable[[torch.Tensor], torch.Tensor]
        ] = "cosine",
        sample_seeds: Optional[torch.Tensor] = None,
        generators: Optional[Sequence[torch.Generator]] = None,
    ) -> torch.LongTensor:
        """Generate raw MAGVIT codes with Show-o/MaskGIT iterative decoding.

        Classifier-free guidance follows official Show-o:
        ``(1 + guidance_scale) * cond - guidance_scale * uncond``.  Proposal
        tokens are sampled from the guided distribution.  At every non-final
        step, low-confidence proposals are re-masked according to the cosine
        schedule, with temperature-scaled Gumbel noise for stochastic ranking.

        Returns:
            A ``[batch, num_image_tokens]`` tensor in raw codebook space
            ``[0, image_vocab_size)``.
        """

        image_offset, _, image_mask_token_id = self._image_layout()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if int(timesteps) <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}")
        if float(temperature) < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        if float(guidance_scale) > 0 and uncond_input_ids is None:
            raise ValueError(
                "guidance_scale > 0 requires unconditional input ids"
            )

        conditional = input_ids.clone()
        if image_token_mask is None:
            if token_types is None:
                raise ValueError(
                    "pass image_token_mask or token_types to locate image tokens"
                )
            image_token_mask = token_types == 1
        image_token_mask = image_token_mask.to(
            device=conditional.device,
            dtype=torch.bool,
        )
        if image_token_mask.shape != conditional.shape:
            raise ValueError("image_token_mask and input_ids must have the same shape")
        conditional = self._normalize_image_ids(
            conditional,
            image_token_mask,
        )

        conditional_image_ids = self._values_at_mask(
            conditional,
            image_token_mask,
        )
        known = conditional_image_ids != image_mask_token_id
        raw_codes = torch.where(
            known,
            conditional_image_ids - image_offset,
            torch.full_like(conditional_image_ids, -1),
        )
        if bool(known.any()):
            known_codes = raw_codes[known]
            if int(known_codes.min()) < 0 or int(known_codes.max()) >= self.image_vocab_size:
                raise ValueError("known image ids are outside the configured codebook")
        initial_unknown_count = (~known).sum(dim=1)

        unconditional = None
        uncond_image_mask = None
        if uncond_input_ids is not None:
            unconditional = uncond_input_ids.to(conditional.device).clone()
            if unconditional.ndim != 2 or unconditional.shape[0] != conditional.shape[0]:
                raise ValueError(
                    "conditional and unconditional inputs must share batch size"
                )
            if uncond_token_types is not None:
                uncond_image_mask = uncond_token_types.to(
                    device=conditional.device
                ) == 1
            elif unconditional.shape == conditional.shape:
                uncond_image_mask = image_token_mask
            else:
                raise ValueError(
                    "uncond_token_types are required when sequence lengths differ"
                )
            if uncond_image_mask.shape != unconditional.shape:
                raise ValueError(
                    "unconditional image mask and input ids must have the same shape"
                )
            uncond_counts = uncond_image_mask.sum(dim=1)
            cond_counts = image_token_mask.sum(dim=1)
            if not bool((uncond_counts == cond_counts).all()):
                raise ValueError(
                    "conditional and unconditional inputs must have the same "
                    "number of image tokens per sample"
                )
            unconditional = self._normalize_image_ids(
                unconditional,
                uncond_image_mask,
            )

        generator, per_sample_generators = self._per_sample_generators(
            conditional,
            generator,
            generators,
            sample_seeds,
        )

        for step in range(int(timesteps)):
            current_unified = torch.where(
                raw_codes >= 0,
                raw_codes + image_offset,
                torch.full_like(raw_codes, image_mask_token_id),
            )
            self._scatter_at_mask(
                conditional,
                image_token_mask,
                current_unified,
            )
            cond_output = self(
                input_ids=conditional,
                token_types=token_types,
                attention_mask=attention_mask,
                use_cache=False,
                return_logits=False,
            )
            cond_hidden = self._values_at_mask(
                cond_output.last_hidden_state,
                image_token_mask,
            )
            cond_logits = self.image_logits(cond_hidden).float()

            guided_logits = cond_logits
            if float(guidance_scale) > 0:
                self._scatter_at_mask(
                    unconditional,
                    uncond_image_mask,
                    current_unified,
                )
                uncond_output = self(
                    input_ids=unconditional,
                    token_types=uncond_token_types,
                    attention_mask=uncond_attention_mask,
                    use_cache=False,
                    return_logits=False,
                )
                uncond_hidden = self._values_at_mask(
                    uncond_output.last_hidden_state,
                    uncond_image_mask,
                )
                uncond_logits = self.image_logits(uncond_hidden).float()
                scale = float(guidance_scale)
                guided_logits = (1.0 + scale) * cond_logits - scale * uncond_logits

            probabilities = guided_logits.softmax(dim=-1)
            proposed_codes = self._multinomial_per_sample(
                probabilities,
                generator,
                per_sample_generators,
            )
            unknown = raw_codes < 0
            proposed_codes = torch.where(unknown, proposed_codes, raw_codes)
            selected_probabilities = probabilities.gather(
                dim=-1,
                index=proposed_codes.unsqueeze(-1),
            ).squeeze(-1)
            selected_probabilities = torch.where(
                unknown,
                selected_probabilities,
                torch.full_like(selected_probabilities, torch.inf),
            )

            # Final iteration commits every remaining proposal.
            if step + 1 == int(timesteps):
                raw_codes = proposed_codes
                break

            progress = torch.tensor(
                float(step + 1) / float(timesteps),
                device=conditional.device,
                dtype=torch.float32,
            )
            mask_ratio = self._mask_ratio(progress, mask_schedule).clamp(0.0, 1.0)
            desired_mask_count = torch.floor(
                initial_unknown_count.float() * mask_ratio
            ).long()
            current_unknown_count = unknown.sum(dim=1)
            max_mask_count = (current_unknown_count - 1).clamp_min(0)
            desired_mask_count = torch.minimum(
                desired_mask_count,
                max_mask_count,
            )
            desired_mask_count = torch.where(
                max_mask_count > 0,
                desired_mask_count.clamp_min(1),
                torch.zeros_like(desired_mask_count),
            )

            gumbel_uniform = self._uniform_per_sample(
                tuple(selected_probabilities.shape),
                device=selected_probabilities.device,
                dtype=selected_probabilities.dtype,
                generator=generator,
                generators=per_sample_generators,
            ).clamp_(1.0e-20, 1.0 - 1.0e-7)
            gumbel_noise = -torch.log(-torch.log(gumbel_uniform))
            ranking_temperature = official_showo_ranking_temperature(
                float(temperature),
                step,
                int(timesteps),
            )
            confidence = (
                selected_probabilities.clamp_min(1.0e-20).log()
                + ranking_temperature * gumbel_noise
            )
            confidence = torch.where(
                unknown,
                confidence,
                torch.full_like(confidence, torch.inf),
            )
            ascending = confidence.argsort(dim=-1)
            ranks = torch.empty_like(ascending)
            rank_values = torch.arange(
                ascending.shape[1],
                device=ascending.device,
            ).expand_as(ascending)
            ranks.scatter_(dim=-1, index=ascending, src=rank_values)
            remask = ranks < desired_mask_count.unsqueeze(-1)
            raw_codes = torch.where(
                remask,
                torch.full_like(proposed_codes, -1),
                proposed_codes,
            )

        if bool((raw_codes < 0).any()):
            raise RuntimeError("MaskGIT sampling ended with unresolved image masks")
        return raw_codes.long()

    @torch.no_grad()
    def sample_image_tokens_maskgit(self, *args, **kwargs) -> torch.LongTensor:
        """Alias retained for sampling scripts."""

        return self.generate_image_tokens_maskgit(*args, **kwargs)


# Compatibility names for the repository's existing model loader conventions.
ShowOVQForCausalLM = QwenShowOForCausalLM
Qwen3ForCausalLM = QwenShowOForCausalLM


__all__ = [
    "QwenShowOForCausalLM",
    "ShowOVQForCausalLM",
    "Qwen3ForCausalLM",
]
