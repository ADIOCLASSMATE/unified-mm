import torch
import random


def assign_sigma_multimodal(token_types, task_mode, seq_len):
    """Per-sample sigma assignment for multimodal training.

    Follows RESEARCH.md §1.1-1.3 spec with three training modes.
    Returns sigma tensor of shape [seq_len] with values in ℝ.

    Token types: 0=text, 1=image, 2=special(BOI/EOI/EOS), 3=padding

    Modes:
      text_to_image: text gets earlier generation-order sigma, image gets later sigma
      image_to_text: image gets earlier generation-order sigma, text gets later sigma
      text_only:     text σ = AR generation order, image σ = -1 (nonexistent)
      interleaved:   per-segment sigma in generation order

    Special tokens: σ follows their document order unless a task mode overrides it
    Padding: σ = seq_len + 1
    """
    device = token_types.device
    sigma = torch.zeros(seq_len, dtype=torch.float32, device=device)

    text_mask = token_types == 0
    image_mask = token_types == 1
    special_mask = token_types == 2
    pad_mask = token_types == 3

    n_text = text_mask.sum().item()
    n_image = image_mask.sum().item()

    if task_mode == "text_to_image":
        # Text is condition: generated first, so it is visible to later image queries.
        if n_text > 0:
            sigma[text_mask] = torch.arange(
                n_text, dtype=torch.float32, device=device
            )
            sigma[text_mask] += torch.rand(n_text, device=device) * 0.1 - 0.05
        if n_image > 0:
            sigma[image_mask] = n_text + torch.rand(n_image, device=device)

    elif task_mode == "image_to_text":
        # Image is condition: generated first, so it is visible to later text queries.
        if n_image > 0:
            sigma[image_mask] = torch.rand(n_image, device=device)
        if n_text > 0:
            sigma[text_mask] = n_image + torch.arange(
                n_text, dtype=torch.float32, device=device
            )
            sigma[text_mask] += torch.rand(n_text, device=device) * 0.1 - 0.05

    elif task_mode == "text_only":
        # Standard AR for pure text
        if n_text > 0:
            sigma[text_mask] = torch.arange(
                n_text, dtype=torch.float32, device=device
            )
        if n_image > 0:
            sigma[image_mask] = -1.0  # shouldn't exist but handle gracefully

    elif task_mode == "interleaved":
        # Mixed mode: alternating condition/target regions
        # Strategy: assign generation-order sigma within each text block,
        # random local order for image spans.
        pos = 0
        text_block_start = None
        while pos < seq_len:
            if pad_mask[pos]:
                break
            if text_mask[pos]:
                if text_block_start is None:
                    text_block_start = pos
            else:
                # End of text block
                if text_block_start is not None:
                    block_len = pos - text_block_start
                    if block_len > 0:
                        sigma[text_block_start:pos] = torch.arange(block_len, dtype=torch.float32, device=device)
                    text_block_start = None
                if image_mask[pos]:
                    block_end = min(pos + 512, seq_len)
                    block_len = block_end - pos
                    sigma[pos:block_end] = pos + torch.rand(block_len, device=device)
            pos += 1
        # Handle trailing text block
        if text_block_start is not None:
            block_len = seq_len - text_block_start
            if block_len > 0:
                sigma[text_block_start:seq_len] = torch.arange(block_len, dtype=torch.float32, device=device)

    else:
        raise ValueError(f"Unknown task_mode: {task_mode}")

    # Special tokens keep their absolute positions by default.
    if special_mask.any():
        sigma[special_mask] = torch.arange(seq_len, dtype=torch.float32, device=device)[special_mask]

    # Padding gets a high sigma so real tokens cannot attend to it as K/V.
    sigma[pad_mask] = float(seq_len + 1)

    # Clamp to avoid extreme values
    sigma = sigma.clamp(0.0, 1e4)

    return sigma


def assign_sigma_multimodal_batch(token_types, task_modes, seq_len):
    """Vectorized batch sigma assignment.

    Args:
        token_types: [B, L] tensor of token type masks
        task_modes: list of B strings
        seq_len: int

    Returns:
        sigma: [B, L] tensor
    """
    B = token_types.shape[0]
    sigma = torch.zeros(B, seq_len, dtype=torch.float32, device=token_types.device)

    for b in range(B):
        sigma[b] = assign_sigma_multimodal(
            token_types=token_types[b],
            task_mode=task_modes[b],
            seq_len=seq_len,
        ).to(token_types.device)

    return sigma


class SelflessSampler():
    def __init__(self, mask_token_id, config) -> None:
        self.mask_token_id = mask_token_id # 126336 is used for [MASK] token
        self.attention_pattern = config.model.attention_pattern
        self.config = config

    @torch.no_grad()
    def sample_mask(self, text_ids, attention_mask=None, eps=1e-3, t_in=None, step_ratio=None, prompt_length=None):
        if self.attention_pattern == "random":
            b, l = text_ids.shape
            t = torch.rand(b, device=text_ids.device) if t_in is None else t_in

            t_sample = (1 - eps) * t + eps
            t_sample = t_sample[:, None].repeat(1, l) # (B, 1)

            v_sample = torch.rand(b, l, device=text_ids.device) # (B, L)
            # ensure first token is not masked
            v_sample[:, 0] = 2

            if prompt_length is not None:
                # ensure prompt_length is of shape (B,)
                if prompt_length.dim() > 1:
                    prompt_length = prompt_length.squeeze(-1)

                v_sample = self.prompt_process(text_ids, v_sample, prompt_length, pad_ids=None)

            masked_indices = v_sample < t_sample

            noisy_ids = torch.where(masked_indices, self.mask_token_id, text_ids) # (B, L)

            return noisy_ids, masked_indices, t_sample, v_sample

        else:
            raise ValueError(f"Wrong attention_pattern: {self.attention_pattern}")

    @torch.no_grad()
    def sample_v(self, text_ids, attention_mask=None, eps=1e-3, pad_ids=None, t_in=None, step_ratio=None, prompt_lengths=None):
        if self.attention_pattern == "random":
            b, l = text_ids.shape
            t = torch.rand(b, device=text_ids.device) if t_in is None else t_in

            t_sample = (1 - eps) * t + eps
            t_sample = t_sample[:, None].repeat(1, l) # (B, 1)

            v_sample = torch.rand(b, l, device=text_ids.device) # (B, L)
            # ensure first token is not masked
            v_sample[:, 0] = 2
        elif self.attention_pattern == "ar":
            # ar: autoregressive generation order.
            # Earlier tokens have smaller sigma and are visible to later tokens.
            b, l = text_ids.shape

            # t_sample set to zero
            t_sample = torch.zeros(b, l, device=text_ids.device)  # (B, L)

            # create position indices [0, 1, 2, ..., l-1]
            pos_idx = torch.arange(l, device=text_ids.device, dtype=torch.float32).unsqueeze(0)  # (1, L)
            if l > 1:
                v_sample = eps + (1 - 2 * eps) * pos_idx / (l - 1)  # (1, L)
            else:
                v_sample = torch.ones(1, 1, device=text_ids.device) * eps
            v_sample = v_sample.expand(b, l)  # (B, L)

        else:
            raise ValueError(f"Wrong attention_pattern: {self.attention_pattern}")

        if prompt_lengths is not None:
            # ensure prompt_length is of shape (B,)
            if prompt_lengths.dim() > 1:
                prompt_lengths = prompt_lengths.squeeze(-1)

            v_sample = self.prompt_process(text_ids, v_sample, prompt_lengths, pad_ids)


        return t_sample, v_sample

    @torch.no_grad()
    def prompt_process(self, text_ids, v_sample, prompt_lengths, pad_ids):
        prompt_attention_pattern = getattr(self.config.training, 'prompt_attention_pattern', None)
        if prompt_attention_pattern == "ar":
            # Prompt tokens use AR generation order and are lower than solution
            # tokens, so the solution can attend to the full prompt.
            # Example: prompt_length=10, solution_length=5, L=15
            # structure: [prompt(0-9) | solution(10-14) | padding(15+)]
            # prompt: [0, 1, ..., 9], solution: 10 + original local sigma.
            B, L = text_ids.shape

            # absolute position index for each token [0, 1, 2, ..., L-1]
            pos_idx = torch.arange(L, device=text_ids.device, dtype=torch.long).unsqueeze(0)  # [1, L]
            # prompt_mask: which positions belong to prompt (position < prompt_length)
            prompt_mask = pos_idx < prompt_lengths.unsqueeze(1)  # [B, L]
            prompt_values = pos_idx.to(v_sample.dtype)
            solution_values = prompt_lengths.unsqueeze(1).to(v_sample.dtype) + v_sample
            v_sample = torch.where(prompt_mask, prompt_values, solution_values)
            if pad_ids is not None:
                v_sample.masked_fill_(text_ids == pad_ids, L + 1)
        elif prompt_attention_pattern == "random":
            # Prompt tokens use random order but remain lower than solution tokens.
            B, L = text_ids.shape

            pos_idx = torch.arange(L, device=text_ids.device, dtype=torch.long).unsqueeze(0)  # [1, L]
            prompt_mask = pos_idx < prompt_lengths.unsqueeze(1)  # [B, L]
            solution_values = 2 + v_sample
            v_sample = torch.where(prompt_mask, v_sample, solution_values)
            if pad_ids is not None:
                v_sample.masked_fill_(text_ids == pad_ids, L + 1)

        else:
            raise ValueError(f"UNKNOWN PROMPT attention_pattern, get {prompt_attention_pattern}")

        return v_sample
