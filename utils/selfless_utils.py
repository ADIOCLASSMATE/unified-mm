import torch
import random


def assign_sigma_multimodal(token_types, task_mode, seq_len):
    """Per-sample sigma assignment for multimodal training.

    Follows RESEARCH.md §1.1-1.3 spec with three training modes.
    Returns sigma tensor of shape [seq_len] with values in ℝ.

    Token types: 0=text, 1=image, 2=special(BOS/BOI/EOI), 3=padding

    Modes:
      text_to_image: text σ ∈ [2, 2+N] (condition), image σ ∈ [0,1] (target)
      image_to_text: image σ ∈ [2, 3] (condition), text σ ∈ [0, N] (target)
      text_only:     text σ = AR descending, image σ = -1 (nonexistent)
      interleaved:   per-segment sigma (condition segments get high σ, target get low)

    Special tokens: σ = max(all_other_sigmas) + 1 (dynamic global max)
    Padding: σ = -1
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
        # Text is condition (high sigma), image is target (low sigma)
        if n_text > 0:
            sigma[text_mask] = 2.0 + n_text - torch.arange(
                n_text, dtype=torch.float32, device=device
            )  # σ ∈ [2, 2+N], AR descending within text
            sigma[text_mask] += torch.rand(n_text, device=device) * 0.1 - 0.05
        if n_image > 0:
            sigma[image_mask] = torch.rand(n_image, device=device)  # σ ∈ [0, 1]

    elif task_mode == "image_to_text":
        # Image is condition (high sigma), text is target (low sigma)
        if n_image > 0:
            sigma[image_mask] = 2.0 + torch.rand(n_image, device=device)  # σ ∈ [2, 3]
        if n_text > 0:
            sigma[text_mask] = n_text - torch.arange(
                n_text, dtype=torch.float32, device=device
            )  # σ ∈ [0, N], AR descending
            sigma[text_mask] += torch.rand(n_text, device=device) * 0.1 - 0.05

    elif task_mode == "text_only":
        # Standard AR for pure text
        if n_text > 0:
            sigma[text_mask] = n_text - torch.arange(
                n_text, dtype=torch.float32, device=device
            )
        if n_image > 0:
            sigma[image_mask] = -1.0  # shouldn't exist but handle gracefully

    elif task_mode == "interleaved":
        # Mixed mode: alternating condition/target regions
        # Strategy: assign AR sigma within each text block, random sigma for images
        # Text blocks get descending sigma, images get random in [0,1]
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
                        sigma[text_block_start:pos] = (
                            block_len - torch.arange(block_len, dtype=torch.float32, device=device)
                        )
                    text_block_start = None
                if image_mask[pos]:
                    block_end = min(pos + 512, seq_len)
                    block_len = block_end - pos
                    sigma[pos:block_end] = torch.rand(block_len, device=device)
            pos += 1
        # Handle trailing text block
        if text_block_start is not None:
            block_len = seq_len - text_block_start
            if block_len > 0:
                sigma[text_block_start:seq_len] = (
                    block_len - torch.arange(block_len, dtype=torch.float32, device=device)
                )

    else:
        raise ValueError(f"Unknown task_mode: {task_mode}")

    # Special tokens: set to global max + 1 (visible to all)
    other_sigmas = sigma[~(special_mask | pad_mask)]
    if other_sigmas.numel() > 0:
        max_sigma = other_sigmas.max().item()
    else:
        max_sigma = 1.0
    sigma[special_mask] = max_sigma + 1.0

    # Padding: set to -1 (excluded from attention)
    sigma[pad_mask] = -1.0

    # Clamp to avoid extreme values
    sigma = sigma.clamp(-1.0, 1e4)

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
            # ar: autoregressive — build strictly decreasing v_sample for AR modeling
            # earlier tokens have larger values (processed first), later tokens have smaller values (processed later)
            b, l = text_ids.shape

            # t_sample set to zero
            t_sample = torch.zeros(b, l, device=text_ids.device)  # (B, L)

            # create position indices [0, 1, 2, ..., l-1]
            pos_idx = torch.arange(l, device=text_ids.device, dtype=torch.float32).unsqueeze(0)  # (1, L)
            # build strictly decreasing v_sample: from near 1 to near 0
            # first token has largest v (near 1), last token has smallest v (near 0)
            # linear interpolation: from 1-eps decreasing to eps
            if l > 1:
                # linear decrease from 1-eps to eps
                v_sample = 1 - eps - (1 - 2 * eps) * pos_idx / (l - 1)  # (1, L)
            else:
                # when l=1, only one token, set to 1-eps
                v_sample = torch.ones(1, 1, device=text_ids.device) * (1 - eps)
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
            # prompt tokens have decreasing v values, min value = 2
            #
            # Example: prompt_length=10, solution_length=5, L=15
            # structure: [prompt(0-9) | solution(10-14) | padding(15+)]
            # for prompt: v_sample = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2] (decreasing, min=2)
            # solution keeps random values [0,1]
            # padding set to -1
            #
            B, L = text_ids.shape

            # absolute position index for each token [0, 1, 2, ..., L-1]
            pos_idx = torch.arange(L, device=text_ids.device, dtype=torch.long).unsqueeze(0)  # [1, L]
            # prompt_mask: which positions belong to prompt (position < prompt_length)
            prompt_mask = pos_idx < prompt_lengths.unsqueeze(1)  # [B, L]
            # offset within the prompt
            # for prompt_length=10, positions [0,1,2,...,9] have offset [0, 1, 2, ..., 9]
            prompt_offset = pos_idx  # [1, L]
            # decreasing index: from prompt_length-1 down to 0
            # for prompt_length=10, reverse_idx = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            prompt_reverse_idx = prompt_lengths.unsqueeze(1) - 1 - prompt_offset  # [B, L]
            # final v values: 2 + reverse_idx, min value = 2 (when offset=prompt_length-1)
            # for prompt_length=10, values = 2 + [9,8,7,6,5,4,3,2,1,0] = [11,10,9,8,7,6,5,4,3,2]
            prompt_values = 2 + prompt_reverse_idx  # [B, L]
            # only apply new values to prompt portion, solution keeps original (random [0,1])
            v_sample = torch.where(prompt_mask, prompt_values, v_sample)
            # set padding to -1
            if pad_ids is not None:
                v_sample.masked_fill_(text_ids == pad_ids, -1)
        elif prompt_attention_pattern == "random":
            # prompt tokens have random v values > 2, meaning prompt uses random order (non-autoregressive)
            B, L = text_ids.shape

            pos_idx = torch.arange(L, device=text_ids.device, dtype=torch.long).unsqueeze(0)  # [1, L]
            prompt_mask = pos_idx < prompt_lengths.unsqueeze(1)  # [B, L]
            # add 2 to v values in prompt region, shifting range from [0,1] to [2,3]
            v_sample = torch.where(prompt_mask, v_sample + 2, v_sample)
            # set padding to -1
            if pad_ids is not None:
                v_sample.masked_fill_(text_ids == pad_ids, -1)

        else:
            raise ValueError(f"UNKNOWN PROMPT attention_pattern, get {prompt_attention_pattern}")

        return v_sample