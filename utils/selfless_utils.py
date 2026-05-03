import torch


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