"""Independent modality batches and frozen backbone feature collection."""
import math

import torch

from .representation_protocol import PREFIX, SUFFIX


def make_batch(rows, modality, tokenizer, model, cache, device, seed):
    from utils.evaluation.multimodal_likelihood import (
        build_attention_masks,
        build_image_sigma,
        image_order_mc_seed,
    )

    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    sequences = []
    for row in rows:
        if modality == "image":
            start = len(prefix) + 1
            body = [model.config.image_mask_token_id] * 256
            ids = (
                prefix
                + [model.config.boi_token_id]
                + body
                + [model.config.eoi_token_id]
            )
            types = [0] * len(prefix) + [2] + [1] * 256 + [2]
            sigma = list(range(len(ids)))
            sigma[-1] = len(prefix) + 1
            order = build_image_sigma(
                256, order="random", seed=image_order_mc_seed(seed, row["image_id"], 0)
            )
            sigma[start : start + 256] = [len(prefix) + 2 + value for value in order]
            data_positions = list(range(start, start + 256))
        else:
            body = tokenizer.encode(row["text"], add_special_tokens=False)
            if not body:
                raise ValueError("empty text")
            start = len(prefix)
            ids = prefix + body
            types = [0] * len(ids)
            sigma = list(range(len(ids)))
            data_positions = list(range(start, len(ids)))
        tail = suffix + [tokenizer.eos_token_id]
        sigma.extend(range(len(ids), len(ids) + len(tail)))
        ids.extend(tail)
        types.extend([0] * len(tail))
        sequences.append((ids, types, sigma, data_positions, start))
    length = math.ceil(max(len(row[0]) for row in sequences) / 64) * 64
    batch = len(rows)
    ids = torch.full((batch, length), tokenizer.eos_token_id, dtype=torch.long)
    types = torch.full_like(ids, 3)
    sigma = torch.zeros_like(ids)
    segments = torch.full_like(ids, -1)
    data_mask = torch.zeros((batch, length), dtype=torch.bool)
    readout = torch.zeros(batch, dtype=torch.long)
    latents = torch.zeros((batch, length, 16), dtype=torch.bfloat16)
    spans = []
    for index, (item, row) in enumerate(zip(sequences, rows)):
        row_ids, row_types, row_sigma, positions, start = item
        size = len(row_ids)
        ids[index, :size] = torch.tensor(row_ids)
        types[index, :size] = torch.tensor(row_types)
        sigma[index, :size] = torch.tensor(row_sigma)
        segments[index, :size] = 0
        data_mask[index, positions] = True
        readout[index] = size - 1
        if modality == "image":
            latents[index, start : start + 256] = cache.sample(row["image_id"])
            spans.append([index, 0, start, start + 256])
    ids, types, sigma, segments = [
        tensor.to(device) for tensor in (ids, types, sigma, segments)
    ]
    query_mask, content_mask = build_attention_masks(
        sigma=sigma,
        segment_ids=segments,
        token_types=types,
        input_ids=ids,
        boi_token_id=model.config.boi_token_id,
        attention_contract=model.config.dual_stream_attention_contract,
        device=device,
    )
    kwargs = {
        "X0_input_ids": ids,
        "token_types": types,
        "flow_sigma": sigma,
        "attention_mask": query_mask,
        "content_attention_mask": content_mask,
        "calculate_likelihood": True,
        "use_cache": False,
        "return_x0_hidden_state": True,
        "image_span_table": torch.tensor(
            spans, device=device, dtype=torch.long
        ).reshape(-1, 4),
    }
    if modality == "image":
        kwargs.update(image_latents=latents.to(device), image_latent_mask=types.eq(1))
    return kwargs, data_mask.to(device), readout.to(device)


class FeatureCollector:
    def __init__(self, backbone):
        self.backbone = backbone
        self.handles = []
        self.handles.append(backbone.layers[0].register_forward_pre_hook(self.before))
        for index, layer in enumerate(backbone.layers):
            self.handles.append(layer.register_forward_hook(self.after(index + 1)))

    def begin(self, mask, readout):
        self.mask = mask
        self.readout = readout
        self.values = []

    def collect(self, x0, xt):
        count = self.mask.sum(dim=1, keepdim=True).float()
        content = (x0.float() * self.mask.unsqueeze(-1)).sum(dim=1) / count
        query = xt[torch.arange(xt.shape[0], device=xt.device), self.readout].float()
        self.values.append(torch.stack([content, query], dim=1))

    def before(self, module, args):
        self.collect(args[0], args[1])

    def after(self, index):
        def hook(module, args, output):
            self.collect(output[0], output[1])
            if index == len(self.backbone.layers):
                self.collect(
                    self.backbone.norm(output[0]), self.backbone.norm(output[1])
                )

        return hook

    def finish(self):
        values = torch.stack(self.values, dim=1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("nonfinite extracted features")
        return values.to(device="cpu", dtype=torch.bfloat16)
