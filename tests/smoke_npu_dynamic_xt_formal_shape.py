"""One-NPU formal-shape forward/backward smoke for Dynamic-XT."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from omegaconf import OmegaConf

from pretrain.train_selfless_flow_dynamic_xt import (
    load_dynamic_xt_model_tokenizer,
)
from utils.utils import get_selfless_mask


def main() -> None:
    if not torch.npu.is_available():
        raise SystemExit("Ascend NPU is required")
    device = torch.device("npu:0")
    batch_size = int(os.environ.get("SMOKE_BATCH_SIZE", "1"))
    seq_len = 320
    image_start = 2
    image_tokens = 256
    image_end = image_start + image_tokens
    valid_end = image_end + 2

    config = OmegaConf.load(
        "configs/selfless/imagenet1k_class_dynamic_xt_800ep.yaml"
    )
    model, _ = load_dynamic_xt_model_tokenizer(
        config,
        model_dtype=torch.bfloat16,
    )
    model = model.to(device=device, dtype=torch.bfloat16).train()
    for module in model.model.backbone_flow_time_embedder.mlp:
        if isinstance(module, torch.nn.Linear):
            if not bool(torch.isfinite(module.weight).all().item()):
                raise AssertionError("Dynamic-XT time weight is non-finite")
            if not bool(torch.equal(module.bias, torch.zeros_like(module.bias))):
                raise AssertionError("Dynamic-XT time bias is not zero-initialized")

    input_ids = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
    input_ids[:, 0] = 3
    input_ids[:, 1] = int(model.config.boi_token_id)
    input_ids[:, image_start:image_end] = int(model.config.image_mask_token_id)
    input_ids[:, image_end] = int(model.config.eoi_token_id)
    input_ids[:, image_end + 1] = int(model.config.eos_token_id)
    token_types = torch.full(
        (batch_size, seq_len), 3, device=device, dtype=torch.uint8
    )
    token_types[:, 0] = 0
    token_types[:, 1] = 2
    token_types[:, image_start:image_end] = 1
    token_types[:, image_end:valid_end] = 2
    segment_ids = torch.full(
        (batch_size, seq_len), -1, device=device, dtype=torch.long
    )
    segment_ids[:, :valid_end] = 0
    sigma = torch.full(
        (batch_size, seq_len), float("inf"), device=device
    )
    sigma[:, :valid_end] = torch.arange(valid_end, device=device).float()
    image_latents = torch.zeros(
        batch_size,
        seq_len,
        16,
        device=device,
        dtype=torch.bfloat16,
    )
    image_latents[:, image_start:image_end] = torch.randn(
        batch_size,
        image_tokens,
        16,
        device=device,
        dtype=torch.bfloat16,
    )
    local_positions = torch.full(
        (batch_size, seq_len), -1, device=device, dtype=torch.long
    )
    local_positions[:, image_start:image_end] = torch.arange(
        image_tokens, device=device
    )
    span_table = torch.stack(
        [
            torch.arange(batch_size, device=device),
            torch.arange(batch_size, device=device),
            torch.full((batch_size,), image_start, device=device),
            torch.full((batch_size,), image_end, device=device),
            torch.zeros(batch_size, device=device),
        ],
        dim=1,
    )
    attention_mask = get_selfless_mask(
        sigma,
        seq_len,
        device,
        segment_ids=segment_ids,
    )
    output = model(
        X0_input_ids=input_ids,
        labels=input_ids,
        attention_mask=attention_mask,
        token_types=token_types,
        image_latents=image_latents,
        image_local_positions=local_positions,
        image_span_table=span_table,
        flow_sigma=sigma,
        record_flow_stats=True,
    )
    if not bool(torch.isfinite(output.loss).item()):
        raise AssertionError("Dynamic-XT formal-shape loss is non-finite")
    output.loss.backward()
    torch.npu.synchronize()
    max_memory = int(torch.npu.max_memory_allocated(device))
    print(
        {
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "flow_batch_mul": int(model.image_flow_batch_mul),
            "loss": float(output.loss.detach().cpu()),
            "dynamic_xt_parameters": model.dynamic_xt_parameter_count(),
            "max_memory_allocated_bytes": max_memory,
        }
    )
    print("DYNAMIC-XT FORMAL-SHAPE NPU SMOKE PASS")


if __name__ == "__main__":
    main()
