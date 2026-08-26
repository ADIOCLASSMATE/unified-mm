"""Single-NPU forward/backward smoke for joint I2T + T2I training.

Run only inside the fixed Ascend development Notebook:

    .venv/bin/python tests/smoke_npu_joint_caption.py
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
from transformers import Qwen3Config

from models.modeling_model.image_position_utils import (
    build_row_col_position_ids,
)
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.utils import get_selfless_mask


def tiny_config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 32
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "10"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "euler"
    config.image_input_noise_strength = 0.0
    config.image_uncond_prob = 0.0
    config.backbone_attention_output_gate = "none"
    config.lambda_text = 0.05
    config.lambda_image = 1.0
    config.use_flex_attention = True
    config.use_cache = False
    return config


def joint_batch(device: torch.device) -> dict[str, torch.Tensor]:
    # Row 0 is T2I and has only image-flow targets. Row 1 is I2T and has
    # only caption/EOS targets. Both image spans remain visible backbone input.
    input_ids = torch.tensor(
        [
            [20, 21, 11, 8, 8, 8, 8, 12, 2, 0, 0],
            [22, 23, 11, 8, 8, 8, 8, 12, 24, 25, 2],
        ],
        device=device,
    )
    token_types = torch.tensor(
        [
            [0, 0, 2, 1, 1, 1, 1, 2, 2, 3, 3],
            [0, 0, 2, 1, 1, 1, 1, 2, 0, 0, 2],
        ],
        device=device,
        dtype=torch.uint8,
    )
    sigma = torch.tensor(
        [
            [0, 1, 2, 4, 5, 6, 7, 3, 8, 11, 11],
            [0, 1, 2, 4, 5, 6, 7, 3, 8, 9, 10],
        ],
        device=device,
        dtype=torch.float32,
    )
    labels = torch.full_like(input_ids, -100)
    labels[1, 8:] = input_ids[1, 8:]
    image_loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    image_loss_mask[0, 3:7] = True
    image_latents = torch.zeros(
        2,
        11,
        4,
        device=device,
        dtype=torch.bfloat16,
    )
    image_latents[:, 3:7] = torch.randn(
        2,
        4,
        4,
        device=device,
        dtype=torch.bfloat16,
    )
    image_local_positions = torch.full(
        (2, 11), -1, device=device, dtype=torch.long
    )
    image_local_positions[:, 3:7] = torch.arange(4, device=device)
    image_span_table = torch.tensor(
        [[0, 0, 3, 7, 1], [1, 0, 3, 7, 2]],
        device=device,
    )
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "labels": labels,
        "image_loss_mask": image_loss_mask,
        "image_latents": image_latents,
        "image_local_positions": image_local_positions,
        "image_span_table": image_span_table,
        "position_ids": build_row_col_position_ids(token_types, 4),
    }


def main() -> None:
    if not torch.npu.is_available():
        raise SystemExit("Ascend NPU is required")
    device = torch.device("npu:0")
    payload = joint_batch(device)
    model = Qwen3ForCausalLM(tiny_config()).to(
        device=device,
        dtype=torch.bfloat16,
    ).train()
    attention_mask = get_selfless_mask(
        sigma=payload["sigma"],
        seq_len=payload["input_ids"].shape[1],
        device=device,
        input_ids=payload["input_ids"],
        token_types=payload["token_types"],
        boi_token_id=11,
    )
    output = model(
        X0_input_ids=payload["input_ids"],
        labels=payload["labels"],
        attention_mask=attention_mask,
        position_ids=payload["position_ids"],
        token_types=payload["token_types"],
        image_latents=payload["image_latents"],
        image_local_positions=payload["image_local_positions"],
        image_span_table=payload["image_span_table"],
        image_loss_mask=payload["image_loss_mask"],
        flow_sigma=payload["sigma"],
        record_flow_stats=False,
    )
    losses = output.per_modality_loss
    counts = output.per_modality_count
    expected = 0.2 * losses["text_loss"] + losses["image_loss"]
    if not bool(torch.isfinite(output.loss).item()):
        raise AssertionError("joint loss is not finite")
    if not bool(torch.allclose(output.loss, expected, atol=1.0e-5, rtol=1.0e-5)):
        raise AssertionError("joint loss weights changed")
    if int(counts["text_tokens"].item()) != 3:
        raise AssertionError(f"unexpected caption targets: {counts}")
    if int(counts["image_tokens"].item()) != 4:
        raise AssertionError(f"I2T row leaked into image targets: {counts}")

    output.loss.backward()
    text_grad = model.lm_head.weight.grad
    flow_grad = model.image_flow_head.net.final_layer.linear.weight.grad
    if text_grad is None or not bool(torch.isfinite(text_grad).all().item()):
        raise AssertionError("caption CE backward failed")
    if flow_grad is None or not bool(torch.isfinite(flow_grad).all().item()):
        raise AssertionError("T2I flow backward failed")
    torch.npu.synchronize()
    print(
        {
            "loss": float(output.loss.detach().cpu()),
            "text_loss": float(losses["text_loss"].cpu()),
            "image_loss": float(losses["image_loss"].cpu()),
            "text_tokens": int(counts["text_tokens"].cpu()),
            "image_tokens": int(counts["image_tokens"].cpu()),
        }
    )
    print("JOINT CAPTION NPU SMOKE PASS")


if __name__ == "__main__":
    main()
