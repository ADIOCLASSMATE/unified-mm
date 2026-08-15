"""One-NPU kernel smoke for the two isolated architecture ablations.

Run only inside the fixed Ascend development Notebook:

    .venv/bin/python tests/smoke_npu_ablation_variants.py
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
from transformers import Qwen3Config

from models.modeling_model.modeling_positionwise_flow import (
    PositionwiseFlowQwen3ForCausalLM,
)
from models.modeling_model.modeling_showo2_maskgit import (
    ShowO2MaskGITQwen3ForCausalLM,
)
from utils.showo2_maskgit import get_showo2_attention_mask
from utils.utils import get_selfless_mask


def tiny_config(architecture_variant: str) -> Qwen3Config:
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
        eos_token_id=9,
    )
    config.architecture_variant = architecture_variant
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 32
    config.image_flow_depth = 2
    config.image_flow_num_sampling_steps = "1"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "euler"
    config.image_input_noise_strength = 1.0e-2
    config.image_uncond_prob = 0.1
    config.backbone_attention_output_gate = "none"
    config.maskgit_generation_steps = 2
    config.maskgit_validation_mask_ratio = 0.5
    config.use_flex_attention = True
    config.use_cache = False
    return config


def batch(device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [[3, 11, 8, 8, 8, 8, 12, 9]],
        device=device,
    )
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]],
        device=device,
        dtype=torch.uint8,
    )
    sigma = torch.tensor(
        [[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0]],
        device=device,
    )
    image_latents = torch.zeros(1, 8, 4, device=device, dtype=torch.bfloat16)
    image_latents[:, 2:6] = torch.randn(
        1,
        4,
        4,
        device=device,
        dtype=torch.bfloat16,
    )
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "token_types": token_types,
        "sigma": sigma,
        "image_latents": image_latents,
        "image_local_positions": torch.tensor(
            [[-1, -1, 0, 1, 2, 3, -1, -1]],
            device=device,
        ),
        "image_span_table": torch.tensor(
            [[0, 0, 2, 6, 0]],
            device=device,
        ),
    }


def run_variant(
    model_class,
    architecture_variant: str,
    device: torch.device,
) -> dict[str, object]:
    payload = batch(device)
    model = model_class(tiny_config(architecture_variant)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    if architecture_variant == "showo2_maskgit":
        attention_mask = get_showo2_attention_mask(payload["token_types"])
        attention_kwargs = {"attention_mask_contract": "showo2"}
    else:
        attention_mask = get_selfless_mask(
            payload["sigma"],
            payload["input_ids"].shape[1],
            device,
        )
        attention_kwargs = {}

    model.train()
    output = model(
        X0_input_ids=payload["input_ids"],
        labels=payload["labels"],
        attention_mask=attention_mask,
        token_types=payload["token_types"],
        image_latents=payload["image_latents"],
        image_local_positions=payload["image_local_positions"],
        image_span_table=payload["image_span_table"],
        flow_sigma=payload["sigma"],
        **attention_kwargs,
    )
    if not bool(torch.isfinite(output.loss).item()):
        raise AssertionError(f"{architecture_variant} training loss is not finite")
    output.loss.backward()
    final_grad = model.image_flow_head.net.final_layer.linear.weight.grad
    if final_grad is None or not bool(torch.isfinite(final_grad).all().item()):
        raise AssertionError(f"{architecture_variant} flow backward failed")

    model.eval()
    generated, trace = model.sample_image_latents_single_stream(
        input_ids=payload["input_ids"],
        token_types=payload["token_types"],
        sigma=payload["sigma"],
        spans=[(0, 2, 6)],
        image_latent_dim=4,
        flow_temperature=1.0,
        flow_cfg=1.0,
        flow_solver="euler",
        flow_num_steps=1,
        parallel_rate=1,
        order_strategy=(
            "maskgit" if architecture_variant == "showo2_maskgit" else "sigma"
        ),
        use_backbone_cache=False,
        return_trace=True,
    )
    torch.npu.synchronize()
    if tuple(generated.shape) != (1, 4, 2, 2):
        raise AssertionError(
            f"{architecture_variant} generated shape mismatch: {generated.shape}"
        )
    if not bool(torch.isfinite(generated).all().item()):
        raise AssertionError(f"{architecture_variant} generation is not finite")
    return {
        "architecture_variant": architecture_variant,
        "loss": float(output.loss.detach().cpu()),
        "generation_steps": int(trace["generation_step"].max().cpu()),
        "flow_head_architecture": trace["flow_head_architecture"],
    }


def main() -> None:
    if not torch.npu.is_available():
        raise SystemExit("Ascend NPU is required")
    device = torch.device("npu:0")
    reports = [
        run_variant(
            PositionwiseFlowQwen3ForCausalLM,
            "positionwise_selfless",
            device,
        ),
        run_variant(
            ShowO2MaskGITQwen3ForCausalLM,
            "showo2_maskgit",
            device,
        ),
    ]
    for report in reports:
        print(report)
    print("ABLATION NPU SMOKE PASS")


if __name__ == "__main__":
    main()
