"""One-NPU kernel smoke for isolated architecture ablations.

Run only inside the fixed Ascend development Notebook:

    .venv/bin/python tests/smoke_npu_ablation_variants.py
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import (
    Qwen3ForCausalLM as BaselineQwen3ForCausalLM,
)
from models.modeling_model.modeling_selfless_flow_dynamic_xt import (
    DynamicXtQwen3ForCausalLM,
    SelflessFlowDynamicXtConfig,
)
from models.modeling_model.modeling_selfless_flow_positionwise_on_b import (
    PositionwiseFlowOnBQwen3ForCausalLM,
    SelflessFlowPositionwiseOnBConfig,
)
from utils.utils import get_selfless_mask


def tiny_config(model_label: str) -> Qwen3Config:
    config_classes = {
        "dynamic_xt": SelflessFlowDynamicXtConfig,
        "positionwise_flow_head_on_b": SelflessFlowPositionwiseOnBConfig,
    }
    config_class = config_classes.get(model_label, Qwen3Config)
    config = config_class(
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
    if model_label in {
        "positionwise_selfless",
        "dynamic_xt",
        "positionwise_flow_head_on_b",
    }:
        config.architecture_variant = model_label
    elif model_label == "deterministic_ltr_on_b":
        config.architecture_variant = "selfless_contextual"
    config.training_objective = "selfless_dual_stream"
    config.dual_stream_attention_contract = "xlnet_content_diagonal"
    config.flow_head_attention_contract = (
        "not_applicable"
        if model_label == "positionwise_flow_head_on_b"
        else "xlnet_content_diagonal"
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 32
    config.image_flow_depth = 2
    config.image_flow_num_sampling_steps = "10"
    config.image_flow_batch_mul = 4
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "euler"
    config.image_input_noise_strength = 1.0e-2
    config.image_uncond_prob = 0.1
    config.lambda_text = 0.05
    config.lambda_image = 1.0
    config.backbone_attention_output_gate = "none"
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
    model_label: str,
    device: torch.device,
) -> dict[str, object]:
    payload = batch(device)
    model = model_class(tiny_config(model_label)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    attention_mask = get_selfless_mask(
        payload["sigma"],
        payload["input_ids"].shape[1],
        device,
    )
    content_attention_mask = get_selfless_mask(
        payload["sigma"],
        payload["input_ids"].shape[1],
        device,
        include_diagonal=True,
    )

    model.train()
    backbone_calls = 0

    def count_backbone_calls(_module, _args, _output):
        nonlocal backbone_calls
        backbone_calls += 1

    hook = (
        model.model.register_forward_hook(count_backbone_calls)
        if model_label == "dynamic_xt"
        else None
    )
    output = model(
        X0_input_ids=payload["input_ids"],
        labels=payload["labels"],
        attention_mask=attention_mask,
        content_attention_mask=content_attention_mask,
        token_types=payload["token_types"],
        image_latents=payload["image_latents"],
        image_local_positions=payload["image_local_positions"],
        image_span_table=payload["image_span_table"],
        flow_sigma=payload["sigma"],
    )
    if not bool(torch.isfinite(output.loss).item()):
        raise AssertionError(f"{model_label} training loss is not finite")
    output.loss.backward()
    if hook is not None:
        hook.remove()
        if backbone_calls != 1:
            raise AssertionError(
                "Dynamic-XT training executed the backbone "
                f"{backbone_calls} times"
            )
    final_grad = model.image_flow_head.net.final_layer.linear.weight.grad
    if final_grad is None or not bool(torch.isfinite(final_grad).all().item()):
        raise AssertionError(f"{model_label} flow backward failed")
    if model_label == "dynamic_xt":
        if output.dynamic_xt_query_batch_mul != 4:
            raise AssertionError("Dynamic-XT did not preserve four RF states")
        if int(output.per_modality_count["image_tokens"].item()) != 16:
            raise AssertionError("Dynamic-XT image-token count is not 4x")
        time_grad = (
            model.model.backbone_flow_time_embedder.mlp[0].weight.grad
        )
        if time_grad is None or not bool(torch.isfinite(time_grad).all().item()):
            raise AssertionError("Dynamic-XT time embedder backward failed")

    model.eval()
    generated, trace = model.generate(
        "t2i",
        input_ids=payload["input_ids"],
        token_types=payload["token_types"],
        sigma=payload["sigma"],
        spans=[(0, 2, 6)],
        image_latent_dim=4,
        flow_temperature=1.0,
        flow_cfg=2.0,
        flow_solver="heun" if model_label == "dynamic_xt" else "euler",
        flow_num_steps=1,
        parallel_rate=1,
        order_strategy=(
            "sequential"
            if model_label == "deterministic_ltr_on_b"
            else "spatial_halton"
            if model_label == "positionwise_flow_head_on_b"
            else "sigma"
        ),
        use_cache=True,
        return_trace=True,
        # D needs two serialized tokens to exercise the fused previous-X0
        # backbone commit and the corresponding flow-content cache update.
        _debug_max_generation_steps=(2 if model_label == "dynamic_xt" else None),
    )
    torch.npu.synchronize()
    if tuple(generated.shape) != (1, 4, 2, 2):
        raise AssertionError(
            f"{model_label} generated shape mismatch: {generated.shape}"
        )
    if not bool(torch.isfinite(generated).all().item()):
        raise AssertionError(f"{model_label} generation is not finite")
    if model_label == "dynamic_xt":
        if trace["dynamic_xt_conditional_velocity_evaluations"] != 4:
            raise AssertionError("Heun must recompute conditional XT twice per token")
        if trace["dynamic_xt_unconditional_velocity_evaluations"] != 4:
            raise AssertionError("Heun must recompute unconditional XT twice per token")
        if trace["dynamic_xt_query_cache_policy"] != "read_only_x0_kv":
            raise AssertionError("Dynamic-XT query cache policy changed")
        if trace["dynamic_xt_flow_query_condition"] != "backbone_xt_hidden":
            raise AssertionError("Dynamic-XT flow query condition changed")
        if trace["dynamic_xt_flow_content_condition"] != "backbone_x0_hidden":
            raise AssertionError("Dynamic-XT flow content condition changed")
        if trace["dynamic_xt_flow_content_condition_commits"] != 1:
            raise AssertionError("Dynamic-XT did not commit the fused previous X0")
    return {
        "architecture_variant": model_label,
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
            BaselineQwen3ForCausalLM,
            "deterministic_ltr_on_b",
            device,
        ),
        run_variant(
            PositionwiseFlowOnBQwen3ForCausalLM,
            "positionwise_flow_head_on_b",
            device,
        ),
        run_variant(
            DynamicXtQwen3ForCausalLM,
            "dynamic_xt",
            device,
        ),
    ]
    for report in reports:
        print(report)
    print("ABLATION NPU SMOKE PASS")


if __name__ == "__main__":
    main()
