"""One-NPU A/B/E XT-query/X0-content training and generation smoke.

Run only in the fixed Ascend development Notebook::

    .venv/bin/python tests/smoke_npu_static_x0_content.py
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import (
    X0_CONTENT_FLOW_CONDITION_CONTRACT,
    Qwen3ForCausalLM,
    X0ContentFlowLoss,
)
from utils.utils import get_selfless_mask


def _config(attention_contract: str) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=9,
    )
    values = {
        "mask_token_id": 7,
        "image_mask_token_id": 8,
        "boi_token_id": 11,
        "eoi_token_id": 12,
        "image_latent_dim": 4,
        "image_tokens_per_img": 4,
        "image_flow_width": 32,
        "image_flow_depth": 2,
        "image_flow_num_sampling_steps": "10",
        "image_flow_batch_mul": 4,
        "image_flow_time_scale": 1000.0,
        "image_flow_time_sampling": "uniform",
        "image_flow_time_eps": 1.0e-4,
        "image_flow_time_uniform_mix": 0.0,
        "image_flow_solver": "heun",
        "image_input_noise_strength": 0.0,
        "image_uncond_prob": 0.0,
        "lambda_text": 0.05,
        "lambda_image": 1.0,
        "backbone_attention_output_gate": "none",
        "use_flex_attention": True,
        "use_cache": False,
        "training_objective": "selfless_dual_stream",
        "dual_stream_attention_contract": attention_contract,
        "flow_head_attention_contract": attention_contract,
        "flow_condition_contract": X0_CONTENT_FLOW_CONDITION_CONTRACT,
    }
    for key, value in values.items():
        setattr(config, key, value)
    return config


def _payload(device: torch.device):
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
        # The two unrevealed image positions share the current frontier sigma,
        # exactly as production generation does. A future masked position must
        # not be ordered behind another still-masked position in this parity
        # check, or the dual X0 stream would (correctly) expose content that the
        # hybrid stream has not committed yet.
        [[0.0, 1.0, 4.0, 5.0, 7.0, 7.0, 2.0, 3.0]],
        device=device,
    )
    latents = torch.zeros(1, 8, 4, device=device, dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu").manual_seed(20260904)
    latents[:, 2:6] = torch.randn(
        1, 4, 4, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    return input_ids, token_types, sigma, latents


def _randomize_flow_output(model) -> None:
    # Generate every probe value on CPU so the exact same model and inputs are
    # exercised when the smoke is pinned to different physical NPU cards.
    generator = torch.Generator(device="cpu").manual_seed(20260905)

    def fill(parameter) -> None:
        value = torch.randn(
            parameter.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).mul_(0.1)
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    with torch.no_grad():
        for block in model.image_flow_head.net.blocks:
            fill(block.adaLN_modulation[-1].weight)
            fill(block.adaLN_modulation[-1].bias)
        final = model.image_flow_head.net.final_layer
        fill(final.adaLN_modulation[-1].weight)
        fill(final.adaLN_modulation[-1].bias)
        fill(final.linear.weight)
        fill(final.linear.bias)


def _run_variant(
    label: str,
    attention_contract: str,
    order_strategy: str,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(20260903)
    torch.npu.manual_seed(20260903)
    model = Qwen3ForCausalLM(_config(attention_contract)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    if type(model.image_flow_head) is not X0ContentFlowLoss:
        raise AssertionError(f"{label} did not select X0ContentFlowLoss")
    _randomize_flow_output(model)
    input_ids, token_types, sigma, latents = _payload(device)
    strict = get_selfless_mask(sigma, 8, device)
    include_diagonal = attention_contract == "xlnet_content_diagonal"
    content = get_selfless_mask(
        sigma,
        8,
        device,
        include_diagonal=include_diagonal,
    )

    model.train()
    output = model(
        X0_input_ids=input_ids,
        labels=input_ids.clone(),
        attention_mask=strict,
        content_attention_mask=content,
        token_types=token_types,
        image_latents=latents,
        image_local_positions=torch.tensor(
            [[-1, -1, 0, 1, 2, 3, -1, -1]],
            device=device,
        ),
        image_span_table=torch.tensor(
            [[0, 0, 2, 6, 0]],
            device=device,
        ),
        image_loss_mask=token_types.eq(1),
        flow_sigma=sigma,
        compute_text_loss=False,
        compute_image_loss=True,
        return_logits=False,
    )
    if not bool(torch.isfinite(output.loss).item()):
        raise AssertionError(f"{label} training loss is non-finite")
    if int(output.per_modality_count["image_tokens"].item()) != 16:
        raise AssertionError(f"{label} did not retain four RF states")
    output.loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all().item()
        ):
            raise AssertionError(f"{label} backward produced non-finite gradients")
    model.zero_grad(set_to_none=True)

    # Generation executes one hybrid stream. Compare it with the exact mixture
    # of the backbone's dual X0/XT streams at every valid row.
    model.eval()
    visible = torch.tensor(
        [[False, False, True, True, False, False, False, False]],
        device=device,
    )
    content_queries = token_types.ne(3) & (token_types.ne(1) | visible)
    dual = model.model(
        X0_input_ids=input_ids,
        attention_mask=strict,
        content_attention_mask=content,
        token_types=token_types,
        image_latents=latents,
        calculate_likelihood=True,
        return_x0_hidden_state=True,
    )
    hybrid_mask = get_selfless_mask(
        sigma,
        8,
        device,
        diagonal_query_mask=(content_queries if include_diagonal else None),
    )
    hybrid = model.model(
        X0_input_ids=input_ids,
        attention_mask=hybrid_mask,
        token_types=token_types,
        image_latents=latents,
        image_latent_mask=visible,
        calculate_likelihood=False,
    ).last_hidden_state
    dual_reference = torch.where(
        content_queries.unsqueeze(-1),
        dual["x0_last_hidden_state"],
        dual.last_hidden_state,
    )
    torch.testing.assert_close(hybrid, dual_reference, rtol=0, atol=0)

    generation_kwargs = {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "spans": [(0, 2, 6)],
        "initial_noise_bank": (
            torch.arange(16, device=device, dtype=torch.float32).view(1, 4, 4)
            / 19.0
        ),
        "flow_temperature": 0.7,
        "flow_cfg": 2.0,
        "flow_cfg_schedule": "constant",
        "flow_solver": "heun",
        "flow_num_steps": 2,
        "parallel_rate": 1,
        "order_strategy": order_strategy,
        "return_trace": True,
    }
    full, full_trace = model.generate(
        "t2i",
        **generation_kwargs,
        use_cache=False,
    )
    cached, cached_trace = model.generate(
        "t2i",
        **generation_kwargs,
        use_cache=True,
    )
    torch.testing.assert_close(cached, full, rtol=5.0e-3, atol=5.0e-3)
    if cached_trace["flow_content_condition"] != "backbone_x0_hidden":
        raise AssertionError(f"{label} generation used the wrong content condition")
    if cached_trace["flow_content_cache_tokens_committed"] != 3:
        raise AssertionError(f"{label} did not commit three pending flow tokens")
    if full_trace["single_stream_attention_contract"] != attention_contract:
        raise AssertionError(f"{label} full generation attention changed")
    torch.npu.synchronize()
    return {
        "variant": label,
        "attention_contract": attention_contract,
        "order_strategy": order_strategy,
        "loss": float(output.loss.detach().cpu()),
        "single_dual_max_abs": float(
            (hybrid - dual_reference).abs().max().float().cpu()
        ),
        "cached_full_max_abs": float((cached - full).abs().max().float().cpu()),
        "flow_condition_contract": cached_trace["flow_condition_contract"],
    }


def main() -> None:
    if not torch.npu.is_available():
        raise SystemExit("Ascend NPU is required")
    device = torch.device("npu:0")
    reports = [
        _run_variant("a", "selfless_strict", "spatial_halton", device),
        _run_variant(
            "b",
            "xlnet_content_diagonal",
            "spatial_halton",
            device,
        ),
        _run_variant(
            "e_on_b",
            "xlnet_content_diagonal",
            "sequential",
            device,
        ),
    ]
    for report in reports:
        print(report)
    print("A/B/E X0-CONTENT NPU SMOKE PASS")


if __name__ == "__main__":
    main()
