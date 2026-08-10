"""Targeted correctness checks for Ascend-only operator optimizations."""

from __future__ import annotations

import math

import tbe  # noqa: F401
import torch
import torch_npu


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    reference_fp32 = reference.float()
    candidate_fp32 = candidate.float()
    difference = reference_fp32 - candidate_fp32
    rel_l2 = difference.norm() / reference_fp32.norm().clamp_min(1e-12)
    max_abs = difference.abs().max()
    return float(rel_l2.cpu()), float(max_abs.cpu())


def _check_native_gqa(device: torch.device) -> None:
    batch, query_heads, kv_heads, seq_len, head_dim = 2, 4, 2, 32, 16
    scale = 1.0 / math.sqrt(head_dim)
    query_base = torch.randn(
        batch, query_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
    )
    key_base = torch.randn(
        batch, kv_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16
    )
    value_base = torch.randn_like(key_base)
    output_grad = torch.randn_like(query_base)
    sigma = torch.rand(batch, seq_len, device=device)
    mask = sigma[:, None, None, :] >= sigma[:, None, :, None]
    row_all = mask.all(dim=-1, keepdim=True)
    safe_mask = mask.clone()
    safe_mask[..., 0] = safe_mask[..., 0] & ~row_all.squeeze(-1)
    valid_rows = (~row_all).to(dtype=torch.bfloat16)

    def run(*, expand_kv: bool) -> tuple[torch.Tensor, ...]:
        query = query_base.detach().clone().requires_grad_(True)
        key = key_base.detach().clone().requires_grad_(True)
        value = value_base.detach().clone().requires_grad_(True)
        attention_key = key
        attention_value = value
        if expand_kv:
            repeats = query_heads // kv_heads
            attention_key = key.repeat_interleave(repeats, dim=1)
            attention_value = value.repeat_interleave(repeats, dim=1)
        output = torch_npu.npu_fusion_attention(
            query,
            attention_key,
            attention_value,
            head_num=query_heads,
            input_layout="BNSD",
            atten_mask=safe_mask,
            sparse_mode=0,
            scale=scale,
        )[0]
        output = output * valid_rows
        (output * output_grad).sum().backward()
        torch.npu.synchronize()
        return output.detach(), query.grad.detach(), key.grad.detach(), value.grad.detach()

    expanded = run(expand_kv=True)
    native = run(expand_kv=False)
    tolerances = {
        "output": 0.0,
        "query_grad": 0.0,
        "key_grad": 0.01,
        "value_grad": 0.01,
    }
    for name, reference, candidate in zip(tolerances, expanded, native):
        rel_l2, max_abs = _metrics(reference, candidate)
        if rel_l2 > tolerances[name]:
            raise AssertionError(
                f"native GQA {name} rel_l2={rel_l2} exceeds {tolerances[name]} "
                f"(max_abs={max_abs})"
            )


def _check_prepared_attention_mask(device: torch.device) -> None:
    from models.modeling_model.modeling_selfless_flow import (
        _to_bool_atten_mask,
        compiled_flex_attention,
    )

    batch, query_heads, kv_heads, seq_len, head_dim = 2, 4, 2, 32, 16
    scale = 1.0 / math.sqrt(head_dim)
    query_base = torch.randn(
        batch, query_heads, seq_len, head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    key_base = torch.randn(
        batch, kv_heads, seq_len, head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value_base = torch.randn_like(key_base)
    output_grad = torch.randn_like(query_base)
    sigma = torch.rand(batch, seq_len, device=device)
    mask = sigma[:, None, None, :] >= sigma[:, None, :, None]
    prepared_mask = _to_bool_atten_mask(mask)

    def run(attention_mask) -> tuple[torch.Tensor, ...]:
        query = query_base.detach().clone().requires_grad_(True)
        key = key_base.detach().clone().requires_grad_(True)
        value = value_base.detach().clone().requires_grad_(True)
        output = compiled_flex_attention(
            query,
            key,
            value,
            attention_mask,
            scale,
            True,
        )
        (output * output_grad).sum().backward()
        torch.npu.synchronize()
        return output.detach(), query.grad.detach(), key.grad.detach(), value.grad.detach()

    per_call = run(mask)
    cached = run(prepared_mask)
    for name, reference, candidate in zip(
        ("output", "query_grad", "key_grad", "value_grad"),
        per_call,
        cached,
    ):
        rel_l2, max_abs = _metrics(reference, candidate)
        if rel_l2 != 0.0 or max_abs != 0.0:
            raise AssertionError(
                f"prepared attention mask {name} mismatch: "
                f"rel_l2={rel_l2}, max_abs={max_abs}"
            )


def _check_uint8_token_type_fill(device: torch.device) -> None:
    from models.modeling_model.modeling_selfless_flow import (
        _fill_invalid_token_types,
    )

    token_types = torch.tensor(
        [[0, 2, 1], [0, 3, 1]],
        device=device,
        dtype=torch.uint8,
    )
    valid_mask = torch.tensor(
        [[True, False, True], [False, True, False]],
        device=device,
        dtype=torch.bool,
    )
    actual = _fill_invalid_token_types(token_types, valid_mask)
    expected = torch.tensor(
        [[0, 3, 1], [3, 3, 3]],
        device=device,
        dtype=torch.uint8,
    )
    torch.npu.synchronize()
    if actual.dtype != torch.uint8 or not torch.equal(actual, expected):
        raise AssertionError(
            "uint8 token-type fill mismatch: "
            f"dtype={actual.dtype}, actual={actual.cpu().tolist()}"
        )


@torch.no_grad()
def _check_single_stream_uint8_cache(device: torch.device) -> None:
    from transformers import Qwen3Config

    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM

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
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 4
    config.image_tokens_per_img = 4
    config.image_flow_width = 32
    config.image_flow_depth = 1
    config.image_flow_num_sampling_steps = "1"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "euler"
    config.image_uncond_prob = 0.0
    config.use_flex_attention = True
    model = Qwen3ForCausalLM(config).eval().to(
        device=device,
        dtype=torch.bfloat16,
    )
    input_ids = torch.tensor(
        [[3, 11, 8, 8, 8, 8, 12, 9], [4, 5, 11, 8, 8, 8, 8, 12]],
        device=device,
    )
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0], [0, 0, 2, 1, 1, 1, 1, 2]],
        device=device,
        dtype=torch.uint8,
    )
    sigma = torch.tensor(
        [[0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 2.0, 3.0],
         [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0]],
        device=device,
    )
    initial_noise = (
        torch.arange(32, device=device, dtype=torch.float32).reshape(2, 4, 4)
        / 17.0
    )
    latents, trace = model.sample_image_latents_single_stream(
        input_ids=input_ids,
        token_types=token_types,
        sigma=sigma,
        spans=[(0, 2, 6), (1, 3, 7)],
        initial_noise_bank=initial_noise,
        flow_temperature=0.7,
        flow_cfg=2.5,
        flow_cfg_schedule="constant",
        flow_solver="euler",
        flow_num_steps=1,
        parallel_rate=1,
        order_strategy="spatial_halton",
        use_backbone_cache=True,
        return_trace=True,
    )
    torch.npu.synchronize()
    if latents.shape != (2, 4, 2, 2) or not bool(torch.isfinite(latents).all()):
        raise AssertionError(
            "NPU single-stream cache returned invalid latents: "
            f"shape={tuple(latents.shape)}"
        )
    if trace["backbone_kv_cache_enabled"] is not True:
        raise AssertionError(f"NPU single-stream cache was disabled: {trace}")


def _check_flow_attention_layout(device: torch.device) -> None:
    batch, seq_len, heads, head_dim = 2, 32, 8, 16
    scale = 1.0 / math.sqrt(head_dim)
    query_base = torch.randn(
        batch, seq_len, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    key_base = torch.randn_like(query_base)
    value_base = torch.randn_like(query_base)
    output_grad = torch.randn_like(query_base)
    sigma = torch.rand(batch, seq_len, device=device)
    mask = sigma[:, None, :] >= sigma[:, :, None]
    row_all = mask.all(dim=-1, keepdim=True)
    safe_mask = mask.clone()
    safe_mask[..., 0] = safe_mask[..., 0] & ~row_all.squeeze(-1)
    atten_mask = safe_mask.unsqueeze(1)
    valid_rows = ~row_all

    def run(input_layout: str) -> tuple[torch.Tensor, ...]:
        query = query_base.detach().clone().requires_grad_(True)
        key = key_base.detach().clone().requires_grad_(True)
        value = value_base.detach().clone().requires_grad_(True)
        if input_layout == "BNSD":
            attention_inputs = tuple(
                tensor.transpose(1, 2) for tensor in (query, key, value)
            )
        else:
            attention_inputs = (query, key, value)
        output = torch_npu.npu_fusion_attention(
            *attention_inputs,
            head_num=heads,
            input_layout=input_layout,
            atten_mask=atten_mask,
            sparse_mode=0,
            scale=scale,
        )[0]
        if input_layout == "BNSD":
            output = output.transpose(1, 2)
        output = output * valid_rows.unsqueeze(-1).to(dtype=output.dtype)
        (output * output_grad).sum().backward()
        torch.npu.synchronize()
        return output.detach(), query.grad.detach(), key.grad.detach(), value.grad.detach()

    bnsd = run("BNSD")
    bsnd = run("BSND")
    for name, reference, candidate in zip(
        ("output", "query_grad", "key_grad", "value_grad"), bnsd, bsnd
    ):
        rel_l2, max_abs = _metrics(reference, candidate)
        if rel_l2 != 0.0 or max_abs != 0.0:
            raise AssertionError(
                f"flow BSND {name} mismatch: rel_l2={rel_l2}, max_abs={max_abs}"
            )
    fully_masked = (~valid_rows).unsqueeze(-1).expand_as(bsnd[0])
    if not bool((bsnd[0][fully_masked] == 0).all()):
        raise AssertionError("flow BSND fully-masked rows must be strictly zero")


def _check_prepared_flow_mask(device: torch.device) -> None:
    from models.modeling_model.image_flow_loss import ContextualFlowBlock

    batch, seq_len, channels, heads = 2, 32, 128, 8
    block = ContextualFlowBlock(
        channels=channels,
        num_heads=heads,
        image_tokens_per_img=256,
    ).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(
        batch,
        seq_len,
        channels,
        device=device,
        dtype=torch.bfloat16,
    )
    context = torch.randn_like(x)
    positions = torch.arange(seq_len, device=device).expand(batch, -1)
    sigma = torch.rand(batch, seq_len, device=device)
    mask = sigma[:, None, :] < sigma[:, :, None]
    cache = block.prepare_cross_cache(context, context_positions=positions)
    prepared_mask = block.prepare_context_mask(
        mask,
        batch,
        seq_len,
        seq_len,
        device,
    )
    reference = block._cross_attention(
        x,
        cache,
        mask,
        query_positions=positions,
    )
    candidate = block._cross_attention(
        x,
        cache,
        prepared_mask,
        query_positions=positions,
    )
    torch.npu.synchronize()
    rel_l2, max_abs = _metrics(reference, candidate)
    if rel_l2 != 0.0 or max_abs != 0.0:
        raise AssertionError(
            "prepared flow mask output mismatch: "
            f"rel_l2={rel_l2}, max_abs={max_abs}"
        )


def _check_span_gather(device: torch.device) -> None:
    batch, seq_len, hidden_size = 3, 24, 32
    rows = torch.tensor([0, 0, 2, 1], device=device, dtype=torch.long)
    token_indices = torch.tensor(
        [
            [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10],
            [0, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
        ],
        device=device,
        dtype=torch.long,
    )
    hidden_base = torch.randn(
        batch, seq_len, hidden_size, device=device, dtype=torch.bfloat16
    )
    output_grad = torch.randn(
        rows.numel(), token_indices.shape[1], hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )

    def run(*, use_gather: bool) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden_base.detach().clone().requires_grad_(True)
        if use_gather:
            selected = torch.index_select(hidden, 0, rows)
            gather_indices = token_indices.unsqueeze(-1).expand(
                -1, -1, hidden_size
            )
            output = torch.gather(selected, 1, gather_indices)
        else:
            output = hidden[rows.unsqueeze(1), token_indices]
        (output * output_grad).sum().backward()
        torch.npu.synchronize()
        return output.detach(), hidden.grad.detach()

    advanced = run(use_gather=False)
    gathered = run(use_gather=True)
    for name, reference, candidate in (
        ("output", advanced[0], gathered[0]),
        ("hidden_grad", advanced[1], gathered[1]),
    ):
        rel_l2, max_abs = _metrics(reference, candidate)
        if rel_l2 != 0.0 or max_abs != 0.0:
            raise AssertionError(
                f"span gather {name} mismatch: rel_l2={rel_l2}, max_abs={max_abs}"
            )


def main() -> None:
    torch.manual_seed(20260809)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    _check_native_gqa(device)
    _check_prepared_attention_mask(device)
    _check_uint8_token_type_fill(device)
    _check_single_stream_uint8_cache(device)
    _check_flow_attention_layout(device)
    _check_prepared_flow_mask(device)
    _check_span_gather(device)
    print(
        "PASS native_gqa prepared_mask uint8_token_type_fill "
        "single_stream_cache flow_bsnd prepared_flow_mask span_gather"
    )


if __name__ == "__main__":
    main()
