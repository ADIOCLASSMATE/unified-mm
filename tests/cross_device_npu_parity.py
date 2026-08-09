"""Deterministic CUDA/NPU parity probe for the Selfless-Flow migration.

The CUDA run creates the fixture once.  Both backends then load the exact same
CPU weights and inputs, so device RNG differences cannot hide migration bugs.
This file is test-only: it does not change model or training implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Qwen3Config


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SEED = 20260809


def _config() -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config.mask_token_id = 7
    config.image_mask_token_id = 8
    config.boi_token_id = 11
    config.eoi_token_id = 12
    config.image_latent_dim = 8
    config.image_tokens_per_img = 16
    config.image_flow_width = 128
    config.image_flow_depth = 2
    config.image_flow_num_sampling_steps = "2"
    config.image_flow_batch_mul = 1
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "uniform"
    config.image_flow_time_eps = 1.0e-4
    config.image_flow_time_uniform_mix = 0.0
    config.image_flow_solver = "heun"
    config.image_input_noise_strength = 0.0
    config.image_uncond_prob = 0.0
    config.backbone_attention_output_gate = "none"
    config.use_flex_attention = True
    return config


def _stable_reinitialize(module: torch.nn.Module, seed: int) -> None:
    """Fill every parameter from one CPU generator, including zero-init gates."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
            if name.endswith("norm.weight") or "layernorm.weight" in name:
                value.normal_(mean=1.0, std=0.02, generator=generator)
            elif parameter.ndim >= 2:
                value.normal_(mean=0.0, std=0.035, generator=generator)
            else:
                value.normal_(mean=0.0, std=0.02, generator=generator)
            parameter.copy_(value.to(dtype=parameter.dtype))


def _cpu_randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _build_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    batch, image_tokens = 2, 16
    seq = 128
    valid_tokens = image_tokens + 4
    padding = seq - valid_tokens
    input_ids = torch.tensor(
        [
            [21, 11, *([8] * image_tokens), 12, 2, *([0] * padding)],
            [22, 11, *([8] * image_tokens), 12, 2, *([0] * padding)],
        ],
        dtype=torch.long,
    )
    token_types = torch.tensor(
        [[0, 2, *([1] * image_tokens), 2, 0, *([0] * padding)]] * batch,
        dtype=torch.uint8,
    )
    image_order = torch.tensor(
        [
            [0, 7, 3, 12, 1, 15, 5, 9, 2, 14, 6, 10, 4, 13, 8, 11],
            [15, 2, 9, 4, 12, 0, 7, 14, 5, 10, 1, 8, 13, 3, 11, 6],
        ],
        dtype=torch.float32,
    )
    sigma = torch.ones(batch, seq, dtype=torch.float32)
    sigma[:, 0] = 0.0
    sigma[:, 1] = 0.0
    sigma[:, 2 : 2 + image_tokens] = image_order / float(image_tokens - 1)
    sigma[:, 2 + image_tokens] = 0.0
    sigma[:, 3 + image_tokens] = 1.0
    image_latents = torch.zeros(batch, seq, 8, dtype=torch.float32)
    image_latents[:, 2 : 2 + image_tokens] = _cpu_randn(
        (batch, image_tokens, 8), generator
    )
    image_span_table = torch.tensor(
        [[0, 0, 2, 18, 1], [1, 0, 2, 18, 1]], dtype=torch.long
    )
    image_local_positions = torch.tensor(
        [[-1, -1, *range(image_tokens), -1, -1, *([-1] * padding)]] * batch,
        dtype=torch.long,
    )
    labels = torch.full_like(input_ids, -100)
    flow_t = torch.linspace(0.08, 0.92, image_tokens).repeat(batch, 1)
    flow_noise = _cpu_randn((batch, image_tokens, 8), generator)
    flow_probe = _cpu_randn((batch, image_tokens, 8), generator)
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "image_latents": image_latents,
        "image_span_table": image_span_table,
        "image_local_positions": image_local_positions,
        "labels": labels,
        "flow_t": flow_t,
        "flow_noise": flow_noise,
        "flow_probe": flow_probe,
    }


def _build_attention_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 2)
    batch, q_heads, kv_heads, seq, head_dim = 2, 8, 4, 16, 16
    sigma = torch.tensor(
        [
            [0, 7, 3, 12, 1, 15, 5, 9, 2, 14, 6, 10, 4, 13, 8, 11],
            [15, 2, 9, 4, 12, 0, 7, 14, 5, 10, 1, 8, 13, 3, 11, 6],
        ],
        dtype=torch.float32,
    ) / 15.0
    return {
        "q": _cpu_randn((batch, q_heads, seq, head_dim), generator),
        "k": _cpu_randn((batch, kv_heads, seq, head_dim), generator),
        "v": _cpu_randn((batch, kv_heads, seq, head_dim), generator),
        "probe": _cpu_randn((batch, q_heads, seq, head_dim), generator),
        "sigma": sigma,
    }


def prepare_fixture(path: Path) -> None:
    from models.modeling_model.image_flow_loss import FlowLoss
    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM

    torch.manual_seed(SEED)
    model = Qwen3ForCausalLM(_config()).float()
    _stable_reinitialize(model, SEED + 3)
    flow = FlowLoss(
        target_channels=8,
        z_channels=128,
        depth=2,
        width=128,
        num_sampling_steps=2,
        time_sampling="uniform",
        uniform_mix=0.0,
        image_tokens_per_img=16,
    ).float()
    _stable_reinitialize(flow, SEED + 4)
    payload = {
        "version": 1,
        "seed": SEED,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "flow_state": {k: v.detach().cpu() for k, v in flow.state_dict().items()},
        "model_inputs": _build_inputs(),
        "attention_inputs": _build_attention_inputs(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(json.dumps({"fixture": str(path), "bytes": path.stat().st_size}))


def _device(backend: str, index: int) -> torch.device:
    if backend == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(index)
        return torch.device("npu", index)
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    else:
        torch.cuda.synchronize(device)


def _cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu()


def _allowed_mask(sigma: torch.Tensor) -> torch.Tensor:
    return sigma.unsqueeze(1) < sigma.unsqueeze(2)


def run_attention(
    fixture: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    from models.modeling_model.modeling_selfless_flow import compiled_flex_attention
    from utils.utils import get_selfless_mask

    source = fixture["attention_inputs"]
    q = source["q"].to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    k = source["k"].to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    v = source["v"].to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    probe = source["probe"].to(device=device, dtype=torch.bfloat16)
    sigma = source["sigma"].to(device=device)
    mask = get_selfless_mask(sigma, sigma.shape[1], device)
    scale = 1.0 / math.sqrt(q.shape[-1])
    output = compiled_flex_attention(q, k, v, mask, scale, True)
    loss = (output.float() * probe.float()).mean()
    loss.backward()

    repeated_k = k.detach().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    repeated_v = v.detach().repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    allowed = _allowed_mask(sigma).unsqueeze(1)
    reference = F.scaled_dot_product_attention(
        q.detach(), repeated_k, repeated_v, attn_mask=allowed, scale=scale
    )
    valid_rows = allowed.any(dim=-1).expand(-1, q.shape[1], -1)
    return {
        "output": _cpu(output),
        "reference": _cpu(reference),
        "valid_rows": _cpu(valid_rows),
        "q_grad": _cpu(q.grad),
        "k_grad": _cpu(k.grad),
        "v_grad": _cpu(v.grad),
    }


def _collect_parameter_grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def run_flow(
    fixture: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    from models.modeling_model.image_flow_loss import FlowLoss

    flow = FlowLoss(
        target_channels=8,
        z_channels=128,
        depth=2,
        width=128,
        num_sampling_steps=2,
        time_sampling="uniform",
        uniform_mix=0.0,
        image_tokens_per_img=16,
    )
    flow.load_state_dict(fixture["flow_state"], strict=True)
    flow = flow.to(device=device, dtype=torch.bfloat16).train()
    source = fixture["model_inputs"]
    x = source["flow_noise"].to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    z = _cpu_randn((2, 16, 128), torch.Generator().manual_seed(SEED + 5)).to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    context = source["image_latents"][:, 2:18].to(
        device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    t = source["flow_t"].to(device=device)
    positions = torch.arange(16, device=device).repeat(2, 1)
    sigma = source["sigma"][:, 2:18].to(device=device)
    context_mask = _allowed_mask(sigma)
    probe = source["flow_probe"].to(device=device, dtype=torch.bfloat16)
    output = flow.velocity(
        x,
        t,
        z,
        context_latents=context,
        context_mask=context_mask,
        query_positions=positions,
        context_positions=positions,
        context_conditions=z,
        record_stats=False,
    )
    scalar = (output.float() * probe.float()).mean()
    scalar.backward()
    return {
        "output": _cpu(output),
        "loss": _cpu(scalar),
        "x_grad": _cpu(x.grad),
        "z_grad": _cpu(z.grad),
        "context_grad": _cpu(context.grad),
        "parameter_grads": _collect_parameter_grads(flow),
    }


@contextlib.contextmanager
def _fixed_flow_randomness(flow, t: torch.Tensor, noise: torch.Tensor):
    import models.modeling_model.image_flow_loss as flow_module

    original_sampler = flow._sample_times
    original_randn = flow_module.torch.randn

    def sample_times(batch_size: int, device) -> torch.Tensor:
        flat = t.reshape(-1)
        if batch_size != flat.numel():
            raise AssertionError((batch_size, flat.numel()))
        return flat.to(device=device)

    def randn(*args, **kwargs):
        shape = tuple(args[0]) if args and isinstance(args[0], (tuple, list)) else tuple(args)
        if shape == tuple(noise.shape):
            return noise.to(
                device=kwargs.get("device", noise.device),
                dtype=kwargs.get("dtype", noise.dtype),
            )
        return original_randn(*args, **kwargs)

    flow._sample_times = sample_times
    flow_module.torch.randn = randn
    try:
        yield
    finally:
        flow._sample_times = original_sampler
        flow_module.torch.randn = original_randn


def run_model(
    fixture: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
    from models.modeling_model.image_position_utils import build_row_col_position_ids
    from utils.utils import get_selfless_mask

    model = Qwen3ForCausalLM(_config())
    model.load_state_dict(fixture["model_state"], strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    source = fixture["model_inputs"]
    inputs = {
        key: value.to(device=device)
        for key, value in source.items()
        if key
        in {
            "input_ids",
            "token_types",
            "sigma",
            "image_latents",
            "image_span_table",
            "image_local_positions",
            "labels",
        }
    }
    position_ids = build_row_col_position_ids(inputs["token_types"], 16)
    attention_mask = get_selfless_mask(
        sigma=inputs["sigma"],
        seq_len=inputs["input_ids"].shape[1],
        device=device,
        input_ids=inputs["input_ids"],
        token_types=inputs["token_types"],
        boi_token_id=11,
    )
    activations: dict[str, list[torch.Tensor]] = {}
    handles = []

    def capture(name: str):
        def hook(_module, _args, output):
            values = output if isinstance(output, tuple) else (output,)
            activations.setdefault(name, []).extend(
                value.detach().cpu() for value in values if isinstance(value, torch.Tensor)
            )

        return hook

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(capture(f"attention.{index}")))
        handles.append(layer.register_forward_hook(capture(f"decoder.{index}")))
    for index, block in enumerate(model.image_flow_head.net.blocks):
        handles.append(block.register_forward_hook(capture(f"flow_block.{index}")))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=0.01
    )
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    with _fixed_flow_randomness(
        model.image_flow_head,
        source["flow_t"],
        source["flow_noise"],
    ):
        output = model(
            X0_input_ids=inputs["input_ids"],
            labels=inputs["labels"],
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_types=inputs["token_types"],
            image_span_table=inputs["image_span_table"],
            image_local_positions=inputs["image_local_positions"],
            image_latents=inputs["image_latents"],
            flow_sigma=inputs["sigma"],
            use_cache=False,
            record_flow_stats=False,
            record_backbone_gate_stats=False,
            return_logits=False,
        )
    output.loss.backward()
    gradients = _collect_parameter_grads(model)
    optimizer.step()
    deltas = {
        name: parameter.detach().cpu() - before[name]
        for name, parameter in model.named_parameters()
    }
    for handle in handles:
        handle.remove()
    return {
        "loss": _cpu(output.loss),
        "last_hidden_state": _cpu(output.last_hidden_state),
        "activations": activations,
        "parameter_grads": gradients,
        "parameter_deltas": deltas,
    }


def run_training_trajectory(
    fixture: dict[str, Any], device: torch.device, steps: int
) -> dict[str, Any]:
    from models.modeling_model.image_position_utils import build_row_col_position_ids
    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
    from utils.utils import get_selfless_mask

    model = Qwen3ForCausalLM(_config())
    model.load_state_dict(fixture["model_state"], strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).train()
    source = fixture["model_inputs"]
    inputs = {
        key: value.to(device=device)
        for key, value in source.items()
        if key
        in {
            "input_ids",
            "token_types",
            "sigma",
            "image_latents",
            "image_span_table",
            "image_local_positions",
            "labels",
        }
    }
    position_ids = build_row_col_position_ids(inputs["token_types"], 16)
    attention_mask = get_selfless_mask(
        sigma=inputs["sigma"],
        seq_len=inputs["input_ids"].shape[1],
        device=device,
        input_ids=inputs["input_ids"],
        token_types=inputs["token_types"],
        boi_token_id=11,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=0.01
    )
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    losses = []
    grad_norms = []
    hidden_projections = []
    projection = torch.linspace(
        -1.0,
        1.0,
        inputs["input_ids"].numel() * model.config.hidden_size,
        device=device,
        dtype=torch.float32,
    ).view(*inputs["input_ids"].shape, model.config.hidden_size)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with _fixed_flow_randomness(
            model.image_flow_head,
            source["flow_t"],
            source["flow_noise"],
        ):
            output = model(
                X0_input_ids=inputs["input_ids"],
                labels=inputs["labels"],
                attention_mask=attention_mask,
                position_ids=position_ids,
                token_types=inputs["token_types"],
                image_span_table=inputs["image_span_table"],
                image_local_positions=inputs["image_local_positions"],
                image_latents=inputs["image_latents"],
                flow_sigma=inputs["sigma"],
                use_cache=False,
                record_flow_stats=False,
                record_backbone_gate_stats=False,
                return_logits=False,
            )
        output.loss.backward()
        squared_norm = torch.zeros((), device=device, dtype=torch.float32)
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().float().square().sum()
        losses.append(output.loss.detach().float().cpu())
        grad_norms.append(squared_norm.sqrt().cpu())
        hidden_projections.append(
            (output.last_hidden_state.detach().float() * projection).mean().cpu()
        )
        optimizer.step()
    final_deltas = {
        name: parameter.detach().cpu() - initial[name]
        for name, parameter in model.named_parameters()
    }
    return {
        "losses": torch.stack(losses),
        "grad_norms": torch.stack(grad_norms),
        "hidden_projections": torch.stack(hidden_projections),
        "final_parameter_deltas": final_deltas,
    }


def run(args: argparse.Namespace) -> None:
    device = _device(args.backend, args.device)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    results: dict[str, Any] = {
        "backend": args.backend,
        "device": str(device),
        "torch_version": str(torch.__version__),
    }
    if args.phase in {"attention", "all"}:
        results["attention"] = run_attention(fixture, device)
    if args.phase in {"flow", "all"}:
        results["flow"] = run_flow(fixture, device)
    if args.phase in {"model", "all"}:
        results["model"] = run_model(fixture, device)
    if args.phase == "trajectory":
        results["trajectory"] = run_training_trajectory(fixture, device, args.steps)
    _sync(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, args.output)
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}))


def _flatten(prefix: str, value: Any, out: dict[str, torch.Tensor]) -> None:
    if isinstance(value, torch.Tensor):
        out[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _flatten(f"{prefix}.{index}", child, out)


def _comparison_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    delta = right - left
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    delta_norm = torch.linalg.vector_norm(delta)
    cosine = (
        F.cosine_similarity(left, right, dim=0).item()
        if left.numel() and left_norm.item() and right_norm.item()
        else None
    )
    return {
        "numel": left.numel(),
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean().item()) if delta.numel() else 0.0,
        "rmse": float(delta.square().mean().sqrt().item()) if delta.numel() else 0.0,
        "rel_l2": float((delta_norm / left_norm.clamp_min(1.0e-30)).item()),
        "cosine": cosine,
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


def _load_result(path: Path) -> dict[str, Any]:
    # Older locally-generated results stored torch.__version__ as TorchVersion.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _attention_row_comparisons(
    cuda_payload: dict[str, Any], npu_payload: dict[str, Any]
) -> dict[str, Any]:
    cuda_attention = cuda_payload["attention"]
    npu_attention = npu_payload["attention"]
    valid = cuda_attention["valid_rows"].bool()
    npu_valid = npu_attention["valid_rows"].bool()
    if not torch.equal(valid, npu_valid):
        return {"valid_row_mask_equal": False}
    valid_values = valid.unsqueeze(-1).expand_as(cuda_attention["output"])
    invalid_values = ~valid_values
    return {
        "valid_row_mask_equal": True,
        "cross_device_output_valid_rows": _comparison_metrics(
            cuda_attention["output"][valid_values],
            npu_attention["output"][valid_values],
        ),
        "cross_device_output_fully_masked_rows": _comparison_metrics(
            cuda_attention["output"][invalid_values],
            npu_attention["output"][invalid_values],
        ),
        "cuda_output_vs_reference_valid_rows": _comparison_metrics(
            cuda_attention["reference"][valid_values],
            cuda_attention["output"][valid_values],
        ),
        "cuda_output_vs_reference_fully_masked_rows": _comparison_metrics(
            cuda_attention["reference"][invalid_values],
            cuda_attention["output"][invalid_values],
        ),
        "npu_output_vs_reference_valid_rows": _comparison_metrics(
            npu_attention["reference"][valid_values],
            npu_attention["output"][valid_values],
        ),
        "npu_output_vs_reference_fully_masked_rows": _comparison_metrics(
            npu_attention["reference"][invalid_values],
            npu_attention["output"][invalid_values],
        ),
    }


def compare(cuda_path: Path, npu_path: Path, report_path: Path | None) -> None:
    cuda_payload = _load_result(cuda_path)
    npu_payload = _load_result(npu_path)
    cuda_tensors: dict[str, torch.Tensor] = {}
    npu_tensors: dict[str, torch.Tensor] = {}
    _flatten("", cuda_payload, cuda_tensors)
    _flatten("", npu_payload, npu_tensors)
    rows = []
    for name in sorted(cuda_tensors.keys() & npu_tensors.keys()):
        rows.append({"name": name, **_comparison_metrics(cuda_tensors[name], npu_tensors[name])})
    report = {
        "cuda": str(cuda_path),
        "npu": str(npu_path),
        "missing_on_npu": sorted(cuda_tensors.keys() - npu_tensors.keys()),
        "missing_on_cuda": sorted(npu_tensors.keys() - cuda_tensors.keys()),
        "attention_rows": _attention_row_comparisons(cuda_payload, npu_payload),
        "tensors": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--fixture", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backend", choices=("cuda", "npu"), required=True)
    run_parser.add_argument("--device", type=int, default=0)
    run_parser.add_argument("--fixture", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument(
        "--phase",
        choices=("attention", "flow", "model", "trajectory", "all"),
        default="all",
    )
    run_parser.add_argument("--steps", type=int, default=5)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--cuda", type=Path, required=True)
    compare_parser.add_argument("--npu", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_fixture(args.fixture)
    elif args.command == "run":
        run(args)
    else:
        compare(args.cuda, args.npu, args.report)


if __name__ == "__main__":
    main()
