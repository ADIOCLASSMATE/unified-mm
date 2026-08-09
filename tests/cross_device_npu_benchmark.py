"""Same-shape CUDA/NPU benchmark for the production Selfless-Flow workload.

The fixture is prepared once on CPU and copied to both machines.  Attention
and flow-head runs therefore use exactly the same tensors; the flow-head state
is also shared.  The full-model benchmark uses the same architecture and input
fixture, and uses real DDP when launched with torchrun.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch
from transformers import Qwen3Config


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SEED = 20260809


def _production_config(flow_mul: int) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=32768,
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
    config.image_latent_dim = 16
    config.image_tokens_per_img = 256
    config.image_flow_width = 1280
    config.image_flow_depth = 8
    config.image_flow_num_sampling_steps = "100"
    config.image_flow_batch_mul = int(flow_mul)
    config.image_flow_time_scale = 1000.0
    config.image_flow_time_sampling = "logit_normal"
    config.image_flow_logit_mean = 0.0
    config.image_flow_logit_std = 1.0
    config.image_flow_time_eps = 1.0e-5
    config.image_flow_time_uniform_mix = 0.1
    config.image_flow_solver = "heun"
    config.image_input_noise_strength = 1.0e-2
    config.image_uncond_prob = 0.1
    config.backbone_attention_output_gate = "none"
    config.use_flex_attention = True
    return config


def _randn(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).bfloat16()


def _stable_reinitialize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            value = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
            if name.endswith("norm.weight") or "layernorm.weight" in name:
                value.normal_(mean=1.0, std=0.02, generator=generator)
            elif parameter.ndim >= 2:
                value.normal_(mean=0.0, std=0.02, generator=generator)
            else:
                value.normal_(mean=0.0, std=0.02, generator=generator)
            parameter.copy_(value.to(dtype=parameter.dtype))


def _sigma(batch: int, seq: int, image_tokens: int, generator: torch.Generator) -> torch.Tensor:
    sigma = torch.ones(batch, seq, dtype=torch.float32)
    sigma[:, :2] = 0.0
    for row in range(batch):
        sigma[row, 2 : 2 + image_tokens] = (
            torch.randperm(image_tokens, generator=generator).float()
            / float(image_tokens - 1)
        )
    sigma[:, 2 + image_tokens] = 0.0
    return sigma


def prepare_fixture(path: Path, batch: int, flow_mul: int) -> None:
    from models.modeling_model.image_flow_loss import FlowLoss
    from models.modeling_model.image_position_utils import build_row_col_position_ids

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    seq, image_tokens = 320, 256
    sigma = _sigma(batch, seq, image_tokens, generator)
    padding = seq - (image_tokens + 4)
    input_ids = torch.tensor(
        [[21, 11, *([8] * image_tokens), 12, 2, *([0] * padding)]] * batch,
        dtype=torch.long,
    )
    token_types = torch.tensor(
        [[0, 2, *([1] * image_tokens), 2, 0, *([0] * padding)]] * batch,
        dtype=torch.uint8,
    )
    image_latents = torch.zeros(batch, seq, 16, dtype=torch.bfloat16)
    image_latents[:, 2 : 2 + image_tokens] = _randn(
        (batch, image_tokens, 16), generator
    )
    image_span_table = torch.tensor(
        [[row, 0, 2, 2 + image_tokens, 1] for row in range(batch)],
        dtype=torch.long,
    )
    image_local_positions = torch.tensor(
        [[-1, -1, *range(image_tokens), -1, -1, *([-1] * padding)]] * batch,
        dtype=torch.long,
    )
    labels = torch.full_like(input_ids, -100)
    position_ids = build_row_col_position_ids(token_types, image_tokens)

    attention = {
        "q": _randn((batch, 16, seq, 128), generator),
        "k": _randn((batch, 8, seq, 128), generator),
        "v": _randn((batch, 8, seq, 128), generator),
        "probe": _randn((batch, 16, seq, 128), generator),
        "sigma": sigma.clone(),
    }
    flow_batch = batch * flow_mul
    image_sigma = sigma[:, 2 : 2 + image_tokens].repeat(flow_mul, 1)
    flow = {
        "x": _randn((flow_batch, image_tokens, 16), generator),
        "t": torch.linspace(0.05, 0.95, image_tokens).repeat(flow_batch, 1),
        "z": _randn((flow_batch, image_tokens, 1024), generator),
        "context": _randn((flow_batch, image_tokens, 16), generator),
        "probe": _randn((flow_batch, image_tokens, 16), generator),
        "sigma": image_sigma,
        "positions": torch.arange(image_tokens).repeat(flow_batch, 1),
    }
    flow_module = FlowLoss(
        target_channels=16,
        z_channels=1024,
        depth=8,
        width=1280,
        num_sampling_steps=100,
        time_scale=1000.0,
        time_sampling="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        time_eps=1.0e-5,
        uniform_mix=0.1,
        solver="heun",
        image_tokens_per_img=256,
    ).bfloat16()
    _stable_reinitialize(flow_module, SEED + 1)
    payload = {
        "version": 1,
        "batch": batch,
        "flow_mul": flow_mul,
        "attention": attention,
        "flow": flow,
        "flow_state": {
            name: value.detach().cpu() for name, value in flow_module.state_dict().items()
        },
        "model": {
            "input_ids": input_ids,
            "token_types": token_types,
            "sigma": sigma,
            "image_latents": image_latents,
            "image_span_table": image_span_table,
            "image_local_positions": image_local_positions,
            "labels": labels,
            "position_ids": position_ids,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(json.dumps({"fixture": str(path), "bytes": path.stat().st_size}))


def prepare_model_state(fixture_path: Path, state_path: Path, flow_mul: int) -> None:
    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM

    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    torch.manual_seed(SEED + 2)
    model = Qwen3ForCausalLM(_production_config(flow_mul)).bfloat16()
    model.image_flow_head.load_state_dict(fixture["flow_state"], strict=True)
    payload = {
        "version": 1,
        "flow_mul": flow_mul,
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, state_path)
    print(json.dumps({"model_state": str(state_path), "bytes": state_path.stat().st_size}))


def _setup_backend(backend: str) -> tuple[torch.device, int, int, bool]:
    if backend == "npu":
        import torch_npu  # noqa: F401

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group(backend="hccl" if backend == "npu" else "nccl")
    if backend == "npu":
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    return device, rank, world, distributed


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    else:
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)
    else:
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gib(device: torch.device) -> float:
    if device.type == "npu":
        value = torch.npu.max_memory_allocated(device)
    else:
        value = torch.cuda.max_memory_allocated(device)
    return float(value / 1024**3)


def _device_name(device: torch.device) -> str:
    if device.type == "npu":
        return torch.npu.get_device_name(device)
    return torch.cuda.get_device_name(device)


def _reduce_max_seconds(seconds: float, device: torch.device, distributed: bool) -> float:
    # HCCL 2.6 does not support FP64 reductions; FP32 is ample for step timing.
    value = torch.tensor(seconds, device=device, dtype=torch.float32)
    if distributed:
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return float(value.item())


def _measure(
    step: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    distributed: bool,
    warmup: int,
    steps: int,
) -> tuple[list[float], float]:
    for _ in range(warmup):
        step()
    _sync(device)
    if distributed:
        torch.distributed.barrier()
    _reset_peak(device)
    times = []
    last_scalar = 0.0
    for _ in range(steps):
        if distributed:
            torch.distributed.barrier()
        _sync(device)
        started = time.perf_counter()
        value = step()
        _sync(device)
        elapsed = time.perf_counter() - started
        times.append(_reduce_max_seconds(elapsed, device, distributed))
        last_scalar = float(value.detach().float().item())
    return times, last_scalar


def _parameter_grad_summaries(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = []
    norms = []
    maxima = []
    means = []
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        names.append(name)
        norms.append(torch.linalg.vector_norm(grad).cpu())
        maxima.append(grad.abs().max().cpu())
        means.append(grad.mean().cpu())
    return {
        "names": names,
        "norms": torch.stack(norms),
        "absmax": torch.stack(maxima),
        "means": torch.stack(means),
    }


@contextlib.contextmanager
def _fixed_model_randomness(
    flow,
    t: torch.Tensor,
    noise: torch.Tensor,
    image_input_noise: torch.Tensor,
):
    import models.modeling_model.image_flow_loss as flow_module
    import models.modeling_model.modeling_selfless_flow as model_module

    original_sampler = flow._sample_times
    original_randn = flow_module.torch.randn
    original_randn_like = model_module.torch.randn_like

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

    def randn_like(input_tensor, *args, **kwargs):
        if tuple(input_tensor.shape) == tuple(image_input_noise.shape):
            return image_input_noise.to(
                device=kwargs.get("device", input_tensor.device),
                dtype=kwargs.get("dtype", input_tensor.dtype),
            )
        return original_randn_like(input_tensor, *args, **kwargs)

    flow._sample_times = sample_times
    flow_module.torch.randn = randn
    model_module.torch.randn_like = randn_like
    try:
        yield
    finally:
        flow._sample_times = original_sampler
        flow_module.torch.randn = original_randn
        model_module.torch.randn_like = original_randn_like


def _build_attention_step(fixture: dict[str, Any], device: torch.device):
    from models.modeling_model.modeling_selfless_flow import compiled_flex_attention
    from utils.utils import get_selfless_mask

    source = fixture["attention"]
    q = source["q"].to(device).requires_grad_(True)
    k = source["k"].to(device).requires_grad_(True)
    v = source["v"].to(device).requires_grad_(True)
    probe = source["probe"].to(device)
    sigma = source["sigma"].to(device)
    mask = get_selfless_mask(sigma, sigma.shape[1], device)
    scale = 1.0 / math.sqrt(q.shape[-1])
    captured = {}

    def step():
        q.grad = None
        k.grad = None
        v.grad = None
        output = compiled_flex_attention(q, k, v, mask, scale, True)
        scalar = (output.float() * probe.float()).mean()
        scalar.backward()
        captured["output"] = output
        return scalar

    def capture():
        return {
            "output": captured["output"].detach().cpu(),
            "q_grad": q.grad.detach().cpu(),
            "k_grad": k.grad.detach().cpu(),
            "v_grad": v.grad.detach().cpu(),
        }

    return step, capture


def _build_flow_step(fixture: dict[str, Any], device: torch.device):
    from models.modeling_model.image_flow_loss import FlowLoss

    flow = FlowLoss(
        target_channels=16,
        z_channels=1024,
        depth=8,
        width=1280,
        num_sampling_steps=100,
        time_scale=1000.0,
        time_sampling="logit_normal",
        logit_mean=0.0,
        logit_std=1.0,
        time_eps=1.0e-5,
        uniform_mix=0.1,
        solver="heun",
        image_tokens_per_img=256,
    ).bfloat16()
    flow.load_state_dict(fixture["flow_state"], strict=True)
    flow = flow.to(device).train()
    source = fixture["flow"]
    x = source["x"].to(device).requires_grad_(True)
    z = source["z"].to(device).requires_grad_(True)
    context = source["context"].to(device).requires_grad_(True)
    t = source["t"].to(device)
    sigma = source["sigma"].to(device)
    positions = source["positions"].to(device)
    probe = source["probe"].to(device)
    context_mask = sigma.unsqueeze(1) < sigma.unsqueeze(2)
    captured = {}

    def step():
        flow.zero_grad(set_to_none=True)
        x.grad = None
        z.grad = None
        context.grad = None
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
        captured["output"] = output
        return scalar

    def capture():
        return {
            "output": captured["output"].detach().cpu(),
            "x_grad": x.grad.detach().cpu(),
            "z_grad": z.grad.detach().cpu(),
            "context_grad": context.grad.detach().cpu(),
            "parameter_grads": _parameter_grad_summaries(flow),
        }

    return step, capture


def _build_model_step(
    fixture: dict[str, Any],
    device: torch.device,
    flow_mul: int,
    distributed: bool,
    local_rank: int,
    model_state: dict[str, Any] | None = None,
):
    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
    from utils.utils import get_selfless_mask

    torch.manual_seed(SEED)
    unwrapped_model = Qwen3ForCausalLM(_production_config(flow_mul)).bfloat16()
    if model_state is not None:
        unwrapped_model.load_state_dict(model_state["model_state"], strict=True)
    unwrapped_model = unwrapped_model.to(device).train()
    activation_tensors: dict[str, torch.Tensor] = {}
    handles = []
    if model_state is not None:
        def capture_activation(name: str):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                activation_tensors[name] = value.detach()

            return hook

        for index, layer in enumerate(unwrapped_model.model.layers):
            handles.append(
                layer.self_attn.register_forward_hook(
                    capture_activation(f"attention.{index}")
                )
            )
            handles.append(
                layer.register_forward_hook(capture_activation(f"decoder.{index}"))
            )
    model = unwrapped_model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=0.01
    )
    source = fixture["model"]
    inputs = {name: value.to(device) for name, value in source.items()}
    attention_mask = get_selfless_mask(
        inputs["sigma"], inputs["input_ids"].shape[1], device
    )
    deterministic_flow = fixture["flow"] if model_state is not None else None
    captured = {}

    def step():
        optimizer.zero_grad(set_to_none=True)
        randomness = (
            _fixed_model_randomness(
                unwrapped_model.image_flow_head,
                deterministic_flow["t"],
                deterministic_flow["x"],
                source["image_latents"],
            )
            if deterministic_flow is not None
            else contextlib.nullcontext()
        )
        with randomness:
            output = model(
                X0_input_ids=inputs["input_ids"],
                labels=inputs["labels"],
                attention_mask=attention_mask,
                position_ids=inputs["position_ids"],
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
        captured["loss"] = output.loss.detach()
        captured["last_hidden_state"] = output.last_hidden_state.detach()
        optimizer.step()
        return output.loss

    def capture():
        result = {
            "loss": captured["loss"].cpu(),
            "last_hidden_state": captured["last_hidden_state"].cpu(),
            "parameter_grads": _parameter_grad_summaries(unwrapped_model),
            "activations": {
                name: value.cpu() for name, value in activation_tensors.items()
            },
        }
        for handle in handles:
            handle.remove()
        return result

    return step, capture


def run(args: argparse.Namespace) -> None:
    device, rank, world, distributed = _setup_backend(args.backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if int(fixture["batch"]) != args.batch or int(fixture["flow_mul"]) != args.flow_mul:
        raise ValueError(
            f"fixture batch/flow_mul={(fixture['batch'], fixture['flow_mul'])}, "
            f"requested={(args.batch, args.flow_mul)}"
        )
    if args.component == "attention":
        step, capture = _build_attention_step(fixture, device)
    elif args.component == "flow":
        step, capture = _build_flow_step(fixture, device)
    else:
        model_state = (
            torch.load(args.model_state, map_location="cpu", weights_only=True)
            if args.model_state is not None
            else None
        )
        step, capture = _build_model_step(
            fixture,
            device,
            args.flow_mul,
            distributed,
            local_rank,
            model_state=model_state,
        )
    times, scalar = _measure(
        step,
        device=device,
        distributed=distributed,
        warmup=args.warmup,
        steps=args.steps,
    )
    mean_seconds = statistics.fmean(times)
    report = {
        "backend": args.backend,
        "component": args.component,
        "device_name": _device_name(device),
        "torch_version": str(torch.__version__),
        "world_size": world,
        "batch_per_rank": args.batch,
        "flow_mul": args.flow_mul,
        "warmup": args.warmup,
        "steps": args.steps,
        "times_seconds": times,
        "mean_seconds": mean_seconds,
        "median_seconds": statistics.median(times),
        "samples_per_second_per_rank": args.batch / mean_seconds,
        "samples_per_second_global": args.batch * world / mean_seconds,
        "peak_memory_gib": _peak_gib(device),
        "last_scalar": scalar,
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if args.capture is not None:
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            torch.save(capture(), args.capture)
        print(json.dumps(report, indent=2, sort_keys=True))
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--fixture", type=Path, required=True)
    prepare_parser.add_argument("--batch", type=int, default=16)
    prepare_parser.add_argument("--flow-mul", type=int, default=4)
    state_parser = subparsers.add_parser("prepare-model-state")
    state_parser.add_argument("--fixture", type=Path, required=True)
    state_parser.add_argument("--state", type=Path, required=True)
    state_parser.add_argument("--flow-mul", type=int, default=4)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backend", choices=("cuda", "npu"), required=True)
    run_parser.add_argument("--component", choices=("attention", "flow", "model"), required=True)
    run_parser.add_argument("--fixture", type=Path, required=True)
    run_parser.add_argument("--model-state", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--capture", type=Path)
    run_parser.add_argument("--batch", type=int, default=16)
    run_parser.add_argument("--flow-mul", type=int, default=4)
    run_parser.add_argument("--warmup", type=int, default=2)
    run_parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_fixture(args.fixture, args.batch, args.flow_mul)
    elif args.command == "prepare-model-state":
        prepare_model_state(args.fixture, args.state, args.flow_mul)
    else:
        run(args)


if __name__ == "__main__":
    main()
