#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch_npu  # noqa: F401
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset_utils import get_dataloaders
from utils.sharded_ema import load_sharded_ema_checkpoint
from utils.utils import get_selfless_mask, load_model_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-sequence and incremental-backbone-cache image "
            "generation using the same checkpoint and initial noise."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ema_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--model_dtype",
        choices=("bf16", "fp32"),
        default="fp32",
        help=(
            "Use FP32 for semantic cache/full equivalence; BF16 may diverge "
            "because Q=full and Q=1/2 FlexAttention kernels have different "
            "reduction numerics."
        ),
    )
    parser.add_argument("--sampling_steps", type=int, default=10)
    parser.add_argument("--max_generation_steps", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument(
        "--mode",
        choices=("both", "full", "cache", "hidden"),
        default="both",
    )
    parser.add_argument(
        "--hidden_reveal_counts",
        default="0,1,32,128,255",
        help=(
            "Comma-separated visible-image-token counts for mode=hidden. "
            "Every count must leave at least one masked query token."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=2.0e-3)
    parser.add_argument("--rtol", type=float, default=2.0e-3)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def image_spans(token_types: torch.Tensor, image_tokens: int):
    spans = []
    for row in range(token_types.shape[0]):
        positions = (token_types[row] == 1).nonzero(as_tuple=True)[0]
        if positions.numel() != image_tokens:
            raise ValueError(
                f"row {row} has {positions.numel()} image tokens; "
                f"expected {image_tokens}"
            )
        start = int(positions[0].item())
        end = int(positions[-1].item()) + 1
        if end - start != image_tokens:
            raise ValueError(f"row {row} image span is not contiguous")
        spans.append((row, start, end))
    return spans


def _halton(index: int, base: int) -> float:
    value = 0.0
    scale = 1.0 / float(base)
    while index > 0:
        value += (index % base) * scale
        index //= base
        scale /= float(base)
    return value


def spatial_halton_order(side: int, device: torch.device) -> torch.Tensor:
    image_tokens = int(side) * int(side)
    seen = set()
    order = []
    index = 1
    while len(order) < image_tokens and index < image_tokens * 32:
        row = min(side - 1, int(_halton(index, 2) * side))
        col = min(side - 1, int(_halton(index, 3) * side))
        local_position = row * side + col
        if local_position not in seen:
            seen.add(local_position)
            order.append(local_position)
        index += 1
    order.extend(
        local_position
        for local_position in range(image_tokens)
        if local_position not in seen
    )
    return torch.tensor(order, device=device, dtype=torch.long)


def parse_reveal_counts(raw: str, image_tokens: int) -> list[int]:
    counts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not counts:
        raise ValueError("hidden_reveal_counts must not be empty")
    invalid = [count for count in counts if count < 0 or count >= image_tokens]
    if invalid:
        raise ValueError(
            "hidden reveal counts must lie in "
            f"[0, {image_tokens - 1}], got {invalid}"
        )
    return list(dict.fromkeys(counts))


@torch.no_grad()
def compare_single_dual_hidden(
    *,
    model,
    input_ids: torch.Tensor,
    token_types: torch.Tensor,
    sigma: torch.Tensor,
    image_latents: torch.Tensor,
    spans: list[tuple[int, int, int]],
    reveal_counts: list[int],
) -> dict:
    """Bitwise-compare B's mixed single stream with its dual-stream reference."""

    attention_contract = str(
        getattr(
            model.config,
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if attention_contract != "xlnet_content_diagonal":
        raise ValueError(
            "hidden equivalence mode requires B's "
            "xlnet_content_diagonal contract, got "
            f"{attention_contract!r}"
        )

    device = input_ids.device
    image_tokens = int(model.config.image_tokens_per_img)
    side = int(image_tokens**0.5)
    if side * side != image_tokens:
        raise ValueError(f"image_tokens_per_img={image_tokens} is not square")
    order = spatial_halton_order(side, device)
    selected_rows = torch.tensor(
        [row for row, _, _ in spans], device=device, dtype=torch.long
    )
    selected_input_ids = torch.index_select(input_ids, 0, selected_rows)
    selected_token_types = torch.index_select(token_types, 0, selected_rows)
    selected_sigma = torch.index_select(sigma, 0, selected_rows).float()
    selected_image_latents = torch.index_select(
        image_latents, 0, selected_rows
    ).to(dtype=next(model.parameters()).dtype)
    local_spans = [
        (sample_index, start, end)
        for sample_index, (_, start, end) in enumerate(spans)
    ]
    valid_rows = selected_token_types.ne(3)
    reports = []
    generation_hidden_all_exact = True
    mixed_sequence_all_exact = True

    for reveal_count in reveal_counts:
        generation_sigma = selected_sigma.clone()
        visible_image = torch.zeros_like(
            selected_token_types, dtype=torch.bool
        )
        work_latents = torch.zeros_like(selected_image_latents)
        current_positions = []
        visible_local_positions = order[:reveal_count]
        current_local_position = order[reveal_count]
        for sample_index, start, end in local_spans:
            original_image_sigma = selected_sigma[sample_index, start:end]
            start_order = float(original_image_sigma.min().item())
            visible_context = (
                selected_token_types[sample_index].ne(1)
                & selected_token_types[sample_index].ne(3)
                & selected_sigma[sample_index].lt(start_order)
            )
            if bool(visible_context.any().item()):
                start_order = max(
                    start_order,
                    float(
                        selected_sigma[sample_index, visible_context]
                        .max()
                        .item()
                        + 1.0
                    ),
                )
            generation_sigma[sample_index, start:end] = (
                start_order + image_tokens
            )
            if reveal_count:
                sequence_positions = start + visible_local_positions
                visible_image[sample_index, sequence_positions] = True
                generation_sigma[sample_index, sequence_positions] = (
                    start_order
                    + torch.arange(
                        reveal_count,
                        device=device,
                        dtype=generation_sigma.dtype,
                    )
                )
                work_latents[sample_index, sequence_positions] = (
                    selected_image_latents[sample_index, sequence_positions]
                )
            current_positions.append(start + current_local_position)

        content_rows = valid_rows & (
            selected_token_types.ne(1) | visible_image
        )
        query_mask = get_selfless_mask(
            sigma=generation_sigma,
            seq_len=selected_input_ids.shape[1],
            device=device,
        )
        content_mask = get_selfless_mask(
            sigma=generation_sigma,
            seq_len=selected_input_ids.shape[1],
            device=device,
            include_diagonal=True,
        )
        hybrid_mask = get_selfless_mask(
            sigma=generation_sigma,
            seq_len=selected_input_ids.shape[1],
            device=device,
            diagonal_query_mask=content_rows,
        )

        dual_layers = []
        handles = [
            layer.register_forward_hook(
                lambda _module, _inputs, output, storage=dual_layers: storage.append(
                    (output[0].detach().clone(), output[1].detach().clone())
                )
            )
            for layer in model.model.layers
        ]
        try:
            dual_query_final = model.model(
                X0_input_ids=selected_input_ids,
                attention_mask=query_mask,
                content_attention_mask=content_mask,
                token_types=selected_token_types,
                image_latents=work_latents,
                image_latent_mask=visible_image,
                image_reveal_sigma=generation_sigma,
                calculate_likelihood=True,
            ).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()

        hybrid_layers = []
        handles = [
            layer.register_forward_hook(
                lambda _module, _inputs, output, storage=hybrid_layers: storage.append(
                    output[0].detach().clone()
                )
            )
            for layer in model.model.layers
        ]
        try:
            hybrid_final = model.model(
                X0_input_ids=selected_input_ids,
                attention_mask=hybrid_mask,
                token_types=selected_token_types,
                image_latents=work_latents,
                image_latent_mask=visible_image,
                image_reveal_sigma=generation_sigma,
                calculate_likelihood=False,
            ).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()

        layer_reports = []
        batch_indices = torch.arange(
            len(local_spans), device=device, dtype=torch.long
        )
        current_positions_tensor = torch.tensor(
            current_positions, device=device, dtype=torch.long
        )
        for layer_index, (
            (dual_content, dual_query),
            hybrid_hidden,
        ) in enumerate(zip(dual_layers, hybrid_layers, strict=True)):
            reference = torch.where(
                content_rows.unsqueeze(-1), dual_content, dual_query
            )
            difference = (
                hybrid_hidden[valid_rows].float()
                - reference[valid_rows].float()
            ).abs()
            mismatch_count = int(torch.count_nonzero(difference).item())
            exact = mismatch_count == 0
            mixed_sequence_all_exact = mixed_sequence_all_exact and exact
            current_difference = (
                hybrid_hidden[
                    batch_indices, current_positions_tensor
                ].float()
                - dual_query[
                    batch_indices, current_positions_tensor
                ].float()
            ).abs()
            current_mismatch_count = int(
                torch.count_nonzero(current_difference).item()
            )
            current_exact = current_mismatch_count == 0
            generation_hidden_all_exact = (
                generation_hidden_all_exact and current_exact
            )
            layer_reports.append(
                {
                    "layer": layer_index,
                    "current_query_exact": current_exact,
                    "current_query_mismatch_count": current_mismatch_count,
                    "current_query_max_abs_error": float(
                        current_difference.max().item()
                    ),
                    "mixed_sequence_exact": exact,
                    "mixed_sequence_mismatch_count": mismatch_count,
                    "mixed_sequence_max_abs_error": float(
                        difference.max().item()
                    ),
                }
            )

        dual_content_final = model.model.norm(dual_layers[-1][0])
        final_reference = torch.where(
            content_rows.unsqueeze(-1),
            dual_content_final,
            dual_query_final,
        )
        final_difference = (
            hybrid_final[valid_rows].float()
            - final_reference[valid_rows].float()
        ).abs()
        final_mismatch_count = int(
            torch.count_nonzero(final_difference).item()
        )
        final_exact = final_mismatch_count == 0
        mixed_sequence_all_exact = mixed_sequence_all_exact and final_exact
        current_difference = (
            hybrid_final[batch_indices, current_positions_tensor].float()
            - dual_query_final[batch_indices, current_positions_tensor].float()
        ).abs()
        current_query_mismatch_count = int(
            torch.count_nonzero(current_difference).item()
        )
        current_query_exact = current_query_mismatch_count == 0
        generation_hidden_all_exact = (
            generation_hidden_all_exact and current_query_exact
        )
        reports.append(
            {
                "reveal_count": int(reveal_count),
                "current_local_position": int(current_local_position.item()),
                "layers": layer_reports,
                "current_query_all_layers_exact": all(
                    item["current_query_exact"] for item in layer_reports
                ),
                "current_query_final_exact": current_query_exact,
                "current_query_final_mismatch_count": (
                    current_query_mismatch_count
                ),
                "current_query_max_abs_error": float(
                    current_difference.max().item()
                ),
                "mixed_sequence_all_layers_exact": all(
                    item["mixed_sequence_exact"] for item in layer_reports
                ),
                "mixed_sequence_final_exact": final_exact,
                "mixed_sequence_final_mismatch_count": final_mismatch_count,
                "mixed_sequence_final_max_abs_error": float(
                    final_difference.max().item()
                ),
            }
        )

    return {
        "schema": "single_dual_hidden_equivalence_v2",
        "attention_contract": attention_contract,
        "comparison": (
            "generation-consumed single-stream query versus dual-stream XT"
        ),
        "exact_equality_required": True,
        "all_exact": bool(generation_hidden_all_exact),
        "mixed_sequence_all_exact": bool(mixed_sequence_all_exact),
        "mixed_sequence_note": (
            "Auxiliary only: unused dual X0/XT rows do not have to equal the "
            "single stream; generation consumes only the current query row."
        ),
        "samples": len(spans),
        "sequence_length": int(selected_input_ids.shape[1]),
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "reveal_stages": reports,
    }


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("This checkpoint validation requires Ascend NPU.")
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    config = OmegaConf.load(args.config)
    config.training.batch_size = int(args.batch_size)
    config.training.dataloader_workers = 0
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)

    model, tokenizer = load_model_tokenizer(
        config,
        model_dtype={
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[args.model_dtype],
    )
    checkpoint_report = load_sharded_ema_checkpoint(
        model,
        args.ema_checkpoint,
    )
    model = model.to(device=device).eval()
    _, val_loader = get_dataloaders(config, tokenizer)
    batch = next(iter(val_loader))
    input_ids = batch["input_ids"].to(device)
    token_types = batch["token_types"].to(device)
    sigma = batch["sigma"].to(device)
    image_latents = batch["image_latents"].to(device)
    image_tokens = int(config.model.image_tokens_per_img)
    latent_dim = int(config.model.image_latent_dim)
    spans = image_spans(token_types, image_tokens)

    if args.mode == "hidden":
        reveal_counts = parse_reveal_counts(
            args.hidden_reveal_counts, image_tokens
        )
        hidden_report = compare_single_dual_hidden(
            model=model,
            input_ids=input_ids,
            token_types=token_types,
            sigma=sigma,
            image_latents=image_latents,
            spans=spans,
            reveal_counts=reveal_counts,
        )
        hidden_report.update(
            {
                "config": str(Path(args.config).resolve()),
                "checkpoint": checkpoint_report,
                "device": torch.npu.get_device_name(device),
                "dtype": str(next(model.parameters()).dtype),
            }
        )
        payload = json.dumps(hidden_report, indent=2, sort_keys=True)
        print(payload)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        if not hidden_report["all_exact"]:
            raise SystemExit(1)
        return

    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(int(args.seed))
    initial_noise = torch.randn(
        len(spans),
        image_tokens,
        latent_dim,
        generator=noise_generator,
        dtype=torch.float32,
    ).to(device)

    common = {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "spans": spans,
        "initial_noise_bank": initial_noise,
        "flow_temperature": 1.0,
        "flow_cfg": float(args.cfg),
        "flow_cfg_schedule": "constant",
        "flow_solver": "euler",
        "flow_num_steps": int(args.sampling_steps),
        "parallel_rate": 1,
        "order_strategy": "spatial_halton",
        "return_trace": True,
        "_debug_max_generation_steps": (
            int(args.max_generation_steps)
            if int(args.max_generation_steps) > 0
            else None
        ),
    }

    def run(use_cache: bool, *, measured: bool = True):
        torch.npu.synchronize(device)
        baseline = torch.npu.memory_allocated(device)
        if measured:
            torch.npu.reset_peak_memory_stats(device)
        started = time.perf_counter()
        generated, trace = model.generate(
            "t2i",
            **common,
            use_cache=use_cache,
        )
        torch.npu.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_delta = (
            torch.npu.max_memory_allocated(device) - baseline
            if measured
            else 0
        )
        return generated, trace, elapsed, peak_delta

    if int(args.warmup_steps) > 0:
        previous_limit = common["_debug_max_generation_steps"]
        common["_debug_max_generation_steps"] = int(args.warmup_steps)
        if args.mode in {"both", "full"}:
            run(False, measured=False)
        if args.mode in {"both", "cache"}:
            run(True, measured=False)
        common["_debug_max_generation_steps"] = previous_limit

    if args.mode != "both":
        use_cache = args.mode == "cache"
        generated, trace, elapsed, peak_delta = run(use_cache)
        report = {
            "schema": "backbone_kv_cache_probe_v1",
            "config": str(Path(args.config).resolve()),
            "checkpoint": checkpoint_report,
            "device": torch.npu.get_device_name(device),
            "dtype": str(next(model.parameters()).dtype),
            "mode": str(args.mode),
            "batch_size": len(spans),
            "sampling_steps": int(args.sampling_steps),
            "max_generation_steps": int(args.max_generation_steps),
            "warmup_steps": int(args.warmup_steps),
            "cfg": float(args.cfg),
            "seconds": float(elapsed),
            "peak_delta_mib": float(peak_delta / (1024.0**2)),
            "generated_rms": float(
                generated.float().pow(2).mean().sqrt().item()
            ),
            "trace": {
                key: value
                for key, value in trace.items()
                if key.startswith("backbone_kv_cache")
            },
        }
        payload = json.dumps(report, indent=2, sort_keys=True)
        print(payload)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        return

    full, full_trace, full_seconds, full_peak_delta = run(False)
    cached, cached_trace, cached_seconds, cached_peak_delta = run(True)
    difference = (cached.float() - full.float()).abs()
    hidden_errors = {}
    cached_hidden_all_exact = True
    compared_hidden_streams = 0
    for stream in ("conditional", "unconditional"):
        trace_key = f"debug_{stream}_backbone_hidden"
        full_hidden = full_trace.get(trace_key)
        cached_hidden = cached_trace.get(trace_key)
        if isinstance(full_hidden, torch.Tensor) and isinstance(
            cached_hidden, torch.Tensor
        ):
            hidden_difference = (cached_hidden - full_hidden).abs()
            hidden_mismatch_count = int(
                torch.count_nonzero(hidden_difference).item()
            )
            hidden_exact = hidden_mismatch_count == 0
            cached_hidden_all_exact = cached_hidden_all_exact and hidden_exact
            compared_hidden_streams += 1
            hidden_errors[f"{stream}_hidden_exact"] = hidden_exact
            hidden_errors[f"{stream}_hidden_mismatch_count"] = (
                hidden_mismatch_count
            )
            hidden_errors[f"{stream}_hidden_max_abs_error"] = float(
                hidden_difference.max().item()
            )
            hidden_errors[f"{stream}_hidden_mean_abs_error"] = float(
                hidden_difference.mean().item()
            )
            hidden_errors[f"{stream}_hidden_reference_rms"] = float(
                full_hidden.float().pow(2).mean().sqrt().item()
            )
    hidden_errors["cached_hidden_all_exact"] = bool(
        compared_hidden_streams > 0 and cached_hidden_all_exact
    )
    hidden_errors["compared_hidden_streams"] = int(compared_hidden_streams)
    close = torch.allclose(
        cached.float(),
        full.float(),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    report = {
        "schema": "backbone_kv_cache_validation_v1",
        "config": str(Path(args.config).resolve()),
        "checkpoint": checkpoint_report,
        "device": torch.npu.get_device_name(device),
        "dtype": str(next(model.parameters()).dtype),
        "batch_size": len(spans),
        "sampling_steps": int(args.sampling_steps),
        "max_generation_steps": int(args.max_generation_steps),
        "warmup_steps": int(args.warmup_steps),
        "cfg": float(args.cfg),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "allclose": bool(close),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        **hidden_errors,
        "full_seconds": float(full_seconds),
        "cached_seconds": float(cached_seconds),
        "speedup": float(full_seconds / cached_seconds),
        "full_peak_delta_mib": float(full_peak_delta / (1024.0**2)),
        "cached_peak_delta_mib": float(cached_peak_delta / (1024.0**2)),
        "full_trace": {
            key: value
            for key, value in full_trace.items()
            if key.startswith("backbone_kv_cache")
        },
        "cached_trace": {
            key: value
            for key, value in cached_trace.items()
            if key.startswith("backbone_kv_cache")
        },
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    if not close:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
