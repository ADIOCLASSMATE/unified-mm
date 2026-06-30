#!/usr/bin/env python3
"""Regression check for the refactored selfless-flow text forward path.

This script intentionally stays out of the default pytest suite because it loads
a large local checkpoint and FineWeb-Edu Arrow shards. It verifies:
  1. Text-only loss path used by pretrain/train_selfless_text.py.
  2. Flow text-batch loss path where token_types are present.
  3. Eval-time single-stream text decoding with calculate_likelihood=False.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.dataset_combined_flow import TextArrowDataset, collate_text_arrow
from utils.utils import get_selfless_mask


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "output/selfless-diffusion-0.6B-text2048-finewebedu-adapt/hf_model-final"
)
DEFAULT_DATASET = REPO_ROOT / "public/.cache/huggingface/datasets/fwb-arrow-eos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-loss-batches", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--rows-per-shard", type=int, default=48804)
    parser.add_argument("--pad-to-multiple-of", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-loss", type=float, default=8.0)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_model(checkpoint: Path, device: torch.device, dtype_name: str) -> tuple[Qwen3ForCausalLM, AutoTokenizer]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    dtype = resolve_dtype(dtype_name)
    load_kwargs = {"trust_remote_code": True}
    if dtype is not None:
        load_kwargs["dtype"] = dtype

    model = Qwen3ForCausalLM.from_pretrained(str(checkpoint), **load_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def tokenizer_pad_id(tokenizer, model: Qwen3ForCausalLM) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = getattr(model.config, "eos_token_id", None)
    return int(0 if pad_id is None else pad_id)


def build_loader(args: argparse.Namespace, tokenizer, model: Qwen3ForCausalLM) -> DataLoader:
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")

    needed = max(1, args.batch_size * args.num_loss_batches)
    dataset = TextArrowDataset(
        tokenized_path=str(args.dataset_path),
        max_seq_length=args.max_seq_length,
        pad_token_id=tokenizer_pad_id(tokenizer, model),
        sigma_mode="ar",
        seed=args.seed,
        max_samples=needed,
        rows_per_shard=args.rows_per_shard,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_text_arrow(
            batch,
            pad_token_id=tokenizer_pad_id(tokenizer, model),
            pad_to_multiple_of=args.pad_to_multiple_of,
        ),
    )


@torch.no_grad()
def evaluate_loss(
    model: Qwen3ForCausalLM,
    loader: DataLoader,
    device: torch.device,
    include_token_types: bool,
) -> dict[str, float]:
    loss_sum = torch.tensor(0.0, device=device)
    token_sum = torch.tensor(0.0, device=device)
    batch_losses: list[float] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        sigma = batch["sigma"].to(device, non_blocking=True)
        attention_mask = get_selfless_mask(
            sigma=sigma,
            seq_len=input_ids.shape[1],
            device=device,
        )
        kwargs = {}
        if include_token_types:
            kwargs["token_types"] = batch["token_types"].to(device, non_blocking=True)

        output = model(
            X0_input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            calculate_likelihood=True,
            **kwargs,
        )
        loss = output.loss.detach().float()
        valid_tokens = (labels != -100).sum().float().to(device)
        loss_sum += loss * valid_tokens
        token_sum += valid_tokens
        batch_losses.append(float(loss.item()))

    avg_loss = float((loss_sum / token_sum.clamp_min(1.0)).item())
    return {
        "loss": avg_loss,
        "ppl": float(math.exp(avg_loss)) if avg_loss < 100 else float("inf"),
        "tokens": float(token_sum.item()),
        "batch_min": min(batch_losses),
        "batch_max": max(batch_losses),
    }


def suppress_special_logits(logits: torch.Tensor, model: Qwen3ForCausalLM) -> torch.Tensor:
    logits = logits.clone()
    for attr in ("mask_token_id", "boi_token_id", "eoi_token_id", "image_mask_token_id"):
        token_id = getattr(model.config, attr, None)
        if token_id is not None and 0 <= int(token_id) < logits.numel():
            logits[int(token_id)] = -torch.inf
    return logits


def sample_next_token(
    logits: torch.Tensor,
    model: Qwen3ForCausalLM,
    temperature: float,
    top_k: int,
) -> int:
    logits = suppress_special_logits(logits.float(), model)
    if temperature <= 1e-6:
        return int(torch.argmax(logits).item())

    logits = logits / float(temperature)
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, k=top_k)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_single_stream(
    model: Qwen3ForCausalLM,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, object]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].to(device=device, dtype=torch.long)
    if prompt_ids.numel() == 0:
        fallback_id = tokenizer.eos_token_id or getattr(model.config, "eos_token_id", 0) or 0
        prompt_ids = torch.tensor([int(fallback_id)], device=device, dtype=torch.long)

    seq = prompt_ids.clone()
    eos_token_id = getattr(model.config, "eos_token_id", tokenizer.eos_token_id)
    mask_token_id = int(getattr(model.config, "mask_token_id"))

    for _ in range(max_new_tokens):
        candidate = torch.cat(
            [seq, torch.tensor([mask_token_id], device=device, dtype=torch.long)]
        )
        sigma = torch.arange(candidate.numel(), device=device, dtype=torch.float32).unsqueeze(0)
        attention_mask = get_selfless_mask(
            sigma=sigma,
            seq_len=candidate.numel(),
            device=device,
        )
        output = model(
            X0_input_ids=candidate.unsqueeze(0),
            attention_mask=attention_mask,
            calculate_likelihood=False,
        )
        logits = output.logits[0, -1]
        next_token = sample_next_token(logits, model, temperature, top_k)
        seq = torch.cat([seq, torch.tensor([next_token], device=device, dtype=torch.long)])
        if eos_token_id is not None and next_token == int(eos_token_id):
            break

    generated_ids = seq[prompt_ids.numel() :].detach().cpu().tolist()
    return {
        "prompt": prompt,
        "generated_token_count": len(generated_ids),
        "generated_ids": generated_ids,
        "text": tokenizer.decode(seq.detach().cpu().tolist(), skip_special_tokens=True),
    }


def assert_loss_ok(name: str, metrics: dict[str, float], max_loss: float) -> None:
    loss = metrics["loss"]
    if not math.isfinite(loss):
        raise AssertionError(f"{name} loss is not finite: {loss}")
    if loss <= 0.0:
        raise AssertionError(f"{name} loss must be positive, got {loss}")
    if loss > max_loss:
        raise AssertionError(f"{name} loss {loss:.4f} exceeds --max-loss {max_loss:.4f}")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device)
    model, tokenizer = load_model(args.checkpoint, device, args.dtype)
    loader = build_loader(args, tokenizer, model)

    text_only_metrics = evaluate_loss(model, loader, device, include_token_types=False)
    loader = build_loader(args, tokenizer, model)
    flow_text_metrics = evaluate_loss(model, loader, device, include_token_types=True)

    assert_loss_ok("text_only", text_only_metrics, args.max_loss)
    assert_loss_ok("flow_text", flow_text_metrics, args.max_loss)

    prompts = args.prompt or [
        "The main idea behind photosynthesis is",
        "In a classroom, a science teacher explains that",
    ]
    generations = [
        generate_single_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        for prompt in prompts
    ]

    for item in generations:
        if item["generated_token_count"] <= 0:
            raise AssertionError(f"Generation produced no tokens for prompt: {item['prompt']!r}")

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_path": str(args.dataset_path),
        "device": str(device),
        "dtype": args.dtype,
        "max_seq_length": args.max_seq_length,
        "num_loss_batches": args.num_loss_batches,
        "batch_size": args.batch_size,
        "text_only": text_only_metrics,
        "flow_text": flow_text_metrics,
        "generations": generations,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
