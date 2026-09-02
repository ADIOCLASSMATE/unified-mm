#!/usr/bin/env python3
"""Generate images in the official GenEval, DPG-Bench, or MJHQ layout."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_single_stream_fid_is import (
    build_canonical_initial_noise_bank,
    decode_latents_in_microbatches,
    distributed_barrier,
    init_distributed,
    metric_images,
)
from scripts.official_t2i_benchmarks import (
    BenchmarkPrompt,
    benchmark_revision,
    expected_images_per_prompt,
    index_images,
    load_benchmark_prompts,
    require_exact_ids,
)
from utils.dataset_imagenet_flow_cache import DEFAULT_CAPTION_PREFIX
from utils.evaluation_model_source import (
    configure_model_source,
    load_model_source_weights,
    resolve_evaluation_model_source,
)
from utils.image_generation_io import load_vae
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.utils import load_model_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("geneval", "dpgbench", "mjhq"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model_source", type=Path, required=True)
    parser.add_argument(
        "--prompt_source",
        type=Path,
        required=True,
        help="Pinned official GenEval, ELLA, or MJHQ-30K git checkout.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--vae_dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--batch_size_per_rank", type=int, default=4)
    parser.add_argument("--vae_decode_batch_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_steps", default="10")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--cfg_schedule", choices=("constant", "linear"), default="constant")
    parser.add_argument("--flow_solver", choices=("heun", "euler"), default="heun")
    parser.add_argument("--parallel_rate", type=int, default=1)
    parser.add_argument("--strategy", default="spatial_halton")
    parser.add_argument(
        "--image_sigma_order", choices=("random", "sequential"), default="random"
    )
    parser.add_argument("--disable_backbone_kv_cache", action="store_true")
    return parser.parse_args()


def _dataset_params(config):
    if str(config.dataset.class_name) == "UnifiedMixedDataset":
        return config.dataset.params.image
    return config.dataset.params


def _build_t2i_item(
    prompt: str,
    *,
    tokenizer,
    config,
    prompt_prefix: str,
    task_index: int,
    image_sigma_order: str,
    seed: int,
) -> dict[str, Any]:
    serialized = f"{prompt_prefix} {prompt}".strip()
    text_ids = torch.tensor(
        tokenizer.encode(serialized, add_special_tokens=False), dtype=torch.long
    )
    image_tokens = int(config.model.image_tokens_per_img)
    latent_dim = int(config.model.image_latent_dim)
    input_ids = torch.cat(
        [
            text_ids,
            torch.tensor([int(config.model.boi_token_id)], dtype=torch.long),
            torch.full(
                (image_tokens,), int(config.model.mask_token_id), dtype=torch.long
            ),
            torch.tensor(
                [int(config.model.eoi_token_id), int(tokenizer.eos_token_id)],
                dtype=torch.long,
            ),
        ]
    )
    token_types = torch.cat(
        [
            torch.zeros(text_ids.numel(), dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
            torch.ones(image_tokens, dtype=torch.uint8),
            torch.tensor([2, 2], dtype=torch.uint8),
        ]
    )
    image_start = int(text_ids.numel()) + 1
    image_loss_mask = torch.zeros(input_ids.numel(), dtype=torch.bool)
    image_loss_mask[image_start : image_start + image_tokens] = True
    reveal_seed = (int(seed) + int(task_index)) % (2**63)
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "labels": torch.full_like(input_ids, -100),
        "image_loss_mask": image_loss_mask,
        "image_latents": torch.zeros(image_tokens, latent_dim),
        "prompt_len": torch.tensor(text_ids.numel(), dtype=torch.long),
        "suffix_len": torch.tensor(0, dtype=torch.long),
        "image_start": torch.tensor(image_start, dtype=torch.long),
        "img_id": torch.tensor(task_index, dtype=torch.long),
        "task_mode": "t2i",
        "reveal_seed": torch.tensor(reveal_seed, dtype=torch.long),
        "image_sigma_order": image_sigma_order,
    }


def _output_path(
    benchmark: str,
    output_dir: Path,
    prompt: BenchmarkPrompt,
    sample_index: int,
) -> Path:
    if benchmark == "geneval":
        return output_dir / "images" / prompt.prompt_id / "samples" / f"{sample_index:05d}.png"
    if benchmark == "dpgbench":
        return output_dir / "samples" / prompt.prompt_id / f"{sample_index:05d}.png"
    if prompt.category is None:
        raise ValueError("MJHQ prompt lacks a category")
    return output_dir / "images" / prompt.category / f"{prompt.prompt_id}.png"


def _prepare_output(
    benchmark: str, output_dir: Path, prompts: list[BenchmarkPrompt]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"benchmark output directory must be empty: {output_dir}"
        )
    (output_dir / "images").mkdir()
    if benchmark == "geneval":
        for prompt in prompts:
            prompt_dir = output_dir / "images" / prompt.prompt_id
            (prompt_dir / "samples").mkdir(parents=True)
            (prompt_dir / "metadata.jsonl").write_text(
                json.dumps(prompt.metadata) + "\n", encoding="utf-8"
            )
    elif benchmark == "dpgbench":
        for prompt in prompts:
            (output_dir / "samples" / prompt.prompt_id).mkdir(parents=True)
    else:
        for category in sorted({prompt.category for prompt in prompts}):
            if category is None:
                raise ValueError("MJHQ prompt lacks a category")
            (output_dir / "images" / category).mkdir(parents=True)


def _compose_dpg_grids(
    output_dir: Path, prompts: list[BenchmarkPrompt]
) -> tuple[int, int]:
    observed_size: tuple[int, int] | None = None
    for prompt in tqdm(prompts, desc="Composing official DPG-Bench grids"):
        paths = [
            output_dir / "samples" / prompt.prompt_id / f"{index:05d}.png"
            for index in range(4)
        ]
        images = [Image.open(path).convert("RGB") for path in paths]
        sizes = {image.size for image in images}
        if len(sizes) != 1:
            raise ValueError(f"DPG-Bench samples have mixed sizes: {prompt.prompt_id}")
        width, height = sizes.pop()
        if width != height:
            raise ValueError("official DPG-Bench crops must be square")
        if observed_size is None:
            observed_size = (width, height)
        elif observed_size != (width, height):
            raise ValueError("DPG-Bench prompts have mixed sample resolutions")
        grid = Image.new("RGB", (2 * width, 2 * height))
        for index, image in enumerate(images):
            grid.paste(image, ((index % 2) * width, (index // 2) * height))
            image.close()
        grid.save(output_dir / "images" / f"{prompt.prompt_id}.png")
        grid.close()
    if observed_size is None:
        raise ValueError("no DPG-Bench images were composed")
    return observed_size


def _validate_generated_layout(
    benchmark: str,
    output_dir: Path,
    prompts: list[BenchmarkPrompt],
    images_per_prompt: int,
) -> None:
    if benchmark == "geneval":
        for prompt in prompts:
            sample_dir = output_dir / "images" / prompt.prompt_id / "samples"
            expected = {f"{index:05d}" for index in range(images_per_prompt)}
            require_exact_ids(
                index_images(sample_dir), expected, f"GenEval prompt {prompt.prompt_id}"
            )
        return
    indexed = index_images(output_dir / "images")
    require_exact_ids(
        indexed,
        (prompt.prompt_id for prompt in prompts),
        f"{benchmark} generated images",
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size_per_rank <= 0:
        raise ValueError("--batch_size_per_rank must be positive")
    if args.vae_decode_batch_size < 0:
        raise ValueError("--vae_decode_batch_size must be nonnegative")

    distributed, rank, world_size, _, device = init_distributed(args.device)
    prompts = load_benchmark_prompts(args.benchmark, args.prompt_source)
    images_per_prompt = expected_images_per_prompt(args.benchmark)
    if rank == 0:
        _prepare_output(args.benchmark, args.output_dir, prompts)
    distributed_barrier(distributed, device)

    config = OmegaConf.load(args.config)
    config.training.runtime_hashing_enabled = False
    source = resolve_evaluation_model_source(args.model_source)
    configure_model_source(config, source)
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)
    model_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[
        args.model_dtype
    ]
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    source_report = load_model_source_weights(model, source)
    model = model.to(device).eval()
    vae = load_vae(config, device, args.vae_dtype)
    scaling_factor = float(config.experiment.validation_vae_scaling_factor)
    dataset_params = _dataset_params(config)
    if str(dataset_params.get("conditioning_mode", "")).strip().lower() != "caption":
        raise ValueError("official text-to-image benchmarks require caption conditioning")
    prompt_prefix = str(
        dataset_params.get("caption_t2i_prefix", DEFAULT_CAPTION_PREFIX)
    ).strip()
    max_length = int(
        dataset_params.get(
            "max_seq_length", config.dataset.preprocessing.max_seq_length
        )
    )

    flat_tasks = [
        (prompt.index * images_per_prompt + sample_index, prompt, sample_index)
        for prompt in prompts
        for sample_index in range(images_per_prompt)
    ]
    local_tasks = flat_tasks[rank::world_size]
    local_sizes: set[tuple[int, int]] = set()
    progress = tqdm(
        range(0, len(local_tasks), args.batch_size_per_rank),
        desc=f"Generating {args.benchmark}",
        disable=rank != 0,
    )
    for start in progress:
        chunk = local_tasks[start : start + args.batch_size_per_rank]
        items = [
            _build_t2i_item(
                prompt.prompt,
                tokenizer=tokenizer,
                config=config,
                prompt_prefix=prompt_prefix,
                task_index=task_index,
                image_sigma_order=args.image_sigma_order,
                seed=args.seed,
            )
            for task_index, prompt, _ in chunk
        ]
        if any(item["input_ids"].numel() > max_length for item in items):
            raise ValueError(
                "a benchmark prompt exceeds dataset.preprocessing.max_seq_length"
            )
        batch = collate_imagenet_flow_cache(items)
        spans = [
            (int(row[0]), int(row[2]), int(row[3]))
            for row in batch["image_span_table"].tolist()
        ]
        input_ids = batch["input_ids"].to(device)
        token_types = batch["token_types"].to(device)
        sigma = batch["sigma"].to(device)
        task_indices = [task_index for task_index, _, _ in chunk]
        initial_noise_bank, _ = build_canonical_initial_noise_bank(
            task_indices, evaluation_seed=args.seed
        )
        generated_latents, _ = model.generate(
            "t2i",
            input_ids=input_ids,
            token_types=token_types,
            sigma=sigma,
            spans=spans,
            segment_ids=None,
            image_latent_dim=int(config.model.image_latent_dim),
            initial_noise_bank=initial_noise_bank,
            flow_temperature=float(args.temperature),
            flow_cfg=float(args.cfg),
            flow_cfg_schedule=str(args.cfg_schedule),
            flow_solver=str(args.flow_solver),
            parallel_rate=int(args.parallel_rate),
            order_strategy=str(args.strategy),
            use_cache=not args.disable_backbone_kv_cache,
            return_trace=True,
        )
        images = metric_images(
            decode_latents_in_microbatches(
                vae,
                generated_latents.float(),
                scaling_factor,
                batch_size=args.vae_decode_batch_size,
            )
        )
        if not torch.isfinite(images).all():
            raise FloatingPointError("generated benchmark images contain NaN/Inf")
        for image, (_, prompt, sample_index) in zip(images, chunk):
            path = _output_path(
                args.benchmark, args.output_dir, prompt, sample_index
            )
            save_image(image, path)
            local_sizes.add((int(image.shape[-1]), int(image.shape[-2])))

    distributed_barrier(distributed, device)
    all_sizes: list[set[tuple[int, int]]] = [set() for _ in range(world_size)]
    if distributed:
        dist.all_gather_object(all_sizes, local_sizes)
    else:
        all_sizes[0] = local_sizes
    observed_sizes = set().union(*all_sizes)
    if len(observed_sizes) != 1:
        raise ValueError(f"generated images have mixed resolutions: {observed_sizes}")
    native_resolution = observed_sizes.pop()

    if rank == 0:
        if args.benchmark == "dpgbench":
            composed_resolution = _compose_dpg_grids(args.output_dir, prompts)
            if composed_resolution != native_resolution:
                raise ValueError("DPG-Bench grid resolution validation failed")
        _validate_generated_layout(
            args.benchmark, args.output_dir, prompts, images_per_prompt
        )
        manifest = {
            "schema": "official_t2i_benchmark_generation_v1",
            "complete": True,
            "benchmark": args.benchmark,
            "official_source": {
                "repository": str(args.prompt_source.resolve()),
                "commit": benchmark_revision(args.benchmark),
            },
            "model_source": source_report,
            "prompt_count": len(prompts),
            "images_per_prompt": images_per_prompt,
            "generated_image_count": len(flat_tasks),
            "native_resolution": list(native_resolution),
            "model_input_prefix": prompt_prefix,
            "generation": {
                "seed": args.seed,
                "noise_seed_formula": "seed + prompt_index * images_per_prompt + sample_index",
                "sampling_steps": str(args.sampling_steps),
                "temperature": args.temperature,
                "cfg": args.cfg,
                "cfg_schedule": args.cfg_schedule,
                "flow_solver": args.flow_solver,
                "parallel_rate": args.parallel_rate,
                "strategy": args.strategy,
                "image_sigma_order": args.image_sigma_order,
                "model_dtype": args.model_dtype,
                "vae_dtype": args.vae_dtype,
            },
            "completed_at": datetime.now(UTC).isoformat().replace(
                "+00:00", "Z"
            ),
        }
        (args.output_dir / "generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    distributed_barrier(distributed, device)


if __name__ == "__main__":
    main()
