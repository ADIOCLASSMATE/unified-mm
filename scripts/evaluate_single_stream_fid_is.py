#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_qwen_showo_fid_is import (  # noqa: E402
    FeatureMoments,
    InceptionScoreMoments,
    build_inception_extractor,
    extract_inception_features,
    frechet_distance,
)
from scripts.generate_flow_validation_images import (  # noqa: E402
    decode_latents,
    load_adapter,
    load_ema_state,
    load_model_state,
    load_vae,
)
from utils.dataset_utils import get_dataloaders  # noqa: E402
from utils.utils import load_model_tokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INCEPTION_WEIGHTS = (
    REPO_ROOT
    / "output"
    / "cache"
    / "inception"
    / "weights-inception-2015-12-05-6726825d.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline FID/IS evaluation for selfless single-stream image generation."
    )
    parser.add_argument("--config", default="configs/selfless/imagenet_flow_full_from_qwen3base.yaml")
    parser.add_argument("--model_path_override", default="")
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--model_state", default="")
    parser.add_argument("--ema_state", default="")
    parser.add_argument("--output_dir", default="output/single_stream_fid_is")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model_dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Floating-point dtype used to load and execute the generation model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--sampling_steps", default="50")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--cfg_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--flow_solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--parallel_rate", type=int, default=1)
    parser.add_argument("--strategies", default="spatial_halton")
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--fid_feature", type=int, default=2048)
    parser.add_argument("--is_splits", type=int, default=10)
    parser.add_argument(
        "--inception_weights_path",
        default=os.environ.get(
            "TORCH_FIDELITY_INCEPTION_WEIGHTS",
            str(DEFAULT_INCEPTION_WEIGHTS) if DEFAULT_INCEPTION_WEIGHTS.exists() else "",
        ),
        help="Optional local torch-fidelity InceptionV3 weights path for FID/IS.",
    )
    parser.add_argument(
        "--real_source",
        choices=["vae_decoded_target_latents", "imagenet_original"],
        default="vae_decoded_target_latents",
        help=(
            "Reference distribution for FID. The default compares against VAE-decoded target latents; "
            "imagenet_original loads original ImageNet files from the manifest/source_path mapping."
        ),
    )
    parser.add_argument(
        "--real_stats_path",
        default="",
        help=(
            "Precomputed original-ImageNet Inception moments shared across architectures. "
            "When set, this overrides --real_source and real images are not re-extracted."
        ),
    )
    parser.add_argument(
        "--imagenet_train_dir",
        default="/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train",
        help="Root used to resolve manifest source paths when --real_source=imagenet_original.",
    )
    parser.add_argument(
        "--real_image_size",
        type=int,
        default=256,
        help="Resize/center-crop size for original ImageNet real images.",
    )
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument(
        "--allow_sigma_strategies",
        action="store_true",
        help="Allow sigma/sigma_replay strategies. Disabled by default because real generation cannot know training sigma.",
    )
    parser.add_argument(
        "--require_official_protocol",
        action="store_true",
        help=(
            "Fail unless shared real stats, the full matching fake sample count, "
            "and 10 deterministic IS splits are used."
        ),
    )
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def init_distributed(requested_device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    distributed_env = {
        key: os.environ.get(key)
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        if os.environ.get(key) is not None
    }
    if not distributed and (rank != 0 or local_rank != 0):
        raise RuntimeError(
            "Inconsistent distributed environment: got nonzero RANK/LOCAL_RANK "
            f"but WORLD_SIZE={world_size}. Launch with torchrun/torch.distributed.run "
            f"so every rank receives WORLD_SIZE, or unset rank env vars. env={distributed_env}"
        )

    requested = str(requested_device).lower()
    if requested != "cpu" and torch.cuda.is_available():
        if distributed or requested in {"auto", "cuda"}:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(requested_device)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return distributed, rank, world_size, local_rank, device


def token_type_debug(token_types: torch.Tensor, image_tokens_per_img: int) -> dict:
    token_types_cpu = token_types.detach().cpu()
    image_counts = (token_types_cpu == 1).sum(dim=1).tolist()
    unique_values, unique_counts = torch.unique(token_types_cpu, return_counts=True)
    run_lengths = []
    for row in token_types_cpu:
        pos = 0
        while pos < int(row.numel()):
            if int(row[pos].item()) != 1:
                pos += 1
                continue
            start = pos
            while pos < int(row.numel()) and int(row[pos].item()) == 1:
                pos += 1
            run_lengths.append(pos - start)
    return {
        "shape": list(token_types_cpu.shape),
        "unique_token_types": {
            str(int(value.item())): int(count.item())
            for value, count in zip(unique_values, unique_counts)
        },
        "image_token_counts_first8": [int(value) for value in image_counts[:8]],
        "image_token_count_min": int(min(image_counts)) if image_counts else None,
        "image_token_count_max": int(max(image_counts)) if image_counts else None,
        "image_run_lengths_first8": [int(value) for value in run_lengths[:8]],
        "complete_run_count": int(sum(1 for value in run_lengths if value == image_tokens_per_img)),
    }


def distributed_barrier(distributed: bool, device: torch.device):
    if not distributed or not (dist.is_available() and dist.is_initialized()):
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[int(device.index or 0)])
    else:
        dist.barrier()


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def reduce_max(value: float | None, device: torch.device) -> float | None:
    raw_value = -float("inf") if value is None else float(value)
    tensor = torch.tensor(raw_value, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    reduced = float(tensor.item())
    return None if reduced == -float("inf") else reduced


def image_spans(token_types: torch.Tensor, image_tokens_per_img: int):
    spans = []
    bsz, seq_len = token_types.shape
    for b in range(bsz):
        pos = 0
        while pos < seq_len:
            if int(token_types[b, pos].item()) != 1:
                pos += 1
                continue
            start = pos
            while pos < seq_len and int(token_types[b, pos].item()) == 1:
                pos += 1
            end = pos
            if end - start == image_tokens_per_img:
                spans.append((b, start, end))
    return spans


def span_latents_to_chw(
    image_latents: torch.Tensor,
    spans: list[tuple[int, int, int]],
    side: int,
) -> torch.Tensor:
    latents = []
    for b, start, end in spans:
        latents.append(image_latents[b, start:end].view(side, side, -1).permute(2, 0, 1))
    return torch.stack(latents)


def metric_images(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().clamp(0.0, 1.0)


def save_batch_images(images: torch.Tensor, directory: Path, offset: int):
    directory.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        save_image(image, directory / f"{offset + idx:08d}.png")


def save_indexed_images(images: torch.Tensor, directory: Path, indices: list[int]):
    directory.mkdir(parents=True, exist_ok=True)
    for image, index in zip(images, indices):
        save_image(image, directory / f"{int(index):08d}.png")


def get_base_dataset_and_indices(loader_dataset):
    if hasattr(loader_dataset, "dataset") and hasattr(loader_dataset, "indices"):
        return loader_dataset.dataset, list(loader_dataset.indices)
    return loader_dataset, None


def source_paths_for_spans(loader_dataset, batch_offset: int, spans, imagenet_train_dir: Path):
    base_dataset, subset_indices = get_base_dataset_and_indices(loader_dataset)
    if not hasattr(base_dataset, "img_ids") or not hasattr(base_dataset, "source_paths"):
        raise ValueError("--real_source=imagenet_original requires an ImageNetFlowCacheDataset or Subset of it.")

    paths = []
    for batch_row, _, _ in spans:
        dataset_row = batch_offset + int(batch_row)
        if subset_indices is not None:
            dataset_row = int(subset_indices[dataset_row])
        img_id = int(base_dataset.img_ids[dataset_row].item())
        source_path = base_dataset.source_paths.get(img_id)
        if not source_path:
            raise KeyError(f"No source_path in manifest for img_id={img_id}")
        path = Path(source_path)
        if not path.is_absolute():
            path = imagenet_train_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def build_real_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
        ]
    )


def load_real_images(paths, transform, device):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images, dim=0).to(device=device, dtype=torch.float32)


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def checkpoint_weight_dtypes(model_path: str | Path) -> list[str]:
    weights_path = Path(model_path) / "model.safetensors"
    dtype_names = {
        "BF16": "bf16",
        "F32": "fp32",
        "F16": "fp16",
    }
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        return sorted(
            {
                dtype_names.get(
                    str(handle.get_slice(key).get_dtype()),
                    str(handle.get_slice(key).get_dtype()).lower(),
                )
                for key in handle.keys()
            }
        )


def is_official_flow_protocol(
    *,
    shared_real_count: int | None,
    samples: int,
    is_splits: int,
    parallel_rate: int,
) -> bool:
    return bool(
        shared_real_count is not None
        and int(samples) == int(shared_real_count)
        and int(is_splits) == 10
        and int(parallel_rate) == 1
    )


def validate_strategies(strategies: list[str], allow_sigma_strategies: bool) -> None:
    if allow_sigma_strategies:
        return
    forbidden = {"sigma", "sigma_replay", "causal_sigma"}
    found = sorted({strategy.lower() for strategy in strategies} & forbidden)
    if found:
        raise ValueError(
            "Refusing sigma-based generation strategies for real evaluation: "
            f"{found}. These strategies require training/data sigma. "
            "Pass --allow_sigma_strategies only for debugging."
        )


def resolve_inception_weights_path(path: str) -> str | None:
    if not path:
        return None
    weights_path = Path(path).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"--inception_weights_path does not exist: {weights_path}")
    return str(weights_path)


def shared_feature_moments(payload, *, feature: int, device) -> FeatureMoments:
    stats = payload["stats"]
    count = int(stats["count"])
    feature_sum = torch.as_tensor(stats["sum"], dtype=torch.float64, device=device)
    outer_sum = torch.as_tensor(
        stats["outer_sum"],
        dtype=torch.float64,
        device=device,
    )
    if tuple(feature_sum.shape) != (int(feature),):
        raise ValueError(
            f"shared real feature sum shape={tuple(feature_sum.shape)}; "
            f"expected={(int(feature),)}"
        )
    if tuple(outer_sum.shape) != (int(feature), int(feature)):
        raise ValueError(
            f"shared real outer-sum shape={tuple(outer_sum.shape)}; "
            f"expected={(int(feature), int(feature))}"
        )
    moments = FeatureMoments.zeros(int(feature), device)
    moments.count.fill_(count)
    moments.sum.copy_(feature_sum)
    moments.outer_sum.copy_(outer_sum)
    return moments


def load_shared_original_real_stats(
    path: str,
    *,
    config,
    fid_feature: int,
    real_image_size: int,
    inception_weights_path: str,
):
    from scripts.evaluate_qwen_showo_fid_is import (
        build_expected_real_metadata,
        feature_metadata,
        load_fixed_val_records,
        load_manifest,
        load_synset_names,
        metric_transform_metadata,
        validate_real_stats_metadata,
    )

    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    manifest_path = Path(config.dataset.params.manifest_jsonl)
    split_manifest_path = Path(config.dataset.params.split_manifest_jsonl)
    synset_mapping_path = Path(config.dataset.params.synset_mapping_path)
    records = load_fixed_val_records(
        load_manifest(manifest_path),
        split_manifest_path,
        load_synset_names(synset_mapping_path),
        expected_classes=int(config.dataset.params.get("num_classes", 100)),
        expected_samples_per_class=int(
            config.dataset.params.get("val_samples_per_class", 100)
        ),
    )
    payload = torch.load(stats_path, map_location="cpu")
    expected = build_expected_real_metadata(
        manifest_path=manifest_path,
        split_manifest_path=split_manifest_path,
        selected_records=records,
        transform=metric_transform_metadata(int(real_image_size)),
        feature=feature_metadata(int(fid_feature), inception_weights_path),
        val_samples_per_class=int(
            config.dataset.params.get("val_samples_per_class", 100)
        ),
        split_seed=int(config.dataset.params.get("split_seed", 42)),
    )
    validate_real_stats_metadata(payload["metadata"], expected)
    if int(payload["stats"]["count"]) != len(records):
        raise ValueError(
            f"shared real stats count={payload['stats']['count']} but split has "
            f"{len(records)} images"
        )
    return payload


def warm_inception_cache_if_needed(
    args,
    distributed: bool,
    rank: int,
    device: torch.device,
    weights_path: str | None,
):
    if weights_path is not None:
        return
    if not distributed or is_main_process(rank):
        extractor = build_inception_extractor(
            int(args.fid_feature),
            weights_path,
            device,
        )
        del extractor
    distributed_barrier(distributed, device)


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    distributed, rank, world_size, local_rank, device = init_distributed(args.device)
    if int(args.samples) < int(world_size):
        raise ValueError(
            f"--samples={args.samples} must be at least world_size={world_size}"
        )
    if int(args.samples) < int(args.is_splits):
        raise ValueError(
            f"--samples={args.samples} must be at least --is_splits={args.is_splits}"
        )
    progress = not args.no_progress and is_main_process(rank)
    out_dir = Path(args.output_dir)
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed, device)

    torch.manual_seed(args.seed + rank * 100_003)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if not strategies:
        raise ValueError("--strategies must contain at least one strategy")
    validate_strategies(strategies, args.allow_sigma_strategies)

    config = OmegaConf.load(args.config)
    if args.model_path_override:
        config.model.model_path = args.model_path_override
    config.training.batch_size = int(args.batch_size)
    config.training.dataloader_workers = 0
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)

    if is_main_process(rank):
        print(
            f"Distributed evaluation: world_size={world_size}, "
            f"device={device}, strategies={strategies}"
        )
        print("Loading model/tokenizer...")
    requested_model_dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.model_dtype]
    stored_checkpoint_dtypes = checkpoint_weight_dtypes(config.model.model_path)
    model, tokenizer = load_model_tokenizer(
        config,
        model_dtype=requested_model_dtype,
    )
    if is_main_process(rank):
        print(f"Loading adapter: {args.adapter}")
    adapter_report = load_adapter(model, args.adapter)
    if is_main_process(rank):
        print(f"Loading model state: {args.model_state or 'none'}")
    model_state_report = load_model_state(model, args.model_state)
    if is_main_process(rank):
        print(f"Loading EMA state: {args.ema_state or 'none'}")
    ema_state_report = load_ema_state(model, args.ema_state)
    model = model.to(device).eval()
    parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    )
    expected_parameter_dtype = str(requested_model_dtype)
    if parameter_dtypes != [expected_parameter_dtype]:
        raise RuntimeError(
            "generation model contains unexpected floating parameter dtypes: "
            f"requested={expected_parameter_dtype}, actual={parameter_dtypes}"
        )

    if is_main_process(rank):
        print("Loading KL16 VAE...")
    vae = load_vae(config, device, args.vae_dtype)
    scaling_factor = float(config.experiment.validation_vae_scaling_factor)

    if is_main_process(rank):
        print(f"Loading {args.split} dataloader...")
    train_loader, val_loader = get_dataloaders(config, tokenizer)
    loader = val_loader if args.split == "val" else train_loader
    if args.real_source == "imagenet_original" and args.split != "val":
        raise ValueError("--real_source=imagenet_original currently expects --split val because train loader is shuffled.")
    real_transform = build_real_image_transform(args.real_image_size)
    imagenet_train_dir = Path(args.imagenet_train_dir)

    image_tokens = int(config.model.image_tokens_per_img)
    side = int(image_tokens ** 0.5)
    if side * side != image_tokens:
        raise ValueError(f"image_tokens_per_img={image_tokens} is not a square grid")

    inception_weights_path = resolve_inception_weights_path(args.inception_weights_path)
    real_stats_path = str(
        args.real_stats_path
        or config.get("evaluation", {}).get("real_stats_path", "")
    )
    shared_real_payload = None
    if real_stats_path:
        if inception_weights_path is None:
            raise ValueError(
                "shared original-image stats require an explicit local "
                "--inception_weights_path so its content hash can be verified"
            )
        shared_real_payload = load_shared_original_real_stats(
            real_stats_path,
            config=config,
            fid_feature=int(args.fid_feature),
            real_image_size=int(args.real_image_size),
            inception_weights_path=inception_weights_path,
        )
    shared_real_count = (
        int(shared_real_payload["stats"]["count"])
        if shared_real_payload is not None
        else None
    )
    official_protocol = is_official_flow_protocol(
        shared_real_count=shared_real_count,
        samples=int(args.samples),
        is_splits=int(args.is_splits),
        parallel_rate=int(args.parallel_rate),
    )
    if args.require_official_protocol and not official_protocol:
        raise ValueError(
            "Official flow FID/IS protocol requires shared original-ImageNet "
            f"stats, exactly its {shared_real_count} fake samples, and "
            "--is_splits=10 with --parallel_rate=1; "
            f"got samples={args.samples}, is_splits={args.is_splits}, "
            f"parallel_rate={args.parallel_rate}, real_stats_path={real_stats_path!r}"
        )
    if is_main_process(rank):
        print(
            "Inception weights: "
            f"{inception_weights_path if inception_weights_path is not None else 'torch-fidelity default cache/download'}"
        )
        if shared_real_payload is not None:
            print(f"Shared original-ImageNet real stats: {Path(real_stats_path).resolve()}")
    warm_inception_cache_if_needed(
        args,
        distributed,
        rank,
        device,
        inception_weights_path,
    )
    inception = build_inception_extractor(
        int(args.fid_feature),
        inception_weights_path,
        device,
    )

    metrics = {
        strategy: {
            "fake_moments": FeatureMoments.zeros(int(args.fid_feature), device),
            "score_moments": None,
            "latent_mse_sum": 0.0,
            "latent_rms_sum": 0.0,
            "count": 0,
        }
        for strategy in strategies
    }
    real_moments = (
        None
        if shared_real_payload is not None
        else FeatureMoments.zeros(int(args.fid_feature), device)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generated = 0
    seen_complete_spans = 0
    batches_seen = 0
    batches_with_complete_spans = 0
    selected_span_batches = 0
    first_batch_debug = None
    first_complete_span_debug = None
    iterator = tqdm(loader, desc="single-stream FID/IS", dynamic_ncols=True, disable=not progress)
    batch_offset = 0
    for batch_idx, batch in enumerate(iterator):
        if seen_complete_spans >= args.samples:
            break

        input_ids = batch["input_ids"].to(device)
        token_types = batch["token_types"].to(device)
        sigma = batch["sigma"].to(device)
        image_latents = batch["image_latents"].to(device)
        current_batch_size = int(input_ids.shape[0])
        all_spans = image_spans(token_types, image_tokens)
        batches_seen += 1
        if first_batch_debug is None:
            first_batch_debug = token_type_debug(token_types, image_tokens)
        if not all_spans:
            batch_offset += current_batch_size
            continue
        batches_with_complete_spans += 1
        if first_complete_span_debug is None:
            first_complete_span_debug = {
                "batch_idx": int(batch_idx),
                "batch_offset": int(batch_offset),
                "complete_spans_in_batch": int(len(all_spans)),
                "first_spans": [
                    [int(batch_row), int(start), int(end)]
                    for batch_row, start, end in all_spans[:8]
                ],
            }

        selected = []
        selected_global_indices = []
        for local_span_idx, span in enumerate(all_spans):
            global_index = seen_complete_spans + local_span_idx
            if global_index >= int(args.samples):
                break
            if global_index % world_size == rank:
                selected.append(span)
                selected_global_indices.append(global_index)

        seen_complete_spans += len(all_spans)
        if not selected:
            batch_offset += current_batch_size
            if progress:
                iterator.set_postfix_str(f"seen {min(seen_complete_spans, args.samples)}/{args.samples}", refresh=False)
            continue
        spans = selected
        selected_span_batches += 1

        target_latents = span_latents_to_chw(image_latents, spans, side)
        target_images = metric_images(decode_latents(vae, target_latents.float(), scaling_factor))
        if shared_real_payload is not None:
            real_images = None
        elif args.real_source == "imagenet_original":
            real_paths = source_paths_for_spans(loader.dataset, batch_offset, spans, imagenet_train_dir)
            real_images = load_real_images(real_paths, real_transform, device)
        else:
            real_images = target_images
        if real_moments is not None:
            real_features, _ = extract_inception_features(inception, real_images)
            real_moments.update(real_features)
        if args.save_images:
            save_indexed_images(target_images.cpu(), out_dir / "target_decoded", selected_global_indices)
            if shared_real_payload is None and args.real_source == "imagenet_original":
                save_indexed_images(real_images.cpu(), out_dir / "imagenet_original_real", selected_global_indices)

        for strategy_idx, strategy in enumerate(strategies):
            torch.manual_seed(int(args.seed) + batch_idx * 1009 + strategy_idx * 131071 + rank * 1_000_003)
            single_latents, trace = model.sample_image_latents_single_stream(
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=image_latents.shape[-1],
                flow_temperature=float(args.temperature),
                flow_cfg=float(args.cfg),
                flow_cfg_schedule=str(args.cfg_schedule),
                flow_solver=args.flow_solver,
                parallel_rate=int(args.parallel_rate),
                order_strategy=str(strategy),
                return_trace=True,
            )
            generated_images = metric_images(decode_latents(vae, single_latents.float(), scaling_factor))
            state = metrics[strategy]
            fake_features, fake_logits = extract_inception_features(
                inception,
                generated_images,
            )
            state["fake_moments"].update(fake_features)
            if state["score_moments"] is None:
                state["score_moments"] = InceptionScoreMoments.zeros(
                    int(args.is_splits),
                    int(fake_logits.shape[-1]),
                    device,
                )
            state["score_moments"].update(
                fake_logits,
                selected_global_indices,
                int(args.samples),
            )
            count = int(generated_images.shape[0])
            state["latent_mse_sum"] += float(F.mse_loss(single_latents.float(), target_latents.float()).item()) * count
            state["latent_rms_sum"] += float(single_latents.float().pow(2).mean().sqrt().item()) * count
            state["count"] += count
            if trace and isinstance(trace.get("generation_step"), torch.Tensor):
                state["generation_step_max"] = float(trace["generation_step"].float().max().item())
            if args.save_images:
                save_indexed_images(generated_images.cpu(), out_dir / str(strategy), selected_global_indices)

        generated += len(spans)
        batch_offset += current_batch_size
        if progress:
            iterator.set_postfix_str(
                f"rank0 {generated}; seen {min(seen_complete_spans, args.samples)}/{args.samples}",
                refresh=False,
            )

    total_generated = int(reduce_sum(float(generated), device))
    if total_generated != int(args.samples):
        debug = {
            "rank": int(rank),
            "world_size": int(world_size),
            "local_rank": int(local_rank),
            "distributed": bool(distributed),
            "device": str(device),
            "config": str(args.config),
            "model_path": str(config.model.model_path),
            "split": str(args.split),
            "samples_requested": int(args.samples),
            "batch_size": int(args.batch_size),
            "loader_batches": int(len(loader)) if hasattr(loader, "__len__") else None,
            "loader_dataset_len": int(len(loader.dataset)) if hasattr(loader, "dataset") else None,
            "image_tokens_per_img": int(image_tokens),
            "batches_seen": int(batches_seen),
            "batches_with_complete_spans": int(batches_with_complete_spans),
            "seen_complete_spans": int(seen_complete_spans),
            "selected_spans_local": int(generated),
            "selected_span_batches": int(selected_span_batches),
            "first_batch": first_batch_debug,
            "first_complete_span_batch": first_complete_span_debug,
            "distributed_env": {
                key: os.environ.get(key)
                for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
                if os.environ.get(key) is not None
            },
        }
        raise RuntimeError(
            f"Expected exactly {args.samples} generated samples across all ranks, "
            f"found {total_generated}. Debug: "
            + json.dumps(debug, sort_keys=True)
        )

    if real_moments is not None:
        real_moments.all_reduce_()
        if int(real_moments.count.item()) != int(args.samples):
            raise RuntimeError(
                f"distributed real feature count={int(real_moments.count.item())}; "
                f"expected={args.samples}"
            )
    for strategy, state in metrics.items():
        if state["score_moments"] is None:
            raise RuntimeError(f"strategy={strategy!r} generated no Inception logits")
        state["fake_moments"].all_reduce_()
        state["score_moments"].all_reduce_()
        if int(state["fake_moments"].count.item()) != int(args.samples):
            raise RuntimeError(
                f"strategy={strategy!r} distributed generated feature count="
                f"{int(state['fake_moments'].count.item())}; expected={args.samples}"
            )
        if int(state["score_moments"].count.sum().item()) != int(args.samples):
            raise RuntimeError(
                f"strategy={strategy!r} distributed Inception Score count="
                f"{int(state['score_moments'].count.sum().item())}; "
                f"expected={args.samples}"
            )

    compute_fid = bool(
        shared_real_payload is None
        or int(args.samples) == int(shared_real_count)
    )
    if is_main_process(rank) and compute_fid:
        real_reference_moments = (
            shared_feature_moments(
                shared_real_payload,
                feature=int(args.fid_feature),
                device=torch.device("cpu"),
            )
            if shared_real_payload is not None
            else real_moments
        )
        real_mean, real_cov = real_reference_moments.mean_cov()
    peak_cuda_allocated_mib = reduce_max(
        (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda"
            else None
        ),
        device,
    )
    peak_cuda_reserved_mib = reduce_max(
        (
            float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
            if device.type == "cuda"
            else None
        ),
        device,
    )

    results = {
        "official_protocol": official_protocol,
        "metric_protocol": {
            "fid_reducer": "symmetric_eigendecomposition",
            "is_split_assignment": "contiguous_by_global_sample_index",
            "is_std": "population",
            "is_splits": int(args.is_splits),
        },
        "config": args.config,
        "model_path": str(config.model.model_path),
        "precision_protocol": {
            "schema": "flow_eval_precision_v1",
            "model_dtype": str(args.model_dtype),
            "model_parameter_dtypes": parameter_dtypes,
            "checkpoint_weight_dtypes": stored_checkpoint_dtypes,
            "vae_dtype": str(args.vae_dtype),
            "flow_integrator_dtype": "fp32",
            "autocast_enabled": False,
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "adapter": adapter_report,
        "model_state": model_state_report,
        "ema_state": ema_state_report,
        "split": args.split,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "samples_requested": int(args.samples),
        "samples_evaluated": int(total_generated),
        "distributed": {
            "enabled": bool(distributed),
            "world_size": int(world_size),
            "rank": int(rank),
            "local_rank": int(local_rank),
            "peak_cuda_allocated_mib": peak_cuda_allocated_mib,
            "peak_cuda_reserved_mib": peak_cuda_reserved_mib,
        },
        "real_source": (
            "cached_original_imagenet"
            if shared_real_payload is not None
            else str(args.real_source)
        ),
        "real_stats_path": (
            str(Path(real_stats_path).resolve())
            if shared_real_payload is not None
            else None
        ),
        "real_stats_metadata": (
            shared_real_payload["metadata"]
            if shared_real_payload is not None
            else None
        ),
        "imagenet_train_dir": (
            str(imagenet_train_dir)
            if shared_real_payload is None and args.real_source == "imagenet_original"
            else None
        ),
        "real_image_size": (
            int(args.real_image_size)
            if shared_real_payload is not None or args.real_source == "imagenet_original"
            else None
        ),
        "cfg": float(args.cfg),
        "cfg_schedule": str(args.cfg_schedule),
        "sampling_steps": str(args.sampling_steps),
        "temperature": float(args.temperature),
        "flow_solver": str(args.flow_solver),
        "parallel_rate": int(args.parallel_rate),
        "inception_weights_path": inception_weights_path,
        "strategies": {},
    }

    for strategy, state in metrics.items():
        global_count = int(reduce_sum(float(state["count"]), device))
        global_latent_mse_sum = reduce_sum(state["latent_mse_sum"], device)
        global_latent_rms_sum = reduce_sum(state["latent_rms_sum"], device)
        global_generation_step_max = reduce_max(state.get("generation_step_max"), device)
        if global_count != int(args.samples):
            raise RuntimeError(
                f"strategy={strategy!r} generated count={global_count}; "
                f"expected={args.samples}"
            )
        if is_main_process(rank):
            fake_mean, fake_cov = state["fake_moments"].mean_cov()
            fid_value = (
                frechet_distance(
                    real_mean,
                    real_cov,
                    fake_mean,
                    fake_cov,
                )
                if compute_fid
                else None
            )
            is_mean, is_std, is_per_split = state["score_moments"].compute()
            results["strategies"][strategy] = {
                "count": int(global_count),
                "fid": (
                    finite_or_none(fid_value)
                    if fid_value is not None
                    else None
                ),
                "inception_score_mean": finite_or_none(is_mean),
                "inception_score_std": finite_or_none(is_std),
                "inception_score_splits": [
                    finite_or_none(value) for value in is_per_split
                ],
                "latent_mse_to_target": global_latent_mse_sum / global_count,
                "latent_rms": global_latent_rms_sum / global_count,
                "generation_step_max": global_generation_step_max,
            }

    if is_main_process(rank):
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results["strategies"], indent=2))
        print(f"Saved metrics: {metrics_path}")
    if distributed:
        distributed_barrier(distributed, device)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
