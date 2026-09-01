#!/usr/bin/env python3
"""Generate held-out ImageNet-val I2T captions and score them with CLIP."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset  # noqa: E402
from utils.imagenet_flow_dataloaders import (  # noqa: E402
    _build_cache_dataset,
    _independent_validation_params,
)
from utils.imagenet_flow_sequence import build_selfless_sigma  # noqa: E402
from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
)
from utils.utils import load_model_tokenizer  # noqa: E402


DEFAULT_CONFIG = Path(
    "configs/selfless/imagenet1k_caption_joint_10ep_ascend16_b1024.yaml"
)
DEFAULT_IMAGE_ROOT = Path(
    "public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/val"
)
DEFAULT_CLIP_MODEL = Path("public/models/openai--clip-vit-base-patch32")
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_model_source_argument(parser)
    parser.add_argument("--clip_model_dir", type=Path, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch_size_per_rank", type=int, default=4)
    parser.add_argument("--clip_batch_size_per_rank", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_image_path(
    dataset: ImageNetFlowCacheDataset,
    img_id: int,
    image_root: Path,
) -> Path:
    recorded = dataset.source_paths_full.get(int(img_id), "")
    if recorded:
        direct = Path(recorded)
        if direct.is_file():
            return direct.resolve()
    relative = dataset.source_paths[int(img_id)]
    candidate = image_root / relative
    if candidate.is_file():
        return candidate.resolve()
    # Validation roots commonly already end in ``val`` while relative paths
    # include ILSVRC/Data/CLS-LOC/val.  The filename fallback is unambiguous
    # inside each synset directory.
    fallback = image_root / dataset.synsets[int(img_id)] / Path(relative).name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(candidate)


def clip_weight_path(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = model_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"CLIP model has neither model.safetensors nor pytorch_model.bin: {model_dir}"
    )


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def initialize_device(kind: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if kind == "npu":
        import torch_npu  # noqa: F401

        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        backend = "hccl"
    elif kind == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def barrier(device: torch.device) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if device.type in {"npu", "cuda"}:
        dist.barrier(device_ids=[int(device.index or 0)])
    else:
        dist.barrier()


def evaluation_image_params(config):
    params = config.dataset.params
    if str(config.dataset.class_name) == "UnifiedMixedDataset":
        params = params.image
    return params


def build_dataset(config, tokenizer) -> tuple[ImageNetFlowCacheDataset, list[int]]:
    params = evaluation_image_params(config)
    validation_params = _independent_validation_params(params)
    if validation_params is None:
        raise ValueError(
            "I2T evaluation requires an independent ImageNet val dataset"
        )
    if str(validation_params.get("expected_split", "")).lower() != "val":
        raise ValueError(
            "I2T evaluation dataset must declare expected_split='val'"
        )
    dataset = _build_cache_dataset(config, validation_params, tokenizer)
    dataset.caption_sequence_modes = ("i2t",)
    dataset.set_training_indices([])
    return dataset, list(range(len(dataset)))


def select_balanced_validation_indices(
    dataset: ImageNetFlowCacheDataset,
    validation_indices: list[int],
    samples: int,
) -> list[int]:
    """Select holdout rows round-robin by synset in deterministic order."""

    if samples <= 0 or samples > len(validation_indices):
        raise ValueError(
            f"samples must be in [1, {len(validation_indices)}], got {samples}"
        )
    groups: dict[str, list[int]] = {}
    for dataset_index in validation_indices:
        img_id = int(dataset.img_ids[int(dataset_index)].item())
        groups.setdefault(dataset.synsets[img_id], []).append(int(dataset_index))
    selected: list[int] = []
    depth = 0
    keys = sorted(groups)
    while len(selected) < samples:
        added = 0
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added += 1
                if len(selected) == samples:
                    break
        if added == 0:
            raise RuntimeError("balanced holdout selection exhausted unexpectedly")
        depth += 1
    return selected


def build_i2t_prefix(
    tokenizer,
    *,
    text_prefix: str,
    boi_token_id: int,
    eoi_token_id: int,
    image_mask_token_id: int,
    image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    prefix_ids = torch.tensor(
        tokenizer.encode(str(text_prefix).strip(), add_special_tokens=False),
        dtype=torch.long,
    )
    if prefix_ids.numel() == 0:
        raise ValueError("I2T text prefix tokenized to an empty sequence")
    image_start = int(prefix_ids.numel()) + 1
    input_ids = torch.cat(
        [
            prefix_ids,
            torch.tensor([int(boi_token_id)], dtype=torch.long),
            torch.full((int(image_tokens),), int(image_mask_token_id), dtype=torch.long),
            torch.tensor([int(eoi_token_id)], dtype=torch.long),
        ]
    )
    token_types = torch.cat(
        [
            torch.zeros(prefix_ids.numel(), dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
            torch.ones(int(image_tokens), dtype=torch.uint8),
            torch.tensor([2], dtype=torch.uint8),
        ]
    )
    # Match the training contract: BOI, then EOI, then all image tokens.  New
    # caption queries receive later sigma values and can see the complete image.
    sigma = torch.empty(input_ids.numel(), dtype=torch.float32)
    prefix_len = int(prefix_ids.numel())
    sigma[:prefix_len] = torch.arange(prefix_len, dtype=torch.float32)
    sigma[prefix_len] = float(prefix_len)
    sigma[-1] = float(prefix_len + 1)
    sigma[image_start : image_start + image_tokens] = torch.arange(
        prefix_len + 2,
        prefix_len + 2 + image_tokens,
        dtype=torch.float32,
    )
    return input_ids, token_types, sigma, image_start


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    image_batch: torch.Tensor,
    *,
    text_prefix: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    base_sigma_batch: torch.Tensor | None = None,
) -> tuple[list[str], list[list[int]], list[str]]:
    batch_size, image_tokens, latent_dim = image_batch.shape
    base_ids, base_types, base_sigma, image_start = build_i2t_prefix(
        tokenizer,
        text_prefix=text_prefix,
        boi_token_id=int(model.config.boi_token_id),
        eoi_token_id=int(model.config.eoi_token_id),
        image_mask_token_id=int(model.config.image_mask_token_id),
        image_tokens=int(image_tokens),
    )
    input_ids = base_ids.unsqueeze(0).expand(batch_size, -1).clone().to(device)
    token_types = base_types.unsqueeze(0).expand(batch_size, -1).clone().to(device)
    if base_sigma_batch is None:
        sigma = base_sigma.unsqueeze(0).expand(batch_size, -1).clone()
    else:
        if tuple(base_sigma_batch.shape) != (batch_size, int(base_ids.numel())):
            raise ValueError(
                "base_sigma_batch must align with the serialized I2T prefix: "
                f"got {tuple(base_sigma_batch.shape)}, expected "
                f"{(batch_size, int(base_ids.numel()))}"
            )
        sigma = base_sigma_batch.clone()
    sigma = sigma.to(device=device, dtype=torch.float32)
    image_latents = torch.zeros(
        batch_size,
        input_ids.shape[1],
        latent_dim,
        device=device,
        dtype=image_batch.dtype,
    )
    image_latent_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    image_latents[:, image_start : image_start + image_tokens] = image_batch.to(device)
    image_latent_mask[:, image_start : image_start + image_tokens] = True

    eos_id = int(tokenizer.eos_token_id)
    stop_ids = {eos_id}
    im_end_id = getattr(model.config, "im_end_token_id", None)
    if im_end_id is not None:
        stop_ids.add(int(im_end_id))
    output_ids, trace = model.generate(
        "i2t",
        input_ids=input_ids,
        token_types=token_types,
        sigma=sigma,
        image_latents=image_latents,
        image_latent_mask=image_latent_mask,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        eos_token_id=sorted(stop_ids),
        use_cache=True,
        return_trace=True,
    )
    if trace.get("backbone_kv_cache_enabled") is not True:
        raise RuntimeError("I2T evaluation must use the model KV cache")

    generated: list[list[int]] = []
    stop_reasons: list[str] = []
    prompt_length = int(input_ids.shape[1])
    for suffix in output_ids[:, prompt_length:].detach().cpu().tolist():
        tokens: list[int] = []
        reason = "max_new_tokens"
        for token in suffix:
            token = int(token)
            if token in stop_ids:
                reason = "eos" if token == eos_id else "im_end"
                break
            tokens.append(token)
        generated.append(tokens)
        stop_reasons.append(reason)
    texts = [
        tokenizer.decode(tokens, skip_special_tokens=True).strip()
        for tokens in generated
    ]
    return texts, generated, stop_reasons


def _feature_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if getattr(value, "pooler_output", None) is not None:
        return value.pooler_output
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise TypeError(f"cannot extract feature tensor from {type(value)}")


@torch.inference_mode()
def add_clip_scores(
    rows: list[dict[str, Any]],
    *,
    clip_model_dir: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    from transformers import CLIPModel, CLIPProcessor

    if not clip_model_dir.is_dir():
        raise FileNotFoundError(f"CLIP model directory is missing: {clip_model_dir}")
    model = CLIPModel.from_pretrained(
        clip_model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(
        clip_model_dir,
        local_files_only=True,
    )
    for start in range(0, len(rows), int(batch_size)):
        batch = rows[start : start + int(batch_size)]
        images: list[Image.Image] = []
        try:
            for row in batch:
                with Image.open(row["source_path"]) as source_image:
                    images.append(source_image.convert("RGB"))
            image_inputs = processor(images=images, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(device=device, dtype=torch.float32)
            original_texts = [row["reference_caption"] for row in batch]
            generated_texts = [row["generated_caption"] or " " for row in batch]
            text_inputs = processor.tokenizer(
                original_texts + generated_texts,
                padding=True,
                truncation=True,
                max_length=int(processor.tokenizer.model_max_length),
                return_tensors="pt",
            )
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            image_features = _feature_tensor(
                model.get_image_features(pixel_values=pixel_values)
            )
            text_features = _feature_tensor(model.get_text_features(**text_inputs))
            image_features = F.normalize(image_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)
            count = len(batch)
            reference_scores = (text_features[:count] * image_features).sum(dim=-1)
            generated_scores = (text_features[count:] * image_features).sum(dim=-1)
            for row, reference_score, generated_score in zip(
                batch,
                reference_scores.cpu().tolist(),
                generated_scores.cpu().tolist(),
            ):
                row["reference_clip_image_text_cosine"] = float(reference_score)
                row["generated_clip_image_text_cosine"] = float(generated_score)
                row["clip_delta_vs_reference"] = float(generated_score - reference_score)
        finally:
            for image in images:
                image.close()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.batch_size_per_rank <= 0 or args.clip_batch_size_per_rank <= 0:
        raise ValueError("batch sizes must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive")
    if not math.isfinite(float(args.temperature)) or args.temperature < 0:
        raise ValueError("--temperature must be finite and non-negative")
    for path in (args.config, args.clip_model_dir, args.image_root):
        if not path.exists():
            raise FileNotFoundError(path)
    clip_weight_path(args.clip_model_dir)


def main() -> None:
    args = parse_args()
    validate_args(args)
    source = model_source_from_args(args)
    rank, world_size, _, device = initialize_device(args.device)
    torch.manual_seed(int(args.seed) + rank)
    config = OmegaConf.load(args.config)
    config.training.runtime_hashing_enabled = False
    configure_model_source(config, source)
    attention_contract = str(
        config.model.get(
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if attention_contract not in {
        "selfless_strict",
        "xlnet_content_diagonal",
    }:
        raise ValueError(
            "unsupported dual-stream attention contract: "
            f"{attention_contract!r}"
        )
    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    loaded_attention_contract = str(
        getattr(
            model.config,
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if loaded_attention_contract != attention_contract:
        raise ValueError(
            "loaded model attention contract does not match evaluation config: "
            f"model={loaded_attention_contract!r}, config={attention_contract!r}"
        )
    model_source_report = load_model_source_weights(model, source)
    model.eval().to(device)
    dataset, validation_indices = build_dataset(config, tokenizer)
    params = evaluation_image_params(config)
    if args.samples > len(validation_indices):
        raise ValueError(
            f"requested {args.samples} samples from {len(validation_indices)} holdout rows"
        )
    selected = select_balanced_validation_indices(
        dataset,
        validation_indices,
        int(args.samples),
    )
    local_pairs = [
        (global_index, dataset_index)
        for global_index, dataset_index in enumerate(selected)
        if global_index % world_size == rank
    ]
    local_rows: list[dict[str, Any]] = []
    for start in range(0, len(local_pairs), int(args.batch_size_per_rank)):
        pairs = local_pairs[start : start + int(args.batch_size_per_rank)]
        latents = []
        base_sigmas = []
        identities = []
        for global_index, dataset_index in pairs:
            sample = dataset[dataset_index]
            latent = sample["image_latents"]
            posterior_seed = int(sample["posterior_seed"].item())
            sample_sigma = build_selfless_sigma(
                sample,
                image_tokens=int(dataset.image_tokens_per_img),
            )
            image_start = int(sample["image_start"].item())
            prefix_end = image_start + int(dataset.image_tokens_per_img) + 1
            base_sigma = sample_sigma[:prefix_end]
            img_id = int(dataset.img_ids[dataset_index].item())
            references = dataset._indexed_caption_texts(dataset_index, img_id)
            if not references:
                raise ValueError(
                    f"expected at least one validation caption, got {len(references)}"
                )
            relative_path = dataset.source_paths[img_id]
            source_path = source_image_path(dataset, img_id, args.image_root)
            latents.append(latent)
            base_sigmas.append(base_sigma)
            identities.append(
                {
                    "global_sample_index": int(global_index),
                    "dataset_index": int(dataset_index),
                    "img_id": img_id,
                    "id": Path(relative_path).stem,
                    "path": relative_path,
                    "source_path": str(source_path.resolve()),
                    "synset": dataset.synsets[img_id],
                    "posterior_seed": int(posterior_seed),
                    "image_reveal_seed": int(sample["reveal_seed"].item()),
                    "image_sigma_order": str(sample["image_sigma_order"]),
                    "reference_caption": references[0],
                }
            )
        generated_texts, generated_ids, stop_reasons = generate_batch(
            model,
            tokenizer,
            torch.stack(latents),
            text_prefix=str(params.caption_i2t_prefix),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            device=device,
            base_sigma_batch=torch.stack(base_sigmas),
        )
        for identity, generated_text, token_ids, stop_reason in zip(
            identities,
            generated_texts,
            generated_ids,
            stop_reasons,
        ):
            local_rows.append(
                {
                    "schema": "selfless_imagenet1k_i2t_generation_sample_v1",
                    **identity,
                    "generated_caption": generated_text,
                    "generated_token_ids": token_ids,
                    "generated_token_count": len(token_ids),
                    "generated_word_count": len(WORD_RE.findall(generated_text)),
                    "stop_reason": stop_reason,
                }
            )

    del model
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    add_clip_scores(
        local_rows,
        clip_model_dir=args.clip_model_dir,
        batch_size=int(args.clip_batch_size_per_rank),
        device=device,
    )
    shard_dir = args.output_dir / "shards"
    shard_path = shard_dir / f"rank-{rank:05d}-of-{world_size:05d}.jsonl"
    atomic_write_text(shard_path, jsonl_text(local_rows))
    barrier(device)

    if rank == 0:
        rows: list[dict[str, Any]] = []
        for shard_rank in range(world_size):
            path = shard_dir / f"rank-{shard_rank:05d}-of-{world_size:05d}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        rows.sort(key=lambda row: int(row["global_sample_index"]))
        observed = [int(row["global_sample_index"]) for row in rows]
        if observed != list(range(int(args.samples))):
            raise ValueError("distributed I2T sample coverage is incomplete or duplicated")
        samples_path = args.output_dir / "samples.jsonl"
        atomic_write_text(samples_path, jsonl_text(rows))
        generated_scores = [
            float(row["generated_clip_image_text_cosine"]) for row in rows
        ]
        reference_scores = [
            float(row["reference_clip_image_text_cosine"]) for row in rows
        ]
        class_counts = Counter(str(row["synset"]) for row in rows)
        model_identity = {
            **source.report(),
            "load_report": model_source_report,
        }
        clip_identity = {
            "model": str(args.clip_model_dir.resolve()),
        }
        class_balance = {
            "class_count": len(class_counts),
            "min_samples_per_class": min(class_counts.values()),
            "max_samples_per_class": max(class_counts.values()),
            "counts": dict(sorted(class_counts.items())),
        }
        validation_params = _independent_validation_params(params)
        split_payload = {
            "name": str(getattr(dataset, "dataset_split", "val")),
            "manifest": str(validation_params.manifest_jsonl),
        }
        metrics = {
            "schema": "selfless_imagenet1k_i2t_clip_metrics_v2",
            "runtime_hashing_enabled": False,
            "samples": len(rows),
            "model": model_identity,
            "split": split_payload,
            "class_balance": class_balance,
            "generation": {
                "seed": int(args.seed),
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "image_sigma_order": sorted(
                    {str(row["image_sigma_order"]) for row in rows}
                ),
                "sigma_source": "validation_sample_training_contract",
                "dual_stream_attention_contract": attention_contract,
                "single_stream_visible_content_diagonal": (
                    attention_contract == "xlnet_content_diagonal"
                ),
                "single_stream_current_query_diagonal": False,
                "empty_caption_rate": sum(
                    not str(row["generated_caption"]).strip() for row in rows
                )
                / len(rows),
                "mean_generated_tokens": sum(
                    int(row["generated_token_count"]) for row in rows
                )
                / len(rows),
                "eos_stop_rate": sum(
                    row["stop_reason"] in {"eos", "im_end"} for row in rows
                )
                / len(rows),
            },
            "clip": {
                **clip_identity,
                "caption_clip_score": sum(generated_scores) / len(generated_scores),
                "reference_clip_score": sum(reference_scores) / len(reference_scores),
                "mean_delta_vs_reference": sum(
                    generated - reference
                    for generated, reference in zip(
                        generated_scores, reference_scores
                    )
                )
                / len(rows),
                "win_rate_vs_reference": sum(
                    generated > reference
                    for generated, reference in zip(
                        generated_scores, reference_scores
                    )
                )
                / len(rows),
            },
            "distributed": {
                "world_size": world_size,
                "device": str(device.type),
                "batch_size_per_rank": int(args.batch_size_per_rank),
                "clip_batch_size_per_rank": int(args.clip_batch_size_per_rank),
            },
            "artifacts": {"samples": str(samples_path.resolve())},
        }
        metrics_path = args.output_dir / "metrics.json"
        atomic_write_text(
            metrics_path,
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))
    barrier(device)
    if dataset.synthetic_text_index is not None:
        dataset.synthetic_text_index.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
