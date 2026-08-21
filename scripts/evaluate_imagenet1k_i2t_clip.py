#!/usr/bin/env python3
"""Generate ImageNet-1K I2T captions and score them with a frozen CLIP model.

The evaluator uses the same deterministic 50-images-per-class holdout as the
joint-training validation loader.  It conditions on cached MAR posterior
latents, generates captions autoregressively, and compares each caption with
the corresponding original ImageNet training image.  Distributed ranks write
auditable JSONL shards; rank zero merges them into one metrics artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
from utils.imagenet_flow_dataloaders import _build_split_indices  # noqa: E402
from utils.utils import get_selfless_mask, load_model_tokenizer  # noqa: E402


DEFAULT_CONFIG = Path(
    "configs/selfless/imagenet1k_caption_joint_sweep_10ep_ascend16_b1024.yaml"
)
DEFAULT_IMAGE_ROOT = Path(
    "public/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train"
)
DEFAULT_CLIP_MODEL = Path("public/models/openai--clip-vit-base-patch32")
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model_path", type=Path, required=True)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_dataset(config, tokenizer) -> tuple[ImageNetFlowCacheDataset, list[int]]:
    params = config.dataset.params
    dataset = ImageNetFlowCacheDataset(
        cache_path=params.cache_path,
        tokenizer=tokenizer,
        boi_token_id=config.model.boi_token_id,
        eoi_token_id=config.model.eoi_token_id,
        mask_token_id=config.model.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        image_tokens_per_img=params.get(
            "image_tokens_per_img", config.model.image_tokens_per_img
        ),
        image_latent_dim=params.get(
            "image_latent_dim", config.model.image_latent_dim
        ),
        manifest_jsonl=params.manifest_jsonl,
        synset_mapping_path=params.get("synset_mapping_path", None),
        conditioning_mode="caption",
        caption_jsonl=params.caption_jsonl,
        caption_list_key=params.get("caption_list_key", "captions"),
        caption_list_text_key=params.get("caption_list_text_key", "text"),
        caption_path_key=params.get("caption_path_key", "path"),
        caption_id_key=params.get("caption_id_key", "id"),
        caption_validation_index=0,
        t2i_prompt_validation_index=0,
        caption_sequence_modes=("i2t",),
        synthetic_text_index_manifest=params.synthetic_text_index_manifest,
        caption_t2i_prefix=params.caption_t2i_prefix,
        caption_i2t_prefix=params.caption_i2t_prefix,
        # The original caption is an evaluation reference only.  Joint training
        # still uses the six synthetic captions because its config remains false.
        caption_include_original=True,
        cache_caption_tokens=False,
        max_seq_length=params.max_seq_length,
        model_context_length=params.get("model_context_length", None),
        caption_manifest_sha256=params.caption_manifest_sha256,
        seed=config.training.seed,
    )
    _, validation_indices = _build_split_indices(
        dataset,
        val_ratio=float(params.get("val_ratio", 0.001)),
        seed=int(params.get("split_seed", config.training.seed)),
        strategy=str(params.get("split_strategy", "stratified")),
        val_samples_per_class=int(params.val_samples_per_class),
    )
    dataset.set_training_indices([])
    return dataset, validation_indices


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
    sigma = base_sigma.unsqueeze(0).expand(batch_size, -1).clone().to(device)
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
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    stop_reasons = ["max_new_tokens"] * batch_size
    for _ in range(int(max_new_tokens)):
        input_ids = torch.cat(
            [
                input_ids,
                torch.full(
                    (batch_size, 1),
                    int(model.config.mask_token_id),
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )
        token_types = torch.cat(
            [token_types, torch.zeros(batch_size, 1, dtype=torch.uint8, device=device)],
            dim=1,
        )
        next_sigma = sigma.amax(dim=1, keepdim=True) + 1.0
        sigma = torch.cat([sigma, next_sigma], dim=1)
        image_latents = torch.cat(
            [
                image_latents,
                torch.zeros(
                    batch_size,
                    1,
                    latent_dim,
                    dtype=image_latents.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        image_latent_mask = torch.cat(
            [
                image_latent_mask,
                torch.zeros(batch_size, 1, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        attention_mask = get_selfless_mask(
            sigma=sigma,
            seq_len=int(input_ids.shape[1]),
            device=device,
        )
        hidden = model.model(
            X0_input_ids=input_ids,
            attention_mask=attention_mask,
            token_types=token_types,
            image_latents=image_latents,
            image_latent_mask=image_latent_mask,
            calculate_likelihood=False,
        ).last_hidden_state[:, -1]
        logits = model.lm_head(hidden)
        if float(temperature) <= 1.0e-6:
            next_tokens = logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(logits.float() / float(temperature), dim=-1)
            next_tokens = torch.multinomial(probabilities, 1).squeeze(1)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_id), next_tokens)
        input_ids[:, -1] = next_tokens
        for row, token in enumerate(next_tokens.detach().cpu().tolist()):
            if bool(finished[row]):
                continue
            if int(token) in stop_ids:
                finished[row] = True
                stop_reasons[row] = "eos" if int(token) == eos_id else "im_end"
            else:
                generated[row].append(int(token))
        if bool(finished.all()):
            break
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
    for path in (args.config, args.model_path, args.clip_model_dir, args.image_root):
        if not path.exists():
            raise FileNotFoundError(path)
    clip_weight_path(args.clip_model_dir)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rank, world_size, _, device = initialize_device(args.device)
    torch.manual_seed(int(args.seed) + rank)
    config = OmegaConf.load(args.config)
    config.model.model_path = str(args.model_path)
    config.training.from_scratch = False
    config.training.use_gradient_checkpointing = False
    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    model.eval().to(device)
    dataset, validation_indices = build_dataset(config, tokenizer)
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
        identities = []
        for global_index, dataset_index in pairs:
            latent, posterior_seed = dataset._sample_posterior(dataset_index, 0)
            img_id = int(dataset.img_ids[dataset_index].item())
            references = dataset._indexed_caption_texts(dataset_index, img_id)
            if len(references) != 7:
                raise ValueError(
                    f"expected original + six synthetic captions, got {len(references)}"
                )
            relative_path = dataset.source_paths[img_id]
            source_path = args.image_root / relative_path
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            latents.append(latent)
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
                    "reference_caption": references[0],
                }
            )
        generated_texts, generated_ids, stop_reasons = generate_batch(
            model,
            tokenizer,
            torch.stack(latents),
            text_prefix=str(config.dataset.params.caption_i2t_prefix),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            device=device,
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
        class_counts_payload = json.dumps(
            dict(sorted(class_counts.items())),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        metrics = {
            "schema": "selfless_imagenet1k_i2t_clip_metrics_v1",
            "samples": len(rows),
            "model": {
                "path": str(args.model_path),
                "config_sha256": file_sha256(args.model_path / "config.json"),
                "weights_sha256": file_sha256(
                    args.model_path / "model.safetensors"
                ),
            },
            "split": {
                "strategy": str(config.dataset.params.split_strategy),
                "seed": int(config.dataset.params.split_seed),
                "val_samples_per_class": int(
                    config.dataset.params.val_samples_per_class
                ),
                "validation_overlap_train": False,
            },
            "class_balance": {
                "class_count": len(class_counts),
                "min_samples_per_class": min(class_counts.values()),
                "max_samples_per_class": max(class_counts.values()),
                "counts_sha256": hashlib.sha256(class_counts_payload).hexdigest(),
            },
            "generation": {
                "seed": int(args.seed),
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
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
                "model": str(args.clip_model_dir.resolve()),
                "config_sha256": file_sha256(args.clip_model_dir / "config.json"),
                "weights_sha256": file_sha256(
                    clip_weight_path(args.clip_model_dir)
                ),
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
