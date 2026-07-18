#!/usr/bin/env python3
"""Encode the balanced ImageNet-100 manifest with official Show-o MAGVITv2.

The script is torchrun-compatible. Each rank writes one atomic packed part;
rank zero merges parts in source-manifest order into ``tokens.pt`` and writes
``manifest.json`` describing the cache provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_VOCAB_SIZE = 8192
IMAGE_TOKENS_PER_IMG = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a packed official Show-o MAGVITv2 cache for ImageNet-100."
    )
    parser.add_argument(
        "--manifest_jsonl",
        default=(
            "public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "public/datasets/imagenet_ablation_100c_balanced/"
            "vq_tokens_magvit2_showo_8192"
        ),
    )
    parser.add_argument(
        "--imagenet_root",
        default="",
        help=(
            "Optional replacement ImageNet root when manifest source_path values "
            "point at an unavailable mount."
        ),
    )
    parser.add_argument(
        "--showo_repo",
        default="/inspire/hdd/global_user/wanjiaxin-253108030048/code/Show-o",
    )
    parser.add_argument(
        "--vq_model_path", default="public/models/showlab/magvitv2"
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit CUDA device such as cuda:0.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Only for smoke tests; negative values encode the complete manifest.",
    )
    parser.add_argument("--allow_errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_tail_after_train(source_path: Path) -> Optional[Path]:
    parts = source_path.parts
    lowered = [part.lower() for part in parts]
    if "train" not in lowered:
        return None
    train_index = len(lowered) - 1 - lowered[::-1].index("train")
    tail = parts[train_index + 1 :]
    return Path(*tail) if tail else None


def resolve_source_path(
    source_path: str,
    synset: str,
    imagenet_root: Optional[Path],
) -> Path:
    """Resolve original paths or relocate them under a new ImageNet root."""

    original = Path(source_path).expanduser()
    if original.is_file():
        return original
    if imagenet_root is None:
        return original

    root = imagenet_root.expanduser()
    tail = _source_tail_after_train(original)
    filename = original.name
    candidates: List[Path] = []
    if not original.is_absolute():
        candidates.append(root / original)
    if tail is not None:
        candidates.extend(
            [
                root / tail,
                root / "train" / tail,
                root / "Data" / "CLS-LOC" / "train" / tail,
                root / "ILSVRC" / "Data" / "CLS-LOC" / "train" / tail,
            ]
        )
    if filename:
        candidates.extend(
            [
                root / synset / filename,
                root / "train" / synset / filename,
                root / "Data" / "CLS-LOC" / "train" / synset / filename,
                root
                / "ILSVRC"
                / "Data"
                / "CLS-LOC"
                / "train"
                / synset
                / filename,
            ]
        )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
            if candidate.is_file():
                return candidate
    return unique_candidates[0] if unique_candidates else original


def read_manifest(
    manifest_path: Path,
    imagenet_root: Optional[Path],
    max_samples: int = -1,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "img_id" not in row or "source_path" not in row or "synset" not in row:
                raise ValueError(
                    f"{manifest_path}:{line_number} must contain "
                    "img_id, source_path, and synset"
                )
            image_id = int(row["img_id"])
            if image_id in seen_ids:
                raise ValueError(
                    f"duplicate img_id={image_id} in {manifest_path}"
                )
            seen_ids.add(image_id)
            synset = str(row["synset"])
            source_path = str(row["source_path"])
            rows.append(
                {
                    "manifest_index": len(rows),
                    "image_id": image_id,
                    "synset": synset,
                    "source_path": source_path,
                    "resolved_path": resolve_source_path(
                        source_path, synset, imagenet_root
                    ),
                }
            )
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
    if not rows:
        raise ValueError(f"no samples found in {manifest_path}")
    return rows


class ManifestImageDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], resolution: int = 256):
        self.rows = list(rows)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    int(resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop((int(resolution), int(resolution))),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    inplace=True,
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[int(index)]
        try:
            with Image.open(row["resolved_path"]) as image:
                pixel_values = self.transform(image.convert("RGB"))
            return (
                int(row["manifest_index"]),
                int(row["image_id"]),
                pixel_values,
                "",
            )
        except Exception as error:
            return (
                int(row["manifest_index"]),
                int(row["image_id"]),
                torch.empty(0),
                f"{row['resolved_path']}: {error}",
            )


def collate_images(items):
    manifest_indices: List[int] = []
    image_ids: List[int] = []
    images: List[torch.Tensor] = []
    errors: List[Tuple[int, int, str]] = []
    for manifest_index, image_id, image, error in items:
        if error:
            errors.append((manifest_index, image_id, error))
        else:
            manifest_indices.append(manifest_index)
            image_ids.append(image_id)
            images.append(image)
    pixel_values = torch.stack(images) if images else torch.empty(0)
    return manifest_indices, image_ids, pixel_values, errors


def load_showo_magvit2(
    showo_repo: Path,
    vq_model_path: str,
    device: torch.device,
):
    showo_repo = showo_repo.resolve()
    source_file = showo_repo / "models" / "modeling_magvitv2.py"
    if not source_file.exists():
        raise FileNotFoundError(
            f"official Show-o MAGVITv2 source not found: {source_file}"
        )
    sys.path.insert(0, str(showo_repo))
    from models.modeling_magvitv2 import MAGVITv2

    path = Path(vq_model_path)
    if path.exists():
        model = MAGVITv2()
        safetensors_path = path / "pytorch_model.safetensors"
        bin_path = path / "pytorch_model.bin"
        if safetensors_path.exists():
            from safetensors.torch import load_file as load_safetensors

            state_dict = load_safetensors(
                str(safetensors_path), device="cpu"
            )
        elif bin_path.exists():
            state_dict = torch.load(
                bin_path, map_location="cpu", weights_only=True
            )
        else:
            raise FileNotFoundError(
                f"no pytorch_model.safetensors or pytorch_model.bin under {path}"
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "MAGVITv2 state_dict mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    else:
        model = MAGVITv2.from_pretrained(vq_model_path)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_batch(
    vq_model,
    pixel_values: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    tokens = vq_model.get_code(
        pixel_values.to(device, non_blocking=True)
    ).long()
    tokens = tokens.detach().cpu()
    if tokens.ndim != 2 or tokens.shape[1] != IMAGE_TOKENS_PER_IMG:
        raise ValueError(
            f"expected MAGVITv2 codes [B,{IMAGE_TOKENS_PER_IMG}], "
            f"got {tuple(tokens.shape)}"
        )
    if tokens.numel():
        token_min = int(tokens.min().item())
        token_max = int(tokens.max().item())
        if token_min < 0 or token_max >= IMAGE_VOCAB_SIZE:
            raise ValueError(
                f"MAGVITv2 codes [{token_min},{token_max}] are outside "
                f"[0,{IMAGE_VOCAB_SIZE})"
            )
    return tokens


def distributed_environment(requested_device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = str(requested_device).lower()
    if device_name == "auto":
        device_name = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    elif device_name == "cuda":
        device_name = f"cuda:{local_rank}"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return device, rank, world_size


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def merge_parts(
    *,
    output_dir: Path,
    world_size: int,
    expected_rows: int,
    manifest_path: Path,
    imagenet_root: Optional[Path],
    showo_repo: Path,
    vq_model_path: str,
    resolution: int,
    allow_errors: bool,
) -> None:
    parts = [
        torch.load(
            output_dir
            / "parts"
            / f"part-{rank:05d}-of-{world_size:05d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for rank in range(world_size)
    ]
    manifest_indices = torch.cat(
        [part["manifest_indices"].long() for part in parts]
    )
    image_ids = torch.cat([part["image_ids"].long() for part in parts])
    tokens = torch.cat([part["tokens"].to(torch.int16) for part in parts])
    errors = [
        error for part in parts for error in part.get("errors", [])
    ]
    if manifest_indices.numel() != image_ids.numel() or tokens.shape[0] != image_ids.numel():
        raise ValueError("packed part tensors have inconsistent sample counts")
    if manifest_indices.numel() != torch.unique(manifest_indices).numel():
        raise ValueError("duplicate source-manifest indices found while merging")
    if image_ids.numel() != torch.unique(image_ids).numel():
        raise ValueError("duplicate image IDs found while merging")
    if not allow_errors and image_ids.numel() != int(expected_rows):
        raise ValueError(
            f"expected {expected_rows} encoded images, got {image_ids.numel()}"
        )
    order = torch.argsort(manifest_indices)
    manifest_indices = manifest_indices[order].contiguous()
    image_ids = image_ids[order].contiguous()
    tokens = tokens[order].contiguous()
    if tokens.ndim != 2 or tokens.shape[1] != IMAGE_TOKENS_PER_IMG:
        raise ValueError(f"invalid merged token shape: {tuple(tokens.shape)}")
    _atomic_torch_save(
        {"image_ids": image_ids, "tokens": tokens},
        output_dir / "tokens.pt",
    )

    cache_manifest = {
        "format": "qwen-showo-imagenet-magvitv2-packed-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_images": int(expected_rows),
        "encoded_images": int(image_ids.numel()),
        "failed_images": int(len(errors)),
        "image_tokens_per_img": IMAGE_TOKENS_PER_IMG,
        "image_vocab_size": IMAGE_VOCAB_SIZE,
        "storage_dtype": "int16",
        "ordering": "source_manifest",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "imagenet_root": (
            str(imagenet_root.resolve()) if imagenet_root is not None else None
        ),
        "showo_repo": str(showo_repo.resolve()),
        "showo_commit": _git_commit(showo_repo),
        "vq_model_path": str(Path(vq_model_path).resolve())
        if Path(vq_model_path).exists()
        else vq_model_path,
        "resolution": int(resolution),
        "transform": [
            f"Resize({int(resolution)}, bicubic)",
            f"CenterCrop({int(resolution)})",
            "ToTensor()",
            "Normalize(mean=0.5,std=0.5)",
        ],
        "world_size": int(world_size),
        "errors": errors,
    }
    _atomic_json_dump(cache_manifest, output_dir / "manifest.json")


def main() -> None:
    args = parse_args()
    device, rank, world_size = distributed_environment(args.device)
    manifest_path = Path(args.manifest_jsonl)
    output_dir = Path(args.output_dir)
    showo_repo = Path(args.showo_repo)
    imagenet_root = Path(args.imagenet_root) if args.imagenet_root else None
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"

    if rank == 0 and args.overwrite:
        for path in (output_dir / "tokens.pt", output_dir / "manifest.json"):
            path.unlink(missing_ok=True)
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
    if dist.is_initialized():
        dist.barrier()
    final_path = output_dir / "tokens.pt"
    if final_path.exists() and not args.overwrite:
        if rank == 0:
            print(f"packed cache already exists: {final_path}")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    rows = read_manifest(
        manifest_path=manifest_path,
        imagenet_root=imagenet_root,
        max_samples=int(args.max_samples),
    )
    rank_rows = rows[rank::world_size]
    print(
        f"rank={rank}/{world_size} device={device} "
        f"samples={len(rank_rows)}/{len(rows)}"
    )

    vq_model = load_showo_magvit2(
        showo_repo=showo_repo,
        vq_model_path=args.vq_model_path,
        device=device,
    )
    dataset = ManifestImageDataset(rank_rows, resolution=int(args.resolution))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=collate_images,
    )

    encoded_manifest_indices: List[int] = []
    encoded_image_ids: List[int] = []
    encoded_tokens: List[torch.Tensor] = []
    errors: List[Dict[str, Any]] = []
    progress = tqdm(loader, desc=f"rank {rank}: MAGVITv2", disable=rank != 0)
    for manifest_indices, image_ids, pixel_values, load_errors in progress:
        for manifest_index, image_id, message in load_errors:
            errors.append(
                {
                    "manifest_index": int(manifest_index),
                    "image_id": int(image_id),
                    "error": str(message),
                }
            )
        if not image_ids:
            continue
        try:
            token_batch = encode_batch(vq_model, pixel_values, device)
            encoded_manifest_indices.extend(int(x) for x in manifest_indices)
            encoded_image_ids.extend(int(x) for x in image_ids)
            encoded_tokens.extend(
                row.to(torch.int16).contiguous() for row in token_batch
            )
        except Exception as batch_error:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"rank={rank} batch encode failed ({len(image_ids)} images): "
                f"{batch_error}; retrying individually"
            )
            for manifest_index, image_id, pixel_value in zip(
                manifest_indices, image_ids, pixel_values
            ):
                try:
                    token_row = encode_batch(
                        vq_model, pixel_value.unsqueeze(0), device
                    )[0]
                    encoded_manifest_indices.append(int(manifest_index))
                    encoded_image_ids.append(int(image_id))
                    encoded_tokens.append(
                        token_row.to(torch.int16).contiguous()
                    )
                except Exception as item_error:
                    errors.append(
                        {
                            "manifest_index": int(manifest_index),
                            "image_id": int(image_id),
                            "error": str(item_error),
                        }
                    )

    error_count = torch.tensor(
        len(errors), device=device, dtype=torch.long
    )
    if dist.is_initialized():
        dist.all_reduce(error_count, op=dist.ReduceOp.SUM)
    if int(error_count.item()) and not args.allow_errors:
        raise RuntimeError(
            f"MAGVITv2 encoding failed for {int(error_count.item())} images; "
            "rerun after fixing paths, or pass --allow_errors explicitly"
        )

    parts_dir.mkdir(parents=True, exist_ok=True)
    part_tokens = (
        torch.stack(encoded_tokens)
        if encoded_tokens
        else torch.empty(
            (0, IMAGE_TOKENS_PER_IMG), dtype=torch.int16
        )
    )
    part_path = (
        parts_dir / f"part-{rank:05d}-of-{world_size:05d}.pt"
    )
    _atomic_torch_save(
        {
            "manifest_indices": torch.tensor(
                encoded_manifest_indices, dtype=torch.long
            ),
            "image_ids": torch.tensor(encoded_image_ids, dtype=torch.long),
            "tokens": part_tokens,
            "errors": errors,
        },
        part_path,
    )
    if dist.is_initialized():
        dist.barrier()

    merge_status: List[Optional[str]] = [None]
    if rank == 0:
        try:
            merge_parts(
                output_dir=output_dir,
                world_size=world_size,
                expected_rows=len(rows),
                manifest_path=manifest_path,
                imagenet_root=imagenet_root,
                showo_repo=showo_repo,
                vq_model_path=args.vq_model_path,
                resolution=int(args.resolution),
                allow_errors=bool(args.allow_errors),
            )
            shutil.rmtree(parts_dir)
        except Exception as error:
            merge_status[0] = f"{type(error).__name__}: {error}"
    if dist.is_initialized():
        dist.broadcast_object_list(merge_status, src=0)
    if merge_status[0] is not None:
        raise RuntimeError(f"failed to merge MAGVITv2 cache: {merge_status[0]}")

    print(
        f"rank={rank} done encoded={len(encoded_image_ids)} "
        f"errors={len(errors)} output={output_dir}"
    )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
