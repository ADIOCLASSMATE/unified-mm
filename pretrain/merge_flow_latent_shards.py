"""Merge and validate ImageNet KL16 posterior-cache shards."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm

from utils.kl16_layout import kl16_layout
from utils.sharded_posterior import SCHEMA as SHARD_INDEX_SCHEMA

POSTERIOR_CACHE_FORMAT = "imagenet_kl16_scaled_posterior_v1"
POSTERIOR_STATS_LAYOUT = "scaled_mean_then_scaled_std"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_img_ids(path: Path) -> torch.Tensor:
    img_ids = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                img_ids.append(int(json.loads(line)["img_id"]))
    if len(img_ids) != len(set(img_ids)):
        raise ValueError(f"Manifest contains duplicate img_ids: {path}")
    return torch.tensor(sorted(img_ids), dtype=torch.long)


def validate_stats(path: Path, posterior_stats: torch.Tensor) -> None:
    if posterior_stats.dtype != torch.float16:
        raise ValueError(
            f"posterior_stats in {path} must be float16, got {posterior_stats.dtype}"
        )
    chunk_rows = 512
    for start in range(0, posterior_stats.shape[0], chunk_rows):
        chunk = posterior_stats[start : start + chunk_rows]
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError(
                f"posterior_stats contains NaN/Inf in {path} at row {start}"
            )
        if bool((chunk[..., 16:] < 0).any()):
            raise ValueError(f"posterior std is negative in {path} at row {start}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--manifest_jsonl", default=None)
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--index_only", action="store_true",
                        help="Publish a JSON shard index; avoid copying TB-scale posterior tensors.")
    parser.add_argument("--row_index_path", default=None,
                        help="With --index_only, store the binary row map outside the text publication.")
    parser.add_argument(
        "--no_hash",
        action="store_true",
        help="Do not calculate VAE, manifest, or output file digests.",
    )
    args = parser.parse_args(argv)

    shard_dir = Path(args.shard_dir)
    output_path = Path(args.output_path)
    if args.index_only and output_path.suffix != ".json":
        raise ValueError("--index_only requires a .json output_path")
    if args.row_index_path and not args.index_only:
        raise ValueError("--row_index_path requires --index_only")
    if args.row_index_path and Path(args.row_index_path).suffix != ".pt":
        raise ValueError("--row_index_path requires a .pt file separate from the JSON index")
    shard_paths = sorted(shard_dir.glob("shard-*-of-*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard-*-of-*.pt files found in {shard_dir}")

    stats_parts = []
    id_parts = []
    source_metadata = []
    expected_shards = None
    shard_indices = set()
    for path in tqdm(shard_paths, desc="Loading posterior cache shards", unit="shard"):
        payload = torch.load(
            str(path),
            map_location="cpu",
            mmap=args.mmap or args.index_only,
            weights_only=True,
        )
        metadata = payload.get("metadata", {})
        if metadata.get("format") != POSTERIOR_CACHE_FORMAT:
            raise ValueError(f"Unexpected cache format in {path}: {metadata}")
        if metadata.get("stats_layout") != POSTERIOR_STATS_LAYOUT:
            raise ValueError(f"Unexpected stats layout in {path}: {metadata}")
        posterior_stats = payload["posterior_stats"]
        img_ids = payload["img_ids"]
        _, image_tokens = kl16_layout(metadata["image_size"])
        if posterior_stats.ndim != 3 or tuple(posterior_stats.shape[1:]) != (
            image_tokens,
            32,
        ):
            raise ValueError(
                f"Unexpected posterior_stats shape in {path}: "
                f"{tuple(posterior_stats.shape)}"
            )
        if img_ids.shape != (posterior_stats.shape[0],):
            raise ValueError(f"img_ids/posterior_stats size mismatch in {path}")
        if img_ids.dtype != torch.int64:
            raise ValueError(f"img_ids in {path} must be int64, got {img_ids.dtype}")
        validate_stats(path, posterior_stats)
        shard_count = int(metadata["num_shards"])
        expected_shards = shard_count if expected_shards is None else expected_shards
        if shard_count != expected_shards:
            raise ValueError(f"Inconsistent num_shards in {path}")
        shard_index = int(metadata["shard_index"])
        if shard_index in shard_indices:
            raise ValueError(f"Duplicate shard_index={shard_index} in {path}")
        shard_indices.add(shard_index)
        stats_parts.append(posterior_stats)
        id_parts.append(img_ids)
        source_metadata.append(metadata)

    if len(shard_paths) != expected_shards:
        raise ValueError(
            f"Found {len(shard_paths)} shards, metadata requires {expected_shards}"
        )
    if shard_indices != set(range(expected_shards)):
        raise ValueError(
            f"Shard indices are incomplete: found={sorted(shard_indices)}, "
            f"expected={list(range(expected_shards))}"
        )

    consistency_fields = (
        "format",
        "stats_layout",
        "stats_are_scaled",
        "source_mode",
        "source_manifest_jsonl",
        "source_manifest_sha256",
        "source_image_root",
        "vae_checkpoint_sha256",
        "vae_module_sha256",
        "scaling_factor",
        "image_size",
        "storage_dtype",
        "vae_dtype",
        "runtime_hashing_enabled",
        "frozen_views",
        "source_view_hashes_verified",
        "token_shape",
        "posterior_shape",
    )
    reference_metadata = source_metadata[0]
    for metadata in source_metadata[1:]:
        for field in consistency_fields:
            if metadata.get(field) != reference_metadata.get(field):
                raise ValueError(
                    f"Inconsistent shard metadata field {field!r}: "
                    f"{reference_metadata.get(field)!r} != {metadata.get(field)!r}"
                )

    img_ids = torch.cat(id_parts, dim=0)
    order = torch.argsort(img_ids)
    img_ids = img_ids[order].contiguous()
    if img_ids.numel() and bool(torch.any(img_ids[1:] <= img_ids[:-1])):
        raise ValueError("Merged cache img_ids are not unique and increasing")

    manifest_path = Path(args.manifest_jsonl) if args.manifest_jsonl else None
    if manifest_path is not None:
        manifest_img_ids = load_manifest_img_ids(manifest_path)
        if not torch.equal(img_ids, manifest_img_ids):
            raise ValueError(
                f"Merged cache ids do not exactly match {manifest_path}: "
                f"cache={img_ids.numel()}, manifest={manifest_img_ids.numel()}"
            )

    vae_checkpoint = Path(source_metadata[0]["vae_checkpoint"])
    if not vae_checkpoint.exists():
        raise FileNotFoundError(f"Missing VAE checkpoint: {vae_checkpoint}")
    if args.no_hash and reference_metadata.get("runtime_hashing_enabled") is not False:
        raise ValueError(
            "--no_hash requires shards prepared with --no_hash"
        )
    metadata = {
        "format": POSTERIOR_CACHE_FORMAT,
        "stats_layout": POSTERIOR_STATS_LAYOUT,
        "stats_are_scaled": True,
        "num_images": int(img_ids.numel()),
        "image_size": int(reference_metadata["image_size"]),
        "frozen_views": bool(reference_metadata.get("frozen_views", False)),
        "source_view_hashes_verified": bool(reference_metadata.get("source_view_hashes_verified", False)),
        "image_tokens_per_img": image_tokens,
        "image_latent_dim": 16,
        "posterior_stats_dim": 32,
        "storage_dtype": "float16",
        "vae": "mar-kl16",
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": (
            None if args.no_hash else sha256_file(vae_checkpoint)
        ),
        "vae_module_root": source_metadata[0].get("vae_module_root"),
        "vae_module_sha256": source_metadata[0].get("vae_module_sha256"),
        "encoder_device_types": sorted(
            {str(item.get("device_type")) for item in source_metadata}
        ),
        "vae_dtype": source_metadata[0].get("vae_dtype"),
        "scaling_factor": float(source_metadata[0]["scaling_factor"]),
        "source_manifest_sha256": source_metadata[0].get("source_manifest_sha256"),
        "source_image_root": source_metadata[0].get("source_image_root"),
        "source_shard_dir": str(shard_dir),
        "source_shards": [str(path) for path in shard_paths],
        "runtime_hashing_enabled": not args.no_hash,
    }
    if manifest_path is not None:
        metadata["manifest_jsonl"] = str(manifest_path)
        metadata["manifest_sha256"] = (
            None if args.no_hash else sha256_file(manifest_path)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if args.index_only:
        if output_path.exists():
            raise FileExistsError(f"use a new output path for immutable publication: {output_path}")
        row_parts = [
            torch.stack((torch.full((len(ids),), shard, dtype=torch.int64), torch.arange(len(ids))), dim=1)
            for shard, ids in enumerate(id_parts)
        ]
        row_path = Path(args.row_index_path).resolve() if args.row_index_path else output_path.with_suffix(".rows.pt")
        if row_path.exists():
            raise FileExistsError(f"use a new path for the immutable row index: {row_path}")
        row_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_rows = row_path.with_suffix(".pt.tmp")
        torch.save({"img_ids": img_ids, "shard_rows": torch.cat(row_parts)[order]}, temporary_rows)
        temporary_rows.replace(row_path)
        temporary_path.write_text(json.dumps({
            "schema": SHARD_INDEX_SCHEMA,
            "row_index": str(row_path) if args.row_index_path else row_path.name,
            "shards": [str(path.resolve()) for path in shard_paths],
            "token_shape": [image_tokens, 32],
            "metadata": metadata,
        }, indent=2) + "\n")
        temporary_path.replace(output_path)
        print(f"Indexed {len(shard_paths)} shards / {len(img_ids)} images: {output_path}")
        return
    posterior_stats = torch.cat(stats_parts, dim=0)[order].contiguous()
    torch.save(
        {
            "posterior_stats": posterior_stats,
            "img_ids": img_ids,
            "metadata": metadata,
        },
        temporary_path,
    )
    temporary_path.replace(output_path)

    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    with metadata_path.open("w") as handle:
        json.dump(
            {
                **metadata,
                "output_path": str(output_path),
                "first_img_id": int(img_ids[0]) if img_ids.numel() else None,
                "last_img_id": int(img_ids[-1]) if img_ids.numel() else None,
            },
            handle,
            indent=2,
        )
    print(
        f"Merged {len(shard_paths)} shards into {output_path} "
        f"({posterior_stats.shape[0]} images, hashing={not args.no_hash})"
    )


if __name__ == "__main__":
    main()
