"""Merge and validate ImageNet KL16 posterior-cache shards."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tqdm import tqdm

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--manifest_jsonl", default=None)
    parser.add_argument("--mmap", action="store_true")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output_path = Path(args.output_path)
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
            mmap=args.mmap,
            weights_only=True,
        )
        metadata = payload.get("metadata", {})
        if metadata.get("format") != POSTERIOR_CACHE_FORMAT:
            raise ValueError(f"Unexpected cache format in {path}: {metadata}")
        if metadata.get("stats_layout") != POSTERIOR_STATS_LAYOUT:
            raise ValueError(f"Unexpected stats layout in {path}: {metadata}")
        posterior_stats = payload["posterior_stats"]
        img_ids = payload["img_ids"]
        if posterior_stats.ndim != 3 or tuple(posterior_stats.shape[1:]) != (
            256,
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
    )
    reference_metadata = source_metadata[0]
    for metadata in source_metadata[1:]:
        for field in consistency_fields:
            if metadata.get(field) != reference_metadata.get(field):
                raise ValueError(
                    f"Inconsistent shard metadata field {field!r}: "
                    f"{reference_metadata.get(field)!r} != {metadata.get(field)!r}"
                )

    posterior_stats = torch.cat(stats_parts, dim=0)
    img_ids = torch.cat(id_parts, dim=0)
    order = torch.argsort(img_ids)
    posterior_stats = posterior_stats[order].contiguous()
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
        raise FileNotFoundError(f"Cannot fingerprint VAE checkpoint: {vae_checkpoint}")
    metadata = {
        "format": POSTERIOR_CACHE_FORMAT,
        "stats_layout": POSTERIOR_STATS_LAYOUT,
        "stats_are_scaled": True,
        "num_images": int(posterior_stats.shape[0]),
        "image_tokens_per_img": 256,
        "image_latent_dim": 16,
        "posterior_stats_dim": 32,
        "storage_dtype": str(posterior_stats.dtype).removeprefix("torch."),
        "vae": "mar-kl16",
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": sha256_file(vae_checkpoint),
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
    }
    if manifest_path is not None:
        metadata["manifest_jsonl"] = str(manifest_path)
        metadata["manifest_sha256"] = sha256_file(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {
            "posterior_stats": posterior_stats,
            "img_ids": img_ids,
            "metadata": metadata,
        },
        temporary_path,
    )
    temporary_path.replace(output_path)

    output_sha256 = sha256_file(output_path)

    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    with metadata_path.open("w") as handle:
        json.dump(
            {
                **metadata,
                "output_path": str(output_path),
                "output_sha256": output_sha256,
                "first_img_id": int(img_ids[0]) if img_ids.numel() else None,
                "last_img_id": int(img_ids[-1]) if img_ids.numel() else None,
            },
            handle,
            indent=2,
        )
    print(
        f"Merged {len(shard_paths)} shards into {output_path} "
        f"({posterior_stats.shape[0]} images, sha256={output_sha256})"
    )


if __name__ == "__main__":
    main()
