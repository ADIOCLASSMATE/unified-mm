"""Create an ImageNet subset posterior cache from the sorted full cache."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


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
    seen = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            img_id = int(json.loads(line)["img_id"])
            if img_id in seen:
                raise ValueError(
                    f"duplicate img_id={img_id} in {path}:{line_number}"
                )
            seen.add(img_id)
            img_ids.append(img_id)
    return torch.tensor(sorted(img_ids), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_cache_path", required=True)
    parser.add_argument("--subset_manifest_jsonl", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    full_cache_path = Path(args.full_cache_path)
    subset_manifest = Path(args.subset_manifest_jsonl)
    output_path = Path(args.output_path)
    payload = torch.load(
        full_cache_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    source_metadata = payload.get("metadata", {})
    if source_metadata.get("format") != POSTERIOR_CACHE_FORMAT:
        raise ValueError(f"Unsupported full cache format: {source_metadata}")
    if source_metadata.get("stats_layout") != POSTERIOR_STATS_LAYOUT:
        raise ValueError(f"Unsupported full cache layout: {source_metadata}")

    full_img_ids = payload["img_ids"]
    if full_img_ids.numel() and bool(
        torch.any(full_img_ids[1:] <= full_img_ids[:-1])
    ):
        raise ValueError("full cache img_ids must be unique and increasing")
    subset_img_ids = load_manifest_img_ids(subset_manifest)
    positions = torch.searchsorted(full_img_ids, subset_img_ids)
    if bool(torch.any(positions >= full_img_ids.numel())) or not torch.equal(
        full_img_ids[positions], subset_img_ids
    ):
        valid = positions < full_img_ids.numel()
        matched = torch.zeros_like(valid)
        matched[valid] = full_img_ids[positions[valid]] == subset_img_ids[valid]
        missing = subset_img_ids[~matched][:8].tolist()
        raise ValueError(
            f"Subset manifest contains ids absent from full cache: {missing}"
        )

    posterior_stats = payload["posterior_stats"][positions].contiguous()
    metadata = {
        **source_metadata,
        "num_images": int(subset_img_ids.numel()),
        "source_full_cache": str(full_cache_path),
        "subset_manifest_jsonl": str(subset_manifest),
        "subset_manifest_sha256": sha256_file(subset_manifest),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(
        {
            "posterior_stats": posterior_stats,
            "img_ids": subset_img_ids,
            "metadata": metadata,
        },
        temporary_path,
    )
    temporary_path.replace(output_path)
    print(
        f"Saved {subset_img_ids.numel()} posterior rows from {full_cache_path} "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
