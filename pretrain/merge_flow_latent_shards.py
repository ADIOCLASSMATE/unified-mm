import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mmap", action="store_true")
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output_path = Path(args.output_path)
    shard_paths = sorted(shard_dir.glob("shard-*-of-*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard-*-of-*.pt files found in {shard_dir}")

    latents = []
    img_ids = []
    metadata = []
    for path in tqdm(shard_paths, desc="Loading latent cache shards", unit="shard"):
        payload = torch.load(path, map_location="cpu", mmap=args.mmap)
        latents.append(payload["latents"])
        img_ids.append(payload["img_ids"])
        metadata.append(payload.get("metadata", {}))

    latents = torch.cat(latents, dim=0)
    img_ids = torch.cat(img_ids, dim=0)
    order = torch.argsort(img_ids)
    latents = latents[order].contiguous()
    img_ids = img_ids[order].contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": latents,
            "img_ids": img_ids,
            "metadata": {
                "num_images": int(latents.shape[0]),
                "image_tokens_per_img": int(latents.shape[1]),
                "image_latent_dim": int(latents.shape[2]),
                "source_shard_dir": str(shard_dir),
                "source_shards": [str(path) for path in shard_paths],
                "source_metadata": metadata,
            },
        },
        output_path,
    )
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    with metadata_path.open("w") as f:
        json.dump(
            {
                "num_images": int(latents.shape[0]),
                "image_tokens_per_img": int(latents.shape[1]),
                "image_latent_dim": int(latents.shape[2]),
                "output_path": str(output_path),
                "source_shards": [str(path) for path in shard_paths],
            },
            f,
            indent=2,
        )
    print(f"Merged {len(shard_paths)} shards into {output_path} ({latents.shape[0]} images)")


if __name__ == "__main__":
    main()
