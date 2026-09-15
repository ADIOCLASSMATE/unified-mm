"""Download only the complete pinned Long pool using the shared direct downloader."""
import argparse
import asyncio
import json
from pathlib import Path
import ssl

import httpx

from data_synthesis.blip3o import REPO_ID, REVISION, build_long_catalogue
from data_synthesis.config import DEFAULT_CONFIG, load_config
from data_synthesis.io import atomic_json
from scripts.download_b512_corners_v3 import init_catalogue, run
from utils.direct_network import check_direct_routes, direct_ssl_context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "download"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metadata", type=Path, help="Previously fetched pinned HF metadata with LFS blobs")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--files", type=int, default=16)
    parser.add_argument("--ranges", type=int, default=4)
    args = parser.parse_args(argv)
    if args.files < 1 or args.ranges < 1:
        parser.error("concurrency must be positive")
    config = load_config(args.config)
    args.root = args.root or Path(config["preparation_root"]) / "downloads/blip3o_long_v1"
    images = args.image_root or Path(config["image_root"])
    metadata_path = args.root / "upstream_metadata.json"
    args.root.mkdir(parents=True, exist_ok=True)
    if args.metadata:
        metadata = json.loads(args.metadata.read_text())
    elif metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        check_direct_routes()
        context = direct_ssl_context()
        context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
        with httpx.Client(trust_env=False, proxy=None, http2=False, verify=context,
                          follow_redirects=True, timeout=60) as client:
            response = client.get(f"https://huggingface.co/api/datasets/{REPO_ID}/revision/{REVISION}",
                                  params={"blobs": "true"})
            response.raise_for_status()
            metadata = response.json()
    catalogue = build_long_catalogue(metadata, images)
    if not metadata_path.exists():
        atomic_json(metadata_path, metadata)
    selected = args.root / "selected_catalogue.json"
    if selected.exists() and json.loads(selected.read_text()) != catalogue:
        raise ValueError("Long destination/scope changed; choose a new download root")
    atomic_json(selected, catalogue)
    args.catalogue = selected
    init_catalogue(args.root, selected)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
