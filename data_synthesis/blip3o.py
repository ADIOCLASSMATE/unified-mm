"""Pinned BLIP3o Long archive selection and image/text identity mapping."""
from collections import Counter
from pathlib import Path, PurePosixPath
import re
import tarfile
from urllib.parse import quote

from data_synthesis.io import sha

REPO_ID = "BLIP3o/BLIP3o-Pretrain-Long-Caption"
REVISION = "e4d07091a466d1a1e35a9b0c61caddc78d14a059"
EXPECTED_GROUPS = {"sa1b": 1000, "cc12m": 1472, "journeydb": 419}
EXPECTED_TAR_BYTES = 1374049964032


def archive_group(name):
    for prefix, group in (("sa_", "sa1b"), ("webdataset_shard_", "cc12m"),
                          ("webdataset_JDB_", "journeydb")):
        if re.fullmatch(re.escape(prefix) + r"\d+\.tar", name):
            return group
    raise ValueError(f"unexpected Long archive name: {name}")


def build_long_catalogue(metadata, image_root):
    if metadata.get("id") != REPO_ID or metadata.get("sha") != REVISION:
        raise ValueError("Long source identity or pinned revision changed")
    files, groups, seen = [], Counter(), set()
    destination = Path(image_root).resolve() / "source_archives/blip3o_long"
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        if not name.endswith(".tar"):
            continue
        group = archive_group(name)
        if name in seen:
            raise ValueError("duplicate Long archive")
        seen.add(name)
        lfs = entry.get("lfs", {})
        size = entry.get("size", lfs.get("size"))
        digest = lfs.get("sha256", "")
        if not isinstance(size, int) or size <= 0 or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Long archive needs upstream length and SHA256")
        if lfs.get("size", size) != size:
            raise ValueError("Long archive size fields disagree")
        groups[group] += 1
        files.append({"id": "blip3o_long:" + name, "source": "blip3o_long",
                      "source_group": group, "upstream_path": name,
                      "path": str(destination / name), "revision": REVISION,
                      "bytes": size, "sha256": digest, "reuse_existing": False,
                      "url": f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{quote(name)}"})
    if dict(groups) != EXPECTED_GROUPS or sum(f["bytes"] for f in files) != EXPECTED_TAR_BYTES:
        raise ValueError("Long metadata is not the complete pinned 2,891-archive scope")
    return {"schema": "b512_selected_archives_v1", "source": REPO_ID, "revision": REVISION,
            "files": files, "declared_bytes": EXPECTED_TAR_BYTES,
            "source_groups": dict(groups), "nominal_images": 27000000,
            "actual_accepted_images": None, "short_and_standalone_journeydb_included": False}


def archive_image_rows(archive, receipt, compute_hashes=True):
    """Reference original tar members without extracting/copying image payloads.

    Keep the existing text bound to its own image key. One tar is indexed at a
    time; neither images nor a full-library manifest are held in memory.
    """
    archive = Path(archive).resolve()
    if receipt.get("revision") != REVISION or receipt.get("size") != archive.stat().st_size:
        raise ValueError("Long intake requires a verified pinned archive receipt")
    if receipt.get("mtime_ns") != archive.stat().st_mtime_ns or (compute_hashes and not receipt.get("sha256")):
        raise ValueError("Long archive changed since verification")
    group = archive_group(archive.name)
    with tarfile.open(archive, "r:") as handle:
        images, captions = {}, {}
        for member in handle:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe Long member path")
            suffix = path.suffix.lower()
            key = str(path.with_suffix(""))
            if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                if key in images:
                    raise ValueError("ambiguous image key in Long archive")
                images[key] = (member.name, member.offset_data, member.size)
            elif suffix == ".txt":
                if key in captions or member.size > 1 << 20:
                    raise ValueError("ambiguous or oversized Long caption")
                captions[key] = handle.extractfile(member).read().decode("utf-8")
        for key, (name, offset, size) in images.items():
            identity = archive.name + "/" + key
            reference = f"tar:{archive}::{offset}:{size}/{name}"
            row = {"source": "blip3o_long", "source_id": identity, "split": "train",
                   "local_path": reference, "view_policy": "fit_pad", "min_short_side": 512,
                   "selection_bucket": "blip3o_long_" + group, "capabilities": ["general"],
                   "annotations": [], "caption_candidates": [],
                   "upstream_split_basis": "publisher_train_pool_requires_benchmark_exclusion"}
            if captions.get(key, "").strip():
                row["caption_candidates"] = [{"image_identity": "blip3o_long:" + identity,
                    "kind": "curated_caption", "author": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "text": captions[key], "provenance": {"dataset": REPO_ID, "revision": REVISION,
                        "archive": archive.name, "archive_sha256": receipt.get("sha256"),
                        "field": key + ".txt"}}]
            yield {"key": sha(("blip3o_long:" + identity).encode()), "reference": reference, "row": row}
