"""Historical September 12 full-source catalogue builder; not used by current downloads."""
import json
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from scripts.download_b512_corners_v3 import PUBLIC, POOL, atomic_json, digest


def init_catalogue(root):
    destination = root / "archive_catalogue.json"
    if destination.exists():
        return json.loads(destination.read_text())
    discovery = root / "source_discovery"
    files = []
    for repo, family, include in (
        ("allenai/pixmo-cap", "pixmo_cap", lambda name: name.startswith("data/train-")),
        ("allenai/pixmo-points", "pixmo_points", lambda name: name.startswith("data/train-")),
        ("huggan/wikiart", "wikiart", lambda name: name.startswith("data/train-") or name == "dataset_infos.json"),
        ("ahmed-masry/ChartQA", "chartqa", lambda name: name == "ChartQA Dataset.zip"),
    ):
        info = json.loads((discovery / (repo.replace("/", "_") + ".json")).read_text())
        for entry in info["siblings"]:
            name = entry["rfilename"]
            if name != "README.md" and not include(name):
                continue
            path = POOL / "source_archives" / family / Path(name).name
            # Reuse verified original bytes; do not duplicate large old archives.
            candidates = [PUBLIC / "datasets/unified_image_pool_512_v1/source_archives" /
                          ("pixmo" if family == "pixmo_cap" else family) / Path(name).name]
            if family == "pixmo_cap" and name.endswith("00000-of-00004.parquet"):
                candidates.append(PUBLIC / "data_preparation/unified_b_512_v2/sources/pixmo_train_00000.parquet")
            if family == "wikiart" and name.endswith("00000-of-00072.parquet"):
                candidates.append(PUBLIC / "data_preparation/unified_b_512_v2/sources/wikiart_train_00000.parquet")
            existing = next((p for p in candidates if p.is_file() and p.stat().st_size == entry["size"]), None)
            files.append({"id": family + ":" + name, "source": family, "upstream_path": name,
                          "url": f"https://huggingface.co/datasets/{repo}/resolve/{info['sha']}/{quote(name)}",
                          "revision": info["sha"], "bytes": entry["size"],
                          "sha256": entry.get("lfs", {}).get("sha256"),
                          "path": str(existing or path), "reuse_existing": bool(existing)})
    anyword = json.loads((discovery / "anyword.json").read_text())
    for entry in anyword["Data"]["Files"]:
        if entry["Type"] != "blob" or entry["Path"].startswith("."):
            continue
        name = entry["Path"]
        files.append({"id": "anyword3m:" + name, "source": "anyword3m", "upstream_path": name,
                      "url": "https://modelscope.cn/api/v1/datasets/iic/AnyWord-3M/repo?" +
                             urlencode({"Revision": entry["Revision"], "FilePath": name}),
                      "revision": entry["Revision"], "bytes": entry["Size"], "sha256": entry["Sha256"],
                      "path": str(POOL / "source_archives/anyword3m" / name), "reuse_existing": False})
    docci = json.loads((discovery / "docci_files.json").read_text())
    if {r["filename"] for r in docci} != {"docci_images.tar.gz", "docci_descriptions.jsonlines", "docci_metadata.jsonlines"}:
        raise ValueError("DOCCI object discovery is incomplete")
    for entry in docci:
        if not entry["revision"]:
            raise ValueError("GCS object generation must be frozen")
        md5 = next((part.strip()[4:] for part in entry.get("md5", "").split(",")
                    if part.strip().startswith("md5=")), None)
        files.append({"id": "docci:" + entry["filename"], "source": "docci",
                      "upstream_path": entry["filename"], "url": entry["url"] + "?generation=" + entry["revision"],
                      "revision": entry["revision"], "bytes": entry["bytes"], "md5_base64": md5,
                      "path": str(POOL / "source_archives/docci" / entry["filename"]), "reuse_existing": False})
    for source, path in (
        ("openimages_relationships", PUBLIC / "data_preparation/unified_b_512_v2/sources/openimages_train_relationships.csv"),
        ("textocr", PUBLIC / "datasets/unified_image_pool_512_v1/source_archives/textocr/TextOCR_0.1_train.json"),
    ):
        files.append({"id": source + ":" + path.name, "source": source, "upstream_path": path.name,
                      "path": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
                      "revision": "frozen_local_sha256", "reuse_existing": True, "url": None})
    value = {"version": "b512-corners-download-first-v3", "created_at": time.time(),
             "imagenet_included": False, "random_sample_cap": None,
             "proxy": "disabled_per_process", "files": files,
             "declared_bytes": sum(r["bytes"] for r in files),
             "scope_note": "Complete declared train sources; DOCCI/ChartQA distribution archives also contain held-out splits, excluded at indexing. URL-backed PixMo/OpenImages/TextOCR images require the separate image download receipt."}
    atomic_json(destination, value)
    return value
