"""Explicit data-hashing policy and non-hash image/file references."""
import json
from pathlib import Path


def hashing_enabled(config=None):
    if config is None:
        path = Path(__file__).resolve().parents[1] / "configs/data_synthesis/b512_sii_v1.json"
        config = json.loads(path.read_text())
    return config.get("compute_hashes", True)


def view_reference(view):
    return view.get("view_id") or view["source_path"]


def view_binding(view, enabled=True):
    return {"view_sha256": view.get("view_sha256") if enabled else None,
            "view_id": view_reference(view)}


def same_view_binding(record, view, enabled=True):
    if enabled:
        return bool(view.get("view_sha256")) and record.get("view_sha256") == view["view_sha256"]
    return record.get("view_id") == view_reference(view)


def file_size(path):
    return Path(path).stat().st_size


def check_file_size(path, expected=None):
    size = file_size(path)
    if expected is not None and size != expected:
        raise ValueError(f"file length changed: {path}")
    return size
