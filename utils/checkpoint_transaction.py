"""Publish complete step checkpoints without exposing partially written state."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from utils.distributed_io import run_io_phase

INVENTORY_NAME = "checkpoint_files.json"


def publish_directory(staging: Path, destination: Path, *, replace_existing=False) -> None:
    """Keep the previous directory until the staged replacement is published."""
    previous = destination.with_name(f".{destination.name}.previous")
    if previous.exists():
        raise FileExistsError(f"publication recovery required: {previous}")
    if destination.exists():
        if not replace_existing:
            raise FileExistsError(f"destination already exists: {destination}")
        os.replace(destination, previous)
    try:
        os.replace(staging, destination)
    except Exception:
        if previous.exists() and not destination.exists():
            os.replace(previous, destination)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def write_checkpoint_inventory(directory: Path, *, world_size: int, mixed_data: bool, ema: bool) -> None:
    required = {"metadata.json"}
    for rank in range(world_size):
        required.add(f"random_states_{rank}.pkl")
        if mixed_data:
            required.add(f"data_state_rank_{rank:05d}.pt")
        if ema:
            required.add(f"ema_shard_rank_{rank:05d}.safetensors")
    if ema:
        required.add("ema_manifest.json")
    files = {str(path.relative_to(directory)): path.stat().st_size
             for path in directory.rglob("*") if path.is_file()
             and path.name not in {INVENTORY_NAME, "checkpoint_complete.json"}}
    missing = sorted(required - files.keys())
    empty = sorted(name for name, size in files.items() if size <= 0)
    if missing or empty:
        raise ValueError(f"checkpoint payload missing={missing}, empty={empty}")
    (directory / INVENTORY_NAME).write_text(json.dumps(
        {"schema": "checkpoint_files_v1", "world_size": world_size,
         "files": files, "required_files": sorted(required)}, indent=2, sort_keys=True) + "\n")


def validate_checkpoint_inventory(directory: Path) -> None:
    payload = json.loads((directory / INVENTORY_NAME).read_text())
    if payload.get("schema") != "checkpoint_files_v1" or not payload.get("files"):
        raise ValueError("invalid checkpoint file inventory")
    for name, size in payload["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid checkpoint file path: {name}")
        path = directory / relative
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"missing or truncated checkpoint file: {path}")


def checkpoint_staging_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.partial")


def prepare_checkpoint(destination: Path, accelerator, *, global_step: int) -> Path:
    """Use a sibling staging directory; never erase a published checkpoint."""
    staging = checkpoint_staging_path(destination)

    def prepare():
        if destination.exists():
            marker = destination / "checkpoint_complete.json"
            # Same-step writes must be explicit. Keeping the old checkpoint
            # silently would confuse a divergent resumed trajectory with it.
            if marker.is_file():
                raise FileExistsError(f"complete checkpoint already exists: {destination}")
            raise FileExistsError(f"incomplete legacy checkpoint requires recovery: {destination}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.parent.mkdir(parents=True, exist_ok=True)

    run_io_phase(accelerator, prepare, description=f"prepare checkpoint {global_step}", main_process_only=True)
    return staging


def publish_checkpoint(staging: Path, destination: Path, accelerator, *, global_step: int) -> None:
    def publish():
        marker = json.loads((staging / "checkpoint_complete.json").read_text())
        if int(marker["global_step"]) != int(global_step):
            raise ValueError("checkpoint completion step mismatch")
        validate_checkpoint_inventory(staging)
        if destination.exists():
            raise FileExistsError(f"checkpoint appeared during publication: {destination}")
        # Both paths share a filesystem; readers see either no destination or
        # the fully committed directory. Old checkpoints remain until pruning.
        publish_directory(staging, destination)

    run_io_phase(accelerator, publish, description=f"publish checkpoint {global_step}", main_process_only=True)
