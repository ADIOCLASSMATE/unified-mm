from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import canonical_json, sha256_file, sha256_text, utc_now
from .manifest import build_canonical_manifest
from .model import fake_model_info, validate_model_snapshot
from .queue import TaskStore


CONFIG_SCHEMA = "imagenet_qwen_caption_farm_config_v1"
RUN_SCHEMA = "imagenet_qwen_caption_farm_run_v1"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"configuration references unset environment variable {name}")
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("farm configuration must be a JSON object")
    value = _expand(value)
    if value.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"expected config schema {CONFIG_SCHEMA}")
    return value


def prompt_fingerprint(caption: dict[str, Any]) -> str:
    payload = {
        "prompt_version": caption["prompt_version"],
        "system_prompt": caption["system_prompt"],
        "prompt_variants": caption["prompt_variants"],
        "target_min_words": int(caption["target_min_words"]),
        "target_max_words": int(caption["target_max_words"]),
    }
    return sha256_text(canonical_json(payload))


def validate_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    caption = config["caption"]
    queue = config["queue"]
    platform = config["platform"]
    controller = config["controller"]
    model = config["model"]
    for name in ("source_manifest", "original_captions"):
        if not Path(dataset[name]).is_file():
            raise ValueError(f"dataset.{name} does not exist: {dataset[name]}")
    if str(dataset["dataset_id"]) != "imagenet" or str(dataset["version_id"]) != "v1":
        raise ValueError("the official ImageNet attachment must be dataset=imagenet version=v1")
    if str(dataset["platform_path"]) != "rclone-worker-1/imagenet/v1":
        raise ValueError("unexpected official ImageNet platform path")
    if str(dataset["container_path"]) != "/inspire/dataset/imagenet/v1":
        raise ValueError("unexpected official ImageNet container path")
    if int(caption["captions_per_image"]) < 1:
        raise ValueError("captions_per_image must be positive")
    if not caption.get("prompt_variants"):
        raise ValueError("at least one prompt variant is required")
    if float(queue["lease_seconds"]) <= float(queue["heartbeat_seconds"]) * 2:
        raise ValueError("lease_seconds must exceed twice heartbeat_seconds")
    if float(queue.get("lock_stale_seconds", 120)) >= float(
        queue.get("lock_timeout_seconds", 60)
    ):
        raise ValueError("lock_stale_seconds must be smaller than lock_timeout_seconds")
    if int(queue["claim_batch_size"]) < 1 or int(queue["max_attempts"]) < 1:
        raise ValueError("queue batch and attempt limits must be positive")
    if model["engine"] != "fake" and not model.get("snapshot_path"):
        raise ValueError("a local snapshot_path is required for a real model")
    if bool(platform.get("enabled", True)):
        if int(platform["priority"]) != 1:
            raise ValueError("caption farm jobs are hard-limited to priority=1")
        if str(platform["gpu_type"]).upper() != "H100":
            raise ValueError("caption farm jobs are hard-limited to H100")
        if str(platform["quota"]).split(",", 1)[0] != "1":
            raise ValueError("caption farm workers must use a one-GPU quota")
        if int(platform.get("nodes", 1)) != 1:
            raise ValueError("caption farm workers must use one single-GPU node")
        if platform["image"] != "docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1":
            raise ValueError("caption farm jobs must use the fixed dev-wjx:v-2.1 image")
        if not platform.get("projects"):
            raise ValueError("the project whitelist is empty")
        if not platform.get("targets"):
            raise ValueError("the H100 target whitelist is empty")
    global_max = int(controller["global_max_active_jobs"])
    target = int(controller["target_active_jobs"])
    if not (1 <= target <= global_max):
        raise ValueError("target_active_jobs must be within the global active limit")
    if float(controller.get("zero_active_attention_seconds", 300)) <= 0:
        raise ValueError("zero_active_attention_seconds must be positive")


def prepare_run(config_path: Path, run_dir: Path, *, rebuild_manifest: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config)
    if (run_dir / "run.json").exists():
        existing = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if existing.get("source_config_sha256") != sha256_file(config_path):
            raise ValueError("run directory already belongs to a different configuration")
        return existing

    model_config = config["model"]
    if model_config["engine"] == "fake":
        model_info = fake_model_info(model_config.get("repository", "fake-caption-model-v1"))
    else:
        model_info = validate_model_snapshot(
            Path(model_config["snapshot_path"]),
            expected_quant_method=model_config.get("expected_quant_method", "fp8"),
        )

    manifest_path = run_dir / "canonical_manifest.jsonl"
    if manifest_path.exists() and not rebuild_manifest:
        raise ValueError("orphan canonical manifest exists without run.json; pass --rebuild-manifest")
    manifest_info = build_canonical_manifest(
        Path(config["dataset"]["source_manifest"]),
        Path(config["dataset"]["original_captions"]),
        manifest_path,
        dataset_mount=Path(config["dataset"]["container_path"]),
    )

    caption = deepcopy(config["caption"])
    caption["prompt_fingerprint"] = prompt_fingerprint(caption)
    model = deepcopy(model_config)
    model.update(model_info)
    run_fingerprint_payload = {
        "manifest_sha256": manifest_info.sha256,
        "model_fingerprint": model["fingerprint"],
        "prompt_fingerprint": caption["prompt_fingerprint"],
        "captions_per_image": int(caption["captions_per_image"]),
        "output_version": config["output"]["version"],
    }
    run_fingerprint = sha256_text(canonical_json(run_fingerprint_payload))
    run = {
        "schema": RUN_SCHEMA,
        "created_at": utc_now(),
        "source_config": str(config_path.resolve()),
        "source_config_sha256": sha256_file(config_path),
        "run_fingerprint": run_fingerprint,
        "run_slug": run_fingerprint[:12],
        "dataset": config["dataset"],
        "manifest": {
            "path": str(manifest_info.path.resolve()),
            "offsets_path": str(manifest_info.offsets_path.resolve()),
            "metadata_path": str(manifest_info.metadata_path.resolve()),
            "row_count": manifest_info.row_count,
            "sha256": manifest_info.sha256,
            "source_manifest_sha256": manifest_info.source_manifest_sha256,
            "original_captions_sha256": manifest_info.original_captions_sha256,
        },
        "model": model,
        "caption": caption,
        "queue": config["queue"],
        "platform": config["platform"],
        "controller": config["controller"],
        "output": config["output"],
    }
    TaskStore.initialize(run_dir, run).close()
    return run
