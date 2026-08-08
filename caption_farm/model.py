from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import canonical_json, sha256_file, sha256_text


REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
)


def validate_model_snapshot(path: Path, *, expected_quant_method: str = "fp8") -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"model snapshot is not a directory: {path}")
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"model snapshot is missing required files: {', '.join(missing)}")
    incomplete = sorted(candidate.name for candidate in path.glob("*.incomplete"))
    if incomplete:
        raise ValueError(f"model snapshot has incomplete files: {', '.join(incomplete)}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config") or {}
    quant_method = str(quantization.get("quant_method") or "").lower()
    if quant_method != expected_quant_method.lower():
        raise ValueError(f"expected quant_method={expected_quant_method}, found {quant_method or '<missing>'}")
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    shard_names = sorted(set(str(name) for name in weight_map.values()))
    shard_stats: list[dict[str, Any]] = []
    total_bytes = 0
    for name in shard_names:
        shard = path / name
        if not shard.is_file():
            raise ValueError(f"model index references missing shard: {name}")
        size = shard.stat().st_size
        if size <= 8:
            raise ValueError(f"model shard is implausibly small: {name} ({size} bytes)")
        total_bytes += size
        shard_stats.append({"name": name, "bytes": size})
    fingerprints = {
        name: sha256_file(path / name)
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "chat_template.jinja",
            "model.safetensors.index.json",
        )
    }
    fingerprint_payload = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "quant_method": quant_method,
        "files": fingerprints,
        "shards": shard_stats,
    }
    return {
        **fingerprint_payload,
        "path": str(path.resolve()),
        "shard_count": len(shard_names),
        "weight_bytes": total_bytes,
        "tensor_count": len(weight_map),
        "fingerprint": sha256_text(canonical_json(fingerprint_payload)),
    }


def fake_model_info(name: str = "fake-caption-model-v1") -> dict[str, Any]:
    fingerprint = sha256_text(name)
    return {
        "path": None,
        "architectures": ["FakeCaptionModel"],
        "model_type": "fake",
        "quant_method": "none",
        "shard_count": 0,
        "weight_bytes": 0,
        "tensor_count": 0,
        "fingerprint": fingerprint,
    }
