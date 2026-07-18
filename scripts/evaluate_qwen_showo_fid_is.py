#!/usr/bin/env python3
"""Evaluate Qwen-Show-o on the fixed ImageNet-100 validation protocol.

This evaluator is deliberately independent of the continuous flow/KL-VAE path:

* prompts are class names from the balanced ImageNet-100 manifest;
* image tokens are sampled with the model's MaskGIT + CFG API;
* tokens are decoded by the official Show-o MAGVITv2 (8192-code) tokenizer;
* FID uses a separately cached *original ImageNet* reference distribution;
* IS and generated FID moments are reduced exactly across distributed ranks.

Heavy ML imports are delayed until ``main`` so the manifest/protocol helpers can
be unit-tested in lightweight environments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl"
DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT
    / "public/datasets/imagenet_ablation_100c_balanced/split_seed42_val100.jsonl"
)
DEFAULT_SYNSET_MAPPING = REPO_ROOT / "public/datasets/imagenet/LOC_synset_mapping.txt"
DEFAULT_REAL_STATS = (
    REPO_ROOT
    / "public/datasets/imagenet_ablation_100c_balanced/fid_stats"
    / "inception_v3_2048_original_256.pt"
)
DEFAULT_MAGVIT_PATH = REPO_ROOT / "public/models/showlab/magvitv2"
DEFAULT_SHOWO_ROOT = Path(
    "/inspire/hdd/global_user/wanjiaxin-253108030048/code/Show-o"
)
DEFAULT_INCEPTION_WEIGHTS = (
    REPO_ROOT
    / "output/cache/inception/weights-inception-2015-12-05-6726825d.pth"
)
REAL_STATS_SCHEMA = "qwen_showo_imagenet100_real_stats_v1"
PROTOCOL_NAME = "imagenet100-balanced-val100-per-class-class-name-v1"
IMAGE_TOKEN_COUNT = 256
IMAGE_VOCAB_SIZE = 8192


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "img_id" not in row or "synset" not in row or "source_path" not in row:
                raise ValueError(
                    f"{path}:{line_number} must contain img_id, synset, and source_path"
                )
            img_id = int(row["img_id"])
            if img_id in seen_ids:
                raise ValueError(f"duplicate img_id={img_id} in {path}")
            seen_ids.add(img_id)
            records.append(
                {
                    "manifest_index": len(records),
                    "img_id": img_id,
                    "synset": str(row["synset"]),
                    "source_path": str(row["source_path"]),
                }
            )
    if not records:
        raise ValueError(f"empty ImageNet manifest: {path}")
    return records


def load_synset_names(path: str | Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            synset, separator, raw_names = line.partition(" ")
            if not separator:
                raise ValueError(f"invalid synset mapping row: {line!r}")
            class_name = raw_names.split(",", 1)[0].strip()
            names[synset] = class_name or synset
    if not names:
        raise ValueError(f"empty synset mapping: {path}")
    return names


def build_fixed_val_records(
    records: Sequence[Mapping[str, Any]],
    synset_names: Mapping[str, str],
    *,
    val_samples_per_class: int = 100,
    split_seed: int = 42,
) -> list[dict[str, Any]]:
    """Reproduce ``ImageNetFlowCacheDataset``'s stratified validation split."""
    if val_samples_per_class <= 0:
        raise ValueError("val_samples_per_class must be positive")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(str(record["synset"]), []).append(record)
    if len(groups) != 100:
        raise ValueError(
            f"{PROTOCOL_NAME} requires exactly 100 classes, found {len(groups)}"
        )

    missing_names = sorted(set(groups) - set(synset_names))
    if missing_names:
        raise ValueError(f"missing class names for synsets: {missing_names[:8]}")

    rng = random.Random(int(split_seed))
    selected: list[dict[str, Any]] = []
    for synset in sorted(groups):
        group = list(groups[synset])
        if len(group) <= val_samples_per_class:
            raise ValueError(
                f"synset {synset} has {len(group)} rows; need more than "
                f"{val_samples_per_class} to preserve a training split"
            )
        rng.shuffle(group)
        for record in group[:val_samples_per_class]:
            selected.append(
                {
                    "manifest_index": int(record["manifest_index"]),
                    "img_id": int(record["img_id"]),
                    "synset": synset,
                    "source_path": str(record["source_path"]),
                    "class_name": str(synset_names[synset]),
                    "prompt": str(synset_names[synset]),
                }
            )
    # This final shuffle is also part of the training dataloader split helper.
    rng.shuffle(selected)
    for evaluation_index, record in enumerate(selected):
        record["evaluation_index"] = evaluation_index
    return selected


def load_fixed_val_records(
    manifest_records: Sequence[Mapping[str, Any]],
    split_manifest_path: str | Path,
    synset_names: Mapping[str, str],
    *,
    expected_classes: int = 100,
    expected_samples_per_class: int = 100,
) -> list[dict[str, Any]]:
    """Load the authoritative cache-order-derived validation split manifest."""
    split_path = Path(split_manifest_path)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"authoritative validation split manifest not found: {split_path}. "
            "Build the ImageNet-100 MAGVIT cache/split before caching FID stats."
        )
    by_id = {int(record["img_id"]): record for record in manifest_records}
    by_index = {
        int(record["manifest_index"]): record for record in manifest_records
    }
    rows: list[Mapping[str, Any]] = []
    with split_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping) and "validation_img_ids" in payload:
                rows.extend(
                    {"img_id": int(img_id), "split": "validation"}
                    for img_id in payload["validation_img_ids"]
                )
                continue
            if not isinstance(payload, Mapping):
                raise ValueError(f"{split_path}:{line_number} must be a JSON object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"empty validation split manifest: {split_path}")

    validation_names = {"validation", "val", "valid"}
    validation_rows = [
        row
        for row in rows
        if str(row.get("split", "validation")).lower() in validation_names
    ]
    has_split_indices = [row.get("split_index") is not None for row in validation_rows]
    if any(has_split_indices):
        if not all(has_split_indices):
            raise ValueError(
                f"{split_path}: every validation row must contain split_index"
            )
        validation_rows.sort(key=lambda row: int(row["split_index"]))
        for expected_index, row in enumerate(validation_rows):
            if int(row["split_index"]) != expected_index:
                raise ValueError(
                    f"{split_path}: validation split_index values must be contiguous "
                    f"from 0; found {row['split_index']} at sorted position "
                    f"{expected_index}"
                )

    selected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row_number, row in enumerate(validation_rows):
        if row.get("img_id") is not None:
            img_id = int(row["img_id"])
            source = by_id.get(img_id)
        elif row.get("manifest_index") is not None:
            source = by_index.get(int(row["manifest_index"]))
            img_id = int(source["img_id"]) if source is not None else -1
        else:
            raise ValueError(
                f"{split_path}: validation row {row_number + 1} must contain "
                "img_id or manifest_index"
            )
        if source is None:
            raise ValueError(
                f"{split_path}: img_id={img_id} is absent from the dataset manifest"
            )
        if img_id in seen_ids:
            raise ValueError(f"duplicate validation img_id={img_id} in {split_path}")
        seen_ids.add(img_id)
        synset = str(source["synset"])
        if row.get("synset") is not None and str(row["synset"]) != synset:
            raise ValueError(
                f"{split_path}: img_id={img_id} synset={row['synset']!r} "
                f"does not match manifest synset={synset!r}"
            )
        if synset not in synset_names:
            raise ValueError(f"class name missing for validation synset={synset}")
        evaluation_index = len(selected)
        declared_index = row.get("evaluation_index", row.get("split_index"))
        if declared_index is not None and int(declared_index) != evaluation_index:
            raise ValueError(
                f"{split_path}: validation index must match sorted split order; row "
                f"{row_number + 1} has {declared_index}, expected "
                f"{evaluation_index}"
            )
        selected.append(
            {
                "manifest_index": int(source["manifest_index"]),
                "img_id": img_id,
                "synset": synset,
                "source_path": str(source["source_path"]),
                "class_name": str(synset_names[synset]),
                "prompt": str(synset_names[synset]),
                "evaluation_index": evaluation_index,
            }
        )

    class_counts: dict[str, int] = {}
    for record in selected:
        synset = str(record["synset"])
        class_counts[synset] = class_counts.get(synset, 0) + 1
    expected_total = int(expected_classes) * int(expected_samples_per_class)
    if len(selected) != expected_total:
        raise ValueError(
            f"{split_path} contains {len(selected)} validation rows; "
            f"expected {expected_total}"
        )
    if len(class_counts) != int(expected_classes):
        raise ValueError(
            f"{split_path} contains {len(class_counts)} classes; "
            f"expected {expected_classes}"
        )
    bad_counts = {
        synset: count
        for synset, count in class_counts.items()
        if count != int(expected_samples_per_class)
    }
    if bad_counts:
        raise ValueError(
            f"{split_path} is not balanced at {expected_samples_per_class} "
            f"validation rows per class: {dict(list(sorted(bad_counts.items()))[:8])}"
        )
    return selected


def selection_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "evaluation_index": int(record["evaluation_index"]),
            "manifest_index": int(record["manifest_index"]),
            "img_id": int(record["img_id"]),
            "synset": str(record["synset"]),
            "source_path": str(record["source_path"]),
            "class_name": str(record["class_name"]),
            "prompt": str(record["prompt"]),
        }
        for record in records
    ]
    return sha256_json(canonical)


def metric_transform_metadata(image_size: int = 256) -> dict[str, Any]:
    return {
        "input_color": "RGB",
        "resize": {
            "type": "shorter_side",
            "size": int(image_size),
            "interpolation": "bicubic",
            "antialias": True,
        },
        "crop": {"type": "center", "height": int(image_size), "width": int(image_size)},
        "tensor_range": [0.0, 1.0],
        "inception_input": "uint8_0_255",
        "quantization": "multiply_255_then_uint8_truncate",
        "inception_resize": {
            "size": 299,
            "interpolation": "bilinear",
            "antialias": True,
        },
    }


def feature_metadata(
    feature: int,
    inception_weights_path: str | Path,
) -> dict[str, Any]:
    weights = Path(inception_weights_path).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Inception weights do not exist: {weights}")
    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "backend": "torchmetrics.NoTrainInceptionV3/torch-fidelity",
        "extractor_antialias": True,
        "feature": int(feature),
        "feature_name": str(int(feature)),
        "logits_name": "logits_unbiased",
        "weights_sha256": sha256_file(weights),
        "weights_filename": weights.name,
        # Informational only. Validation keys on the content hash, not machine path.
        "weights_path": str(weights),
        "software": {
            "torch": package_version("torch"),
            "torchmetrics": package_version("torchmetrics"),
            "torch-fidelity": package_version("torch-fidelity"),
        },
    }


def build_expected_real_metadata(
    *,
    manifest_path: str | Path,
    split_manifest_path: str | Path,
    selected_records: Sequence[Mapping[str, Any]],
    transform: Mapping[str, Any],
    feature: Mapping[str, Any],
    val_samples_per_class: int = 100,
    split_seed: int = 42,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    class_counts: dict[str, int] = {}
    for record in selected_records:
        synset = str(record["synset"])
        class_counts[synset] = class_counts.get(synset, 0) + 1
    return {
        "schema": REAL_STATS_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "real_source": "original_imagenet",
        "manifest_sha256": sha256_file(manifest),
        "manifest_filename": manifest.name,
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_manifest_filename": Path(split_manifest_path).name,
        "selection_sha256": selection_fingerprint(selected_records),
        "num_samples": len(selected_records),
        "num_classes": len(class_counts),
        "class_counts": dict(sorted(class_counts.items())),
        "split": {
            "source": "authoritative_split_manifest",
            "order": "validation_split_index",
            "strategy": "stratified",
            "seed": int(split_seed),
            "val_samples_per_class": int(val_samples_per_class),
        },
        "prompt": {
            "type": "class_name",
            "mapping": "first comma-separated LOC_synset_mapping name",
        },
        "transform": dict(transform),
        "feature": dict(feature),
    }


def validate_real_stats_metadata(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Strictly reject reference stats produced by a different protocol."""
    required_paths = (
        ("schema",),
        ("protocol",),
        ("real_source",),
        ("manifest_sha256",),
        ("split_manifest_sha256",),
        ("selection_sha256",),
        ("num_samples",),
        ("num_classes",),
        ("class_counts",),
        ("split",),
        ("prompt",),
        ("transform",),
        ("feature", "backend"),
        ("feature", "extractor_antialias"),
        ("feature", "feature"),
        ("feature", "feature_name"),
        ("feature", "logits_name"),
        ("feature", "weights_sha256"),
        ("feature", "software"),
    )

    missing = object()

    def lookup(root: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        value: Any = root
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                return missing
            value = value[key]
        return value

    mismatches = []
    for path in required_paths:
        actual_value = lookup(actual, path)
        expected_value = lookup(expected, path)
        if actual_value != expected_value:
            mismatches.append(
                f"{'.'.join(path)}: "
                f"cached={'<missing>' if actual_value is missing else repr(actual_value)}, "
                f"expected={'<missing>' if expected_value is missing else repr(expected_value)}"
            )
    if mismatches:
        raise ValueError(
            "Real Inception statistics are incompatible with this evaluation:\n  - "
            + "\n  - ".join(mismatches)
            + "\nRebuild them with scripts/cache_imagenet100_real_stats.py."
        )


def deterministic_sample_seed(base_seed: int, record: Mapping[str, Any]) -> int:
    payload = (
        f"{PROTOCOL_NAME}|{int(base_seed)}|{int(record['evaluation_index'])}|"
        f"{int(record['img_id'])}|{record['synset']}"
    ).encode("utf-8")
    # Keep seeds in the range accepted by torch.Generator.manual_seed.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def resolve_original_image_path(
    record: Mapping[str, Any],
    imagenet_train_dir: str | Path,
) -> Path:
    """Resolve old absolute manifest paths against a potentially new mount."""
    source = Path(str(record["source_path"]))
    if source.is_file():
        return source
    root = Path(imagenet_train_dir)
    candidates = [
        root / str(record["synset"]) / source.name,
        root / source.name,
        root / source,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"ImageNet source for img_id={record['img_id']} not found. "
        f"manifest={source}; tried={[str(path) for path in candidates]}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed Qwen-Show-o ImageNet-100 MaskGIT FID/IS evaluation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="HF checkpoint directory, or an Accelerator/PyTorch model state.",
    )
    parser.add_argument("--output_dir", default="output/qwen_showo_imagenet100_fid_is")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split_manifest", default=str(DEFAULT_SPLIT_MANIFEST))
    parser.add_argument("--synset_mapping", default=str(DEFAULT_SYNSET_MAPPING))
    parser.add_argument("--real_stats", default=str(DEFAULT_REAL_STATS))
    parser.add_argument(
        "--inception_weights_path",
        default="",
        help="Optional override. If omitted, use the path recorded by --real_stats.",
    )
    parser.add_argument("--showo_root", default=str(DEFAULT_SHOWO_ROOT))
    parser.add_argument("--magvit_path", default=str(DEFAULT_MAGVIT_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_samples_per_class", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--fid_feature", type=int, default=2048)
    parser.add_argument("--is_splits", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=12)
    parser.add_argument("--guidance_scale", type=float, default=11.75)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mask_schedule", choices=["cosine"], default="cosine")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Smoke-test escape hatch. The official score uses the default full 10K set.",
    )
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def init_distributed(requested_device: str):
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if not distributed and (rank != 0 or local_rank != 0):
        raise RuntimeError(
            "RANK/LOCAL_RANK is set but WORLD_SIZE is not >1; launch with torchrun "
            "or clear the distributed environment."
        )
    requested = str(requested_device).lower()
    if requested != "cpu" and torch.cuda.is_available():
        if distributed or requested in {"auto", "cuda"}:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(requested_device)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return distributed, rank, world_size, local_rank, device


def distributed_barrier(distributed: bool, device) -> None:
    import torch.distributed as dist

    if not distributed:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[int(device.index or 0)])
    else:
        dist.barrier()


def build_metric_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                int(image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
        ]
    )


def build_inception_extractor(
    feature: int,
    weights_path: str | Path | None,
    device,
):
    from torchmetrics.image.fid import NoTrainInceptionV3

    class FeaturesAndLogitsInceptionV3(NoTrainInceptionV3):
        """Expose all requested torch-fidelity outputs, not only the first."""

        def forward(self, images):
            return self._torch_fidelity_forward(images)

    extractor = FeaturesAndLogitsInceptionV3(
        name="inception-v3-compat",
        features_list=[str(int(feature)), "logits_unbiased"],
        feature_extractor_weights_path=(
            str(weights_path) if weights_path is not None else None
        ),
        antialias=True,
    )
    return extractor.to(device).eval()


def extract_inception_features(extractor, images):
    """Return float64 FID features and float64 unbiased logits."""
    import torch

    # Match torchmetrics normalize=True exactly: multiplication followed by
    # uint8 conversion (truncation, not rounding).
    uint8_images = images.detach().float().clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    outputs = extractor(uint8_images)
    if isinstance(outputs, torch.Tensor):
        raise RuntimeError(
            "Inception extractor returned one tensor; expected FID features and logits."
        )
    if isinstance(outputs, Mapping):
        feature_keys = [key for key in outputs if key != "logits_unbiased"]
        feature = outputs.get(feature_keys[0]) if len(feature_keys) == 1 else None
        logits = outputs.get("logits_unbiased")
    else:
        feature, logits = outputs
    if feature is None or logits is None:
        raise RuntimeError("Inception extractor did not return features and logits_unbiased")
    return feature.reshape(feature.shape[0], -1).double(), logits.reshape(
        logits.shape[0], -1
    ).double()


@dataclass
class FeatureMoments:
    count: Any
    sum: Any
    outer_sum: Any

    @classmethod
    def zeros(cls, dimension: int, device):
        import torch

        return cls(
            count=torch.zeros((), dtype=torch.long, device=device),
            sum=torch.zeros(int(dimension), dtype=torch.float64, device=device),
            outer_sum=torch.zeros(
                int(dimension), int(dimension), dtype=torch.float64, device=device
            ),
        )

    def update(self, features) -> None:
        features = features.to(device=self.sum.device, dtype=self.sum.dtype)
        self.count += int(features.shape[0])
        self.sum += features.sum(dim=0)
        self.outer_sum += features.T @ features

    def all_reduce_(self) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self.count)
        dist.all_reduce(self.sum)
        dist.all_reduce(self.outer_sum)

    def mean_cov(self):
        count = int(self.count.item())
        if count < 2:
            raise ValueError(f"at least two features are required, found {count}")
        mean = self.sum / count
        covariance = (self.outer_sum - count * mean[:, None] * mean[None, :]) / (
            count - 1
        )
        return mean, (covariance + covariance.T) * 0.5


@dataclass
class InceptionScoreMoments:
    count: Any
    probability_sum: Any
    probability_log_probability_sum: Any

    @classmethod
    def zeros(cls, splits: int, classes: int, device):
        import torch

        return cls(
            count=torch.zeros(int(splits), dtype=torch.long, device=device),
            probability_sum=torch.zeros(
                int(splits), int(classes), dtype=torch.float64, device=device
            ),
            probability_log_probability_sum=torch.zeros(
                int(splits), dtype=torch.float64, device=device
            ),
        )

    def update(
        self,
        logits,
        global_indices: Sequence[int],
        total_samples: int,
    ) -> None:
        import torch

        probabilities = logits.double().softmax(dim=-1)
        indices = torch.as_tensor(global_indices, device=logits.device, dtype=torch.long)
        split_ids = torch.div(
            indices * int(self.count.numel()),
            int(total_samples),
            rounding_mode="floor",
        ).clamp_max(self.count.numel() - 1)
        p_log_p = (
            probabilities
            * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=-1)
        for split in split_ids.unique().tolist():
            mask = split_ids == int(split)
            self.count[split] += int(mask.sum().item())
            self.probability_sum[split] += probabilities[mask].sum(dim=0)
            self.probability_log_probability_sum[split] += p_log_p[mask].sum()

    def all_reduce_(self) -> None:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.all_reduce(self.count)
        dist.all_reduce(self.probability_sum)
        dist.all_reduce(self.probability_log_probability_sum)

    def compute(self) -> tuple[float, float, list[float]]:
        import torch

        scores = []
        for split in range(int(self.count.numel())):
            count = int(self.count[split].item())
            if count <= 0:
                raise ValueError(f"Inception Score split {split} is empty")
            marginal = self.probability_sum[split] / count
            expected_p_log_p = self.probability_log_probability_sum[split] / count
            marginal_entropy_term = (
                marginal
                * marginal.clamp_min(torch.finfo(marginal.dtype).tiny).log()
            ).sum()
            scores.append(float(torch.exp(expected_p_log_p - marginal_entropy_term)))
        values = torch.tensor(scores, dtype=torch.float64)
        return float(values.mean()), float(values.std(unbiased=False)), scores


def frechet_distance(real_mean, real_cov, fake_mean, fake_cov) -> float:
    """Numerically stable FID using symmetric eigendecompositions."""
    import torch

    real_mean = real_mean.detach().cpu().double()
    fake_mean = fake_mean.detach().cpu().double()
    real_cov = (real_cov.detach().cpu().double() + real_cov.T.detach().cpu().double()) * 0.5
    fake_cov = (fake_cov.detach().cpu().double() + fake_cov.T.detach().cpu().double()) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(real_cov)
    real_sqrt = (eigenvectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(0)) @ eigenvectors.T
    middle = real_sqrt @ fake_cov @ real_sqrt
    middle = (middle + middle.T) * 0.5
    trace_sqrt_product = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum()
    difference = real_mean - fake_mean
    fid = difference.dot(difference) + torch.trace(real_cov) + torch.trace(
        fake_cov
    ) - 2.0 * trace_sqrt_product
    return float(fid.clamp_min(0))


def _model_dtype(model):
    import torch

    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def build_prompt_batch(
    prompts: Sequence[str],
    tokenizer,
    model_config,
    *,
    t2i_prefix: str,
    class_prompt_template: str,
    pad_to_multiple_of: int | None,
    device,
) -> dict[str, Any]:
    from utils.dataset_qwen_showo_imagenet import (
        build_qwen_showo_generation_batch,
    )

    image_tokens = int(getattr(model_config, "image_tokens_per_img", IMAGE_TOKEN_COUNT))
    if image_tokens != IMAGE_TOKEN_COUNT:
        raise ValueError(
            f"official MAGVITv2 protocol requires 256 tokens, got {image_tokens}"
        )
    batch = build_qwen_showo_generation_batch(
        prompts,
        tokenizer,
        t2i_token_id=int(model_config.t2i_token_id),
        boi_token_id=int(model_config.boi_token_id),
        eoi_token_id=int(model_config.eoi_token_id),
        image_mask_token_id=int(model_config.image_mask_token_id),
        image_tokens_per_img=image_tokens,
        class_prompt_template=str(class_prompt_template),
        t2i_prefix=str(t2i_prefix),
        pad_to_multiple_of=pad_to_multiple_of,
        attention_dtype=_model_dtype_from_config(model_config),
    )
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _model_dtype_from_config(model_config):
    import torch

    raw = str(getattr(model_config, "torch_dtype", "")).lower()
    if "bfloat16" in raw:
        return torch.bfloat16
    if "float16" in raw or raw.endswith("half"):
        return torch.float16
    return torch.float32


def _slice_batch(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {key: value[index : index + 1] for key, value in batch.items()}


def _accepted_kwargs(method, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(method)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _normalize_sampled_tokens(output, image_token_mask, expected_tokens: int):
    import torch

    if isinstance(output, Mapping):
        for key in ("image_tokens", "generated_image_tokens", "token_ids", "tokens"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"MaskGIT sampler returned unsupported value: {type(output)!r}")
    if output.ndim != 2:
        raise ValueError(f"sampled image tokens must be rank 2, got {tuple(output.shape)}")
    if output.shape[1] != expected_tokens:
        if output.shape == image_token_mask.shape:
            output = torch.stack(
                [row[mask] for row, mask in zip(output, image_token_mask)], dim=0
            )
        else:
            raise ValueError(
                f"sampler returned {output.shape[1]} tokens; expected {expected_tokens}"
            )
    if int(output.min()) < 0 or int(output.max()) >= IMAGE_VOCAB_SIZE:
        raise ValueError(
            "MaskGIT returned token IDs outside official MAGVITv2 range "
            f"[0,{IMAGE_VOCAB_SIZE}): [{int(output.min())},{int(output.max())}]"
        )
    return output.long()


def call_maskgit_sampler(
    model,
    conditional_batch: Mapping[str, Any],
    unconditional_batch: Mapping[str, Any],
    *,
    sample_seeds: Sequence[int],
    timesteps: int,
    guidance_scale: float,
    temperature: float,
    mask_schedule: str,
):
    """Call the model-owned MaskGIT sampler with deterministic per-sample RNG."""
    import torch

    method = None
    for name in (
        "generate_image_tokens_maskgit",
        "sample_image_tokens_maskgit",
        "t2i_generate_maskgit",
        "t2i_generate",
    ):
        candidate = getattr(model, name, None)
        if callable(candidate):
            method = candidate
            break
    if method is None:
        raise AttributeError(
            "Qwen-Show-o model must expose generate_image_tokens_maskgit "
            "(preferred), sample_image_tokens_maskgit, t2i_generate_maskgit, or t2i_generate."
        )

    signature = inspect.signature(method)
    supports_batched_seeds = "sample_seeds" in signature.parameters
    supports_generators = "generators" in signature.parameters

    def invoke(cond, uncond, seeds):
        generators = []
        for seed in seeds:
            generator = torch.Generator(device=cond["input_ids"].device)
            generator.manual_seed(int(seed))
            generators.append(generator)
        kwargs = {
            "input_ids": cond["input_ids"],
            "token_types": cond["token_types"],
            "attention_mask": cond["attention_mask"].to(_model_dtype(model)),
            "image_token_mask": cond["image_token_mask"],
            "uncond_input_ids": uncond["input_ids"],
            "uncond_token_types": uncond["token_types"],
            "uncond_attention_mask": uncond["attention_mask"].to(_model_dtype(model)),
            "uncond_image_token_mask": uncond["image_token_mask"],
            "timesteps": int(timesteps),
            "generation_timesteps": int(timesteps),
            "guidance_scale": float(guidance_scale),
            "temperature": float(temperature),
            "mask_schedule": str(mask_schedule),
            "noise_schedule": str(mask_schedule),
            "seq_len": IMAGE_TOKEN_COUNT,
        }
        if supports_batched_seeds:
            kwargs["sample_seeds"] = torch.tensor(
                seeds, dtype=torch.long, device=cond["input_ids"].device
            )
        elif supports_generators:
            kwargs["generators"] = generators
        else:
            kwargs["generator"] = generators[0]
        result = method(**_accepted_kwargs(method, kwargs))
        return _normalize_sampled_tokens(
            result, cond["image_token_mask"], IMAGE_TOKEN_COUNT
        )

    if supports_batched_seeds or supports_generators or len(sample_seeds) == 1:
        return invoke(conditional_batch, unconditional_batch, sample_seeds)

    # Exact, rank/batch-size invariant fallback for APIs accepting one generator.
    rows = []
    for index, seed in enumerate(sample_seeds):
        rows.append(
            invoke(
                _slice_batch(conditional_batch, index),
                _slice_batch(unconditional_batch, index),
                [seed],
            )
        )
    return torch.cat(rows, dim=0)


def load_official_magvit(
    showo_root: str | Path,
    pretrained_path: str | Path,
    device,
):
    """Load Show-o's local MAGVITv2 without colliding with this repo's models package."""
    import importlib.util
    import types

    package_name = "_qwen_showo_official_models"
    models_root = Path(showo_root).expanduser().resolve() / "models"
    model_file = models_root / "modeling_magvitv2.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"official Show-o MAGVITv2 source not found: {model_file}")
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(models_root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    module_name = f"{package_name}.modeling_magvitv2"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import official MAGVITv2 from {model_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    cls = sys.modules[module_name].MAGVITv2
    pretrained = Path(pretrained_path).expanduser().resolve()
    weights = pretrained / "pytorch_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"official MAGVITv2 weights not found: {weights}")
    # Show-o vendors an older diffusers ModelMixin whose from_pretrained currently
    # calls a removed ``load_state_dict(..., variant=...)`` signature. Loading the
    # exact safetensors state directly is version-independent.
    from safetensors.torch import load_file

    model = cls()
    state = load_file(str(weights), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"official MAGVITv2 state mismatch: missing={missing[:12]}, "
            f"unexpected={unexpected[:12]}"
        )
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def load_qwen_showo_model(config_path: str | Path, checkpoint: str | Path, device):
    from omegaconf import OmegaConf

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from pretrain.train_qwen_showo import load_qwen_showo_model_tokenizer

    config = OmegaConf.load(config_path)
    checkpoint_path = Path(checkpoint).expanduser()
    is_hf_checkpoint = checkpoint_path.is_dir() and (
        checkpoint_path / "config.json"
    ).is_file()
    if is_hf_checkpoint:
        model, tokenizer = load_qwen_showo_model_tokenizer(
            config, model_path=str(checkpoint_path)
        )
    else:
        model, tokenizer = load_qwen_showo_model_tokenizer(config)
    if not is_hf_checkpoint:
        import torch

        state_path = checkpoint_path
        if checkpoint_path.is_dir():
            candidates = [
                checkpoint_path / "model.safetensors",
                checkpoint_path / "pytorch_model.bin",
            ]
            state_path = next((path for path in candidates if path.is_file()), state_path)
        if not state_path.is_file():
            raise FileNotFoundError(
                f"checkpoint is neither an HF directory nor a model state: {checkpoint_path}"
            )
        if state_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(state_path), device="cpu")
        else:
            state = torch.load(state_path, map_location="cpu")
            if isinstance(state, Mapping) and "state_dict" in state:
                state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint state mismatch: missing={missing[:12]}, "
                f"unexpected={unexpected[:12]}"
            )
    return model.to(device).eval(), tokenizer, config


def save_generated_images(images, records, output_dir: Path) -> None:
    from torchvision.utils import save_image

    output_dir.mkdir(parents=True, exist_ok=True)
    for image, record in zip(images, records):
        save_image(
            image.detach().cpu().float().clamp(0, 1),
            output_dir / f"{int(record['evaluation_index']):08d}.png",
        )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_batches(records: Sequence[Mapping[str, Any]], batch_size: int):
    for start in range(0, len(records), int(batch_size)):
        yield records[start : start + int(batch_size)]


def _resolve_weights_from_cache(
    cached_metadata: Mapping[str, Any],
    override: str,
) -> Path:
    if override:
        path = Path(override).expanduser().resolve()
    else:
        recorded = cached_metadata.get("feature", {}).get("weights_path", "")
        path = Path(recorded).expanduser().resolve() if recorded else DEFAULT_INCEPTION_WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(
            f"Inception weights are unavailable at {path}. Pass "
            "--inception_weights_path with the same file used to cache real stats."
        )
    return path


def _cache_moments(payload: Mapping[str, Any], device) -> FeatureMoments:
    import torch

    stats = payload.get("stats", payload)
    return FeatureMoments(
        count=torch.as_tensor(stats["count"], dtype=torch.long, device=device),
        sum=torch.as_tensor(stats["sum"], dtype=torch.float64, device=device),
        outer_sum=torch.as_tensor(
            stats["outer_sum"], dtype=torch.float64, device=device
        ),
    )


def validate_cached_moment_shapes(
    moments: FeatureMoments,
    expected_dimension: int,
) -> None:
    dimension = int(expected_dimension)
    if tuple(moments.sum.shape) != (dimension,):
        raise ValueError(
            f"cached feature sum has shape {tuple(moments.sum.shape)}; "
            f"expected {(dimension,)}"
        )
    if tuple(moments.outer_sum.shape) != (dimension, dimension):
        raise ValueError(
            f"cached feature outer_sum has shape {tuple(moments.outer_sum.shape)}; "
            f"expected {(dimension, dimension)}"
        )


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def validate_protocol_settings(
    *,
    split_seed: int,
    val_samples_per_class: int,
    image_size: int,
    fid_feature: int,
) -> None:
    expected = {
        "split_seed": 42,
        "val_samples_per_class": 100,
        "image_size": 256,
        "fid_feature": 2048,
    }
    actual = {
        "split_seed": int(split_seed),
        "val_samples_per_class": int(val_samples_per_class),
        "image_size": int(image_size),
        "fid_feature": int(fid_feature),
    }
    mismatches = [
        f"{key}={actual[key]} (expected {value})"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        raise ValueError(
            f"{PROTOCOL_NAME} has fixed metric settings: " + ", ".join(mismatches)
        )


def main() -> None:
    import torch
    import torch.distributed as dist
    from tqdm.auto import tqdm

    args = parse_args()
    validate_protocol_settings(
        split_seed=args.split_seed,
        val_samples_per_class=args.val_samples_per_class,
        image_size=args.image_size,
        fid_feature=args.fid_feature,
    )
    if int(args.is_splits) != 10:
        raise ValueError(f"{PROTOCOL_NAME} requires --is_splits=10")
    distributed, rank, world_size, local_rank, device = init_distributed(args.device)
    is_main = rank == 0
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed, device)

    manifest_records = load_manifest(args.manifest)
    synset_names = load_synset_names(args.synset_mapping)
    selected_records = load_fixed_val_records(
        manifest_records,
        args.split_manifest,
        synset_names,
        expected_samples_per_class=int(args.val_samples_per_class),
    )
    official_sample_count = len(selected_records)
    if official_sample_count != 10_000:
        raise ValueError(
            f"official ImageNet-100 validation protocol must contain 10,000 rows, "
            f"found {official_sample_count}"
        )
    if int(args.max_samples) > 0:
        selected_records = selected_records[: int(args.max_samples)]
    total_samples = len(selected_records)
    if total_samples < world_size:
        raise ValueError(
            f"samples={total_samples} must be at least distributed world_size={world_size}"
        )
    if total_samples < int(args.is_splits):
        raise ValueError(
            f"samples={total_samples} must be at least is_splits={args.is_splits}"
        )
    local_records = [
        record
        for record in selected_records
        if int(record["evaluation_index"]) % world_size == rank
    ]

    real_stats_path = Path(args.real_stats)
    if not real_stats_path.is_file():
        raise FileNotFoundError(
            f"cached original-ImageNet stats not found: {real_stats_path}. Run "
            "scripts/cache_imagenet100_real_stats.py first."
        )
    real_payload = torch.load(real_stats_path, map_location="cpu")
    cached_metadata = real_payload["metadata"]
    inception_weights = _resolve_weights_from_cache(
        cached_metadata, args.inception_weights_path
    )
    expected_metadata = build_expected_real_metadata(
        manifest_path=args.manifest,
        split_manifest_path=args.split_manifest,
        selected_records=selected_records
        if official_sample_count == len(selected_records)
        else load_fixed_val_records(
            manifest_records,
            args.split_manifest,
            synset_names,
            expected_samples_per_class=int(args.val_samples_per_class),
        ),
        transform=metric_transform_metadata(args.image_size),
        feature=feature_metadata(args.fid_feature, inception_weights),
        val_samples_per_class=int(args.val_samples_per_class),
        split_seed=int(args.split_seed),
    )
    validate_real_stats_metadata(cached_metadata, expected_metadata)
    if int(real_payload["stats"]["count"]) != official_sample_count:
        raise ValueError(
            f"cached real count={real_payload['stats']['count']} but protocol count="
            f"{official_sample_count}"
        )
    official_protocol = total_samples == official_sample_count

    if is_main:
        sample_manifest = output_dir / "samples.jsonl"
        with sample_manifest.open("w", encoding="utf-8") as handle:
            for record in selected_records:
                row = dict(record)
                row["sample_seed"] = deterministic_sample_seed(args.seed, record)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    model, tokenizer, config = load_qwen_showo_model(
        args.config, args.checkpoint, device
    )
    if int(getattr(model.config, "image_vocab_size", IMAGE_VOCAB_SIZE)) != IMAGE_VOCAB_SIZE:
        raise ValueError("Qwen-Show-o evaluation requires image_vocab_size=8192")
    magvit = load_official_magvit(args.showo_root, args.magvit_path, device)
    inception = build_inception_extractor(
        args.fid_feature, inception_weights, device
    )
    fake_moments = FeatureMoments.zeros(args.fid_feature, device)
    score_moments: InceptionScoreMoments | None = None
    t2i_prefix = str(config.dataset.params.get("t2i_prefix", ""))
    class_prompt_template = str(
        config.dataset.params.get("class_prompt_template", "{class_name}")
    )
    if t2i_prefix.strip():
        raise ValueError(
            "The fixed ImageNet-100 protocol is pure class-name conditioning; "
            f"dataset.params.t2i_prefix must be empty, got {t2i_prefix!r}."
        )
    if class_prompt_template not in {"{}", "{class_name}"}:
        raise ValueError(
            "The fixed ImageNet-100 protocol requires a bare class-name prompt; "
            "class_prompt_template must be '{}' or '{class_name}', got "
            f"{class_prompt_template!r}."
        )
    pad_to_multiple_of = config.dataset.params.get("pad_to_multiple_of", 64)
    autocast_dtype = _model_dtype(model)
    autocast_enabled = device.type == "cuda" and autocast_dtype in (
        torch.float16,
        torch.bfloat16,
    )

    iterator: Iterable[Sequence[Mapping[str, Any]]] = _record_batches(
        local_records, args.batch_size
    )
    iterator = tqdm(
        iterator,
        total=math.ceil(len(local_records) / args.batch_size),
        disable=args.no_progress or not is_main,
        desc="Qwen-Show-o FID/IS",
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        for batch_records in iterator:
            prompts = [str(record["prompt"]) for record in batch_records]
            seeds = [
                deterministic_sample_seed(args.seed, record)
                for record in batch_records
            ]
            conditional = build_prompt_batch(
                prompts,
                tokenizer,
                model.config,
                t2i_prefix=t2i_prefix,
                class_prompt_template=class_prompt_template,
                pad_to_multiple_of=pad_to_multiple_of,
                device=device,
            )
            unconditional = {
                "input_ids": conditional["uncond_input_ids"],
                "token_types": conditional["uncond_token_types"],
                "attention_mask": conditional["uncond_attention_mask"],
                "image_token_mask": conditional["uncond_image_token_mask"],
            }
            conditional = {
                "input_ids": conditional["input_ids"],
                "token_types": conditional["token_types"],
                "attention_mask": conditional["attention_mask"],
                "image_token_mask": conditional["image_token_mask"],
            }
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                image_tokens = call_maskgit_sampler(
                    model,
                    conditional,
                    unconditional,
                    sample_seeds=seeds,
                    timesteps=args.timesteps,
                    guidance_scale=args.guidance_scale,
                    temperature=args.temperature,
                    mask_schedule=args.mask_schedule,
                )
            # Keep VQ reconstruction identical across backbone precision modes.
            with torch.autocast(device_type=device.type, enabled=False):
                decoded = magvit.decode_code(image_tokens)
            images = decoded.float().add(1.0).mul(0.5).clamp(0.0, 1.0)
            if tuple(images.shape[1:]) != (3, args.image_size, args.image_size):
                raise ValueError(
                    f"official MAGVITv2 decode must return [B,3,256,256], got "
                    f"{tuple(images.shape)}"
                )
            features, logits = extract_inception_features(inception, images)
            fake_moments.update(features)
            if score_moments is None:
                score_moments = InceptionScoreMoments.zeros(
                    args.is_splits, logits.shape[-1], device
                )
            score_moments.update(
                logits,
                [int(record["evaluation_index"]) for record in batch_records],
                total_samples,
            )
            if args.save_images:
                save_generated_images(images, batch_records, output_dir / "generated")
    if score_moments is None:
        raise RuntimeError(f"rank {rank} generated no images")
    fake_moments.all_reduce_()
    score_moments.all_reduce_()
    if int(fake_moments.count.item()) != total_samples:
        raise RuntimeError(
            f"distributed generated feature count={int(fake_moments.count.item())}; "
            f"expected={total_samples}"
        )
    if int(score_moments.count.sum().item()) != total_samples:
        raise RuntimeError(
            f"distributed Inception Score count="
            f"{int(score_moments.count.sum().item())}; expected={total_samples}"
        )
    distributed_barrier(distributed, device)

    if is_main:
        real_moments = _cache_moments(real_payload, torch.device("cpu"))
        validate_cached_moment_shapes(real_moments, args.fid_feature)
        fake_mean, fake_cov = fake_moments.mean_cov()
        fid = None
        if official_protocol:
            real_mean, real_cov = real_moments.mean_cov()
            fid = frechet_distance(real_mean, real_cov, fake_mean, fake_cov)
        is_mean, is_std, is_per_split = score_moments.compute()
        results = {
            "protocol": PROTOCOL_NAME,
            "official_protocol": bool(official_protocol),
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "samples": int(fake_moments.count.item()),
            "seed": int(args.seed),
            "sampling": {
                "method": "maskgit",
                "timesteps": int(args.timesteps),
                "guidance_scale": float(args.guidance_scale),
                "common_cfg_scale": 1.0 + float(args.guidance_scale),
                "guidance_formula": "(1+s)*conditional-s*unconditional",
                "temperature": float(args.temperature),
                "temperature_schedule": "official_showo_cumulative_one_minus_ratio",
                "mask_schedule": str(args.mask_schedule),
            },
            "tokenizer": {
                "type": "official_showo_magvitv2",
                "path": str(Path(args.magvit_path).resolve()),
                "image_vocab_size": IMAGE_VOCAB_SIZE,
                "tokens_per_image": IMAGE_TOKEN_COUNT,
                "decode_dtype": "float32",
            },
            "real_stats": {
                "path": str(real_stats_path.resolve()),
                "metadata": cached_metadata,
            },
            "metrics": {
                "fid": _finite(fid) if fid is not None else None,
                "fid_feature": int(args.fid_feature),
                "inception_score_mean": _finite(is_mean),
                "inception_score_std": _finite(is_std),
                "inception_score_splits": [_finite(value) for value in is_per_split],
            },
            "distributed": {
                "world_size": int(world_size),
                "local_batch_size": int(args.batch_size),
            },
            "saved_images": bool(args.save_images),
        }
        write_json(output_dir / "metrics.json", results)
        print(json.dumps(results["metrics"], indent=2, sort_keys=True))
        print(f"Saved metrics to {output_dir / 'metrics.json'}")

    distributed_barrier(distributed, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
