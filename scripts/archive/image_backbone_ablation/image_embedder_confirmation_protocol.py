#!/usr/bin/env python3
# Historical provenance implementation retained for evidence audit only.
"""Strict provenance helpers for image-embedder confirmation runs.

The seed-42 architecture screen predates this protocol.  These helpers are
therefore deliberately opt-in and only apply to the preregistered confirmation
seeds.  A confirmation run must create its provenance before the first
training batch; the resulting digest is then bound to checkpoints and formal
evaluation output.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import RandomSampler, Subset


REPO_ROOT = Path(__file__).resolve().parents[3]
SCREEN_SUMMARY_SCHEMA = "selfless_flow_image_embedder_ablation_summary_v3"
CONFIRMATION_MANIFEST_SCHEMA = "selfless_flow_image_embedder_confirmation_candidates_v1"
CONFIRMATION_DECLARATION_SCHEMA = "selfless_flow_image_embedder_confirmation_declaration_v1"
CONFIRMATION_PROVENANCE_SCHEMA = "selfless_flow_image_embedder_confirmation_training_provenance_v1"
CONFIRMATION_SEEDS = frozenset((43, 44, 45))
EVALUATOR_RNG_CONTRACT_SCHEMA = "canonical_image_flow_initial_noise_v1"
EVALUATOR_RNG_CONTRACT_SHA256 = (
    "5a622fe0cc134735c3047c9addcd4e36f1991d4911cdefe38c2ac57f7dbdb86a"
)
EXPANDED_MATRIX_IDS = (
    "E0", "E1", "E2a", "E2b", "E2", "E3", "E4a", "E4b",
    "E4", "E5", "E6a", "E6b", "E6", "E7a", "E7b", "E7",
)

INITIALIZATION_CONTRACT = {
    "schema": "image_embedder_module_keyed_initialization_v1",
    "special_tokens": "name-keyed CPU float32 normal; image_mask copies mask",
    "image_modules": (
        "module-keyed torch RNG fork; trainable parameter state is byte-paired within "
        "an S2D layout; deterministic architecture-specific buffers are excluded"
    ),
    "cross_layout_scope": (
        "training seed and initialization policy are paired; differently shaped flow-head "
        "parameters and downstream same-name parameters are not claimed byte-paired"
    ),
}
TRAIN_ORDER_CONTRACT = {
    "schema": "image_embedder_random_sampler_order_v1",
    "sampler": "torch.utils.data.RandomSampler_without_replacement",
    "generator_policy": "dedicated_CPU_generator_seeded_with_training_seed",
    "dataloader_base_seed_consumption": "one_int64_draw_before_sampler_iteration",
    "distributed_policy": "Accelerate BatchSamplerShard over identical global batch stream",
}
AUGMENTATION_CONTRACT = {
    "schema": "imagenet_flow_cache_stateless_hflip_v1",
    "formula": "random.Random(seed + epoch*1000003 + dataset_index*9176 + 13579)",
    "layout": "canonical_HWC_before_space_to_depth",
    "epoch_state": "shared_int64_visible_to_persistent_workers",
}

RUNTIME_SOURCE_FILES = (
    "configs/ablation/imagenet_flow_image_embedder_100c_80ep.yaml",
    "script/ablation/train_image_embedder_ablation.sh",
    "script/ablation/pretraining_imagenet_flow_100c_80ep.sh",
    "script/ablation/evaluate_image_embedder_ablation.sh",
    "script/ablation/evaluate_imagenet_flow_100c.sh",
    "scripts/image_embedder_ablation_matrix.py",
    "scripts/image_embedder_confirmation_protocol.py",
    "pretrain/train_selfless_flow.py",
    "utils/dataset_imagenet_flow_cache.py",
    "utils/utils.py",
    "models/modeling_model/modeling_selfless_flow.py",
    "models/modeling_model/image_flow_loss.py",
    "models/modeling_model/image_position_utils.py",
    "models/modeling_model/image_latent_layout.py",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    value = str(value)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest, got {value!r}")
    return value


def _validate_screen_manifest(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("schema") != SCREEN_SUMMARY_SCHEMA or summary.get("expected") != "expanded":
        raise ValueError("confirmation screen must be the formal expanded seed-42 summary")
    manifest = summary.get("confirmation_candidate_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("confirmation screen is missing confirmation_candidate_manifest")
    manifest = dict(manifest)
    expected_fixed = {
        "schema": CONFIRMATION_MANIFEST_SCHEMA,
        "screen_summary_schema": SCREEN_SUMMARY_SCHEMA,
        "screen_training_seed": 42,
        "confirmation_training_seeds": sorted(CONFIRMATION_SEEDS),
    }
    for key, expected in expected_fixed.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"confirmation screen manifest {key}={manifest.get(key)!r}; expected {expected!r}"
            )

    runs = summary.get("runs")
    aggregates = summary.get("aggregates")
    if not isinstance(runs, list) or not isinstance(aggregates, list):
        raise ValueError("confirmation screen must contain run and aggregate evidence")
    run_pairs = {(row.get("id"), row.get("training_seed")) for row in runs if isinstance(row, Mapping)}
    if run_pairs != {(variant_id, 42) for variant_id in EXPANDED_MATRIX_IDS}:
        raise ValueError("confirmation screen does not contain exactly the expanded seed-42 matrix")
    aggregate_ids = [row.get("id") for row in aggregates if isinstance(row, Mapping)]
    if set(aggregate_ids) != set(EXPANDED_MATRIX_IDS) or len(aggregate_ids) != len(EXPANDED_MATRIX_IDS):
        raise ValueError("confirmation screen aggregates do not contain the expanded matrix exactly once")

    selector_fields = (
        "near_best_fid_ids",
        "fid_is_pareto_ids",
        "speed_pareto_ids_meeting_threshold",
    )
    selected = {"E0"}
    for field in selector_fields:
        values = manifest.get(field)
        if not isinstance(values, list) or any(value not in EXPANDED_MATRIX_IDS for value in values):
            raise ValueError(f"confirmation screen manifest {field} is invalid")
        selected.update(values)
    expected_candidates = [value for value in EXPANDED_MATRIX_IDS if value in selected]
    if manifest.get("candidate_ids") != expected_candidates:
        raise ValueError(
            "confirmation screen candidate_ids do not equal the preregistered selector union"
        )
    return manifest


def load_confirmation_screen(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    try:
        raw = path.read_bytes()
        summary = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read confirmation screen {path}: {exc}") from exc
    if not isinstance(summary, Mapping):
        raise ValueError("confirmation screen must be a JSON object")
    manifest = _validate_screen_manifest(summary)
    return manifest, hashlib.sha256(raw).hexdigest()


def build_confirmation_declaration(
    *,
    variant_id: str,
    seed: int,
    screen_path: str | Path,
) -> dict[str, Any]:
    seed = int(seed)
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError(
            f"confirmation training seed must be one of {sorted(CONFIRMATION_SEEDS)}, got {seed}"
        )
    manifest, screen_sha256 = load_confirmation_screen(screen_path)
    if variant_id not in manifest["candidate_ids"]:
        raise ValueError(
            f"{variant_id} is not in the frozen confirmation candidate manifest"
        )
    declaration = {
        "schema": CONFIRMATION_DECLARATION_SCHEMA,
        "ablation_id": str(variant_id),
        "training_seed": seed,
        "dataloader_shuffle_seed": seed,
        "evaluation_seed": 42,
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "screen_summary_path": str(Path(screen_path)),
        "screen_summary_sha256": screen_sha256,
        "candidate_manifest": manifest,
        "candidate_manifest_sha256": canonical_sha256(manifest),
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
    }
    declaration["declaration_sha256"] = canonical_sha256(declaration)
    return declaration


def validate_confirmation_declaration(
    declaration: Mapping[str, Any],
    *,
    variant_id: str,
    seed: int,
) -> dict[str, Any]:
    if not isinstance(declaration, Mapping):
        raise ValueError("experiment.confirmation_protocol must be an object")
    declaration = dict(declaration)
    stored_digest = declaration.pop("declaration_sha256", None)
    _require_sha256(stored_digest, "confirmation declaration digest")
    if canonical_sha256(declaration) != stored_digest:
        raise ValueError("confirmation declaration digest mismatch")
    declaration["declaration_sha256"] = stored_digest
    expected = {
        "schema": CONFIRMATION_DECLARATION_SCHEMA,
        "ablation_id": str(variant_id),
        "training_seed": int(seed),
        "dataloader_shuffle_seed": int(seed),
        "evaluation_seed": 42,
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
    }
    for key, value in expected.items():
        if declaration.get(key) != value:
            raise ValueError(f"confirmation declaration {key} drifted")
    manifest = declaration.get("candidate_manifest")
    if not isinstance(manifest, Mapping) or variant_id not in manifest.get("candidate_ids", []):
        raise ValueError("confirmation declaration does not authorize this ablation ID")
    if canonical_sha256(manifest) != declaration.get("candidate_manifest_sha256"):
        raise ValueError("confirmation candidate manifest digest mismatch")
    _require_sha256(declaration.get("screen_summary_sha256"), "screen summary digest")
    return declaration


def is_confirmation_config(config: DictConfig) -> bool:
    experiment = config.get("experiment", None)
    return bool(
        experiment is not None
        and str(experiment.get("ablation_phase", "screen")) == "confirmation"
    )


def stable_named_seed(seed: int, name: str) -> int:
    payload = f"image-embedder-confirmation-v1:{int(seed)}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    # ``view(dtype)`` cannot reinterpret a zero-dimensional tensor when the
    # element sizes differ (for example a scalar bf16 parameter as bytes).
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(tensor.shape)))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def named_tensor_evidence(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    entries = []
    combined = hashlib.sha256()
    schema = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        entry = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": tensor_sha256(tensor),
        }
        entries.append(entry)
        schema.update(canonical_json_bytes({key: entry[key] for key in ("name", "shape", "dtype")}))
        combined.update(canonical_json_bytes(entry))
    return {
        "parameter_count": len(entries),
        "parameter_schema_sha256": schema.hexdigest(),
        "state_sha256": combined.hexdigest(),
        "parameters": entries,
    }


def initial_state_evidence(model, special_token_ids: Mapping[str, int]) -> dict[str, Any]:
    modules = {
        "image_flow_head": model.image_flow_head,
        "image_flow_condition_proj": model.image_flow_condition_proj,
        "image_token_embedder": model.image_token_embedder,
    }
    image_parameters = []
    for module_name, module in modules.items():
        image_parameters.extend(
            (f"{module_name}.{name}", value)
            for name, value in module.named_parameters()
        )
    embedding = model.model.embed_tokens.weight
    token_rows = torch.stack(
        [embedding[int(token_id)] for _, token_id in sorted(special_token_ids.items())]
    )
    return {
        "contract": INITIALIZATION_CONTRACT,
        "image_modules": named_tensor_evidence(image_parameters),
        "special_token_names_and_ids": [
            [name, int(token_id)] for name, token_id in sorted(special_token_ids.items())
        ],
        "special_token_rows_sha256": tensor_sha256(token_rows),
    }


def _dataset_identity(dataset) -> tuple[list[int], Any]:
    if isinstance(dataset, Subset):
        return [int(value) for value in dataset.indices], dataset.dataset
    return list(range(len(dataset))), dataset


def _sample_identity(base_dataset, dataset_index: int) -> dict[str, Any]:
    img_ids = getattr(base_dataset, "img_ids", None)
    img_id = int(img_ids[dataset_index].item()) if img_ids is not None else dataset_index
    source_paths = getattr(base_dataset, "source_paths", {})
    return {
        "dataset_index": int(dataset_index),
        "img_id": img_id,
        "source_path": str(source_paths.get(img_id, "")),
    }


def train_data_evidence(train_loader, config: DictConfig) -> dict[str, Any]:
    generator = getattr(train_loader, "generator", None)
    if generator is None:
        raise ValueError("confirmation train loader must have a dedicated generator")
    expected_seed = int(config.training.seed)
    if int(config.training.dataloader_shuffle_seed) != expected_seed:
        raise ValueError("confirmation dataloader seed must equal training seed")

    clone = torch.Generator(device="cpu")
    clone.set_state(generator.get_state())
    initial_generator_state_sha256 = tensor_sha256(clone.get_state())
    base_seed = int(
        torch.empty((), dtype=torch.int64).random_(generator=clone).item()
    )
    sampler = RandomSampler(train_loader.dataset, replacement=False, generator=clone)
    order = [int(value) for value in sampler]
    if len(order) != len(train_loader.dataset) or len(set(order)) != len(order):
        raise ValueError("confirmation epoch-0 RandomSampler order is not a permutation")

    subset_indices, base_dataset = _dataset_identity(train_loader.dataset)
    identities = [_sample_identity(base_dataset, subset_indices[position]) for position in order]
    identity_digest = hashlib.sha256()
    augmentation_digest = hashlib.sha256()
    augmentation_seed = int(getattr(base_dataset, "seed", expected_seed))
    hflip_probability = float(getattr(base_dataset, "latent_hflip_prob", 0.0))
    for identity in identities:
        identity_digest.update(canonical_json_bytes(identity))
        dataset_index = int(identity["dataset_index"])
        flip = (
            random.Random(augmentation_seed + dataset_index * 9_176 + 13_579).random()
            < hflip_probability
        )
        augmentation_digest.update(
            canonical_json_bytes(
                {"epoch": 0, "dataset_index": dataset_index, "hflip": bool(flip)}
            )
        )

    params = config.dataset.params
    input_files = {}
    for label, key in (
        ("cache", "cache_path"),
        ("manifest", "manifest_jsonl"),
        ("split_manifest", "split_manifest_jsonl"),
        ("synset_mapping", "synset_mapping_path"),
    ):
        path = Path(str(params.get(key, "")))
        if not path.is_file():
            raise ValueError(f"confirmation dataset input is missing: {path}")
        input_files[label] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return {
        "contract": TRAIN_ORDER_CONTRACT,
        "dataloader_shuffle_seed": expected_seed,
        "initial_generator_state_sha256": initial_generator_state_sha256,
        "dataloader_base_seed": base_seed,
        "dataset_length": len(train_loader.dataset),
        "epoch0_ordered_sample_identity_sha256": identity_digest.hexdigest(),
        "augmentation_contract": AUGMENTATION_CONTRACT,
        "epoch0_augmentation_decisions_sha256": augmentation_digest.hexdigest(),
        "augmentation_seed": augmentation_seed,
        "latent_hflip_probability": hflip_probability,
        "batch_size_per_rank": int(config.training.batch_size),
        "total_batch_size": int(config.training.total_batch_size),
        "drop_last": bool(train_loader.drop_last),
        "num_workers": int(train_loader.num_workers),
        "persistent_workers": bool(train_loader.persistent_workers),
        "input_files": input_files,
    }


def runtime_source_evidence(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root)
    entries = []
    for relative in RUNTIME_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"confirmation runtime source is missing: {path}")
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {"files": entries, "manifest_sha256": canonical_sha256(entries)}


def base_model_evidence(config: DictConfig) -> dict[str, Any]:
    root = Path(str(config.model.model_path))
    candidates = sorted(root.glob("*.safetensors")) + sorted(root.glob("*.json"))
    if not candidates:
        raise ValueError(f"base model contains no local safetensors/JSON evidence: {root}")
    entries = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in candidates
    ]
    return {"files": entries, "manifest_sha256": canonical_sha256(entries)}


def build_training_provenance(
    *,
    config: DictConfig,
    model,
    train_loader,
    special_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    if not is_confirmation_config(config):
        raise ValueError("training provenance is only defined for confirmation configs")
    declaration = validate_confirmation_declaration(
        OmegaConf.to_container(config.experiment.confirmation_protocol, resolve=True),
        variant_id=str(config.experiment.ablation_id),
        seed=int(config.training.seed),
    )
    screen_manifest, screen_sha256 = load_confirmation_screen(
        declaration["screen_summary_path"]
    )
    if (
        screen_sha256 != declaration["screen_summary_sha256"]
        or canonical_sha256(screen_manifest)
        != declaration["candidate_manifest_sha256"]
    ):
        raise ValueError(
            "confirmation screen evidence changed after config preregistration"
        )
    provenance = {
        "schema": CONFIRMATION_PROVENANCE_SCHEMA,
        "ablation_id": str(config.experiment.ablation_id),
        "training_seed": int(config.training.seed),
        "space_to_depth_factor": int(config.model.image_space_to_depth_factor),
        "confirmation_declaration": declaration,
        "confirmation_declaration_sha256": declaration["declaration_sha256"],
        "initial_state": initial_state_evidence(model, special_token_ids),
        "train_data": train_data_evidence(train_loader, config),
        "base_model": base_model_evidence(config),
        "runtime_source": runtime_source_evidence(),
    }
    provenance["provenance_sha256"] = canonical_sha256(provenance)
    return provenance


def write_training_provenance(path: str | Path, provenance: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(provenance)
    stored = payload.pop("provenance_sha256", None)
    _require_sha256(stored, "training provenance digest")
    if canonical_sha256(payload) != stored:
        raise ValueError("refusing to write training provenance with an invalid digest")
    payload["provenance_sha256"] = stored
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return stored


def load_and_validate_training_provenance(
    path: str | Path,
    *,
    expected_sha256: str,
    variant_id: str,
    seed: int,
) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid confirmation training provenance: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("confirmation training provenance must be an object")
    payload = dict(payload)
    stored = payload.pop("provenance_sha256", None)
    _require_sha256(stored, "training provenance digest")
    if stored != expected_sha256 or canonical_sha256(payload) != stored:
        raise ValueError("confirmation training provenance digest mismatch")
    payload["provenance_sha256"] = stored
    if payload.get("schema") != CONFIRMATION_PROVENANCE_SCHEMA:
        raise ValueError("confirmation training provenance schema mismatch")
    if payload.get("ablation_id") != variant_id or payload.get("training_seed") != int(seed):
        raise ValueError("confirmation training provenance run identity mismatch")
    return payload
