#!/usr/bin/env python3
"""Build and validate the matched backbone × DF1 flow-head seed-42 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_NAME = "backbone_flow_head_joint"
PROTOCOL_VERSION = "v1"
SCREEN_SEED = 42
BASE_CONFIG = REPO_ROOT / "configs/ablation/imagenet_flow_100c_80ep.yaml"
CONFIG_DIR = (
    REPO_ROOT
    / "configs/ablation/archive/backbone_flow_head_joint_ablation/screen"
)
EVIDENCE_DIR = REPO_ROOT / "output/backbone_flow_head_joint_ablation/evidence"
SOURCE_MANIFEST_PATH = EVIDENCE_DIR / "runtime_source_manifest.json"
MATRIX_MANIFEST_PATH = EVIDENCE_DIR / "matrix_manifest.json"
SUMMARY_PATH = EVIDENCE_DIR / "summary_seed42.json"

BACKBONE_VARIANTS = ("E2-Q1", "E2-Q0", "E2b-Q0")
FLOW_POSITION_VARIANTS = ("FH0", "FH4")
FLOW_HEAD_VARIANT = "DF1"
CELLS = tuple(
    f"{backbone}__{FLOW_HEAD_VARIANT}-{position}"
    for backbone in BACKBONE_VARIANTS
    for position in FLOW_POSITION_VARIANTS
)

CONCEPTUAL_ORDER = (
    "E2-Q0__DF1-FH4",
    "E2-Q1__DF1-FH4",
    "E2-Q0__DF1-FH0",
    "E2-Q1__DF1-FH0",
    "E2b-Q0__DF1-FH4",
    "E2b-Q0__DF1-FH0",
)
FID_NONINFERIORITY_MARGIN = 0.50
IS_NONINFERIORITY_MARGIN = 1.00
EXPECTED_FLOW_HEAD_PARAMETERS = 164_072_976

RUNTIME_SOURCE_FILES = (
    "configs/ablation/imagenet_flow_100c_80ep.yaml",
    "docs/SELFLESS_FLOW_BACKBONE_FLOW_HEAD_JOINT_ABLATION_PROPOSAL.md",
    "script/ablation/train_eval_backbone_flow_head_joint_ablation.sh",
    "scripts/backbone_flow_head_joint_ablation.py",
    "scripts/evaluate_single_stream_fid_is.py",
    "pretrain/train_selfless_flow.py",
    "utils/dataset_imagenet_flow_cache.py",
    "utils/dataset_utils.py",
    "utils/utils.py",
    "models/modeling_model/image_backbone.py",
    "models/modeling_model/image_flow_loss.py",
    "models/modeling_model/image_flow_position.py",
    "models/modeling_model/image_position_utils.py",
    "models/modeling_model/modeling_selfless_flow.py",
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


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def normalize_backbone(value: str) -> str:
    aliases = {
        variant.lower().replace("_", "-"): variant
        for variant in BACKBONE_VARIANTS
    }
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in aliases:
        raise ValueError(
            f"Unknown backbone {value!r}; expected {BACKBONE_VARIANTS}."
        )
    return aliases[normalized]


def normalize_position(value: str) -> str:
    normalized = str(value).strip().upper().replace("_", "")
    if normalized not in FLOW_POSITION_VARIANTS:
        raise ValueError(
            f"Unknown flow position {value!r}; "
            f"expected {FLOW_POSITION_VARIANTS}."
        )
    return normalized


def cell_id(backbone: str, position: str) -> str:
    return (
        f"{normalize_backbone(backbone)}__"
        f"{FLOW_HEAD_VARIANT}-{normalize_position(position)}"
    )


def parse_cell(value: str) -> tuple[str, str]:
    normalized = str(value).strip()
    if "__" not in normalized:
        raise ValueError(
            f"Cell {value!r} must have BACKBONE__DF1-FH form."
        )
    backbone, flow = normalized.split("__", 1)
    flow = flow.upper().replace("_", "-")
    if not flow.startswith(f"{FLOW_HEAD_VARIANT}-"):
        raise ValueError(f"Cell {value!r} must use {FLOW_HEAD_VARIANT}.")
    return normalize_backbone(backbone), normalize_position(
        flow[len(FLOW_HEAD_VARIANT) + 1 :]
    )


def run_slug(backbone: str, position: str, seed: int = SCREEN_SEED) -> str:
    backbone_slug = normalize_backbone(backbone).lower().replace("-", "")
    position_slug = normalize_position(position).lower()
    return (
        f"selfless-flow-bfh-{backbone_slug}-"
        f"{FLOW_HEAD_VARIANT.lower()}-{position_slug}-s{int(seed)}"
    )


def config_filename(
    backbone: str,
    position: str,
    seed: int = SCREEN_SEED,
) -> str:
    backbone_slug = normalize_backbone(backbone).replace("-", "_")
    return (
        f"{backbone_slug}_{FLOW_HEAD_VARIANT}_"
        f"{normalize_position(position)}_s{int(seed)}.yaml"
    )


def runtime_source_evidence(
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root)
    entries = []
    for relative in RUNTIME_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Runtime source is missing: {path}")
        entries.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema": "selfless_backbone_flow_head_runtime_source_v1",
        "files": entries,
        "manifest_sha256": canonical_sha256(entries),
    }


def validate_runtime_manifest(
    manifest_path: str | Path = SOURCE_MANIFEST_PATH,
    *,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = runtime_source_evidence(repo_root)
    if payload != actual:
        expected_by_path = {
            row["path"]: row for row in payload.get("files", [])
        }
        actual_by_path = {
            row["path"]: row for row in actual.get("files", [])
        }
        drifted = sorted(
            key
            for key in set(expected_by_path) | set(actual_by_path)
            if expected_by_path.get(key) != actual_by_path.get(key)
        )
        raise ValueError(
            "Runtime source drifted from the frozen manifest: "
            + ", ".join(drifted)
        )
    return actual


def _resolved_payload(config: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Resolved config must be a mapping.")
    payload.pop("config", None)
    return payload


def config_fingerprint(config: DictConfig) -> str:
    payload = _resolved_payload(config)
    experiment = payload.get("experiment", {})
    if isinstance(experiment, dict):
        experiment.pop("config_fingerprint", None)
    model = payload.get("model", {})
    if isinstance(model, dict):
        for key in (
            "mask_token_id",
            "boi_token_id",
            "eoi_token_id",
            "image_mask_token_id",
            "image_offset",
        ):
            model.pop(key, None)
    return canonical_sha256(payload)


def control_fingerprint(config: DictConfig) -> str:
    payload = _resolved_payload(config)
    experiment = payload["experiment"]
    for key in (
        "project",
        "name",
        "ablation_id",
        "backbone_id",
        "flow_head_id",
        "flow_position_id",
        "config_fingerprint",
    ):
        experiment.pop(key, None)
    model = payload["model"]
    model.pop("image_backbone_variant", None)
    model.pop("image_flow_position_variant", None)
    payload["evaluation"].pop("checkpoint", None)
    return canonical_sha256(payload)


def build_config(
    backbone: str,
    position: str,
    *,
    source_manifest_sha256: str,
    seed: int = SCREEN_SEED,
) -> DictConfig:
    backbone = normalize_backbone(backbone)
    position = normalize_position(position)
    seed = int(seed)
    config = OmegaConf.load(BASE_CONFIG)
    cell = cell_id(backbone, position)
    project = run_slug(backbone, position, seed)

    config.experiment.project = project
    config.experiment.name = (
        f"imagenet100-bfh-{backbone.lower()}-"
        f"{FLOW_HEAD_VARIANT.lower()}-{position.lower()}-"
        f"seed{seed}-qwen3base-8gpu-b256-80ep"
    )
    config.experiment.ablation_study = STUDY_NAME
    config.experiment.ablation_phase = "seed42_screen"
    config.experiment.ablation_id = cell
    config.experiment.backbone_id = backbone
    config.experiment.flow_head_id = f"{FLOW_HEAD_VARIANT}-{position}"
    config.experiment.flow_position_id = position
    config.experiment.protocol_version = PROTOCOL_VERSION
    config.experiment.runtime_source_manifest_sha256 = (
        str(source_manifest_sha256)
    )

    config.model.image_backbone_variant = backbone
    config.model.image_flow_head_arch = "contextual"
    config.model.image_flow_head_variant = FLOW_HEAD_VARIANT
    config.model.image_flow_position_variant = position
    for retired in (
        "image_flow_query_position_mode",
        "image_flow_context_position_mode",
        "image_flow_rope_mode",
        "image_flow_rope_axis_dims",
        "image_flow_rope_rotate_value",
    ):
        if retired in config.model:
            del config.model[retired]

    config.training.seed = seed
    config.training.dataloader_shuffle_seed = seed
    config.evaluation.checkpoint = (
        f"output/{project}/hf_model-final-ema"
    )
    config.evaluation.seed = SCREEN_SEED
    config.experiment.config_fingerprint = config_fingerprint(config)
    validate_config(config, expected_cell=cell)
    return config


def _expect(
    config: DictConfig,
    path: str,
    expected: Any,
    errors: list[str],
) -> None:
    actual = OmegaConf.select(config, path)
    if actual != expected:
        errors.append(f"{path}={actual!r}, expected {expected!r}")


def validate_config(
    config_or_path: DictConfig | str | Path,
    *,
    expected_cell: str | None = None,
) -> dict[str, Any]:
    config = (
        OmegaConf.load(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )
    declared = str(config.experiment.ablation_id)
    backbone, position = parse_cell(expected_cell or declared)
    expected = cell_id(backbone, position)
    errors: list[str] = []

    checks = {
        "experiment.ablation_study": STUDY_NAME,
        "experiment.ablation_phase": "seed42_screen",
        "experiment.ablation_id": expected,
        "experiment.backbone_id": backbone,
        "experiment.flow_head_id": f"{FLOW_HEAD_VARIANT}-{position}",
        "experiment.flow_position_id": position,
        "experiment.protocol_version": PROTOCOL_VERSION,
        "model.image_backbone_variant": backbone,
        "model.image_flow_head_arch": "contextual",
        "model.image_flow_head_variant": FLOW_HEAD_VARIANT,
        "model.image_flow_position_variant": position,
        "model.image_tokens_per_img": 256,
        "model.image_latent_dim": 16,
        "model.image_flow_depth": 8,
        "model.image_flow_width": 1280,
        "model.image_flow_mlp_ratio": 1.0,
        "model.image_flow_latent_mixer_heads": 8,
        "model.image_flow_latent_mixer_dropout": 0.0,
        "model.image_flow_latent_mixer_zero_init_gate": True,
        "training.seed": SCREEN_SEED,
        "training.dataloader_shuffle_seed": SCREEN_SEED,
        "training.total_batch_size": 256,
        "training.mixed_precision": "bf16",
        "training.max_train_steps": 35_920,
        "training.use_ema": True,
        "evaluation.model_dtype": "bf16",
        "evaluation.samples": 10_000,
        "evaluation.batch_size": 512,
        "evaluation.sampling_steps": 100,
        "evaluation.temperature": 1.0,
        "evaluation.cfg": 3.5,
        "evaluation.cfg_schedule": "constant",
        "evaluation.flow_solver": "heun",
        "evaluation.parallel_rate": 1,
        "evaluation.strategies": "spatial_halton",
        "evaluation.seed": SCREEN_SEED,
    }
    for path, value in checks.items():
        _expect(config, path, value, errors)

    for forbidden in (
        "image_flow_query_position_mode",
        "image_flow_context_position_mode",
        "image_flow_rope_mode",
        "image_flow_rope_axis_dims",
        "image_flow_rope_rotate_value",
    ):
        if forbidden in config.model:
            errors.append(
                f"model.{forbidden} is a retired low-level position knob"
            )

    expected_project = run_slug(backbone, position, SCREEN_SEED)
    _expect(config, "experiment.project", expected_project, errors)
    _expect(
        config,
        "evaluation.checkpoint",
        f"output/{expected_project}/hf_model-final-ema",
        errors,
    )
    declared_fingerprint = str(
        config.experiment.get("config_fingerprint", "")
    )
    actual_fingerprint = config_fingerprint(config)
    if declared_fingerprint != actual_fingerprint:
        errors.append(
            "experiment.config_fingerprint does not match resolved config"
        )
    source_digest = str(
        config.experiment.get("runtime_source_manifest_sha256", "")
    )
    if len(source_digest) != 64:
        errors.append(
            "experiment.runtime_source_manifest_sha256 is not a SHA256"
        )

    if errors:
        raise ValueError(
            f"Invalid joint-ablation config {expected}: "
            + "; ".join(errors)
        )
    return {
        "cell_id": expected,
        "backbone": backbone,
        "flow_head": FLOW_HEAD_VARIANT,
        "position": position,
        "project": expected_project,
        "seed": SCREEN_SEED,
        "config_fingerprint": actual_fingerprint,
        "control_fingerprint": control_fingerprint(config),
        "runtime_source_manifest_sha256": source_digest,
    }


def build_matrix() -> dict[str, Any]:
    source = runtime_source_evidence()
    _write_json(SOURCE_MANIFEST_PATH, source)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    controls = set()
    for backbone in BACKBONE_VARIANTS:
        for position in FLOW_POSITION_VARIANTS:
            config = build_config(
                backbone,
                position,
                source_manifest_sha256=source["manifest_sha256"],
            )
            path = CONFIG_DIR / config_filename(backbone, position)
            OmegaConf.save(config, path)
            validated = validate_config(path)
            controls.add(validated["control_fingerprint"])
            rows.append(
                {
                    **validated,
                    "config_path": str(path.relative_to(REPO_ROOT)),
                    "config_sha256": file_sha256(path),
                    "checkpoint": str(config.evaluation.checkpoint),
                    "metrics": (
                        f"output/{validated['project']}/"
                        "fid_is_cfg3p5_10k_ema/metrics.json"
                    ),
                }
            )
    if len(controls) != 1:
        raise ValueError(
            f"Non-factor controls differ across matrix: {sorted(controls)}"
        )
    manifest = {
        "schema": "selfless_backbone_flow_head_joint_matrix_v1",
        "study": STUDY_NAME,
        "phase": "seed42_screen",
        "seed": SCREEN_SEED,
        "cells": list(CELLS),
        "fresh_training_required": True,
        "runtime_source_manifest_path": str(
            SOURCE_MANIFEST_PATH.relative_to(REPO_ROOT)
        ),
        "runtime_source_manifest_sha256": source["manifest_sha256"],
        "control_fingerprint": next(iter(controls)),
        "rows": rows,
        "selector": {
            "fid_noninferiority_margin": FID_NONINFERIORITY_MARGIN,
            "is_noninferiority_margin": IS_NONINFERIORITY_MARGIN,
            "conceptual_order": list(CONCEPTUAL_ORDER),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write_json(MATRIX_MANIFEST_PATH, manifest)
    return manifest


def _load_metrics(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics must be a JSON object: {path}")
    return payload


def validate_metrics(
    config_path: str | Path,
    metrics_path: str | Path,
) -> dict[str, Any]:
    config = OmegaConf.load(config_path)
    contract = validate_config(config)
    metrics = _load_metrics(metrics_path)
    errors: list[str] = []

    architecture = metrics.get("architecture", {})
    flow_head = architecture.get("flow_head", {})
    position_contract = flow_head.get("position_contract", {})
    precision = metrics.get("precision_protocol", {})
    strategy = metrics.get("strategies", {}).get("spatial_halton")
    expected_position_contract = {
        "FH0": {
            "variant": "FH0",
            "A_q": 1,
            "A_c": 1,
            "R_f": 0,
            "query_position_mode": "additive_2d",
            "context_position_mode": "additive_2d",
            "rope_mode": "none",
        },
        "FH4": {
            "variant": "FH4",
            "A_q": 0,
            "A_c": 0,
            "R_f": 1,
            "query_position_mode": "none",
            "context_position_mode": "none",
            "rope_mode": "row_col_2d",
        },
    }[contract["position"]]

    exact = {
        "official_protocol": (
            metrics.get("official_protocol"),
            True,
        ),
        "seed": (metrics.get("seed"), SCREEN_SEED),
        "batch_size": (metrics.get("batch_size"), 512),
        "samples_requested": (metrics.get("samples_requested"), 10_000),
        "samples_evaluated": (metrics.get("samples_evaluated"), 10_000),
        "image_backbone_variant": (
            architecture.get("image_backbone_variant"),
            contract["backbone"],
        ),
        "flow_head.arch": (flow_head.get("arch"), "contextual"),
        "flow_head.variant": (
            flow_head.get("variant"),
            FLOW_HEAD_VARIANT,
        ),
        "flow_head_parameters": (
            metrics.get("parameters", {}).get("flow_head"),
            EXPECTED_FLOW_HEAD_PARAMETERS,
        ),
        "model_dtype": (precision.get("model_dtype"), "bf16"),
        "vae_dtype": (precision.get("vae_dtype"), "fp32"),
        "flow_integrator_dtype": (
            precision.get("flow_integrator_dtype"),
            "fp32",
        ),
    }
    for label, (actual, expected) in exact.items():
        if actual != expected:
            errors.append(f"{label}={actual!r}, expected {expected!r}")
    for key, expected in expected_position_contract.items():
        if position_contract.get(key) != expected:
            errors.append(
                f"position_contract.{key}="
                f"{position_contract.get(key)!r}, expected {expected!r}"
            )
    if position_contract.get("rope_axis_dims") != [80, 80]:
        errors.append("position_contract.rope_axis_dims must be [80, 80]")
    if not isinstance(strategy, Mapping):
        errors.append("missing spatial_halton strategy metrics")
    elif int(strategy.get("count", -1)) != 10_000:
        errors.append(
            f"spatial_halton.count={strategy.get('count')!r}, expected 10000"
        )
    finite_rate = (
        metrics.get("mechanism_diagnostics", {})
        .get("generated_latent_finite_rate")
    )
    if finite_rate != 1.0:
        errors.append(
            f"generated_latent_finite_rate={finite_rate!r}, expected 1.0"
        )
    if errors:
        raise ValueError(
            f"Invalid metrics for {contract['cell_id']}: "
            + "; ".join(errors)
        )
    return {
        **contract,
        "metrics_path": str(Path(metrics_path).resolve()),
        "metrics_sha256": file_sha256(metrics_path),
        "fid": float(strategy["fid"]),
        "inception_score_mean": float(
            strategy["inception_score_mean"]
        ),
        "inception_score_std": float(
            strategy["inception_score_std"]
        ),
        "generation_wall_seconds": float(
            strategy["generation_wall_seconds"]
        ),
        "generation_samples_per_second": float(
            strategy["generation_samples_per_second"]
        ),
        "flow_content_cache_peak_mib_per_sample": float(
            strategy.get("flow_content_cache_peak_mib_per_sample", 0.0)
        ),
    }


def _delta(
    lookup: Mapping[str, Mapping[str, Any]],
    left: str,
    right: str,
) -> dict[str, float]:
    return {
        "fid": float(lookup[left]["fid"] - lookup[right]["fid"]),
        "inception_score_mean": float(
            lookup[left]["inception_score_mean"]
            - lookup[right]["inception_score_mean"]
        ),
    }


def select_winner(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {str(row["cell_id"]): row for row in rows}
    missing = [cell for cell in CELLS if cell not in lookup]
    if missing:
        raise ValueError(f"Missing result cells: {missing}")
    best_fid = min(float(row["fid"]) for row in rows)
    best_is = max(float(row["inception_score_mean"]) for row in rows)
    qualified = [
        cell
        for cell in CELLS
        if float(lookup[cell]["fid"])
        <= best_fid + FID_NONINFERIORITY_MARGIN
        and float(lookup[cell]["inception_score_mean"])
        >= best_is - IS_NONINFERIORITY_MARGIN
    ]
    if qualified:
        selected = next(
            cell for cell in CONCEPTUAL_ORDER if cell in qualified
        )
        rationale = (
            "Selected the conceptually cleanest/scaling-preferred cell "
            "inside the preregistered FID/IS noninferiority set."
        )
        fallback = False
    else:
        pareto = []
        for cell in CELLS:
            row = lookup[cell]
            dominated = any(
                other != cell
                and float(lookup[other]["fid"]) <= float(row["fid"])
                and float(lookup[other]["inception_score_mean"])
                >= float(row["inception_score_mean"])
                and (
                    float(lookup[other]["fid"]) < float(row["fid"])
                    or float(lookup[other]["inception_score_mean"])
                    > float(row["inception_score_mean"])
                )
                for other in CELLS
            )
            if not dominated:
                pareto.append(cell)
        selected = min(
            pareto,
            key=lambda cell: (
                float(lookup[cell]["fid"]),
                CONCEPTUAL_ORDER.index(cell),
            ),
        )
        qualified = []
        rationale = (
            "The joint FID/IS noninferiority set was empty; selected the "
            "lowest-FID cell on the Pareto frontier."
        )
        fallback = True
    return {
        "selected": selected,
        "quality_noninferior_cells": qualified,
        "best_fid": best_fid,
        "best_inception_score_mean": best_is,
        "fid_noninferiority_margin": FID_NONINFERIORITY_MARGIN,
        "is_noninferiority_margin": IS_NONINFERIORITY_MARGIN,
        "conceptual_order": list(CONCEPTUAL_ORDER),
        "fallback_used": fallback,
        "rationale": rationale,
    }


def summarize(
    matrix_manifest_path: str | Path = MATRIX_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = json.loads(
        Path(matrix_manifest_path).read_text(encoding="utf-8")
    )
    rows = []
    for item in manifest["rows"]:
        rows.append(
            validate_metrics(
                REPO_ROOT / item["config_path"],
                REPO_ROOT / item["metrics"],
            )
        )
    lookup = {row["cell_id"]: row for row in rows}
    flow_position_effects = {}
    for backbone in BACKBONE_VARIANTS:
        flow_position_effects[backbone] = _delta(
            lookup,
            cell_id(backbone, "FH4"),
            cell_id(backbone, "FH0"),
        )
    backbone_effects = {}
    for position in FLOW_POSITION_VARIANTS:
        reference = cell_id("E2-Q1", position)
        backbone_effects[position] = {
            backbone: _delta(
                lookup,
                cell_id(backbone, position),
                reference,
            )
            for backbone in ("E2-Q0", "E2b-Q0")
        }
    reference_flow_effect = flow_position_effects["E2-Q1"]
    interactions = {
        backbone: {
            metric: (
                flow_position_effects[backbone][metric]
                - reference_flow_effect[metric]
            )
            for metric in ("fid", "inception_score_mean")
        }
        for backbone in ("E2-Q0", "E2b-Q0")
    }
    summary = {
        "schema": "selfless_backbone_flow_head_joint_summary_v1",
        "study": STUDY_NAME,
        "phase": "seed42_screen",
        "seed": SCREEN_SEED,
        "matrix_manifest_path": str(
            Path(matrix_manifest_path).resolve()
        ),
        "matrix_manifest_sha256": file_sha256(matrix_manifest_path),
        "rows": rows,
        "estimands": {
            "fh4_minus_fh0_by_backbone": flow_position_effects,
            "backbone_minus_e2_q1_by_flow_position": backbone_effects,
            "architecture_position_interaction_vs_e2_q1": interactions,
        },
        "decision": select_winner(rows),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _write_json(SUMMARY_PATH, summary)
    return summary


def write_preflight(
    config_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    runtime = validate_runtime_manifest(manifest_path)
    contract = validate_config(config_path)
    if (
        contract["runtime_source_manifest_sha256"]
        != runtime["manifest_sha256"]
    ):
        raise ValueError(
            "Config runtime-source digest does not match manifest."
        )
    payload = {
        "schema": "selfless_backbone_flow_head_joint_preflight_v1",
        **contract,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "runtime_source_manifest_path": str(
            Path(manifest_path).resolve()
        ),
        "runtime_source_manifest_sha256": runtime["manifest_sha256"],
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite preflight: {output_path}"
        )
    _write_json(output_path, payload)
    return payload


def inspect_config(config_path: str | Path) -> list[str]:
    config = OmegaConf.load(config_path)
    contract = validate_config(config)
    return [
        contract["project"],
        str(contract["seed"]),
        contract["cell_id"],
        str(config.evaluation.checkpoint),
        (
            f"output/{contract['project']}/"
            "fid_is_cfg3p5_10k_ema"
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build")

    validate_config_parser = subparsers.add_parser("validate-config")
    validate_config_parser.add_argument("--config", required=True)

    validate_runtime_parser = subparsers.add_parser("validate-runtime")
    validate_runtime_parser.add_argument(
        "--manifest",
        default=str(SOURCE_MANIFEST_PATH),
    )

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--config", required=True)

    preflight_parser = subparsers.add_parser("write-preflight")
    preflight_parser.add_argument("--config", required=True)
    preflight_parser.add_argument("--manifest", required=True)
    preflight_parser.add_argument("--output", required=True)

    metrics_parser = subparsers.add_parser("validate-metrics")
    metrics_parser.add_argument("--config", required=True)
    metrics_parser.add_argument("--metrics", required=True)
    metrics_parser.add_argument("--output", default="")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument(
        "--matrix-manifest",
        default=str(MATRIX_MANIFEST_PATH),
    )

    args = parser.parse_args()
    if args.command == "build":
        result = build_matrix()
    elif args.command == "validate-config":
        result = validate_config(args.config)
    elif args.command == "validate-runtime":
        result = validate_runtime_manifest(args.manifest)
    elif args.command == "inspect":
        for value in inspect_config(args.config):
            print(value)
        return
    elif args.command == "write-preflight":
        result = write_preflight(
            args.config,
            args.manifest,
            args.output,
        )
    elif args.command == "validate-metrics":
        result = validate_metrics(args.config, args.metrics)
        if args.output:
            _write_json(args.output, result)
    elif args.command == "summarize":
        result = summarize(args.matrix_manifest)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
