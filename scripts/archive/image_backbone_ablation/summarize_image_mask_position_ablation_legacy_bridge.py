#!/usr/bin/env python3
# Historical Q1 evidence bridge retained for audit only.
"""Bridge legacy implicit-Q1 confirmation runs with explicit Q0 runs.

This is intentionally separate from the frozen Q-factor summarizer.  The two
Q1 cells are historical E2b/E2 confirmation checkpoints whose missing
``image_mask_position_mode`` field has legacy ``additive_2d`` semantics.  The
two Q0 cells are post-change Q-factor checkpoints.  They may be compared only
as seed-aligned, cross-source evidence; they are never presented as
same-source training pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from scripts.archive.image_backbone_ablation.image_embedder_ablation_matrix import FLOW_HEAD_INVARIANTS, run_slug
from scripts.archive.image_backbone_ablation.image_embedder_confirmation_protocol import (
    canonical_sha256,
    load_and_validate_training_provenance,
    validate_confirmation_declaration,
)
from scripts.archive.image_backbone_ablation.image_mask_position_ablation_protocol import (
    PARENT_CONFIRMATION_IDS,
    Q_FACTOR_DECISION_RULE,
    Q_FACTOR_IDS,
    Q_FACTOR_RUNTIME_SOURCE_FILES,
    Q_FACTOR_SEEDS,
    Q_FACTOR_VARIANTS,
    load_and_validate_q_factor_training_provenance,
    load_parent_summary_evidence,
    q_factor_run_slug,
)
from scripts.archive.image_backbone_ablation.summarize_image_embedder_ablation import (
    SummaryError as LegacySummaryError,
    _load_run as _load_legacy_metrics,
    _validate_confirmation_run,
)
from scripts.archive.image_backbone_ablation.summarize_image_mask_position_ablation import (
    SummaryError as QFactorSummaryError,
    _load_run as _load_q0_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REUSE_MANIFEST = (
    REPO_ROOT / "configs/ablation/image_mask_position_legacy_q1_reuse_manifest_v1.json"
)
LEGACY_REUSE_MANIFEST_RAW_SHA256 = (
    "92ca9ad50e1d9235673a1f3a1c1c093a8232e7590ed0ca01d5c5213d55eb3352"
)
LEGACY_REUSE_MANIFEST_CANONICAL_SHA256 = (
    "398a6782651514323e2c8e624041dfa95bd695c03b371e75ccd1e9d934b3be94"
)
Q0_ATTESTATION_MANIFEST_RELATIVE_PATH = Path(
    "configs/ablation/image_mask_position_q0_metrics_attestation_v1.json"
)
DEFAULT_Q0_ATTESTATION_MANIFEST = REPO_ROOT / Q0_ATTESTATION_MANIFEST_RELATIVE_PATH
Q0_ATTESTATION_MANIFEST_RAW_SHA256: str | None = (
    "30491a0d5a4e24cbe44a523fa0b644a2a423ed248996f393faeda10049ac9cf3"
)
EQUIVALENCE_REPORT_RELATIVE_PATH = Path(
    "configs/ablation/image_mask_position_legacy_q1_equivalence_v1.json"
)
EQUIVALENCE_REPORT_RAW_SHA256 = (
    "ae8a7e62c8506ce20b51550af1681192d1826427a2911e85f818a12dd61acf2c"
)
EQUIVALENCE_REPORT_CANONICAL_SHA256 = (
    "cf979c11d3df5bb8cca14d4858e87538307882c469464c4085acde2657cedbe8"
)
EVALUATION_SOURCE_EQUIVALENCE_RELATIVE_PATH = Path(
    "output/image_mask_position_ablation/source_drift_waiver/"
    "evaluation_source_equivalence.json"
)
EVALUATION_SOURCE_EQUIVALENCE_RAW_SHA256 = (
    "0f9ac1cb13cd68b0204edfe573c823af76034671c7a7118a20b51b9dabc68a60"
)
EVALUATION_WAIVER_LAUNCHER_RELATIVE_PATH = Path(
    "output/image_mask_position_ablation/run_q0_eval_with_source_waiver.sh"
)
EVALUATION_WAIVER_LAUNCHER_RAW_SHA256 = (
    "718e488663dff5f2ead8b8dcc9ab240668f81336d42f15991d62046381f3603f"
)
EVALUATION_WAIVER_SITECUSTOMIZE_RELATIVE_PATH = Path(
    "output/image_mask_position_ablation/source_drift_waiver/sitecustomize.py"
)
EVALUATION_WAIVER_SITECUSTOMIZE_RAW_SHA256 = (
    "5af599f5f9b5f63e407c43cf146c265a4ae771b4b8d4622eb311e693eee99c47"
)
EVALUATION_WAIVER_FROZEN_BYTECODE_RAW_SHA256 = (
    "d9ec0fa3a3ea3545a15e25ec8cbea939d480213bf722996468ae1508858b3c6f"
)
FROZEN_MODEL_SOURCE_SHA256 = (
    "41e845cd5375f50edb6733985763dd81d006c8d9816a4abc14fb143c50e7fd92"
)
FROZEN_MODEL_SOURCE_SIZE = 134_243
CURRENT_MODEL_SOURCE_SHA256 = (
    "e1fb61dc12bab86158912d6f85467e98974bd28645769b94f995f79370b8e5b3"
)
CURRENT_MODEL_SOURCE_SIZE = 141_449
MODEL_SOURCE_RELATIVE_PATH = Path("models/modeling_model/modeling_selfless_flow.py")
REGISTERED_Q0_SOURCE_SHA256 = (
    "5be769a7f2f5d01b3749844caec79044d135db0f5f54be3cfc4328aad72b0f04"
)

SCHEMA = "selfless_flow_image_mask_position_legacy_bridge_summary_v1"
REUSE_MANIFEST_SCHEMA = "selfless_flow_image_mask_position_legacy_q1_reuse_manifest_v1"
Q0_ATTESTATION_SCHEMA = "selfless_flow_image_mask_position_q0_metrics_attestation_v1"
LEGACY_GATE_SCHEMA = "selfless_flow_image_mask_position_legacy_q1_cohort_gate_v1"
Q0_GATE_SCHEMA = "selfless_flow_image_mask_position_q0_cohort_gate_v1"
CROSS_GATE_SCHEMA = (
    "selfless_flow_image_mask_position_cross_cohort_comparability_gate_v1"
)
STRATEGY = "spatial_halton"
T95_DF2 = 4.302652729911275
FID_SIMPLICITY_MARGIN = float(Q_FACTOR_DECISION_RULE["close_fid_absolute_margin"])
IS_SIMPLICITY_MARGIN = float(
    Q_FACTOR_DECISION_RULE["close_inception_score_absolute_margin"]
)
PARENT_SUMMARY_RELATIVE_PATH = Path(
    "output/image_embedder_ablation/confirmation_d1_summary.json"
)
EVAL_METRICS_SUFFIX = Path("fid_is_selected_cfg3p5_ema/metrics.json")

LEGACY_ANALYSIS_TO_PHYSICAL = {
    "E2b-Q1": "E2b",
    "E2-Q1": "E2",
}
Q0_IDS = ("E2b-Q0", "E2-Q0")
Q0_ATTESTED_EVALUATION_JOBS = {
    ("E2-Q0", 44): "imgemb-qf-e2-q0-s44-ev-fz",
    ("E2-Q0", 45): "imgemb-qf-e2-q0-s45-ev-fz",
}
ANALYSIS_IDS = tuple(Q_FACTOR_IDS)
EXPECTED_PAIRS = frozenset(
    (analysis_id, seed)
    for analysis_id in ANALYSIS_IDS
    for seed in sorted(Q_FACTOR_SEEDS)
)
EXPECTED_Q0_ATTESTATION_PAIRS = frozenset(
    (analysis_id, seed) for analysis_id in Q0_IDS for seed in sorted(Q_FACTOR_SEEDS)
)
LEGACY_EQUIVALENCE = {
    "analysis_factor": "Q1",
    "checkpoint_field_requirement": "image_mask_position_mode_absent",
    "effective_mask_position_mode": "additive_2d",
    "semantics": "legacy_implicit_additive_2d",
    "bitwise_regression": (
        "tests/test_image_mask_position_mode.py::"
        "test_legacy_and_explicit_q1_are_bitwise_identical_for_mask_x0_and_xt"
    ),
    "same_source_training_with_q0": False,
    "formal_q_factor": False,
    "post_hoc_amendment": True,
    "equivalence_report_path": EQUIVALENCE_REPORT_RELATIVE_PATH.as_posix(),
    "equivalence_report_raw_sha256": EQUIVALENCE_REPORT_RAW_SHA256,
    "equivalence_report_sha256": EQUIVALENCE_REPORT_CANONICAL_SHA256,
}


class SummaryError(ValueError):
    """A bridge input violated its frozen source or comparability contract."""


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryError(f"{label} must be an object")
    return dict(value)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SummaryError(f"{label} must be a lowercase SHA256 digest")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise SummaryError(f"{label} must be {qualifier}finite")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:

        def reject_constant(token: str) -> None:
            raise ValueError(f"non-finite JSON constant {token}")

        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise SummaryError(f"failed to read {label} {path}: {exc}") from exc
    return _mapping(value, f"{label} {path}")


def _resolve_exact(path: Path, expected: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise SummaryError(f"missing {label}: {path}: {exc}") from exc
    if resolved != expected_resolved or not resolved.is_file():
        raise SummaryError(
            f"{label} must be exactly {expected_resolved}, got {resolved}"
        )
    return resolved


def _require_regular_artifact(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise SummaryError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SummaryError(f"missing {label}: {path}: {exc}") from exc
    if not resolved.is_file():
        raise SummaryError(f"{label} is not a regular file: {resolved}")
    return resolved


def _validate_checkpoint_model_config(
    model_config: Mapping[str, Any],
    *,
    analysis_id: str,
    legacy: bool,
    source: Path,
) -> None:
    variant = Q_FACTOR_VARIANTS[analysis_id]
    expected = {
        "image_query_stage_mode": "none",
        "image_observed_position_mode": variant.observed_position_mode,
        "image_rope_mode": "row_col_2d",
        "image_space_to_depth_factor": 1,
        "image_flow_head_arch": FLOW_HEAD_INVARIANTS["image_flow_head_arch"],
        "image_flow_depth": FLOW_HEAD_INVARIANTS["image_flow_depth"],
        "image_flow_width": FLOW_HEAD_INVARIANTS["image_flow_width"],
        "image_flow_mlp_ratio": FLOW_HEAD_INVARIANTS["image_flow_mlp_ratio"],
        "image_flow_latent_mixer_heads": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_heads"
        ],
        "image_flow_latent_mixer_dropout": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_dropout"
        ],
        "image_flow_latent_mixer_zero_init_gate": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_zero_init_gate"
        ],
    }
    for key, expected_value in expected.items():
        if model_config.get(key) != expected_value:
            raise SummaryError(
                f"{source}: checkpoint model config expected {key}="
                f"{expected_value!r}, got {model_config.get(key)!r}"
            )
    if legacy:
        if "image_mask_position_mode" in model_config:
            raise SummaryError(
                f"{source}: legacy checkpoint metadata must omit "
                "image_mask_position_mode"
            )
    elif model_config.get("image_mask_position_mode") != "none":
        raise SummaryError(
            f"{source}: Q0 checkpoint metadata must explicitly use mask mode none"
        )


def _legacy_slug(analysis_id: str, seed: int) -> str:
    return run_slug(LEGACY_ANALYSIS_TO_PHYSICAL[analysis_id], seed)


def _expected_metrics_path(artifact_root: Path, analysis_id: str, seed: int) -> Path:
    slug = (
        _legacy_slug(analysis_id, seed)
        if analysis_id in LEGACY_ANALYSIS_TO_PHYSICAL
        else q_factor_run_slug(analysis_id, seed)
    )
    return artifact_root / "output" / slug / EVAL_METRICS_SUFFIX


def _parse_run(spec: str) -> tuple[str, int, Path]:
    label, separator, raw_path = spec.partition("=")
    raw_id, seed_separator, raw_seed = label.partition("@")
    if not separator or not seed_separator or not raw_path.strip():
        raise SummaryError(
            f"--run must use ANALYSIS_ID@TRAINING_SEED=PATH syntax, got {spec!r}"
        )
    analysis_id = raw_id.strip()
    if analysis_id not in ANALYSIS_IDS:
        raise SummaryError(
            f"unknown bridge analysis ID {analysis_id!r}; expected one of "
            + ", ".join(ANALYSIS_IDS)
        )
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise SummaryError(f"invalid training seed in --run {spec!r}") from exc
    if seed not in Q_FACTOR_SEEDS:
        raise SummaryError(
            f"bridge training seed must be one of {sorted(Q_FACTOR_SEEDS)}, got {seed}"
        )
    return analysis_id, seed, Path(raw_path).expanduser()


def _legacy_manifest_run(
    artifact_root: Path, analysis_id: str, seed: int
) -> dict[str, Any]:
    physical_id = LEGACY_ANALYSIS_TO_PHYSICAL[analysis_id]
    slug = _legacy_slug(analysis_id, seed)
    relative_root = Path("output") / slug
    files = {
        "metrics": relative_root / EVAL_METRICS_SUFFIX,
        "provenance": relative_root / "confirmation_training_provenance.json",
        "hf_provenance": (
            relative_root / "hf_model-final-ema/confirmation_training_provenance.json"
        ),
        "checkpoint_metadata": relative_root / "checkpoint-35920/metadata.json",
        "training_config": relative_root / "config.yaml",
        "hf_config": relative_root / "hf_model-final-ema/config.json",
    }
    resolved_files = {}
    for label, relative_path in files.items():
        path = artifact_root / relative_path
        resolved_files[label] = _require_regular_artifact(path, f"legacy reuse {label}")

    metrics = _read_json(resolved_files["metrics"], "legacy metrics")
    architecture = _mapping(metrics.get("architecture"), "legacy architecture")
    if architecture.get("ablation_id") != physical_id:
        raise SummaryError(
            f"legacy {analysis_id}@{seed} has physical ID "
            f"{architecture.get('ablation_id')!r}, expected {physical_id!r}"
        )
    if "image_mask_position_mode" in architecture:
        raise SummaryError(
            f"legacy {analysis_id}@{seed} must omit image_mask_position_mode"
        )
    training = _mapping(metrics.get("training_protocol"), "legacy training protocol")
    if training.get("training_seed") != seed:
        raise SummaryError(
            f"legacy {analysis_id}@{seed} training seed does not match metrics"
        )
    confirmation = _mapping(training.get("confirmation"), "legacy confirmation")
    expected_provenance_sha = _sha256(
        confirmation.get("provenance_sha256"), "legacy provenance digest"
    )
    _read_json(resolved_files["provenance"], "legacy full provenance")
    try:
        full_provenance = load_and_validate_training_provenance(
            resolved_files["provenance"],
            expected_sha256=expected_provenance_sha,
            variant_id=physical_id,
            seed=seed,
        )
    except ValueError as exc:
        raise SummaryError(
            f"legacy {analysis_id}@{seed} full provenance is invalid: {exc}"
        ) from exc
    if (
        resolved_files["provenance"].read_bytes()
        != resolved_files["hf_provenance"].read_bytes()
    ):
        raise SummaryError(
            f"legacy {analysis_id}@{seed} root/HF provenance copies differ"
        )
    checkpoint = _read_json(
        resolved_files["checkpoint_metadata"], "legacy checkpoint metadata"
    )
    if checkpoint.get("global_step") != 35_920:
        raise SummaryError(f"legacy {analysis_id}@{seed} checkpoint step drifted")
    checkpoint_binding = _mapping(
        checkpoint.get("confirmation_provenance"),
        f"legacy {analysis_id}@{seed} checkpoint provenance binding",
    )
    if checkpoint_binding != {
        "path": files["provenance"].as_posix(),
        "sha256": full_provenance["provenance_sha256"],
        "declaration_sha256": full_provenance["confirmation_declaration_sha256"],
    }:
        raise SummaryError(
            f"legacy {analysis_id}@{seed} checkpoint provenance binding drifted"
        )
    checkpoint_model = _mapping(
        checkpoint.get("model_config"),
        f"legacy {analysis_id}@{seed} checkpoint model config",
    )
    _validate_checkpoint_model_config(
        checkpoint_model,
        analysis_id=analysis_id,
        legacy=True,
        source=resolved_files["checkpoint_metadata"],
    )

    hf_config = _read_json(resolved_files["hf_config"], "legacy HF config")
    if "image_mask_position_mode" in hf_config:
        raise SummaryError(
            f"legacy {analysis_id}@{seed} HF config must omit image_mask_position_mode"
        )
    return {
        "analysis_id": analysis_id,
        "physical_ablation_id": physical_id,
        "training_seed": seed,
        "metrics_path": files["metrics"].as_posix(),
        "metrics_sha256": _file_sha256(resolved_files["metrics"]),
        "provenance_path": files["provenance"].as_posix(),
        "provenance_file_sha256": _file_sha256(resolved_files["provenance"]),
        "hf_provenance_path": files["hf_provenance"].as_posix(),
        "hf_provenance_file_sha256": _file_sha256(resolved_files["hf_provenance"]),
        "checkpoint_metadata_path": files["checkpoint_metadata"].as_posix(),
        "checkpoint_metadata_sha256": _file_sha256(
            resolved_files["checkpoint_metadata"]
        ),
        "training_config_path": files["training_config"].as_posix(),
        "training_config_sha256": _file_sha256(resolved_files["training_config"]),
        "hf_config_path": files["hf_config"].as_posix(),
        "hf_config_sha256": _file_sha256(resolved_files["hf_config"]),
    }


def _validate_legacy_equivalence_report() -> dict[str, Any]:
    path = REPO_ROOT / EQUIVALENCE_REPORT_RELATIVE_PATH
    if path.is_symlink():
        raise SummaryError(
            f"legacy Q1 equivalence report must not be a symlink: {path}"
        )
    if _file_sha256(path) != EQUIVALENCE_REPORT_RAW_SHA256:
        raise SummaryError("legacy Q1 equivalence report raw digest drifted")
    report = _read_json(path, "legacy Q1 equivalence report")
    stored = report.pop("report_sha256", None)
    if stored != EQUIVALENCE_REPORT_CANONICAL_SHA256:
        raise SummaryError("legacy Q1 equivalence report canonical digest drifted")
    if canonical_sha256(report) != stored:
        raise SummaryError("legacy Q1 equivalence report is not self-consistent")
    report["report_sha256"] = stored
    expected = {
        "schema": "selfless_flow_image_mask_position_legacy_q1_equivalence_v1",
        "historical_implementation": {
            "git_blob": "00d1215378daa7d37657e86a5912666668ed9956",
            "path": "models/modeling_model/modeling_selfless_flow.py",
            "raw_sha256": "cfd4056dff1e51478165fe46f21d28a4310f779193500efc3011e13c6e2bb164",
            "size_bytes": 133356,
            "runtime_source_manifest_sha256": "c1568c6aac70a442092901716a0220cc0bbf998b0a95c88d673a4e9fa8e05048",
        },
        "q0_registered_implementation": {
            "git_blob": "a3ca758658681aa10d499df3d182584f1be77202",
            "path": "models/modeling_model/modeling_selfless_flow.py",
            "raw_sha256": "41e845cd5375f50edb6733985763dd81d006c8d9816a4abc14fb143c50e7fd92",
            "size_bytes": 134243,
            "runtime_source_manifest_sha256": REGISTERED_Q0_SOURCE_SHA256,
        },
        "diff_evidence": {
            "classification": "q1_semantics_preserving_mask_mode_gate_only",
            "changed_lines": 48,
            "insertions": 34,
            "deletions": 14,
            "legacy_behavior": "unconditional_position_lookup_and_add",
            "q1_behavior": "the_same_position_lookup_and_add_guarded_by_additive_2d",
            "missing_field_default": "additive_2d",
        },
        "bitwise_regression": {
            "nodeid": LEGACY_EQUIVALENCE["bitwise_regression"],
            "test_file_sha256": "f8080e05dc4c4c1e147a6524ed1ec5b39fa567c837d0e53f8dfab5e2ffd2c66c",
        },
        "claims": {
            "effective_mask_position_mode": "additive_2d",
            "formal_q_factor": False,
            "post_hoc_amendment": True,
            "same_source_training_with_q0": False,
        },
        "report_sha256": EQUIVALENCE_REPORT_CANONICAL_SHA256,
    }
    if report != expected:
        raise SummaryError("legacy Q1 equivalence report content drifted")
    if (
        _file_sha256(
            REPO_ROOT / report["bitwise_regression"]["nodeid"].split("::", 1)[0]
        )
        != report["bitwise_regression"]["test_file_sha256"]
    ):
        raise SummaryError("legacy Q1 bitwise regression file digest drifted")
    return report


def build_legacy_reuse_manifest(
    artifact_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build a self-digested manifest for the six immutable legacy Q1 inputs."""

    root = Path(artifact_root).resolve()
    _validate_legacy_equivalence_report()
    parent_path = root / PARENT_SUMMARY_RELATIVE_PATH
    _read_json(parent_path, "legacy reuse parent summary")
    try:
        parent = load_parent_summary_evidence(parent_path)
    except ValueError as exc:
        raise SummaryError(f"legacy reuse parent summary is invalid: {exc}") from exc
    runs = [
        _legacy_manifest_run(root, analysis_id, seed)
        for analysis_id in LEGACY_ANALYSIS_TO_PHYSICAL
        for seed in sorted(Q_FACTOR_SEEDS)
    ]
    manifest = {
        "schema": REUSE_MANIFEST_SCHEMA,
        "parent_summary": {
            "path": PARENT_SUMMARY_RELATIVE_PATH.as_posix(),
            "sha256": parent["sha256"],
            "schema": parent["schema"],
            "pairing_gate_sha256": parent["pairing_gate_sha256"],
        },
        "legacy_equivalence": dict(LEGACY_EQUIVALENCE),
        "runs": runs,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def load_and_validate_legacy_reuse_manifest(
    path: str | Path,
    *,
    artifact_root: str | Path = REPO_ROOT,
    enforce_production_pin: bool = False,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    root = Path(artifact_root).resolve()
    if enforce_production_pin:
        if root != REPO_ROOT or source != DEFAULT_REUSE_MANIFEST.resolve(strict=True):
            raise SummaryError(
                "production legacy reuse validation requires REPO_ROOT and the "
                "default manifest path"
            )
        if _file_sha256(source) != LEGACY_REUSE_MANIFEST_RAW_SHA256:
            raise SummaryError("production legacy reuse manifest raw digest drifted")
    payload = _read_json(source, "legacy reuse manifest")
    stored = payload.pop("manifest_sha256", None)
    _sha256(stored, "legacy reuse manifest digest")
    if enforce_production_pin and stored != LEGACY_REUSE_MANIFEST_CANONICAL_SHA256:
        raise SummaryError("production legacy reuse manifest canonical digest drifted")
    if canonical_sha256(payload) != stored:
        raise SummaryError(f"{source}: legacy reuse manifest digest mismatch")
    payload["manifest_sha256"] = stored
    expected = build_legacy_reuse_manifest(root)
    if payload != expected:
        raise SummaryError(
            f"{source}: legacy reuse manifest does not match the exact frozen artifacts"
        )
    return payload


def _validate_flow_head(config: Mapping[str, Any], source: Path) -> None:
    expected = {
        "arch": FLOW_HEAD_INVARIANTS["image_flow_head_arch"],
        "depth": FLOW_HEAD_INVARIANTS["image_flow_depth"],
        "width": FLOW_HEAD_INVARIANTS["image_flow_width"],
        "mlp_ratio": FLOW_HEAD_INVARIANTS["image_flow_mlp_ratio"],
        "latent_mixer_heads": FLOW_HEAD_INVARIANTS["image_flow_latent_mixer_heads"],
        "latent_mixer_dropout": FLOW_HEAD_INVARIANTS["image_flow_latent_mixer_dropout"],
        "zero_init_gate": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_zero_init_gate"
        ],
    }
    if dict(config) != expected:
        raise SummaryError(f"{source}: frozen flow-head architecture drifted")


def _validate_hf_config(
    artifact_root: Path,
    *,
    analysis_id: str,
    seed: int,
    legacy: bool,
) -> str:
    slug = (
        _legacy_slug(analysis_id, seed)
        if legacy
        else q_factor_run_slug(analysis_id, seed)
    )
    path = artifact_root / "output" / slug / "hf_model-final-ema/config.json"
    path = _require_regular_artifact(path, "HF config")
    payload = _read_json(path, "HF config")
    variant = Q_FACTOR_VARIANTS[analysis_id]
    expected = {
        "image_query_stage_mode": "none",
        "image_observed_position_mode": variant.observed_position_mode,
        "image_rope_mode": "row_col_2d",
        "image_space_to_depth_factor": 1,
        "image_flow_head_arch": FLOW_HEAD_INVARIANTS["image_flow_head_arch"],
        "image_flow_depth": FLOW_HEAD_INVARIANTS["image_flow_depth"],
        "image_flow_width": FLOW_HEAD_INVARIANTS["image_flow_width"],
        "image_flow_mlp_ratio": FLOW_HEAD_INVARIANTS["image_flow_mlp_ratio"],
        "image_flow_latent_mixer_heads": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_heads"
        ],
        "image_flow_latent_mixer_dropout": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_dropout"
        ],
        "image_flow_latent_mixer_zero_init_gate": FLOW_HEAD_INVARIANTS[
            "image_flow_latent_mixer_zero_init_gate"
        ],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise SummaryError(
                f"{path}: expected {key}={value!r}, got {payload.get(key)!r}"
            )
    if legacy:
        if "image_mask_position_mode" in payload:
            raise SummaryError(
                f"{path}: legacy HF config must omit image_mask_position_mode"
            )
    elif payload.get("image_mask_position_mode") != "none":
        raise SummaryError(f"{path}: Q0 HF config must explicitly use mask mode none")
    return _file_sha256(path)


def _validate_metric_finiteness(payload: Mapping[str, Any], source: Path) -> None:
    distributed = _mapping(payload.get("distributed"), f"{source}: distributed")
    for key in ("peak_cuda_allocated_mib", "peak_cuda_reserved_mib"):
        _finite(distributed.get(key), f"{source}: distributed.{key}", positive=True)
    strategies = _mapping(payload.get("strategies"), f"{source}: strategies")
    if set(strategies) != {STRATEGY}:
        raise SummaryError(f"{source}: expected exactly strategy {STRATEGY!r}")
    metrics = _mapping(strategies[STRATEGY], f"{source}: {STRATEGY}")
    for key in (
        "fid",
        "inception_score_mean",
        "generation_wall_seconds",
        "generation_samples_per_second",
        "latent_mse_to_target",
        "latent_rms",
        "generation_step_max",
    ):
        _finite(metrics.get(key), f"{source}: {key}", positive=True)
    _finite(metrics.get("inception_score_std"), f"{source}: inception_score_std")
    splits = metrics.get("inception_score_splits")
    if not isinstance(splits, list) or len(splits) != 10:
        raise SummaryError(f"{source}: expected exactly 10 finite IS split scores")
    for index, value in enumerate(splits):
        _finite(value, f"{source}: IS split {index}", positive=True)


def _protocol_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    training = _mapping(payload.get("training_protocol"), "training protocol")
    return {
        "official_protocol": payload.get("official_protocol"),
        "metric_protocol": payload.get("metric_protocol"),
        "precision_protocol": payload.get("precision_protocol"),
        "real_source": payload.get("real_source"),
        "real_stats_path": payload.get("real_stats_path"),
        "real_stats_metadata": payload.get("real_stats_metadata"),
        "inception_weights_path": payload.get("inception_weights_path"),
        "split": payload.get("split"),
        "evaluation_seed": payload.get("seed"),
        "batch_size": payload.get("batch_size"),
        "samples_requested": payload.get("samples_requested"),
        "samples_evaluated": payload.get("samples_evaluated"),
        "cfg": payload.get("cfg"),
        "cfg_schedule": payload.get("cfg_schedule"),
        "sampling_steps": payload.get("sampling_steps"),
        "temperature": payload.get("temperature"),
        "flow_solver": payload.get("flow_solver"),
        "parallel_rate": payload.get("parallel_rate"),
        "training_protocol_invariants_sha256": training.get("invariants_sha256"),
    }


def _assert_compact_legacy_matches_full(
    compact: Mapping[str, Any], full: Mapping[str, Any], source: Path
) -> None:
    bindings = {
        "schema": full.get("schema"),
        "ablation_id": full.get("ablation_id"),
        "training_seed": full.get("training_seed"),
        "provenance_sha256": full.get("provenance_sha256"),
        "confirmation_declaration_sha256": full.get("confirmation_declaration_sha256"),
        "space_to_depth_factor": full.get("space_to_depth_factor"),
    }
    for key, expected in bindings.items():
        if compact.get(key) != expected:
            raise SummaryError(f"{source}: compact legacy provenance {key} drifted")
    for key in ("initial_state", "train_data"):
        full_value = _mapping(full.get(key), f"{source}: full provenance {key}")
        compact_value = _mapping(
            compact.get(key), f"{source}: compact provenance {key}"
        )
        for nested_key, value in compact_value.items():
            if nested_key == "image_modules" and isinstance(value, Mapping):
                full_modules = _mapping(
                    full_value.get(nested_key), f"{source}: full image modules"
                )
                if any(full_modules.get(k) != v for k, v in value.items()):
                    raise SummaryError(
                        f"{source}: compact legacy image-module evidence drifted"
                    )
            elif full_value.get(nested_key) != value:
                raise SummaryError(
                    f"{source}: compact legacy provenance {key}.{nested_key} drifted"
                )
    base = _mapping(full.get("base_model"), f"{source}: full base model")
    runtime_source = _mapping(
        full.get("runtime_source"), f"{source}: full runtime source"
    )
    if compact.get("base_model_manifest_sha256") != base.get("manifest_sha256"):
        raise SummaryError(f"{source}: compact legacy base-model binding drifted")
    if compact.get("runtime_source_manifest_sha256") != runtime_source.get(
        "manifest_sha256"
    ):
        raise SummaryError(f"{source}: compact legacy source binding drifted")
    for label, evidence in (("base model", base), ("runtime source", runtime_source)):
        files = evidence.get("files")
        if not isinstance(files, list) or not files:
            raise SummaryError(
                f"{source}: full legacy {label} file evidence is missing"
            )
        if canonical_sha256(files) != evidence.get("manifest_sha256"):
            raise SummaryError(
                f"{source}: full legacy {label} manifest digest mismatch"
            )


def _load_legacy_run(
    analysis_id: str,
    seed: int,
    path: Path,
    *,
    artifact_root: Path,
    manifest_entry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    physical_id = LEGACY_ANALYSIS_TO_PHYSICAL[analysis_id]
    expected_path = _expected_metrics_path(artifact_root, analysis_id, seed)
    resolved = _resolve_exact(path, expected_path, "legacy Q1 metrics path")
    if _file_sha256(resolved) != manifest_entry["metrics_sha256"]:
        raise SummaryError(
            f"{resolved}: legacy metrics hash differs from reuse manifest"
        )
    strict_payload = _read_json(resolved, "legacy metrics")
    try:
        legacy_row, _unused_signature, loaded_payload = _load_legacy_metrics(
            physical_id, seed, resolved
        )
    except LegacySummaryError as exc:
        raise SummaryError(str(exc)) from exc
    if _mapping(loaded_payload, str(resolved)) != strict_payload:
        raise SummaryError(f"{resolved}: legacy metrics parser projection drifted")
    payload = strict_payload
    _validate_metric_finiteness(payload, resolved)
    architecture = _mapping(payload.get("architecture"), f"{resolved}: architecture")
    if "image_mask_position_mode" in architecture:
        raise SummaryError(
            f"{resolved}: reused legacy metrics must omit image_mask_position_mode"
        )
    for forbidden in (
        "parent_ablation_id",
        "q_factor_id",
        "mask_query_position_factor",
    ):
        if forbidden in architecture:
            raise SummaryError(
                f"{resolved}: reused legacy metrics must not fabricate {forbidden}"
            )
    _validate_flow_head(
        _mapping(architecture.get("flow_head"), f"{resolved}: flow head"), resolved
    )
    hf_config_sha = _validate_hf_config(
        artifact_root, analysis_id=analysis_id, seed=seed, legacy=True
    )
    if hf_config_sha != manifest_entry["hf_config_sha256"]:
        raise SummaryError(
            f"{resolved}: legacy HF config hash differs from reuse manifest"
        )

    training = _mapping(payload.get("training_protocol"), f"{resolved}: training")
    confirmation = _mapping(training.get("confirmation"), f"{resolved}: confirmation")
    compact = _mapping(
        confirmation.get("provenance"), f"{resolved}: compact legacy provenance"
    )
    provenance_path = artifact_root / str(manifest_entry["provenance_path"])
    if _file_sha256(provenance_path) != manifest_entry["provenance_file_sha256"]:
        raise SummaryError(f"{resolved}: legacy provenance file hash drifted")
    hf_provenance_path = artifact_root / str(manifest_entry["hf_provenance_path"])
    if (
        _file_sha256(hf_provenance_path) != manifest_entry["hf_provenance_file_sha256"]
        or provenance_path.read_bytes() != hf_provenance_path.read_bytes()
    ):
        raise SummaryError(f"{resolved}: legacy root/HF provenance binding drifted")
    _read_json(provenance_path, "legacy full provenance")
    try:
        full = load_and_validate_training_provenance(
            provenance_path,
            expected_sha256=_sha256(
                confirmation.get("provenance_sha256"),
                f"{resolved}: legacy provenance digest",
            ),
            variant_id=physical_id,
            seed=seed,
        )
        declaration = validate_confirmation_declaration(
            _mapping(
                full.get("confirmation_declaration"),
                f"{resolved}: confirmation declaration",
            ),
            variant_id=physical_id,
            seed=seed,
        )
    except ValueError as exc:
        raise SummaryError(
            f"{resolved}: invalid full legacy provenance: {exc}"
        ) from exc
    _assert_compact_legacy_matches_full(compact, full, resolved)
    try:
        evidence = _validate_confirmation_run(
            legacy_row,
            payload,
            candidate_manifest_sha256=declaration["candidate_manifest_sha256"],
        )
    except LegacySummaryError as exc:
        raise SummaryError(str(exc)) from exc
    evidence.update(
        {
            "source_cohort": "legacy_confirmation_q1",
            "analysis_id": analysis_id,
            "physical_ablation_id": physical_id,
            "source_manifest_sha256": evidence["runtime_source_manifest_sha256"],
            "training_protocol_sha256": legacy_row["training_protocol_sha256"],
            "parameters_sha256": canonical_sha256(
                {
                    key: legacy_row[key]
                    for key in ("total", "trainable", "image_embedder", "flow_head")
                }
            ),
            "special_token_names_and_ids": compact["initial_state"][
                "special_token_names_and_ids"
            ],
            "provenance_file_sha256": manifest_entry["provenance_file_sha256"],
            "hf_provenance_file_sha256": manifest_entry["hf_provenance_file_sha256"],
            "checkpoint_metadata_sha256": manifest_entry["checkpoint_metadata_sha256"],
        }
    )
    strategy = _mapping(payload["strategies"][STRATEGY], f"{resolved}: metrics")
    row = _normalize_row(
        analysis_id=analysis_id,
        physical_id=physical_id,
        source_cohort="legacy_confirmation_q1",
        seed=seed,
        base_row=legacy_row,
        strategy=strategy,
        evidence=evidence,
        metrics_sha256=manifest_entry["metrics_sha256"],
        hf_config_sha256=hf_config_sha,
        checkpoint_hashes={
            "checkpoint_metadata_sha256": manifest_entry["checkpoint_metadata_sha256"]
        },
        evaluation_source_mode="legacy_historical_source",
        evaluation_source_equivalence_sha256=None,
    )
    return row, _protocol_signature(payload), evidence, payload


def _assert_compact_q0_matches_full(
    compact: Mapping[str, Any], full: Mapping[str, Any], source: Path
) -> None:
    for key in (
        "schema",
        "q_factor_id",
        "parent_ablation_id",
        "training_seed",
        "architecture",
        "provenance_sha256",
        "q_factor_declaration_sha256",
        "study_manifest_sha256",
        "parent_summary_sha256",
        "config_contract_sha256",
        "runtime_source_manifest_sha256",
    ):
        if compact.get(key) != full.get(key):
            raise SummaryError(f"{source}: compact Q0 provenance {key} drifted")
    declaration = _mapping(
        full.get("q_factor_declaration"), f"{source}: Q0 declaration"
    )
    if compact.get("config_contract") != declaration.get("config_contract"):
        raise SummaryError(f"{source}: compact Q0 config contract drifted")

    def require_projection(compact_value: Any, full_value: Any, label: str) -> None:
        if isinstance(compact_value, Mapping):
            if not isinstance(full_value, Mapping):
                raise SummaryError(f"{source}: compact Q0 provenance {label} drifted")
            for nested_key, nested_value in compact_value.items():
                if nested_key not in full_value:
                    raise SummaryError(
                        f"{source}: compact Q0 provenance {label}.{nested_key} drifted"
                    )
                require_projection(
                    nested_value, full_value[nested_key], f"{label}.{nested_key}"
                )
        elif compact_value != full_value:
            raise SummaryError(f"{source}: compact Q0 provenance {label} drifted")

    for key in ("initial_state", "train_data", "runtime_context"):
        require_projection(compact.get(key), full.get(key), key)
    base = _mapping(full.get("base_model"), f"{source}: Q0 base model")
    if compact.get("base_model_manifest_sha256") != base.get("manifest_sha256"):
        raise SummaryError(f"{source}: compact Q0 base-model binding drifted")


def _validate_evaluation_source_mode(
    artifact_root: Path,
    *,
    analysis_id: str,
    seed: int,
    metrics_path: Path,
    source_manifest_sha256: str,
) -> tuple[str, str | None]:
    if source_manifest_sha256 != REGISTERED_Q0_SOURCE_SHA256:
        raise SummaryError(
            f"{metrics_path}: Q0 source must be the preregistered manifest "
            f"{REGISTERED_Q0_SOURCE_SHA256}"
        )
    sidecar = metrics_path.parent / "evaluation_source_equivalence.json"
    waiver_required = analysis_id == "E2-Q0" and seed in {44, 45}
    if not waiver_required:
        if sidecar.exists() or sidecar.is_symlink():
            raise SummaryError(
                f"{metrics_path}: {analysis_id}@{seed} must not claim an evaluation "
                "source waiver"
            )
        return "frozen_registered_source", None

    canonical_path = artifact_root / EVALUATION_SOURCE_EQUIVALENCE_RELATIVE_PATH
    for label, candidate in (
        ("canonical waiver", canonical_path),
        ("metrics waiver", sidecar),
    ):
        if candidate.is_symlink():
            raise SummaryError(f"{metrics_path}: {label} must not be a symlink")
        if not candidate.is_file():
            raise SummaryError(f"{metrics_path}: missing {label} {candidate}")
        if _file_sha256(candidate) != EVALUATION_SOURCE_EQUIVALENCE_RAW_SHA256:
            raise SummaryError(f"{metrics_path}: {label} raw digest drifted")
    if canonical_path.read_bytes() != sidecar.read_bytes():
        raise SummaryError(
            f"{metrics_path}: metrics waiver is not byte-identical to canonical waiver"
        )
    waiver = _read_json(sidecar, "evaluation source equivalence waiver")
    if (
        waiver.get("schema")
        != "selfless_flow_q_factor_evaluation_source_equivalence_v1"
    ):
        raise SummaryError(f"{metrics_path}: evaluation waiver schema drifted")
    if waiver.get("scope") != {
        "q_factor_ids": ["E2-Q0"],
        "training_seeds": [44, 45],
        "evaluation_seed": 42,
    }:
        raise SummaryError(f"{metrics_path}: evaluation waiver scope drifted")
    if (
        waiver.get("preregistered_runtime_source_manifest_sha256")
        != REGISTERED_Q0_SOURCE_SHA256
    ):
        raise SummaryError(f"{metrics_path}: evaluation waiver source binding drifted")
    drift = _mapping(waiver.get("drift"), f"{metrics_path}: waiver drift")
    if drift != {
        "path": MODEL_SOURCE_RELATIVE_PATH.as_posix(),
        "preregistered_size_bytes": FROZEN_MODEL_SOURCE_SIZE,
        "preregistered_sha256": FROZEN_MODEL_SOURCE_SHA256,
        "evaluation_size_bytes": CURRENT_MODEL_SOURCE_SIZE,
        "evaluation_sha256": CURRENT_MODEL_SOURCE_SHA256,
        "classification": "formatting_and_unused_import_cleanup",
    }:
        raise SummaryError(
            f"{metrics_path}: evaluation waiver drift description changed"
        )
    proof = _mapping(waiver.get("proof"), f"{metrics_path}: waiver proof")
    expected_proof = {
        "code_object_topology_equal": True,
        "code_object_count": 99,
        "non_module_code_objects_compared": 98,
        "normalized_executable_bodies_equal": True,
        "normalization_ignores": [
            "line_table_metadata",
            "NOP_instructions",
            "bytecode_offsets_after_NOP_normalization",
        ],
        "exception_table_graphs_compared": True,
        "module_level_delta": [
            "import_reordering",
            "split_import_formatting",
            "unused_import_removal",
        ],
        "other_preregistered_runtime_files_match": True,
    }
    if proof != expected_proof:
        raise SummaryError(
            f"{metrics_path}: evaluation waiver executable proof drifted"
        )
    semantics = _mapping(waiver.get("semantics"), f"{metrics_path}: waiver semantics")
    if semantics.get("model_execution_patch_applied") is not False:
        raise SummaryError(
            f"{metrics_path}: waiver must forbid model execution patches"
        )
    if semantics.get("metric_overwrite_allowed") is not False:
        raise SummaryError(f"{metrics_path}: waiver must forbid metric overwrite")
    bytecode = _mapping(
        waiver.get("frozen_bytecode_evidence"), f"{metrics_path}: bytecode evidence"
    )
    if bytecode.get("cache_tag") != "cpython-312" or bytecode.get("header_flags") != 0:
        raise SummaryError(f"{metrics_path}: frozen bytecode header evidence drifted")
    if bytecode.get("header_source_size") != FROZEN_MODEL_SOURCE_SIZE:
        raise SummaryError(f"{metrics_path}: frozen bytecode source-size proof drifted")
    bytecode_sha = _sha256(
        bytecode.get("sha256"), f"{metrics_path}: frozen bytecode digest"
    )
    if bytecode_sha != EVALUATION_WAIVER_FROZEN_BYTECODE_RAW_SHA256:
        raise SummaryError(f"{metrics_path}: frozen bytecode digest binding drifted")
    bytecode_path = artifact_root / str(bytecode.get("path", ""))
    if bytecode_path.is_symlink() or not bytecode_path.is_file():
        raise SummaryError(
            f"{metrics_path}: frozen bytecode proof is missing or symlinked"
        )
    if _file_sha256(bytecode_path) != bytecode_sha:
        raise SummaryError(f"{metrics_path}: frozen bytecode proof digest drifted")
    return (
        "frozen_bytecode_equivalence_waiver",
        EVALUATION_SOURCE_EQUIVALENCE_RAW_SHA256,
    )


def _load_q0_run(
    analysis_id: str,
    seed: int,
    path: Path,
    *,
    artifact_root: Path,
    attestation_entry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_path = _expected_metrics_path(artifact_root, analysis_id, seed)
    resolved = _resolve_exact(path, expected_path, "Q0 metrics path")
    metrics_sha256 = _file_sha256(resolved)
    if attestation_entry is not None:
        expected_relative_path = (
            Path("output") / q_factor_run_slug(analysis_id, seed) / EVAL_METRICS_SUFFIX
        ).as_posix()
        if attestation_entry.get("metrics_path") != expected_relative_path:
            raise SummaryError(
                f"{resolved}: Q0 metrics attestation path binding drifted"
            )
        if attestation_entry.get("metrics_sha256") != metrics_sha256:
            raise SummaryError(
                f"{resolved}: raw metrics digest differs from Q0 attestation"
            )
    payload = _read_json(resolved, "Q0 metrics")
    try:
        q_row, _unused_signature, evidence = _load_q0_metrics(
            analysis_id, seed, resolved
        )
    except QFactorSummaryError as exc:
        raise SummaryError(str(exc)) from exc
    _validate_metric_finiteness(payload, resolved)
    architecture = _mapping(payload.get("architecture"), f"{resolved}: architecture")
    if architecture.get("image_mask_position_mode") != "none":
        raise SummaryError(f"{resolved}: Q0 must explicitly declare mask mode none")
    hf_config_sha = _validate_hf_config(
        artifact_root, analysis_id=analysis_id, seed=seed, legacy=False
    )
    training = _mapping(payload.get("training_protocol"), f"{resolved}: training")
    q_factor = _mapping(training.get("q_factor"), f"{resolved}: q_factor")
    compact = _mapping(q_factor.get("provenance"), f"{resolved}: compact Q0 provenance")
    slug = q_factor_run_slug(analysis_id, seed)
    provenance_path = (
        artifact_root / "output" / slug / "q_factor_training_provenance.json"
    )
    expected_provenance_path = f"output/{slug}/q_factor_training_provenance.json"
    if q_factor.get("provenance_path") != expected_provenance_path:
        raise SummaryError(f"{resolved}: Q0 provenance path drifted")
    artifacts = _mapping(training.get("artifacts"), f"{resolved}: Q0 artifacts")
    hf_provenance_relative = (
        f"output/{slug}/hf_model-final-ema/q_factor_training_provenance.json"
    )
    checkpoint_relative = f"output/{slug}/checkpoint-35920/metadata.json"
    if artifacts.get("q_factor_provenance_path") != expected_provenance_path:
        raise SummaryError(f"{resolved}: Q0 artifact provenance path drifted")
    if artifacts.get("q_factor_hf_provenance_path") != hf_provenance_relative:
        raise SummaryError(f"{resolved}: Q0 HF provenance path drifted")
    if artifacts.get("checkpoint_metadata_path") != checkpoint_relative:
        raise SummaryError(f"{resolved}: Q0 checkpoint metadata path drifted")
    provenance_path = _require_regular_artifact(
        provenance_path, f"{resolved}: Q0 root provenance"
    )
    hf_provenance_path = _require_regular_artifact(
        artifact_root / hf_provenance_relative,
        f"{resolved}: Q0 HF provenance",
    )
    if provenance_path.read_bytes() != hf_provenance_path.read_bytes():
        raise SummaryError(f"{resolved}: Q0 root/HF provenance copies differ")
    _read_json(provenance_path, "Q0 full provenance")
    try:
        full = load_and_validate_q_factor_training_provenance(
            provenance_path,
            expected_sha256=_sha256(
                q_factor.get("provenance_sha256"), f"{resolved}: Q0 provenance digest"
            ),
            variant_id=analysis_id,
            seed=seed,
            repo_root=artifact_root,
            source_files=Q_FACTOR_RUNTIME_SOURCE_FILES,
            validate_parent_summary=False,
            validate_runtime_source=False,
        )
    except ValueError as exc:
        raise SummaryError(f"{resolved}: invalid full Q0 provenance: {exc}") from exc
    _assert_compact_q0_matches_full(compact, full, resolved)
    checkpoint_path = _require_regular_artifact(
        artifact_root / checkpoint_relative,
        f"{resolved}: Q0 checkpoint metadata",
    )
    checkpoint_sha = _file_sha256(checkpoint_path)
    expected_checkpoint_sha = _sha256(
        artifacts.get("checkpoint_metadata_sha256"),
        f"{resolved}: Q0 checkpoint metadata digest",
    )
    if checkpoint_sha != expected_checkpoint_sha:
        raise SummaryError(f"{resolved}: Q0 checkpoint metadata digest drifted")
    checkpoint = _read_json(checkpoint_path, "Q0 checkpoint metadata")
    if checkpoint.get("global_step") != 35_920:
        raise SummaryError(f"{resolved}: Q0 checkpoint step drifted")
    checkpoint_binding = _mapping(
        checkpoint.get("q_factor_provenance"),
        f"{resolved}: checkpoint Q0 provenance binding",
    )
    expected_checkpoint_binding = {
        "path": expected_provenance_path,
        "sha256": full["provenance_sha256"],
        "declaration_sha256": full["q_factor_declaration_sha256"],
        "study_manifest_sha256": full["study_manifest_sha256"],
        "config_contract_sha256": full["config_contract_sha256"],
        "source_manifest_sha256": full["source_manifest_sha256"],
    }
    if checkpoint_binding != expected_checkpoint_binding:
        raise SummaryError(f"{resolved}: checkpoint Q0 provenance binding drifted")
    _validate_checkpoint_model_config(
        _mapping(
            checkpoint.get("model_config"),
            f"{resolved}: checkpoint Q0 model config",
        ),
        analysis_id=analysis_id,
        legacy=False,
        source=checkpoint_path,
    )
    parent_binding = _mapping(
        full["q_factor_declaration"]["study_manifest"]["parent_summary"],
        f"{resolved}: parent summary binding",
    )
    if parent_binding.get("path") != PARENT_SUMMARY_RELATIVE_PATH.as_posix():
        raise SummaryError(f"{resolved}: Q0 parent-summary path drifted")
    evaluation_source_mode, waiver_sha256 = _validate_evaluation_source_mode(
        artifact_root,
        analysis_id=analysis_id,
        seed=seed,
        metrics_path=resolved,
        source_manifest_sha256=evidence["source_manifest_sha256"],
    )
    if attestation_entry is not None:
        attested_source = {
            "source_manifest_sha256": attestation_entry.get("source_manifest_sha256"),
            "evaluation_source_mode": attestation_entry.get("evaluation_source_mode"),
            "evaluation_source_equivalence_sha256": attestation_entry.get(
                "evaluation_source_equivalence_sha256"
            ),
            "evaluation_job_name": attestation_entry.get("evaluation_job_name"),
            "evaluation_waiver": attestation_entry.get("evaluation_waiver"),
        }
        actual_source = {
            "source_manifest_sha256": evidence["source_manifest_sha256"],
            "evaluation_source_mode": evaluation_source_mode,
            "evaluation_source_equivalence_sha256": waiver_sha256,
            "evaluation_job_name": Q0_ATTESTED_EVALUATION_JOBS.get((analysis_id, seed)),
            "evaluation_waiver": (
                _evaluation_waiver_attestation(artifact_root)
                if (analysis_id, seed) in Q0_ATTESTED_EVALUATION_JOBS
                else None
            ),
        }
        if attested_source != actual_source:
            raise SummaryError(
                f"{resolved}: Q0 metrics attestation source binding drifted"
            )
    evidence.update(
        {
            "source_cohort": "q_factor_q0",
            "analysis_id": analysis_id,
            "physical_ablation_id": analysis_id,
            "training_seed": seed,
            "training_protocol_sha256": evidence["training_protocol_sha256"],
            "special_token_names_and_ids": compact["initial_state"][
                "special_token_names_and_ids"
            ],
            "provenance_file_sha256": _file_sha256(provenance_path),
            "hf_provenance_file_sha256": _file_sha256(hf_provenance_path),
            "checkpoint_metadata_sha256": checkpoint_sha,
            "parent_summary_path": parent_binding["path"],
            "evaluation_source_mode": evaluation_source_mode,
            "evaluation_source_equivalence_sha256": waiver_sha256,
        }
    )
    strategy = _mapping(payload["strategies"][STRATEGY], f"{resolved}: metrics")
    row = _normalize_row(
        analysis_id=analysis_id,
        physical_id=analysis_id,
        source_cohort="q_factor_q0",
        seed=seed,
        base_row=q_row,
        strategy=strategy,
        evidence=evidence,
        metrics_sha256=metrics_sha256,
        hf_config_sha256=hf_config_sha,
        checkpoint_hashes=evidence["checkpoint_hashes"],
        evaluation_source_mode=evaluation_source_mode,
        evaluation_source_equivalence_sha256=waiver_sha256,
    )
    return row, _protocol_signature(payload), evidence, payload


def _attested_file(
    artifact_root: Path,
    relative_path: Path,
    *,
    label: str,
    expected_sha256: str,
) -> dict[str, Any]:
    path = _require_regular_artifact(artifact_root / relative_path, label)
    digest = _file_sha256(path)
    if digest != expected_sha256:
        raise SummaryError(f"{label} raw digest drifted")
    return {
        "path": relative_path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def _evaluation_waiver_attestation(artifact_root: Path) -> dict[str, Any]:
    waiver_path = _require_regular_artifact(
        artifact_root / EVALUATION_SOURCE_EQUIVALENCE_RELATIVE_PATH,
        "canonical evaluation source waiver",
    )
    if _file_sha256(waiver_path) != EVALUATION_SOURCE_EQUIVALENCE_RAW_SHA256:
        raise SummaryError("canonical evaluation source waiver raw digest drifted")
    waiver = _read_json(waiver_path, "canonical evaluation source waiver")
    drift = _mapping(waiver.get("drift"), "canonical evaluation source waiver drift")
    bytecode = _mapping(
        waiver.get("frozen_bytecode_evidence"),
        "canonical evaluation source waiver bytecode",
    )
    frozen_bytecode_path = Path(str(bytecode.get("path", "")))
    if bytecode.get("sha256") != EVALUATION_WAIVER_FROZEN_BYTECODE_RAW_SHA256:
        raise SummaryError("canonical waiver frozen bytecode digest drifted")
    current_source = _attested_file(
        artifact_root,
        MODEL_SOURCE_RELATIVE_PATH,
        label="current evaluation model source",
        expected_sha256=CURRENT_MODEL_SOURCE_SHA256,
    )
    if current_source["size_bytes"] != CURRENT_MODEL_SOURCE_SIZE:
        raise SummaryError("current evaluation model source size drifted")
    expected_drift = {
        "path": MODEL_SOURCE_RELATIVE_PATH.as_posix(),
        "preregistered_size_bytes": FROZEN_MODEL_SOURCE_SIZE,
        "preregistered_sha256": FROZEN_MODEL_SOURCE_SHA256,
        "evaluation_size_bytes": CURRENT_MODEL_SOURCE_SIZE,
        "evaluation_sha256": CURRENT_MODEL_SOURCE_SHA256,
        "classification": "formatting_and_unused_import_cleanup",
    }
    if drift != expected_drift:
        raise SummaryError("canonical evaluation source waiver drift binding changed")
    return {
        "waiver": {
            "path": EVALUATION_SOURCE_EQUIVALENCE_RELATIVE_PATH.as_posix(),
            "size_bytes": waiver_path.stat().st_size,
            "sha256": EVALUATION_SOURCE_EQUIVALENCE_RAW_SHA256,
        },
        "launcher": _attested_file(
            artifact_root,
            EVALUATION_WAIVER_LAUNCHER_RELATIVE_PATH,
            label="Q0 waiver evaluation launcher",
            expected_sha256=EVALUATION_WAIVER_LAUNCHER_RAW_SHA256,
        ),
        "sitecustomize": _attested_file(
            artifact_root,
            EVALUATION_WAIVER_SITECUSTOMIZE_RELATIVE_PATH,
            label="Q0 waiver sitecustomize",
            expected_sha256=EVALUATION_WAIVER_SITECUSTOMIZE_RAW_SHA256,
        ),
        "frozen_bytecode": _attested_file(
            artifact_root,
            frozen_bytecode_path,
            label="Q0 waiver frozen bytecode",
            expected_sha256=EVALUATION_WAIVER_FROZEN_BYTECODE_RAW_SHA256,
        ),
        "frozen_source": {
            "path": MODEL_SOURCE_RELATIVE_PATH.as_posix(),
            "size_bytes": FROZEN_MODEL_SOURCE_SIZE,
            "sha256": FROZEN_MODEL_SOURCE_SHA256,
        },
        "current_source": current_source,
    }


def _expected_q0_attestation_metadata(
    analysis_id: str, seed: int, artifact_root: Path
) -> dict[str, Any]:
    job_name = Q0_ATTESTED_EVALUATION_JOBS.get((analysis_id, seed))
    if job_name is not None:
        return {
            "source_manifest_sha256": REGISTERED_Q0_SOURCE_SHA256,
            "evaluation_source_mode": "frozen_bytecode_equivalence_waiver",
            "evaluation_source_equivalence_sha256": (
                EVALUATION_SOURCE_EQUIVALENCE_RAW_SHA256
            ),
            "evaluation_job_name": job_name,
            "evaluation_waiver": _evaluation_waiver_attestation(artifact_root),
        }
    return {
        "source_manifest_sha256": REGISTERED_Q0_SOURCE_SHA256,
        "evaluation_source_mode": "frozen_registered_source",
        "evaluation_source_equivalence_sha256": None,
        "evaluation_job_name": None,
        "evaluation_waiver": None,
    }


def _validate_q0_attestation_shape(
    payload: Mapping[str, Any], source: Path, artifact_root: Path
) -> None:
    if set(payload) != {"schema", "runs", "manifest_sha256"}:
        raise SummaryError(f"{source}: Q0 attestation top-level fields drifted")
    if payload.get("schema") != Q0_ATTESTATION_SCHEMA:
        raise SummaryError(f"{source}: Q0 attestation schema drifted")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 6:
        raise SummaryError(f"{source}: Q0 attestation requires exactly six runs")
    expected_fields = {
        "analysis_id",
        "training_seed",
        "metrics_path",
        "metrics_sha256",
        "source_manifest_sha256",
        "evaluation_source_mode",
        "evaluation_source_equivalence_sha256",
        "evaluation_job_name",
        "evaluation_waiver",
    }
    seen: set[tuple[str, int]] = set()
    for index, raw_entry in enumerate(runs):
        entry = _mapping(raw_entry, f"{source}: Q0 attestation run {index}")
        if set(entry) != expected_fields:
            raise SummaryError(f"{source}: Q0 attestation run {index} fields drifted")
        analysis_id = entry.get("analysis_id")
        seed = entry.get("training_seed")
        if (
            not isinstance(analysis_id, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise SummaryError(
                f"{source}: Q0 attestation run {index} ID/seed is invalid"
            )
        pair = (analysis_id, seed)
        if pair in seen:
            raise SummaryError(f"{source}: duplicate Q0 attestation pair {pair}")
        seen.add(pair)
        if pair not in EXPECTED_Q0_ATTESTATION_PAIRS:
            raise SummaryError(f"{source}: unexpected Q0 attestation pair {pair}")
        expected_path = (
            Path("output") / q_factor_run_slug(analysis_id, seed) / EVAL_METRICS_SUFFIX
        ).as_posix()
        if entry.get("metrics_path") != expected_path:
            raise SummaryError(
                f"{source}: Q0 attestation metrics path drifted for "
                f"{analysis_id}@{seed}"
            )
        _sha256(
            entry.get("metrics_sha256"),
            f"{source}: Q0 attestation metrics digest for {analysis_id}@{seed}",
        )
        expected_metadata = _expected_q0_attestation_metadata(
            analysis_id, seed, artifact_root
        )
        actual_metadata = {key: entry.get(key) for key in expected_metadata}
        if actual_metadata != expected_metadata:
            raise SummaryError(
                f"{source}: Q0 attestation source/job binding drifted for "
                f"{analysis_id}@{seed}"
            )
    if seen != EXPECTED_Q0_ATTESTATION_PAIRS:
        raise SummaryError(
            f"{source}: Q0 attestation must contain the exact 2x3 matrix"
        )


def build_q0_metrics_attestation(
    artifact_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build an attestation for the exact six immutable Q0 metrics files."""

    root = Path(artifact_root).resolve()
    runs = []
    for analysis_id in Q0_IDS:
        for seed in sorted(Q_FACTOR_SEEDS):
            metrics_path = _expected_metrics_path(root, analysis_id, seed)
            row, _signature, evidence, _payload = _load_q0_run(
                analysis_id,
                seed,
                metrics_path,
                artifact_root=root,
            )
            relative_path = (
                Path("output")
                / q_factor_run_slug(analysis_id, seed)
                / EVAL_METRICS_SUFFIX
            ).as_posix()
            metadata = _expected_q0_attestation_metadata(analysis_id, seed, root)
            if {
                "source_manifest_sha256": evidence["source_manifest_sha256"],
                "evaluation_source_mode": row["evaluation_source_mode"],
                "evaluation_source_equivalence_sha256": row[
                    "evaluation_source_equivalence_sha256"
                ],
            } != {
                key: metadata[key]
                for key in (
                    "source_manifest_sha256",
                    "evaluation_source_mode",
                    "evaluation_source_equivalence_sha256",
                )
            }:
                raise SummaryError(
                    f"{metrics_path}: Q0 source metadata cannot be attested"
                )
            runs.append(
                {
                    "analysis_id": analysis_id,
                    "training_seed": seed,
                    "metrics_path": relative_path,
                    "metrics_sha256": row["metrics_sha256"],
                    **metadata,
                }
            )
    manifest = {"schema": Q0_ATTESTATION_SCHEMA, "runs": runs}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _validate_q0_attestation_shape(manifest, Path("<generated Q0 attestation>"), root)
    return manifest


def load_and_validate_q0_metrics_attestation(
    path: str | Path,
    *,
    artifact_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    source = _require_regular_artifact(
        Path(path).expanduser(), "Q0 metrics attestation manifest"
    )
    if Q0_ATTESTATION_MANIFEST_RAW_SHA256 is None:
        raise SummaryError(
            "Q0 metrics attestation raw SHA256 pin is not configured; "
            "summary construction is fail-closed"
        )
    expected_raw_sha256 = _sha256(
        Q0_ATTESTATION_MANIFEST_RAW_SHA256,
        "Q0 metrics attestation raw SHA256 pin",
    )
    if _file_sha256(source) != expected_raw_sha256:
        raise SummaryError(f"{source}: Q0 attestation raw digest differs from pin")
    payload = _read_json(source, "Q0 metrics attestation")
    root = Path(artifact_root).resolve()
    _validate_q0_attestation_shape(payload, source, root)
    stored = payload.get("manifest_sha256")
    _sha256(stored, f"{source}: Q0 attestation manifest digest")
    body = dict(payload)
    body.pop("manifest_sha256")
    if canonical_sha256(body) != stored:
        raise SummaryError(f"{source}: Q0 attestation self-digest mismatch")
    for entry in payload["runs"]:
        analysis_id = str(entry["analysis_id"])
        seed = int(entry["training_seed"])
        expected_path = _expected_metrics_path(root, analysis_id, seed)
        resolved = _resolve_exact(
            root / str(entry["metrics_path"]),
            expected_path,
            "attested Q0 metrics path",
        )
        if _file_sha256(resolved) != entry["metrics_sha256"]:
            raise SummaryError(
                f"{resolved}: raw metrics digest differs from Q0 attestation"
            )
    return payload


def _normalize_row(
    *,
    analysis_id: str,
    physical_id: str,
    source_cohort: str,
    seed: int,
    base_row: Mapping[str, Any],
    strategy: Mapping[str, Any],
    evidence: Mapping[str, Any],
    metrics_sha256: str,
    hf_config_sha256: str,
    checkpoint_hashes: Mapping[str, str] | None,
    evaluation_source_mode: str,
    evaluation_source_equivalence_sha256: str | None,
) -> dict[str, Any]:
    variant = Q_FACTOR_VARIANTS[analysis_id]
    hashes = dict(checkpoint_hashes or {})
    legacy = source_cohort == "legacy_confirmation_q1"
    return {
        "id": analysis_id,
        "analysis_id": analysis_id,
        "physical_ablation_id": physical_id,
        "source_cohort": source_cohort,
        "training_seed": seed,
        "parent_ablation_id": variant.parent_ablation_id,
        "observed_position_mode": variant.observed_position_mode,
        "effective_mask_position_mode": variant.mask_position_mode,
        "declared_mask_position_mode": None if legacy else "none",
        "mask_position_semantics": (
            "legacy_implicit_additive_2d" if legacy else "explicit_none"
        ),
        "fid": float(base_row["fid"]),
        "is_mean": float(base_row["is_mean"]),
        "is_std": float(base_row["is_std"]),
        "sampling_wall_seconds": float(base_row["sampling_wall_seconds"]),
        "sampling_samples_per_second": float(base_row["sampling_samples_per_second"]),
        "latent_mse_to_target": float(strategy["latent_mse_to_target"]),
        "latent_rms": float(strategy["latent_rms"]),
        "generation_step_max": float(strategy["generation_step_max"]),
        "peak_cuda_allocated_mib": float(base_row["peak_cuda_allocated_mib"]),
        "peak_cuda_reserved_mib": float(base_row["peak_cuda_reserved_mib"]),
        "total": int(base_row["total"]),
        "trainable": int(base_row["trainable"]),
        "image_embedder": int(base_row["image_embedder"]),
        "flow_head": int(base_row["flow_head"]),
        "metrics_path": str(base_row["metrics_path"]),
        "metrics_sha256": metrics_sha256,
        "hf_config_sha256": hf_config_sha256,
        "provenance_file_sha256": evidence["provenance_file_sha256"],
        "source_manifest_sha256": evidence["source_manifest_sha256"],
        "evaluation_source_mode": evaluation_source_mode,
        "evaluation_source_equivalence_sha256": (evaluation_source_equivalence_sha256),
        "checkpoint_metadata_sha256": hashes.get("checkpoint_metadata_sha256"),
        "ema_state_sha256": hashes.get("ema_state_sha256"),
        "hf_model_weights_sha256": hashes.get("hf_model_weights_sha256"),
    }


def _parent_summary_payload(
    artifact_root: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[tuple[str, int], Mapping[str, Any]]]:
    path = artifact_root / PARENT_SUMMARY_RELATIVE_PATH
    payload = _read_json(path, "frozen parent summary")
    try:
        evidence = load_parent_summary_evidence(path)
    except ValueError as exc:
        raise SummaryError(f"invalid frozen parent summary: {exc}") from exc
    if evidence["sha256"] != expected_sha256:
        raise SummaryError("frozen parent-summary digest differs from reuse binding")
    lookup = {
        (str(item.get("id")), int(item.get("training_seed"))): item
        for item in payload["runs"]
        if isinstance(item, Mapping)
        and item.get("id") in set(LEGACY_ANALYSIS_TO_PHYSICAL.values())
    }
    expected = {
        (physical_id, seed)
        for physical_id in LEGACY_ANALYSIS_TO_PHYSICAL.values()
        for seed in Q_FACTOR_SEEDS
    }
    if set(lookup) != expected:
        raise SummaryError("parent summary is missing an exact legacy E2b/E2 2x3 slice")
    return payload, lookup


def _validate_parent_row_binding(
    row: Mapping[str, Any], parent_row: Mapping[str, Any]
) -> None:
    fields = (
        "training_seed",
        "fid",
        "is_mean",
        "is_std",
        "sampling_wall_seconds",
        "sampling_samples_per_second",
        "latent_mse_to_target",
        "latent_rms",
        "generation_step_max",
        "peak_cuda_allocated_mib",
        "peak_cuda_reserved_mib",
        "total",
        "trainable",
        "image_embedder",
        "flow_head",
    )
    for field in fields:
        if row.get(field) != parent_row.get(field):
            raise SummaryError(
                f"legacy {row['analysis_id']}@{row['training_seed']} differs from "
                f"frozen parent summary field {field}"
            )
    try:
        parent_path = Path(str(parent_row.get("metrics_path"))).resolve(strict=True)
        row_path = Path(str(row.get("metrics_path"))).resolve(strict=True)
    except OSError as exc:
        raise SummaryError(f"parent-summary metrics path is invalid: {exc}") from exc
    if parent_path != row_path:
        raise SummaryError(
            f"legacy {row['analysis_id']}@{row['training_seed']} path differs from "
            "frozen parent summary"
        )


def _require_same(
    evidence: Sequence[Mapping[str, Any]], fields: Sequence[str], scope: str
) -> None:
    for field in fields:
        values = {json.dumps(item[field], sort_keys=True) for item in evidence}
        if len(values) != 1:
            details = {
                f"{item['analysis_id']}@{item['training_seed']}": item[field]
                for item in evidence
            }
            raise SummaryError(
                f"comparability mismatch for {field} within {scope}: {details}"
            )


def _by_seed(
    evidence: Sequence[Mapping[str, Any]],
) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for item in evidence:
        grouped[int(item["training_seed"])].append(item)
    return grouped


def _validate_legacy_gate(
    evidence: Sequence[Mapping[str, Any]], reuse_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if len(evidence) != 6:
        raise SummaryError("legacy Q1 cohort must contain exactly six runs")
    global_fields = (
        "screen_summary_sha256",
        "candidate_manifest_sha256",
        "dataset_length",
        "input_files_sha256",
        "base_model_manifest_sha256",
        "runtime_source_manifest_sha256",
        "evaluator_rng_contract_sha256",
        "canonical_noise_manifest_sha256",
        "ordered_eval_sample_manifest_sha256",
        "training_protocol_sha256",
        "parameters_sha256",
    )
    _require_same(evidence, global_fields, "legacy Q1 cohort")
    paired_fields = (
        "image_parameter_count",
        "image_parameter_schema_sha256",
        "image_state_sha256",
        "special_token_rows_sha256",
        "special_token_names_and_ids",
        "initial_generator_state_sha256",
        "dataloader_base_seed",
        "epoch0_ordered_sample_identity_sha256",
        "epoch0_augmentation_decisions_sha256",
        "input_files_sha256",
    )
    grouped = _by_seed(evidence)
    for seed, items in sorted(grouped.items()):
        if len(items) != 2:
            raise SummaryError(f"legacy Q1 seed {seed} must contain E2b and E2")
        _require_same(items, paired_fields, f"legacy Q1 seed {seed}")
    if len({items[0]["image_state_sha256"] for items in grouped.values()}) != 3:
        raise SummaryError("legacy Q1 seeds do not have independent initial states")
    return {
        "schema": LEGACY_GATE_SCHEMA,
        "validated_runs": len(evidence),
        "validated_training_seeds": sorted(grouped),
        "source_manifest_sha256": evidence[0]["source_manifest_sha256"],
        "reuse_manifest_sha256": reuse_manifest["manifest_sha256"],
        "legacy_equivalence": dict(LEGACY_EQUIVALENCE),
    }


def _validate_q0_gate(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(evidence) != 6:
        raise SummaryError("Q0 cohort must contain exactly six runs")
    global_fields = (
        "study_manifest_sha256",
        "parent_summary_sha256",
        "source_manifest_sha256",
        "input_files_sha256",
        "base_model_manifest_sha256",
        "runtime_software_sha256",
        "evaluator_rng_contract_sha256",
        "canonical_noise_manifest_sha256",
        "ordered_eval_sample_manifest_sha256",
        "parameters_sha256",
        "config_pairing_projection_sha256",
        "training_protocol_sha256",
    )
    _require_same(evidence, global_fields, "Q0 cohort")
    paired_fields = (
        "image_parameter_count",
        "image_parameter_schema_sha256",
        "image_state_sha256",
        "special_token_rows_sha256",
        "special_token_names_and_ids",
        "initial_generator_state_sha256",
        "dataloader_base_seed",
        "dataset_length",
        "epoch0_ordered_sample_identity_sha256",
        "epoch0_augmentation_decisions_sha256",
        "input_files_sha256",
    )
    grouped = _by_seed(evidence)
    for seed, items in sorted(grouped.items()):
        if len(items) != 2:
            raise SummaryError(f"Q0 seed {seed} must contain E2b-Q0 and E2-Q0")
        _require_same(items, paired_fields, f"Q0 seed {seed}")
    if len({items[0]["image_state_sha256"] for items in grouped.values()}) != 3:
        raise SummaryError("Q0 seeds do not have independent initial states")
    source_modes = {
        f"{item['analysis_id']}@{item['training_seed']}": item["evaluation_source_mode"]
        for item in evidence
    }
    expected_modes = {
        f"{analysis_id}@{seed}": (
            "frozen_bytecode_equivalence_waiver"
            if analysis_id == "E2-Q0" and seed in {44, 45}
            else "frozen_registered_source"
        )
        for analysis_id in Q0_IDS
        for seed in Q_FACTOR_SEEDS
    }
    if source_modes != expected_modes:
        raise SummaryError(f"Q0 evaluation source modes drifted: {source_modes}")
    return {
        "schema": Q0_GATE_SCHEMA,
        "validated_runs": len(evidence),
        "validated_training_seeds": sorted(grouped),
        "source_manifest_sha256": evidence[0]["source_manifest_sha256"],
        "parent_summary_sha256": evidence[0]["parent_summary_sha256"],
        "study_manifest_sha256": evidence[0]["study_manifest_sha256"],
        "evaluation_source_modes": source_modes,
        "evaluation_source_equivalence_sha256": (
            EVALUATION_SOURCE_EQUIVALENCE_RAW_SHA256
        ),
    }


def _validate_cross_gate(
    legacy: Sequence[Mapping[str, Any]], q0: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    equivalence_report = _validate_legacy_equivalence_report()
    evidence = [*legacy, *q0]
    shared_fields = (
        "training_protocol_sha256",
        "input_files_sha256",
        "base_model_manifest_sha256",
        "evaluator_rng_contract_sha256",
        "canonical_noise_manifest_sha256",
        "ordered_eval_sample_manifest_sha256",
        "parameters_sha256",
    )
    _require_same(evidence, shared_fields, "all 12 bridge runs")
    seed_fields = (
        "image_parameter_count",
        "image_parameter_schema_sha256",
        "image_state_sha256",
        "special_token_rows_sha256",
        "special_token_names_and_ids",
        "initial_generator_state_sha256",
        "dataloader_base_seed",
        "dataset_length",
        "epoch0_ordered_sample_identity_sha256",
        "epoch0_augmentation_decisions_sha256",
        "input_files_sha256",
    )
    grouped = _by_seed(evidence)
    for seed, items in sorted(grouped.items()):
        if len(items) != 4:
            raise SummaryError(f"cross-cohort seed {seed} must contain four cells")
        _require_same(items, seed_fields, f"cross-cohort seed {seed}")
    legacy_source = legacy[0]["source_manifest_sha256"]
    q0_source = q0[0]["source_manifest_sha256"]
    expected_legacy_source = equivalence_report["historical_implementation"][
        "runtime_source_manifest_sha256"
    ]
    expected_q0_source = equivalence_report["q0_registered_implementation"][
        "runtime_source_manifest_sha256"
    ]
    if legacy_source != expected_legacy_source or q0_source != expected_q0_source:
        raise SummaryError(
            "bridge source pair differs from the hash-bound legacy/Q0 equivalence report"
        )
    return {
        "schema": CROSS_GATE_SCHEMA,
        "validated_runs": len(evidence),
        "validated_training_seeds": sorted(grouped),
        "comparison_design": "seed_aligned_cross_source",
        "same_source_training": False,
        "causal_scope": "legacy_anchored_source_revision_confounded",
        "legacy_source_manifest_sha256": legacy_source,
        "q0_source_manifest_sha256": q0_source,
        "source_revision_confounded": True,
        "formal_q_factor": False,
        "post_hoc_amendment": True,
        "legacy_q1_equivalence_report": {
            "path": EQUIVALENCE_REPORT_RELATIVE_PATH.as_posix(),
            "raw_sha256": EQUIVALENCE_REPORT_RAW_SHA256,
            "report_sha256": equivalence_report["report_sha256"],
            "diff_classification": equivalence_report["diff_evidence"][
                "classification"
            ],
        },
        "matched_fields": list(shared_fields),
        "seed_aligned_fields": list(seed_fields),
    }


def _statistic(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise SummaryError("bridge statistics require exactly three finite seed values")
    average = mean(values)
    sample_std = stdev(values)
    standard_error = sample_std / math.sqrt(3)
    radius = T95_DF2 * standard_error
    return {
        "n": 3,
        "values": values,
        "mean": average,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "t_critical_95_df2": T95_DF2,
        "ci95": [average - radius, average + radius],
    }


def _effect(
    lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    candidate: str,
    reference: str,
    *,
    comparison_design: str,
    same_source_training: bool,
) -> dict[str, Any]:
    seeds = sorted(Q_FACTOR_SEEDS)

    def differences(field: str) -> list[float]:
        return [
            float(lookup[(candidate, seed)][field])
            - float(lookup[(reference, seed)][field])
            for seed in seeds
        ]

    fid = _statistic(differences("fid"))
    fid["candidate_wins"] = sum(value < 0 for value in fid["values"])
    fid["exact_ties"] = sum(value == 0 for value in fid["values"])
    return {
        "candidate": candidate,
        "reference": reference,
        "seeds": seeds,
        "comparison_design": comparison_design,
        "same_source_training": same_source_training,
        "fid_candidate_minus_reference": fid,
        "is_candidate_minus_reference": _statistic(differences("is_mean")),
        "throughput_candidate_minus_reference": _statistic(
            differences("sampling_samples_per_second")
        ),
    }


def _interaction(lookup: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    seeds = sorted(Q_FACTOR_SEEDS)

    def values(field: str) -> list[float]:
        return [
            (
                float(lookup[("E2-Q0", seed)][field])
                - float(lookup[("E2b-Q0", seed)][field])
            )
            - (
                float(lookup[("E2-Q1", seed)][field])
                - float(lookup[("E2b-Q1", seed)][field])
            )
            for seed in seeds
        ]

    return {
        "formula": "(E2-Q0 - E2b-Q0) - (E2-Q1 - E2b-Q1)",
        "seeds": seeds,
        "comparison_design": "difference_of_within_cohort_paired_effects",
        "same_source_training": False,
        "causal_scope": "cross_source_descriptive_interaction",
        "fid": _statistic(values("fid")),
        "is": _statistic(values("is_mean")),
        "throughput": _statistic(values("sampling_samples_per_second")),
    }


def _aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)
    output = []
    for analysis_id in ANALYSIS_IDS:
        group = grouped[analysis_id]
        if len(group) != 3:
            raise SummaryError(f"{analysis_id} aggregate requires exactly three seeds")
        output.append(
            {
                "id": analysis_id,
                "source_cohort": group[0]["source_cohort"],
                "parent_ablation_id": Q_FACTOR_VARIANTS[analysis_id].parent_ablation_id,
                "observed_position_mode": Q_FACTOR_VARIANTS[
                    analysis_id
                ].observed_position_mode,
                "mask_position_mode": Q_FACTOR_VARIANTS[analysis_id].mask_position_mode,
                "seeds": sorted(int(item["training_seed"]) for item in group),
                "fid_mean": mean(float(item["fid"]) for item in group),
                "fid_sample_std": stdev(float(item["fid"]) for item in group),
                "is_mean": mean(float(item["is_mean"]) for item in group),
                "is_sample_std": stdev(float(item["is_mean"]) for item in group),
                "sampling_samples_per_second_mean": mean(
                    float(item["sampling_samples_per_second"]) for item in group
                ),
                "sampling_samples_per_second_sample_std": stdev(
                    float(item["sampling_samples_per_second"]) for item in group
                ),
            }
        )
    return output


def _pareto(aggregates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in aggregates:
        dominated = any(
            other is not candidate
            and float(other["fid_mean"]) <= float(candidate["fid_mean"])
            and float(other["is_mean"]) >= float(candidate["is_mean"])
            and (
                float(other["fid_mean"]) < float(candidate["fid_mean"])
                or float(other["is_mean"]) > float(candidate["is_mean"])
            )
            for other in aggregates
        )
        if not dominated:
            frontier.append(dict(candidate))
    return sorted(frontier, key=lambda item: (item["fid_mean"], -item["is_mean"]))


def _selection(
    aggregates: Sequence[Mapping[str, Any]],
    parent_summary: Mapping[str, Any],
    parent_summary_sha256: str,
) -> dict[str, Any]:
    bridge_by_id = {str(item["id"]): item for item in aggregates}
    parent_aggregates = parent_summary.get("aggregates")
    if not isinstance(parent_aggregates, list):
        raise SummaryError("frozen parent summary is missing decision aggregates")
    parent_by_id = {
        str(item.get("id")): item
        for item in parent_aggregates
        if isinstance(item, Mapping)
    }
    if set(parent_by_id) != set(PARENT_CONFIRMATION_IDS):
        raise SummaryError("frozen parent decision candidate set drifted")
    candidates = [
        {
            "id": candidate_id,
            "source": "parent_confirmation",
            "fid_mean": _finite(
                parent_by_id[candidate_id].get("fid_mean"),
                f"parent aggregate {candidate_id} FID",
                positive=True,
            ),
            "is_mean": _finite(
                parent_by_id[candidate_id].get("is_mean"),
                f"parent aggregate {candidate_id} IS",
                positive=True,
            ),
        }
        for candidate_id in PARENT_CONFIRMATION_IDS
    ]
    candidates.extend(
        {
            "id": str(item["id"]),
            "source": str(item["source_cohort"]),
            "fid_mean": float(item["fid_mean"]),
            "is_mean": float(item["is_mean"]),
        }
        for item in aggregates
    )
    best = min(candidates, key=lambda item: (item["fid_mean"], -item["is_mean"]))
    simple = bridge_by_id["E2-Q0"]
    fid_delta = float(simple["fid_mean"]) - float(best["fid_mean"])
    is_delta = float(simple["is_mean"]) - float(best["is_mean"])
    close = (
        abs(fid_delta) <= FID_SIMPLICITY_MARGIN
        and abs(is_delta) <= IS_SIMPLICITY_MARGIN
    )
    selected = "E2-Q0" if close else str(best["id"])
    return {
        "decision_rule_schema": Q_FACTOR_DECISION_RULE["schema"],
        "comparison_design": "legacy_anchored_cross_source",
        "same_source_training": False,
        "fid_margin": FID_SIMPLICITY_MARGIN,
        "is_margin": IS_SIMPLICITY_MARGIN,
        "best_fid_id": str(best["id"]),
        "best_fid_source": str(best["source"]),
        "e2_q0_minus_best_fid": fid_delta,
        "e2_q0_minus_best_is": is_delta,
        "within_simplicity_margins": close,
        "selected_id": selected,
        "selected_reason": "simplicity_preference" if close else "best_mean_fid",
        "limitation": "Q0-vs-Q1 selection includes a frozen source revision confound",
        "decision_evidence": {
            "authoritative_candidates": candidates,
            "parent_summary_sha256": parent_summary_sha256,
            "nominal_best": dict(best),
            "e2_q0": {
                "fid_mean": float(simple["fid_mean"]),
                "is_mean": float(simple["is_mean"]),
            },
            "rule": (
                "select E2-Q0 when its absolute mean-FID gap to the nominal "
                "best is <=0.5 and its absolute mean-IS gap is <=1.0"
            ),
        },
    }


def build_summary(
    run_specs: Sequence[str],
    *,
    artifact_root: str | Path = REPO_ROOT,
    reuse_manifest_path: str | Path = DEFAULT_REUSE_MANIFEST,
    q0_attestation_manifest_path: str | Path = DEFAULT_Q0_ATTESTATION_MANIFEST,
    enforce_production_pins: bool = False,
) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    if enforce_production_pins:
        try:
            q0_attestation_source = (
                Path(q0_attestation_manifest_path).expanduser().resolve(strict=True)
            )
            default_q0_attestation_source = DEFAULT_Q0_ATTESTATION_MANIFEST.resolve(
                strict=True
            )
        except OSError as exc:
            raise SummaryError(
                f"missing production Q0 metrics attestation manifest: {exc}"
            ) from exc
        if root != REPO_ROOT or q0_attestation_source != default_q0_attestation_source:
            raise SummaryError(
                "production summary requires REPO_ROOT and the default Q0 "
                "metrics attestation path"
            )
    reuse_manifest = load_and_validate_legacy_reuse_manifest(
        reuse_manifest_path,
        artifact_root=root,
        enforce_production_pin=enforce_production_pins,
    )
    q0_attestation = load_and_validate_q0_metrics_attestation(
        q0_attestation_manifest_path,
        artifact_root=root,
    )
    q0_attestation_entries = {
        (str(item["analysis_id"]), int(item["training_seed"])): item
        for item in q0_attestation["runs"]
    }
    if set(q0_attestation_entries) != EXPECTED_Q0_ATTESTATION_PAIRS:
        raise SummaryError(
            "Q0 metrics attestation does not contain the exact 2x3 slice"
        )
    manifest_entries = {
        (str(item["analysis_id"]), int(item["training_seed"])): item
        for item in reuse_manifest["runs"]
    }
    if set(manifest_entries) != {
        (analysis_id, seed)
        for analysis_id in LEGACY_ANALYSIS_TO_PHYSICAL
        for seed in Q_FACTOR_SEEDS
    }:
        raise SummaryError("legacy reuse manifest does not contain the exact 2x3 slice")

    parsed = []
    seen: set[tuple[str, int]] = set()
    seen_paths: set[Path] = set()
    for spec in run_specs:
        analysis_id, seed, path = _parse_run(spec)
        key = (analysis_id, seed)
        if key in seen:
            raise SummaryError(
                f"duplicate bridge run assignment for {analysis_id}@{seed}"
            )
        expected_path = _expected_metrics_path(root, analysis_id, seed)
        resolved = _resolve_exact(path, expected_path, "bridge metrics path")
        if resolved in seen_paths:
            raise SummaryError(f"duplicate physical metrics path: {resolved}")
        seen.add(key)
        seen_paths.add(resolved)
        parsed.append((analysis_id, seed, resolved))
    if seen != EXPECTED_PAIRS:
        raise SummaryError(
            "legacy bridge requires the exact 4x3 analysis matrix; "
            f"missing={sorted(EXPECTED_PAIRS - seen)}, "
            f"unexpected={sorted(seen - EXPECTED_PAIRS)}"
        )

    parent_sha = reuse_manifest["parent_summary"]["sha256"]
    parent_payload, parent_lookup = _parent_summary_payload(root, parent_sha)
    rows = []
    legacy_evidence = []
    q0_evidence = []
    signature = None
    for analysis_id, seed, path in parsed:
        if analysis_id in LEGACY_ANALYSIS_TO_PHYSICAL:
            row, current_signature, evidence, _payload = _load_legacy_run(
                analysis_id,
                seed,
                path,
                artifact_root=root,
                manifest_entry=manifest_entries[(analysis_id, seed)],
            )
            physical_id = LEGACY_ANALYSIS_TO_PHYSICAL[analysis_id]
            _validate_parent_row_binding(row, parent_lookup[(physical_id, seed)])
            legacy_evidence.append(evidence)
        else:
            row, current_signature, evidence, _payload = _load_q0_run(
                analysis_id,
                seed,
                path,
                artifact_root=root,
                attestation_entry=q0_attestation_entries[(analysis_id, seed)],
            )
            if evidence["parent_summary_sha256"] != parent_sha:
                raise SummaryError(
                    f"{analysis_id}@{seed} Q0 provenance is not bound to the reused "
                    "legacy parent summary"
                )
            q0_evidence.append(evidence)
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise SummaryError(
                f"{path}: frozen evaluation protocol differs across bridge runs"
            )
        rows.append(row)

    order = {analysis_id: index for index, analysis_id in enumerate(ANALYSIS_IDS)}
    rows.sort(key=lambda item: (order[str(item["id"])], int(item["training_seed"])))
    legacy_gate = _validate_legacy_gate(legacy_evidence, reuse_manifest)
    q0_gate = _validate_q0_gate(q0_evidence)
    cross_gate = _validate_cross_gate(legacy_evidence, q0_evidence)
    aggregates = _aggregates(rows)
    ranking_fid = sorted(
        aggregates, key=lambda item: (item["fid_mean"], -item["is_mean"])
    )
    ranking_is = sorted(
        aggregates, key=lambda item: (-item["is_mean"], item["fid_mean"])
    )
    lookup = {(str(item["id"]), int(item["training_seed"])): item for item in rows}
    within_cohort = {
        "observed_none_minus_additive_at_Q1": _effect(
            lookup,
            "E2-Q1",
            "E2b-Q1",
            comparison_design="within_legacy_cohort_paired",
            same_source_training=True,
        ),
        "observed_none_minus_additive_at_Q0": _effect(
            lookup,
            "E2-Q0",
            "E2b-Q0",
            comparison_design="within_q0_cohort_paired",
            same_source_training=True,
        ),
    }
    cross_source = {
        "mask_Q0_minus_Q1_at_E2b": _effect(
            lookup,
            "E2b-Q0",
            "E2b-Q1",
            comparison_design="seed_aligned_cross_source",
            same_source_training=False,
        ),
        "mask_Q0_minus_Q1_at_E2": _effect(
            lookup,
            "E2-Q0",
            "E2-Q1",
            comparison_design="seed_aligned_cross_source",
            same_source_training=False,
        ),
    }
    return {
        "schema": SCHEMA,
        "expected": "exact_legacy_q1_plus_q0_2x2x3",
        "analysis_design": "historical_control_bridge",
        "comparison_kind": "seed_aligned_cross_source_descriptive",
        "prospective_matrix_complete": False,
        "formal_q_factor": False,
        "post_hoc_amendment": True,
        "same_source_training": False,
        "source_revision_confounded": True,
        "protocol": signature,
        "reuse_binding": {
            "manifest_path": str(Path(reuse_manifest_path).resolve()),
            "manifest_sha256": reuse_manifest["manifest_sha256"],
            "parent_summary": reuse_manifest["parent_summary"],
            "legacy_equivalence": reuse_manifest["legacy_equivalence"],
            "validated_legacy_runs": 6,
        },
        "q0_metrics_attestation": {
            "manifest_path": str(Path(q0_attestation_manifest_path).resolve()),
            "manifest_raw_sha256": Q0_ATTESTATION_MANIFEST_RAW_SHA256,
            "manifest_sha256": q0_attestation["manifest_sha256"],
            "validated_q0_runs": 6,
        },
        "runs": rows,
        "aggregates": aggregates,
        "ranking_by_fid": [item["id"] for item in ranking_fid],
        "ranking_by_is": [item["id"] for item in ranking_is],
        "best_by_fid": ranking_fid[0],
        "best_by_is": ranking_is[0],
        "fid_is_pareto_frontier": _pareto(aggregates),
        "cohort_gates": {"legacy_q1": legacy_gate, "q0": q0_gate},
        "cross_cohort_comparability_gate": cross_gate,
        "within_cohort_paired_effects": within_cohort,
        "seed_aligned_cross_source_effects": cross_source,
        "interaction": _interaction(lookup),
        "selection": _selection(aggregates, parent_payload, parent_sha),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--artifact-root", default=str(REPO_ROOT))
    parser.add_argument("--reuse-manifest", default=str(DEFAULT_REUSE_MANIFEST))
    parser.add_argument(
        "--q0-attestation-manifest",
        default=str(DEFAULT_Q0_ATTESTATION_MANIFEST),
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument(
        "--generate-reuse-manifest",
        help="Write a hash-bound manifest for the exact legacy E2b/E2 seed 43/44/45 files.",
    )
    parser.add_argument(
        "--generate-q0-attestation-manifest",
        help="Atomically write the exact six-run Q0 metrics attestation.",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if Path(args.artifact_root).resolve() != REPO_ROOT:
        parser.error("production CLI requires --artifact-root to be REPO_ROOT")
    if args.generate_reuse_manifest and args.generate_q0_attestation_manifest:
        parser.error("only one manifest may be generated at a time")
    if args.generate_reuse_manifest:
        if args.run or args.output_json or args.output_csv:
            parser.error(
                "--generate-reuse-manifest cannot be combined with summary outputs"
            )
        target = Path(args.generate_reuse_manifest).expanduser().resolve()
        if target != DEFAULT_REUSE_MANIFEST:
            parser.error(
                "production CLI writes the legacy reuse manifest only at its "
                "versioned default path"
            )
        manifest = build_legacy_reuse_manifest(args.artifact_root)
        _atomic_write(
            target,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        print(
            json.dumps(
                {
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest_raw_sha256": _file_sha256(target),
                },
                indent=2,
            )
        )
        return
    if args.generate_q0_attestation_manifest:
        if args.run or args.output_json or args.output_csv:
            parser.error(
                "--generate-q0-attestation-manifest cannot be combined with "
                "summary outputs"
            )
        target = Path(args.generate_q0_attestation_manifest).expanduser().resolve()
        if target != DEFAULT_Q0_ATTESTATION_MANIFEST:
            parser.error(
                "production CLI writes the Q0 metrics attestation only at its "
                "versioned default path"
            )
        manifest = build_q0_metrics_attestation(args.artifact_root)
        _atomic_write(
            target,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        print(
            json.dumps(
                {
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest_raw_sha256": _file_sha256(target),
                },
                indent=2,
            )
        )
        return
    if not args.output_json:
        parser.error("--output-json is required when building a bridge summary")
    summary = build_summary(
        args.run,
        artifact_root=args.artifact_root,
        reuse_manifest_path=args.reuse_manifest,
        q0_attestation_manifest_path=args.q0_attestation_manifest,
        enforce_production_pins=True,
    )
    _atomic_write(Path(args.output_json), json.dumps(summary, indent=2) + "\n")
    if args.output_csv:
        _write_csv(Path(args.output_csv), summary["runs"])
    print(json.dumps(summary["selection"], indent=2))


if __name__ == "__main__":
    main()
