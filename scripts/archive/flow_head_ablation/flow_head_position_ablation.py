#!/usr/bin/env python3
"""Closed FH0--FH4 matrix, run provenance, and config generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.modeling_model.image_flow_position import (  # noqa: E402
    FLOW_HEAD_POSITION_SPECS,
    SUPPORTED_FLOW_HEAD_POSITION_VARIANTS,
    resolve_flow_head_position_config,
)
from scripts.archive.image_backbone_ablation.image_embedder_confirmation_protocol import (  # noqa: E402
    base_model_evidence,
    initial_state_evidence,
    train_data_evidence,
)

DEFAULT_BASE_CONFIG = (
    REPO_ROOT / "configs/ablation/imagenet_flow_head_position_100c_80ep.yaml"
)
SCREEN_SEED = 42
CONFIRMATION_SEEDS = (43, 44, 45)
STUDY_NAME = "flow_head_position"
PROVENANCE_FILENAME = "flow_head_position_training_provenance.json"
SUMMARY_SCHEMA = "selfless_flow_head_position_ablation_summary_v1"
SELECTOR_SCHEMA = "selfless_flow_head_position_selector_v1"
PROVENANCE_SCHEMA = "selfless_flow_head_position_training_provenance_v1"
SOURCE_MANIFEST_SCHEMA = "selfless_flow_head_position_source_manifest_v1"

FLOW_HEAD_INVARIANTS = {
    "image_flow_head_arch": "contextual",
    "image_flow_depth": 8,
    "image_flow_width": 1280,
    "image_flow_mlp_ratio": 1.0,
    "image_flow_latent_mixer_heads": 8,
    "image_flow_latent_mixer_dropout": 0.0,
    "image_flow_latent_mixer_zero_init_gate": True,
}

RUNTIME_SOURCE_FILES = (
    "configs/ablation/imagenet_flow_head_position_100c_80ep.yaml",
    "docs/SELFLESS_FLOW_FLOW_HEAD_2D_ROPE_ABLATION_PROPOSAL.md",
    "script/ablation/train_eval_flow_head_position_ablation.sh",
    "scripts/flow_head_position_ablation.py",
    "scripts/summarize_flow_head_position_ablation.py",
    "scripts/evaluate_single_stream_fid_is.py",
    "pretrain/train_selfless_flow.py",
    "utils/dataset_imagenet_flow_cache.py",
    "utils/utils.py",
    "models/modeling_model/image_backbone.py",
    "models/modeling_model/image_flow_loss.py",
    "models/modeling_model/image_flow_position.py",
    "models/modeling_model/image_position_utils.py",
    "models/modeling_model/modeling_selfless_flow.py",
)

INITIALIZATION_CONTRACT = {
    "schema": "flow_head_position_module_keyed_initialization_v1",
    "policy": (
        "training-seed plus stable module name; every FH variant has identical "
        "learned parameter names, shapes, dtypes, and initial bytes"
    ),
}
TRAIN_ORDER_CONTRACT = {
    "schema": "flow_head_position_paired_train_order_v1",
    "policy": (
        "dedicated CPU dataloader generator equals training seed; dataset split "
        "remains seed 42; stateless latent hflip uses the same seed"
    ),
}


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


def normalize_variant_id(value: str) -> str:
    normalized = str(value).strip().upper().replace("_", "")
    if normalized not in FLOW_HEAD_POSITION_SPECS:
        raise ValueError(
            f"Unknown FH variant {value!r}; expected "
            f"{SUPPORTED_FLOW_HEAD_POSITION_VARIANTS}."
        )
    return normalized


def run_slug(variant_id: str, seed: int) -> str:
    return f"selfless-flow-fhpos-{normalize_variant_id(variant_id).lower()}-s{int(seed)}"


def runtime_source_evidence(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root)
    entries = []
    for relative in RUNTIME_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"FH runtime source is missing: {path}")
        entries.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "files": entries,
        "manifest_sha256": canonical_sha256(entries),
    }


def _config_payload(config: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("resolved config must be a mapping")
    # ``get_config`` accepts both ``config=...`` and ``--config ...``.  The
    # former appears as a launcher-only root selector after OmegaConf merging
    # and must never alter the scientific config fingerprint.
    payload.pop("config", None)
    experiment = payload.get("experiment")
    if isinstance(experiment, dict):
        experiment.pop("config_fingerprint", None)
        experiment.pop("flow_head_position_provenance_path", None)
        experiment.pop("flow_head_position_provenance_sha256", None)
        project = str(experiment.get("project", ""))
        output_dir = Path(str(experiment.get("output_dir", "output")))
        if output_dir.name == project:
            experiment["output_dir"] = str(output_dir.parent)
    model = payload.get("model")
    if isinstance(model, dict):
        for key in (
            "mask_token_id",
            "boi_token_id",
            "eoi_token_id",
            "image_mask_token_id",
            "image_offset",
        ):
            model.pop(key, None)
    return payload


def config_fingerprint(config: DictConfig) -> str:
    return canonical_sha256(_config_payload(config))


def architecture_contract(config: DictConfig) -> dict[str, Any]:
    spec, axis_dims = resolve_flow_head_position_config(config)
    if spec is None or axis_dims is None:
        raise ValueError("FH study requires an explicit contextual position contract")
    return {
        "schema": "selfless_flow_head_position_architecture_v1",
        "image_backbone_variant": str(config.model.image_backbone_variant),
        "flow_head_position": spec.as_contract(axis_dims),
        "flow_head_invariants": dict(FLOW_HEAD_INVARIANTS),
        "image_tokens_per_img": int(config.model.image_tokens_per_img),
        "image_latent_dim": int(config.model.image_latent_dim),
    }


def is_flow_head_position_config(config: DictConfig) -> bool:
    experiment = config.get("experiment")
    return bool(
        experiment is not None
        and str(experiment.get("ablation_study", "")).strip().lower() == STUDY_NAME
    )


def _load_screen_selector(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read FH screen summary {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("FH confirmation requires the formal screen summary")
    selector = payload.get("selector")
    if not isinstance(selector, Mapping) or selector.get("schema") != SELECTOR_SCHEMA:
        raise ValueError("FH screen summary is missing the frozen selector")
    if payload.get("phase") != "screen":
        raise ValueError("FH confirmation source must be a screen summary")
    selected = selector.get("selected_ids")
    if (
        not isinstance(selected, list)
        or "FH0" not in selected
        or any(value not in SUPPORTED_FLOW_HEAD_POSITION_VARIANTS for value in selected)
    ):
        raise ValueError("FH screen selected_ids are invalid")
    return dict(selector), hashlib.sha256(raw).hexdigest()


def validate_ablation_config(
    config: DictConfig,
    expected_id: str | None = None,
) -> None:
    if not is_flow_head_position_config(config):
        raise ValueError("experiment.ablation_study must be flow_head_position")
    variant_id = normalize_variant_id(expected_id or config.experiment.ablation_id)
    contract = architecture_contract(config)
    if contract["flow_head_position"]["variant"] != variant_id:
        raise ValueError(
            f"{variant_id} ID does not match its flow-head position contract"
        )
    if str(config.model.image_backbone_variant) != "E2-Q1":
        raise ValueError("The primary FH matrix is fixed to image_backbone_variant=E2-Q1")
    for key, expected in FLOW_HEAD_INVARIANTS.items():
        actual = config.model.get(key)
        if actual != expected:
            raise ValueError(
                f"All FH variants require model.{key}={expected!r}, got {actual!r}"
            )
    fixed_values = {
        "dataset.params.split_seed": 42,
        "evaluation.seed": 42,
        "evaluation.samples": 10_000,
        "evaluation.cfg": 3.5,
        "evaluation.parallel_rate": 1,
        "training.total_batch_size": 256,
        "training.max_train_steps": 35_920,
    }
    for key, expected in fixed_values.items():
        actual = OmegaConf.select(config, key)
        if actual != expected:
            raise ValueError(f"FH protocol requires {key}={expected!r}, got {actual!r}")
    seed = int(config.training.seed)
    if int(config.training.dataloader_shuffle_seed) != seed:
        raise ValueError("FH dataloader shuffle seed must equal the training seed")
    phase = str(config.experiment.ablation_phase)
    if phase == "screen":
        if seed != SCREEN_SEED:
            raise ValueError("FH screen must use seed 42")
    elif phase == "confirmation":
        if seed not in CONFIRMATION_SEEDS:
            raise ValueError(
                f"FH confirmation seed must be one of {CONFIRMATION_SEEDS}"
            )
        selector, summary_sha = _load_screen_selector(
            config.experiment.screen_summary_path
        )
        if variant_id not in selector["selected_ids"]:
            raise ValueError(f"{variant_id} was not selected for FH confirmation")
        if str(config.experiment.screen_summary_sha256) != summary_sha:
            raise ValueError("FH screen summary changed after confirmation config freeze")
        if canonical_sha256(selector) != str(
            config.experiment.selector_manifest_sha256
        ):
            raise ValueError("FH selector manifest digest mismatch")
    else:
        raise ValueError(f"Unknown FH ablation phase {phase!r}")
    expected_slug = run_slug(variant_id, seed)
    if str(config.experiment.project) != expected_slug:
        raise ValueError(
            f"FH run slug mismatch: expected={expected_slug!r}, "
            f"actual={config.experiment.project!r}"
        )
    declared_source = _require_sha256(
        config.experiment.runtime_source_manifest_sha256,
        "FH source manifest",
    )
    current_source = runtime_source_evidence()["manifest_sha256"]
    if declared_source != current_source:
        raise ValueError(
            "FH runtime source changed after config generation: "
            f"declared={declared_source}, current={current_source}"
        )
    declared_fingerprint = _require_sha256(
        config.experiment.config_fingerprint,
        "FH config fingerprint",
    )
    actual_fingerprint = config_fingerprint(config)
    if declared_fingerprint != actual_fingerprint:
        raise ValueError(
            "FH resolved config fingerprint mismatch: "
            f"declared={declared_fingerprint}, actual={actual_fingerprint}"
        )


def build_ablation_config(
    variant_id: str,
    seed: int,
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    screen_summary: str | Path | None = None,
) -> DictConfig:
    variant_id = normalize_variant_id(variant_id)
    seed = int(seed)
    config = OmegaConf.load(Path(base_config))
    spec = FLOW_HEAD_POSITION_SPECS[variant_id]
    slug = run_slug(variant_id, seed)
    config.experiment.project = slug
    config.experiment.name = (
        f"imagenet100-flow-head-position-{variant_id.lower()}-seed{seed}-"
        "qwen3base-8gpu-b256-80ep"
    )
    config.experiment.ablation_study = STUDY_NAME
    config.experiment.ablation_id = variant_id
    config.experiment.ablation_phase = "screen"
    config.training.seed = seed
    config.training.dataloader_shuffle_seed = seed
    config.model.image_backbone_variant = "E2-Q1"
    config.model.image_flow_position_variant = variant_id
    config.model.image_flow_query_position_mode = spec.query_position_mode
    config.model.image_flow_context_position_mode = spec.context_position_mode
    config.model.image_flow_rope_mode = spec.rope_mode
    config.model.image_flow_rope_axis_dims = [80, 80]
    config.model.image_flow_rope_rotate_value = False
    config.evaluation.checkpoint = f"output/{slug}/hf_model-final-ema"
    if screen_summary is not None:
        selector, screen_sha = _load_screen_selector(screen_summary)
        if seed not in CONFIRMATION_SEEDS:
            raise ValueError(
                f"confirmation config seed must be one of {CONFIRMATION_SEEDS}"
            )
        if variant_id not in selector["selected_ids"]:
            raise ValueError(f"{variant_id} was not selected for confirmation")
        config.experiment.ablation_phase = "confirmation"
        config.experiment.screen_summary_path = str(Path(screen_summary).resolve())
        config.experiment.screen_summary_sha256 = screen_sha
        config.experiment.selector_manifest = selector
        config.experiment.selector_manifest_sha256 = canonical_sha256(selector)
    elif seed != SCREEN_SEED:
        raise ValueError("non-seed-42 FH configs require --screen-summary")

    resolve_flow_head_position_config(config)
    config.experiment.runtime_source_manifest_sha256 = runtime_source_evidence()[
        "manifest_sha256"
    ]
    config.experiment.pop("config_fingerprint", None)
    config.experiment.config_fingerprint = config_fingerprint(config)
    validate_ablation_config(config, variant_id)
    return config


def output_run_dir(config: DictConfig) -> Path:
    base = Path(str(config.experiment.output_dir))
    project = str(config.experiment.project)
    return base if base.name == project else base / project


def provenance_path(config: DictConfig) -> Path:
    return output_run_dir(config) / PROVENANCE_FILENAME


def build_training_provenance(
    *,
    config: DictConfig,
    model,
    train_loader,
    special_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    validate_ablation_config(config)
    source = runtime_source_evidence()
    initial = initial_state_evidence(model, special_token_ids)
    initial["contract"] = INITIALIZATION_CONTRACT
    train_data = train_data_evidence(train_loader, config)
    train_data["contract"] = TRAIN_ORDER_CONTRACT
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "ablation_id": str(config.experiment.ablation_id),
        "phase": str(config.experiment.ablation_phase),
        "training_seed": int(config.training.seed),
        "evaluation_seed": int(config.evaluation.seed),
        "config_fingerprint": str(config.experiment.config_fingerprint),
        "architecture": architecture_contract(config),
        "initial_state": initial,
        "train_data": train_data,
        "base_model": base_model_evidence(config),
        "runtime_source": source,
    }
    if str(config.experiment.ablation_phase) == "confirmation":
        payload["screen_summary_sha256"] = str(
            config.experiment.screen_summary_sha256
        )
        payload["selector_manifest_sha256"] = str(
            config.experiment.selector_manifest_sha256
        )
    payload["provenance_sha256"] = canonical_sha256(payload)
    return payload


def write_training_provenance(
    path: str | Path,
    provenance: Mapping[str, Any],
) -> str:
    path = Path(path)
    payload = dict(provenance)
    stored = payload.pop("provenance_sha256", None)
    _require_sha256(stored, "FH training provenance digest")
    if canonical_sha256(payload) != stored:
        raise ValueError("refusing to write invalid FH training provenance")
    payload["provenance_sha256"] = stored
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return str(stored)


def load_and_validate_training_provenance(
    path: str | Path,
    *,
    config: DictConfig | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid FH training provenance: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("FH training provenance must be an object")
    payload = dict(payload)
    stored = payload.pop("provenance_sha256", None)
    _require_sha256(stored, "FH training provenance digest")
    if expected_sha256 is not None and stored != expected_sha256:
        raise ValueError("FH training provenance binding digest mismatch")
    if canonical_sha256(payload) != stored:
        raise ValueError("FH training provenance content digest mismatch")
    payload["provenance_sha256"] = stored
    if payload.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("FH training provenance schema mismatch")
    if config is not None:
        validate_ablation_config(config)
        expected = {
            "ablation_id": str(config.experiment.ablation_id),
            "phase": str(config.experiment.ablation_phase),
            "training_seed": int(config.training.seed),
            "evaluation_seed": int(config.evaluation.seed),
            "config_fingerprint": str(config.experiment.config_fingerprint),
            "architecture": architecture_contract(config),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"FH training provenance {key} drifted")
    return payload


def compact_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    initial = payload["initial_state"]["image_modules"]
    return {
        "schema": payload["schema"],
        "ablation_id": payload["ablation_id"],
        "phase": payload["phase"],
        "training_seed": int(payload["training_seed"]),
        "evaluation_seed": int(payload["evaluation_seed"]),
        "config_fingerprint": payload["config_fingerprint"],
        "architecture": payload["architecture"],
        "provenance_sha256": payload["provenance_sha256"],
        "runtime_source_manifest_sha256": payload["runtime_source"][
            "manifest_sha256"
        ],
        "initial_parameter_count": int(initial["parameter_count"]),
        "initial_parameter_schema_sha256": initial["parameter_schema_sha256"],
        "initial_parameter_state_sha256": initial["state_sha256"],
        "train_order_sha256": payload["train_data"][
            "epoch0_ordered_sample_identity_sha256"
        ],
        "augmentation_sha256": payload["train_data"][
            "epoch0_augmentation_decisions_sha256"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--id", dest="variant_id")
    parser.add_argument("--seed", type=int, default=SCREEN_SEED)
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--screen-summary")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list:
        print(
            json.dumps(
                {
                    key: value.as_contract((80, 80))
                    for key, value in FLOW_HEAD_POSITION_SPECS.items()
                },
                indent=2,
            )
        )
        return
    if not args.variant_id or not args.output:
        raise SystemExit("--id and --output are required unless --list is used")
    config = build_ablation_config(
        args.variant_id,
        args.seed,
        base_config=args.base_config,
        screen_summary=args.screen_summary,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
