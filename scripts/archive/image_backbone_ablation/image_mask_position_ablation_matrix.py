#!/usr/bin/env python3
# Completed Q-factor matrix retained for evidence audit only.
"""Build and strictly validate the independent image-mask-position Q study."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.archive.image_backbone_ablation.image_latent_layout_legacy import (  # noqa: E402
    resolve_image_layout_config,
)
from scripts.archive.image_backbone_ablation.image_embedder_ablation_matrix import (  # noqa: E402
    FLOW_HEAD_INVARIANTS,
    TRAINING_PROTOCOL_INVARIANTS,
)
from scripts.archive.image_backbone_ablation.image_mask_position_ablation_protocol import (  # noqa: E402
    Q_FACTOR_IDS,
    Q_FACTOR_PHASE,
    Q_FACTOR_SEEDS,
    Q_FACTOR_VARIANTS,
    build_q_factor_declaration,
    normalize_q_factor_id,
    q_factor_config_contract,
    q_factor_run_slug,
    validate_q_factor_declaration,
)


DEFAULT_BASE_CONFIG = (
    REPO_ROOT / "configs/ablation/imagenet_flow_image_embedder_100c_80ep.yaml"
)
DEFAULT_PARENT_SUMMARY = (
    REPO_ROOT / "output/image_embedder_ablation/confirmation_d1_summary.json"
)
Q_FACTOR_SEQUENCE_LENGTH = 320


def _require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"Q-factor requires {label}={expected!r}; got {actual!r}")


def validate_q_factor_config(
    config: DictConfig,
    expected_id: str | None = None,
    expected_seed: int | None = None,
) -> None:
    """Reject any resolved-config drift from the preregistered 4x3 study."""

    configured_id = str(config.experiment.ablation_id)
    variant_id = normalize_q_factor_id(expected_id or configured_id)
    _require_equal("experiment.ablation_id", configured_id, variant_id)
    variant = Q_FACTOR_VARIANTS[variant_id]

    seed = int(config.training.seed)
    if seed not in Q_FACTOR_SEEDS:
        raise ValueError(
            f"Q-factor training seed must be one of {sorted(Q_FACTOR_SEEDS)}, got {seed}"
        )
    if expected_seed is not None:
        _require_equal("training.seed", seed, int(expected_seed))

    slug = q_factor_run_slug(variant_id, seed)
    _require_equal(
        "experiment.ablation_phase",
        str(config.experiment.get("ablation_phase", "")),
        Q_FACTOR_PHASE,
    )
    _require_equal(
        "experiment.q_factor_formal",
        bool(config.experiment.get("q_factor_formal", False)),
        True,
    )
    _require_equal(
        "experiment.parent_ablation_id",
        str(config.experiment.get("parent_ablation_id", "")),
        variant.parent_ablation_id,
    )
    _require_equal("experiment.project", str(config.experiment.project), slug)
    if config.experiment.get("confirmation_protocol", None) is not None:
        raise ValueError("Q-factor configs must not carry the historical confirmation protocol")

    _require_equal(
        "training.dataloader_shuffle_seed",
        int(config.training.get("dataloader_shuffle_seed", -1)),
        seed,
    )
    _require_equal("evaluation.seed", int(config.evaluation.seed), 42)
    _require_equal(
        "evaluation.checkpoint",
        str(config.evaluation.checkpoint),
        f"output/{slug}/hf_model-final-ema",
    )

    architecture = {
        "image_query_stage_mode": str(config.model.image_query_stage_mode),
        "image_observed_position_mode": str(
            config.model.image_observed_position_mode
        ),
        "image_mask_position_mode": str(config.model.image_mask_position_mode),
        "image_rope_mode": str(config.model.image_rope_mode),
        "image_space_to_depth_factor": int(
            config.model.image_space_to_depth_factor
        ),
    }
    expected_architecture = {
        "image_query_stage_mode": "none",
        "image_observed_position_mode": variant.observed_position_mode,
        "image_mask_position_mode": variant.mask_position_mode,
        "image_rope_mode": "row_col_2d",
        "image_space_to_depth_factor": 1,
    }
    if architecture != expected_architecture:
        raise ValueError(
            f"{variant_id} Q-factor architecture mismatch: "
            f"expected {expected_architecture}, got {architecture}"
        )

    for key, expected in FLOW_HEAD_INVARIANTS.items():
        _require_equal(f"model.{key}", config.model.get(key), expected)
    _require_equal(
        "model.image_generation_head_type",
        str(config.model.image_generation_head_type),
        "flow",
    )
    for key, expected in TRAINING_PROTOCOL_INVARIANTS.items():
        _require_equal(key, OmegaConf.select(config, key), expected)

    layout = resolve_image_layout_config(config)
    _require_equal("model.image_space_to_depth_factor", layout["factor"], 1)
    _require_equal("model.image_tokens_per_img", layout["image_tokens_per_img"], 256)
    _require_equal("model.image_latent_dim", layout["image_latent_dim"], 16)
    _require_equal(
        "dataset.params.image_space_to_depth_factor",
        int(config.dataset.params.image_space_to_depth_factor),
        1,
    )
    lengths = (
        int(config.dataset.params.max_seq_length),
        int(config.dataset.params.pad_to_length),
        int(config.dataset.preprocessing.max_seq_length),
    )
    _require_equal(
        "all sequence lengths",
        lengths,
        (Q_FACTOR_SEQUENCE_LENGTH,) * 3,
    )
    _require_equal("dataset.params.split_seed", int(config.dataset.params.split_seed), 42)

    declaration = config.experiment.get("q_factor_protocol", None)
    if declaration is None:
        raise ValueError("Q-factor config is missing experiment.q_factor_protocol")
    contract = q_factor_config_contract(config)
    validate_q_factor_declaration(
        OmegaConf.to_container(declaration, resolve=True),
        variant_id=variant_id,
        seed=seed,
        config_contract=contract,
    )


def build_q_factor_config(
    variant_id: str,
    seed: int,
    *,
    base_config: str | Path = DEFAULT_BASE_CONFIG,
    parent_summary_json: str | Path = DEFAULT_PARENT_SUMMARY,
) -> DictConfig:
    """Resolve a Q run, bind its config contract, then create its declaration."""

    variant_id = normalize_q_factor_id(variant_id)
    seed = int(seed)
    if seed not in Q_FACTOR_SEEDS:
        raise ValueError(
            f"Q-factor training seed must be one of {sorted(Q_FACTOR_SEEDS)}, got {seed}"
        )
    variant = Q_FACTOR_VARIANTS[variant_id]
    slug = q_factor_run_slug(variant_id, seed)
    config = OmegaConf.load(Path(base_config))

    config.experiment.project = slug
    config.experiment.name = (
        f"imagenet100-image-mask-position-qf-{variant_id.lower()}-seed{seed}-"
        "qwen3base-8gpu-b256-80ep"
    )
    config.experiment.ablation_id = variant_id
    config.experiment.parent_ablation_id = variant.parent_ablation_id
    config.experiment.ablation_phase = Q_FACTOR_PHASE
    config.experiment.q_factor_formal = True
    config.training.seed = seed
    config.training.dataloader_shuffle_seed = seed

    config.model.image_query_stage_mode = "none"
    config.model.image_observed_position_mode = variant.observed_position_mode
    config.model.image_mask_position_mode = variant.mask_position_mode
    config.model.image_rope_mode = "row_col_2d"
    config.model.image_space_to_depth_factor = 1

    config.dataset.params.max_seq_length = Q_FACTOR_SEQUENCE_LENGTH
    config.dataset.params.pad_to_length = Q_FACTOR_SEQUENCE_LENGTH
    config.dataset.preprocessing.max_seq_length = Q_FACTOR_SEQUENCE_LENGTH
    config.evaluation.seed = 42
    config.evaluation.checkpoint = f"output/{slug}/hf_model-final-ema"

    resolve_image_layout_config(config)

    # This ordering is intentional: the declaration binds the fully resolved
    # config, while q_factor_protocol itself is excluded from the contract.
    contract = q_factor_config_contract(config)
    declaration = build_q_factor_declaration(
        variant_id=variant_id,
        seed=seed,
        config_contract=contract,
        parent_summary_path=parent_summary_json,
    )
    config.experiment.q_factor_protocol = declaration

    validate_q_factor_config(config, variant_id, seed)
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print the 4-run matrix as JSON.")
    parser.add_argument("--id", dest="variant_id", help="Q-factor ID, e.g. E2b-Q0.")
    parser.add_argument("--seed", type=int, help="Training seed: 43, 44, or 45.")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument(
        "--parent-summary-json",
        default=str(DEFAULT_PARENT_SUMMARY),
        help="Strict 6x3 parent confirmation summary.",
    )
    parser.add_argument("--output", help="Resolved YAML output path.")
    parser.add_argument(
        "--validate-config",
        help="Validate an existing resolved YAML instead of building one.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.list:
        print(
            json.dumps(
                {key: asdict(value) for key, value in Q_FACTOR_VARIANTS.items()},
                indent=2,
            )
        )
        return
    if not args.variant_id or args.seed is None:
        raise SystemExit("--id and --seed are required")
    if args.validate_config:
        path = Path(args.validate_config)
        validate_q_factor_config(
            OmegaConf.load(path),
            expected_id=args.variant_id,
            expected_seed=args.seed,
        )
        print(path.resolve())
        return
    if not args.output:
        raise SystemExit("--output is required when building a config")
    config = build_q_factor_config(
        args.variant_id,
        args.seed,
        base_config=args.base_config,
        parent_summary_json=args.parent_summary_json,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
