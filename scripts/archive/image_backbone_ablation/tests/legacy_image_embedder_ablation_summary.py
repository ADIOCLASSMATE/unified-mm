"""Archived tests for the historical screening summary; not part of CI."""
import hashlib
import json
import math
from statistics import stdev

import pytest

from scripts.image_embedder_ablation_matrix import (
    FLOW_HEAD_INVARIANTS,
    STAGE_BUFFER_IMPLEMENTATION_CONTRACT,
    TRAINING_PROTOCOL_INVARIANTS,
    TRAINING_PROTOCOL_SCHEMA,
    VARIANTS,
    run_slug,
    training_protocol_fingerprint,
)
from scripts.summarize_image_embedder_ablation import (
    CONFIRMATION_MANIFEST_SCHEMA,
    CONFIRMATION_SEEDS,
    CONFIRMATION_SCOPE_AMENDMENT_SCHEMA,
    FOUR_FACTOR_SETTINGS,
    SCHEMA,
    SummaryError,
    build_summary,
)
from scripts.image_embedder_confirmation_protocol import (
    AUGMENTATION_CONTRACT,
    CONFIRMATION_PROVENANCE_SCHEMA,
    INITIALIZATION_CONTRACT,
    TRAIN_ORDER_CONTRACT,
    canonical_sha256,
)
from scripts.evaluate_single_stream_fid_is import EVALUATOR_RNG_CONTRACT


def _digest(label):
    return hashlib.sha256(str(label).encode("utf-8")).hexdigest()


def _confirmation_manifest(*candidate_ids):
    selected = set(candidate_ids)
    ordered = [variant_id for variant_id in VARIANTS if variant_id in selected]
    return {
        "schema": CONFIRMATION_MANIFEST_SCHEMA,
        "screen_summary_schema": SCHEMA,
        "screen_training_seed": 42,
        "confirmation_training_seeds": sorted(CONFIRMATION_SEEDS),
        "near_best_fid_margin": 1.0,
        "speed_advantage_ratio_vs_e0": 1.5,
        "near_best_fid_ids": [value for value in ordered if value != "E0"],
        "fid_is_pareto_ids": [],
        "speed_pareto_ids_meeting_threshold": [],
        "candidate_ids": ordered,
    }


def _confirmation_metadata(variant_id, seed, candidate_ids=("E0", "E3")):
    factor = VARIANTS[variant_id].space_to_depth_factor
    slug = run_slug(variant_id, seed)
    declaration_sha256 = _digest(f"declaration-{variant_id}-{seed}")
    provenance_sha256 = _digest(f"provenance-{variant_id}-{seed}")
    manifest_sha256 = canonical_sha256(_confirmation_manifest(*candidate_ids))
    provenance_path = f"output/{slug}/confirmation_training_provenance.json"
    input_files = {
        label: {
            "path": f"public/datasets/{label}",
            "size_bytes": 1000 + index,
            "sha256": _digest(f"input-{label}"),
        }
        for index, label in enumerate(
            ("cache", "manifest", "split_manifest", "synset_mapping")
        )
    }
    return {
        "declaration_sha256": declaration_sha256,
        "screen_summary_sha256": _digest("expanded-screen-summary"),
        "candidate_manifest_sha256": manifest_sha256,
        "evaluator_rng_contract_sha256": canonical_sha256(EVALUATOR_RNG_CONTRACT),
        "dataloader_shuffle_seed": seed,
        "provenance_path": provenance_path,
        "provenance_sha256": provenance_sha256,
        "provenance": {
            "schema": CONFIRMATION_PROVENANCE_SCHEMA,
            "ablation_id": variant_id,
            "training_seed": seed,
            "provenance_sha256": provenance_sha256,
            "confirmation_declaration_sha256": declaration_sha256,
            "space_to_depth_factor": factor,
            "initial_state": {
                "contract": INITIALIZATION_CONTRACT,
                "image_modules": {
                    "parameter_count": 100 + factor,
                    "parameter_schema_sha256": _digest(f"image-schema-factor-{factor}"),
                    "state_sha256": _digest(f"image-state-seed-{seed}-factor-{factor}"),
                },
                "special_token_names_and_ids": [["image", 10], ["mask", 11]],
                "special_token_rows_sha256": _digest(f"special-token-rows-{seed}"),
            },
            "train_data": {
                "contract": TRAIN_ORDER_CONTRACT,
                "dataloader_shuffle_seed": seed,
                "initial_generator_state_sha256": _digest(f"generator-state-{seed}"),
                "dataloader_base_seed": 100_000 + seed,
                "dataset_length": 100_000,
                "epoch0_ordered_sample_identity_sha256": _digest(
                    f"epoch0-order-{seed}"
                ),
                "augmentation_contract": AUGMENTATION_CONTRACT,
                "epoch0_augmentation_decisions_sha256": _digest(
                    f"epoch0-augmentations-{seed}"
                ),
                "augmentation_seed": seed,
                "latent_hflip_probability": 0.5,
                "batch_size_per_rank": 32,
                "total_batch_size": 256,
                "drop_last": True,
                "num_workers": 16,
                "persistent_workers": True,
                "input_files": input_files,
            },
            "base_model_manifest_sha256": _digest("base-model"),
            "runtime_source_manifest_sha256": _digest("runtime-source"),
        },
    }


def _metrics(
    variant_id,
    seed,
    *,
    fid,
    is_mean,
    wall=1000.0,
    confirmation_candidate_ids=("E0", "E3"),
):
    variant = VARIANTS[variant_id]
    factor = variant.space_to_depth_factor
    slug = run_slug(variant_id, seed)
    evaluator_rng_contract = EVALUATOR_RNG_CONTRACT
    metrics = {
        "official_protocol": True,
        "implementation_contracts": {
            "image_stage_buffer": STAGE_BUFFER_IMPLEMENTATION_CONTRACT,
            "evaluator_rng_contract": evaluator_rng_contract,
            "evaluator_rng_contract_sha256": canonical_sha256(
                evaluator_rng_contract
            ),
            "canonical_initial_noise_enabled": seed in CONFIRMATION_SEEDS,
            **(
                {
                    "canonical_noise_manifest_schema": (
                        "canonical_image_flow_noise_manifest_v1"
                    ),
                    "canonical_noise_manifest_sha256": _digest(
                        "canonical-noise-manifest"
                    ),
                    "ordered_eval_sample_manifest_schema": (
                        "ordered_image_embedder_eval_samples_v1"
                    ),
                    "ordered_eval_sample_manifest_sha256": _digest(
                        "ordered-eval-sample-manifest"
                    ),
                    "paired_sample_count": 10_000,
                }
                if seed in CONFIRMATION_SEEDS
                else {}
            ),
        },
        "metric_protocol": {"fid_reducer": "symmetric_eigendecomposition"},
        "config": f"output/{slug}/config.yaml",
        "model_path": f"output/{slug}/hf_model-final-ema",
        "training_protocol": {
            "schema": TRAINING_PROTOCOL_SCHEMA,
            "training_seed": seed,
            "invariants": dict(TRAINING_PROTOCOL_INVARIANTS),
            "invariants_sha256": training_protocol_fingerprint(
                TRAINING_PROTOCOL_INVARIANTS
            ),
            "final_global_step": 35_920,
            "artifacts": {
                "checkpoint_metadata_path": (
                    f"output/{slug}/checkpoint-35920/metadata.json"
                ),
                "ema_state_path": f"output/{slug}/checkpoint-35920/ema_state.pt",
                "ema_state_size_bytes": 3_000_000_000,
                "hf_model_weights_path": (
                    f"output/{slug}/hf_model-final-ema/model.safetensors"
                ),
                "hf_model_weights_size_bytes": 3_000_000_000,
            },
        },
        "architecture": {
            "ablation_id": variant_id,
            "image_query_stage_mode": variant.query_stage_mode,
            "image_observed_position_mode": variant.observed_position_mode,
            "image_rope_mode": variant.rope_mode,
            "image_space_to_depth_factor": factor,
            "image_canonical_grid_side": 16,
            "image_canonical_latent_dim": 16,
            "image_grid_side": 16 // factor,
            "image_tokens_per_img": 256 // factor**2,
            "image_latent_dim": 16 * factor**2,
            "padded_sequence_length": 320 if factor == 1 else 128,
            "flow_head": {
                "arch": FLOW_HEAD_INVARIANTS["image_flow_head_arch"],
                "depth": FLOW_HEAD_INVARIANTS["image_flow_depth"],
                "width": FLOW_HEAD_INVARIANTS["image_flow_width"],
                "mlp_ratio": FLOW_HEAD_INVARIANTS["image_flow_mlp_ratio"],
                "latent_mixer_heads": FLOW_HEAD_INVARIANTS[
                    "image_flow_latent_mixer_heads"
                ],
                "latent_mixer_dropout": FLOW_HEAD_INVARIANTS[
                    "image_flow_latent_mixer_dropout"
                ],
                "zero_init_gate": FLOW_HEAD_INVARIANTS[
                    "image_flow_latent_mixer_zero_init_gate"
                ],
            },
        },
        "parameters": {
            "total": 1_000_000,
            "trainable": 1_000_000,
            "image_embedder": 10_000,
            "flow_head": 500_000,
        },
        "precision_protocol": {
            "schema": "flow_eval_precision_v1",
            "model_dtype": "bf16",
            "model_parameter_dtypes": ["torch.bfloat16"],
            "checkpoint_weight_dtypes": ["fp32"],
            "vae_dtype": "fp32",
            "flow_integrator_dtype": "fp32",
        },
        "adapter": {"adapter": None},
        "model_state": {"model_state": None},
        "ema_state": {"ema_state": None},
        "split": "val",
        "seed": 42,
        "batch_size": 512,
        "samples_requested": 10_000,
        "samples_evaluated": 10_000,
        "distributed": {
            "enabled": True,
            "world_size": 8,
            "rank": 0,
            "local_rank": 0,
            "peak_cuda_allocated_mib": 12_000.0,
            "peak_cuda_reserved_mib": 20_000.0,
        },
        "real_source": "cached_original_imagenet",
        "real_stats_path": "/data/stats.pt",
        "real_stats_metadata": {"manifest_sha256": "a" * 64},
        "inception_weights_path": "/data/inception.pth",
        "cfg": 3.5,
        "cfg_schedule": "constant",
        "sampling_steps": "100",
        "temperature": 1.0,
        "flow_solver": "heun",
        "parallel_rate": 1,
        "strategies": {
            "spatial_halton": {
                "count": 10_000,
                "fid": fid,
                "inception_score_mean": is_mean,
                "inception_score_std": 1.0,
                "latent_mse_to_target": 2.0,
                "latent_rms": 1.0,
                "generation_step_max": 256.0 / factor**2,
                "generation_wall_seconds": wall,
                "generation_samples_per_second": 10_000 / wall,
            }
        },
    }
    if seed in CONFIRMATION_SEEDS:
        confirmation = _confirmation_metadata(
            variant_id, seed, confirmation_candidate_ids
        )
        metrics["training_protocol"]["confirmation"] = confirmation
        metrics["training_protocol"]["artifacts"].update(
            {
                "confirmation_provenance_path": confirmation["provenance_path"],
                "confirmation_hf_provenance_path": (
                    f"output/{slug}/hf_model-final-ema/"
                    "confirmation_training_provenance.json"
                ),
            }
        )
    return metrics


def _write(tmp_path, variant_id, seed, *, fid, is_mean, wall=1000.0):
    path = tmp_path / f"{variant_id}-{seed}.json"
    path.write_text(
        json.dumps(_metrics(variant_id, seed, fid=fid, is_mean=is_mean, wall=wall)),
        encoding="utf-8",
    )
    return f"{variant_id}@{seed}={path}"


def _confirmation_specs(
    tmp_path,
    *,
    candidate_ids=("E0", "E3"),
    provenance_candidate_ids=None,
    mutate=None,
):
    provenance_candidate_ids = provenance_candidate_ids or candidate_ids
    specs = []
    for seed in sorted(CONFIRMATION_SEEDS):
        for index, variant_id in enumerate(candidate_ids):
            payload = _metrics(
                variant_id,
                seed,
                fid=30.0 - index,
                is_mean=50.0 + index,
                confirmation_candidate_ids=provenance_candidate_ids,
            )
            if mutate is not None:
                mutate(variant_id, seed, payload)
            path = tmp_path / f"confirmation-{variant_id}-{seed}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            specs.append(f"{variant_id}@{seed}={path}")
    return specs


def _scope_amendment(original_ids, effective_ids, **updates):
    original_ids = list(original_ids)
    effective_ids = list(effective_ids)
    payload = {
        "schema": CONFIRMATION_SCOPE_AMENDMENT_SCHEMA,
        "created_at_utc": "2026-07-20T14:35:18Z",
        "reason": "S2D is invalid for this confirmation study.",
        "parent_screen_summary_path": "output/screen.json",
        "parent_screen_summary_sha256": _digest("expanded-screen-summary"),
        "parent_candidate_manifest_sha256": canonical_sha256(
            _confirmation_manifest(*original_ids)
        ),
        "original_candidate_ids": original_ids,
        "confirmation_candidate_ids": effective_ids,
        "removed_candidate_ids": [
            value for value in original_ids if value not in effective_ids
        ],
        "confirmation_training_seeds": sorted(CONFIRMATION_SEEDS),
        "required_space_to_depth_factor": 1,
        "confirmation_metrics_observed_before_amendment": False,
    }
    payload.update(updates)
    return payload


def _set_nested(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_first_stage_summary_ranks_and_computes_atomic_effects(tmp_path):
    specs = [
        _write(tmp_path, variant_id, 42, fid=30.0 - index, is_mean=50.0 + index)
        for index, variant_id in enumerate(("E0", "E1", "E2a", "E2b", "E2", "E3"))
    ]
    summary = build_summary(specs, "first-stage")

    assert summary["best_by_fid"]["id"] == "E3"
    assert summary["ranking_by_is"][0] == "E3"
    assert summary["effects"]["stage_E1_minus_E0"]["fid_delta"] == -1.0
    assert summary["effects"]["R_subfactor_interaction"]["fid_mean"] == 1.0


def test_full_summary_reports_averaged_factorial_effects(tmp_path):
    settings = {
        "E0": (0, 0, 0),
        "E1": (1, 0, 0),
        "E2a": (0, 0, 0),
        "E2b": (0, 0, 0),
        "E2": (0, 1, 0),
        "E3": (0, 0, 1),
        "E4": (1, 1, 0),
        "E5": (1, 0, 1),
        "E6": (0, 1, 1),
        "E7": (1, 1, 1),
    }
    specs = []
    for variant_id, (stage, relation, s2d) in settings.items():
        fid = 30.0 - stage - 2.0 * relation - 3.0 * s2d
        specs.append(_write(tmp_path, variant_id, 42, fid=fid, is_mean=50.0))
    summary = build_summary(specs, "full")
    effects = summary["effects"]["factorial_2x2x2"]["fid_mean"]
    assert effects["average_main_effects"] == {"S": -1.0, "R": -2.0, "D": -3.0}
    assert effects["average_two_way_interactions"] == {
        "SxR": 0.0,
        "SxD": 0.0,
        "RxD": 0.0,
    }
    assert effects["three_way_interaction_SxRxD"] == 0.0


def test_expanded_summary_reports_four_factor_effects(tmp_path):
    settings = {
        "E0": (0, 0, 0, 0),
        "E1": (1, 0, 0, 0),
        "E2a": (0, 1, 0, 0),
        "E2b": (0, 0, 1, 0),
        "E2": (0, 1, 1, 0),
        "E3": (0, 0, 0, 1),
        "E4a": (1, 1, 0, 0),
        "E4b": (1, 0, 1, 0),
        "E4": (1, 1, 1, 0),
        "E5": (1, 0, 0, 1),
        "E6a": (0, 1, 0, 1),
        "E6b": (0, 0, 1, 1),
        "E6": (0, 1, 1, 1),
        "E7a": (1, 1, 0, 1),
        "E7b": (1, 0, 1, 1),
        "E7": (1, 1, 1, 1),
    }
    specs = []
    for variant_id, (stage, remove_additive, row_col, s2d) in settings.items():
        fid = (
            30.0
            - stage
            - 2.0 * remove_additive
            - 3.0 * row_col
            - 4.0 * s2d
            - 5.0 * remove_additive * row_col
        )
        specs.append(_write(tmp_path, variant_id, 42, fid=fid, is_mean=50.0))

    summary = build_summary(specs, "expanded")
    effects = summary["effects"]["factorial_2x2x2x2"]["fid_mean"]
    assert effects["average_main_effects"] == {
        "S": -1.0,
        "Ra": -4.5,
        "Rb": -5.5,
        "D": -4.0,
    }
    assert effects["average_two_way_interactions"]["RaxRb"] == -5.0
    assert effects["average_two_way_interactions"]["SxD"] == 0.0
    assert effects["four_way_interaction_SxRaxRbxD"] == 0.0
    assert summary["confirmation_candidate_manifest"]["candidate_ids"] == [
        "E0",
        "E6",
        "E7",
    ]


def test_expanded_summary_preserves_nonzero_high_order_contrast_signs(tmp_path):
    specs = []
    for variant_id, (stage, remove_additive, row_col, s2d) in FOUR_FACTOR_SETTINGS.items():
        fid = (
            30.0
            - 6.0 * stage * remove_additive * s2d
            - 7.0 * stage * remove_additive * row_col * s2d
        )
        specs.append(_write(tmp_path, variant_id, 42, fid=fid, is_mean=50.0))

    effects = build_summary(specs, "expanded")["effects"]["factorial_2x2x2x2"]["fid_mean"]
    assert effects["average_three_way_interactions"]["SxRaxD"] == pytest.approx(-9.5)
    assert effects["four_way_interaction_SxRaxRbxD"] == pytest.approx(-7.0)


def test_confirmation_manifest_keeps_significant_speed_pareto_candidate(tmp_path):
    specs = []
    for index, variant_id in enumerate(FOUR_FACTOR_SETTINGS):
        if variant_id == "E0":
            specs.append(_write(tmp_path, variant_id, 42, fid=30.0, is_mean=50.0))
        elif variant_id == "E3":
            specs.append(
                _write(
                    tmp_path,
                    variant_id,
                    42,
                    fid=60.0,
                    is_mean=40.0,
                    wall=50.0,
                )
            )
        else:
            specs.append(
                _write(
                    tmp_path,
                    variant_id,
                    42,
                    fid=40.0 + index,
                    is_mean=40.0,
                )
            )

    summary = build_summary(specs, "expanded")
    manifest = summary["confirmation_candidate_manifest"]
    assert manifest["speed_pareto_ids_meeting_threshold"] == ["E3"]
    assert manifest["candidate_ids"] == ["E0", "E3"]


def test_summary_aggregates_multiple_training_seeds(tmp_path):
    specs = [
        _write(tmp_path, "E0", 42, fid=30.0, is_mean=50.0),
        _write(tmp_path, "E0", 43, fid=28.0, is_mean=54.0),
    ]
    summary = build_summary(specs)
    aggregate = summary["aggregates"][0]
    assert aggregate["fid_mean"] == 29.0
    assert aggregate["fid_sample_std"] == pytest.approx(2**0.5)
    assert aggregate["is_mean"] == 52.0


def test_first_stage_rejects_mixed_or_non_screen_training_seeds(tmp_path):
    ids = ("E0", "E1", "E2a", "E2b", "E2", "E3")
    specs = [
        _write(
            tmp_path,
            variant_id,
            43 if variant_id == "E3" else 42,
            fid=30.0,
            is_mean=50.0,
        )
        for variant_id in ids
    ]
    with pytest.raises(SummaryError, match="seed-42 architecture screen"):
        build_summary(specs, "first-stage")


def test_confirmation_requires_independent_paired_seeds_and_reports_ci(tmp_path):
    specs = []
    for seed, baseline_fid, candidate_fid in (
        (43, 30.0, 29.0),
        (44, 32.0, 30.0),
        (45, 31.0, 30.5),
    ):
        specs.extend(
            [
                _write(tmp_path, "E0", seed, fid=baseline_fid, is_mean=50.0),
                _write(tmp_path, "E3", seed, fid=candidate_fid, is_mean=52.0),
            ]
        )
    summary = build_summary(
        specs,
        "confirmation",
        confirmation_manifest=_confirmation_manifest("E0", "E3"),
    )
    paired = summary["paired_vs_e0"]["E3"]["fid_candidate_minus_e0"]
    assert paired["values"] == [-1.0, -2.0, -0.5]
    assert paired["mean"] == pytest.approx(-7.0 / 6.0)
    assert paired["candidate_wins"] == 3
    expected_half_width = 4.303 * stdev([-1.0, -2.0, -0.5]) / math.sqrt(3)
    assert paired["ci95"] == pytest.approx(
        [paired["mean"] - expected_half_width, paired["mean"] + expected_half_width]
    )
    assert summary["confirmation_pairing_gate"]["validated_runs"] == 6


def test_confirmation_scope_amendment_keeps_original_provenance_manifest(tmp_path):
    original_ids = ("E0", "E1", "E3")
    effective_ids = ("E0", "E1")
    specs = _confirmation_specs(
        tmp_path,
        candidate_ids=effective_ids,
        provenance_candidate_ids=original_ids,
    )
    manifest = _confirmation_manifest(*original_ids)
    amendment = _scope_amendment(original_ids, effective_ids)

    summary = build_summary(
        specs,
        "confirmation",
        confirmation_manifest=manifest,
        confirmation_scope_amendment=amendment,
    )

    scope = summary["confirmation_scope_manifest"]
    assert scope["candidate_ids"] == list(effective_ids)
    assert scope["removed_candidate_ids"] == ["E3"]
    assert scope["source_candidate_manifest_sha256"] == canonical_sha256(manifest)
    assert scope["validated_parent_screen_summary_sha256"] == _digest(
        "expanded-screen-summary"
    )
    assert summary["confirmation_pairing_gate"]["candidate_manifest_sha256"] == (
        canonical_sha256(manifest)
    )
    assert summary["confirmation_pairing_gate"]["validated_runs"] == 6


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"parent_candidate_manifest_sha256": "0" * 64},
            "frozen candidate manifest",
        ),
        (
            {"confirmation_metrics_observed_before_amendment": True},
            "no confirmation metrics were observed",
        ),
        (
            {
                "confirmation_candidate_ids": ["E0", "E1", "E3"],
                "removed_candidate_ids": [],
            },
            "exactly the factor-1 candidates",
        ),
        (
            {"parent_screen_summary_sha256": "0" * 64},
            "parent screen summary",
        ),
    ],
)
def test_confirmation_scope_amendment_rejects_invalid_narrowing(
    tmp_path, updates, message
):
    original_ids = ("E0", "E1", "E3")
    effective_ids = ("E0", "E1")
    specs = _confirmation_specs(
        tmp_path,
        candidate_ids=effective_ids,
        provenance_candidate_ids=original_ids,
    )
    amendment = _scope_amendment(original_ids, effective_ids, **updates)

    with pytest.raises(SummaryError, match=message):
        build_summary(
            specs,
            "confirmation",
            confirmation_manifest=_confirmation_manifest(*original_ids),
            confirmation_scope_amendment=amendment,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("training_protocol", "confirmation", "candidate_manifest_sha256"),
            "0" * 64,
            "candidate_manifest_sha256",
        ),
        (
            (
                "training_protocol",
                "confirmation",
                "evaluator_rng_contract_sha256",
            ),
            "0" * 64,
            "evaluator_rng_contract_sha256",
        ),
        (
            ("training_protocol", "confirmation", "dataloader_shuffle_seed"),
            999,
            "dataloader_shuffle_seed",
        ),
        (
            ("training_protocol", "confirmation", "provenance_path"),
            "output/wrong-run/confirmation_training_provenance.json",
            "provenance_path",
        ),
        (
            (
                "training_protocol",
                "confirmation",
                "provenance",
                "ablation_id",
            ),
            "E0",
            "ablation_id",
        ),
        (
            (
                "training_protocol",
                "confirmation",
                "provenance",
                "training_seed",
            ),
            45,
            "training_seed",
        ),
        (
            (
                "training_protocol",
                "confirmation",
                "provenance",
                "space_to_depth_factor",
            ),
            99,
            "space_to_depth_factor",
        ),
        (
            ("implementation_contracts", "canonical_initial_noise_enabled"),
            False,
            "canonical_initial_noise_enabled",
        ),
        (
            ("implementation_contracts", "evaluator_rng_contract_sha256"),
            "0" * 64,
            "evaluator RNG contract digest mismatch",
        ),
    ],
)
def test_confirmation_rejects_invalid_per_run_provenance(
    tmp_path, path, value, message
):
    def mutate(variant_id, seed, payload):
        if variant_id == "E3" and seed == 44:
            _set_nested(payload, path, value)

    specs = _confirmation_specs(tmp_path, mutate=mutate)
    with pytest.raises(SummaryError, match=message):
        build_summary(
            specs,
            "confirmation",
            confirmation_manifest=_confirmation_manifest("E0", "E3"),
        )


def test_confirmation_requires_embedded_checkpoint_bound_provenance(tmp_path):
    def mutate(variant_id, seed, payload):
        if variant_id == "E3" and seed == 44:
            payload["training_protocol"]["confirmation"].pop("provenance")

    specs = _confirmation_specs(tmp_path, mutate=mutate)
    with pytest.raises(SummaryError, match="confirmation.provenance"):
        build_summary(
            specs,
            "confirmation",
            confirmation_manifest=_confirmation_manifest("E0", "E3"),
        )


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (
            (
                "training_protocol",
                "confirmation",
                "provenance",
                "initial_state",
                "special_token_rows_sha256",
            ),
            "special_token_rows_sha256",
        ),
        (
            (
                "training_protocol",
                "confirmation",
                "provenance",
                "train_data",
                "epoch0_ordered_sample_identity_sha256",
            ),
            "epoch0_ordered_sample_identity_sha256",
        ),
        (
            (
                "training_protocol",
                "confirmation",
                "provenance",
                "train_data",
                "epoch0_augmentation_decisions_sha256",
            ),
            "epoch0_augmentation_decisions_sha256",
        ),
        (
            (
                "training_protocol",
                "confirmation",
                "provenance",
                "base_model_manifest_sha256",
            ),
            "base_model_manifest_sha256",
        ),
        (
            (
                "training_protocol",
                "confirmation",
                "provenance",
                "runtime_source_manifest_sha256",
            ),
            "runtime_source_manifest_sha256",
        ),
        (
            ("implementation_contracts", "canonical_noise_manifest_sha256"),
            "canonical_noise_manifest_sha256",
        ),
        (
            (
                "implementation_contracts",
                "ordered_eval_sample_manifest_sha256",
            ),
            "ordered_eval_sample_manifest_sha256",
        ),
        (
            ("training_protocol", "confirmation", "screen_summary_sha256"),
            "screen_summary_sha256",
        ),
    ],
)
def test_confirmation_rejects_paired_evidence_mismatch(tmp_path, path, message):
    def mutate(variant_id, seed, payload):
        if variant_id == "E3" and seed == 44:
            _set_nested(payload, path, _digest(f"mismatch-{message}"))

    specs = _confirmation_specs(tmp_path, mutate=mutate)
    with pytest.raises(SummaryError, match=f"pairing mismatch for {message}"):
        build_summary(
            specs,
            "confirmation",
            confirmation_manifest=_confirmation_manifest("E0", "E3"),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("parameter_schema_sha256", "image_parameter_schema_sha256"),
        ("state_sha256", "image_state_sha256"),
    ],
)
def test_confirmation_rejects_image_init_mismatch_within_same_s2d_layout(
    tmp_path, field, message
):
    candidate_ids = ("E0", "E1", "E3")

    def mutate(variant_id, seed, payload):
        if variant_id == "E1" and seed == 44:
            payload["training_protocol"]["confirmation"]["provenance"][
                "initial_state"
            ]["image_modules"][field] = _digest(f"mismatched-{field}")

    specs = _confirmation_specs(
        tmp_path,
        candidate_ids=candidate_ids,
        mutate=mutate,
    )
    with pytest.raises(SummaryError, match=f"pairing mismatch for {message}"):
        build_summary(
            specs,
            "confirmation",
            confirmation_manifest=_confirmation_manifest(*candidate_ids),
        )


def test_confirmation_gate_is_not_applied_to_nonconfirmation_summary(tmp_path):
    payload = _metrics("E0", 43, fid=30.0, is_mean=50.0)
    payload["training_protocol"].pop("confirmation")
    path = tmp_path / "seed43-without-confirmation-provenance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    summary = build_summary([f"E0@43={path}"], "any")
    assert summary["best_by_fid"]["id"] == "E0"
    assert summary["confirmation_pairing_gate"] is None


def test_confirmation_rejects_unpaired_or_screen_seed(tmp_path):
    unpaired = [
        _write(tmp_path, "E0", seed, fid=30.0, is_mean=50.0)
        for seed in (43, 44, 45)
    ] + [
        _write(tmp_path, "E3", seed, fid=29.0, is_mean=52.0)
        for seed in (43, 44, 46)
    ]
    with pytest.raises(SummaryError, match="identical paired"):
        build_summary(
            unpaired,
            "confirmation",
            confirmation_manifest=_confirmation_manifest("E0", "E3"),
        )

    includes_screen_seed = []
    for seed in (42, 43, 44):
        includes_screen_seed.extend(
            [
                _write(tmp_path, "E0", seed, fid=30.0, is_mean=50.0),
                _write(tmp_path, "E3", seed, fid=29.0, is_mean=52.0),
            ]
        )
    with pytest.raises(SummaryError, match="preregistered paired training seeds"):
        build_summary(
            includes_screen_seed,
            "confirmation",
            confirmation_manifest=_confirmation_manifest("E0", "E3"),
        )

    matched_but_not_preregistered = []
    for seed in (43, 44, 46):
        matched_but_not_preregistered.extend(
            [
                _write(tmp_path, "E0", seed, fid=30.0, is_mean=50.0),
                _write(tmp_path, "E3", seed, fid=29.0, is_mean=52.0),
            ]
        )
    with pytest.raises(SummaryError, match="preregistered paired training seeds"):
        build_summary(
            matched_but_not_preregistered,
            "confirmation",
            confirmation_manifest=_confirmation_manifest("E0", "E3"),
        )


def test_confirmation_requires_and_exactly_matches_screen_candidate_manifest(tmp_path):
    specs = []
    for seed in sorted(CONFIRMATION_SEEDS):
        specs.extend(
            [
                _write(tmp_path, "E0", seed, fid=30.0, is_mean=50.0),
                _write(tmp_path, "E3", seed, fid=29.0, is_mean=52.0),
            ]
        )
    with pytest.raises(SummaryError, match="candidate manifest"):
        build_summary(specs, "confirmation")
    with pytest.raises(SummaryError, match="exactly match"):
        build_summary(
            specs,
            "confirmation",
            confirmation_manifest=_confirmation_manifest("E0", "E2", "E3"),
        )


def test_summary_rejects_architecture_or_protocol_drift(tmp_path):
    payload = _metrics("E0", 42, fid=30.0, is_mean=50.0)
    payload["architecture"]["flow_head"]["width"] = 999
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SummaryError, match="flow-head contract"):
        build_summary([f"E0@42={path}"])


def test_summary_rejects_extra_strategy_that_changes_halton_rng(tmp_path):
    payload = _metrics("E0", 42, fid=30.0, is_mean=50.0)
    payload["strategies"] = {
        "spatial_uniform": payload["strategies"]["spatial_halton"],
        **payload["strategies"],
    }
    path = tmp_path / "extra-strategy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SummaryError, match="exactly one strategy"):
        build_summary([f"E0@42={path}"])


def test_summary_rejects_nonfinite_generated_latent_diagnostics(tmp_path):
    payload = _metrics("E1", 42, fid=268.0, is_mean=6.0)
    payload["strategies"]["spatial_halton"]["latent_rms"] = float("nan")
    path = tmp_path / "nonfinite-latents.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SummaryError, match="latent_rms must be positive and finite"):
        build_summary([f"E1@42={path}"])


def test_summary_requires_stage_buffer_fix_provenance_for_stage_variants(tmp_path):
    payload = _metrics("E1", 42, fid=25.0, is_mean=60.0)
    payload.pop("implementation_contracts")
    path = tmp_path / "missing-stage-buffer-contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SummaryError, match="implementation_contracts"):
        build_summary([f"E1@42={path}"])

    baseline = _metrics("E0", 42, fid=30.0, is_mean=50.0)
    baseline.pop("implementation_contracts")
    baseline_path = tmp_path / "legacy-baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert build_summary([f"E0@42={baseline_path}"])["best_by_fid"]["id"] == "E0"


def test_summary_reports_fid_is_pareto_dominance_and_nominal_tie_break(tmp_path):
    specs = [
        _write(tmp_path, "E0", 42, fid=30.0, is_mean=50.0),
        _write(tmp_path, "E1", 42, fid=29.0, is_mean=49.0),
        _write(tmp_path, "E2a", 42, fid=31.0, is_mean=48.0),
        _write(tmp_path, "E2b", 42, fid=30.0, is_mean=51.0),
    ]
    summary = build_summary(specs)
    assert summary["fid_is_pareto_frontier"] == ["E1", "E2b"]
    assert summary["ranking_by_fid"][:2] == ["E1", "E2b"]


def test_summary_rejects_training_seed_or_protocol_drift(tmp_path):
    payload = _metrics("E0", 42, fid=30.0, is_mean=50.0)
    payload["training_protocol"]["training_seed"] = 43
    path = tmp_path / "wrong-training-seed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SummaryError, match="training_seed"):
        build_summary([f"E0@42={path}"])

    payload = _metrics("E0", 42, fid=30.0, is_mean=50.0)
    payload["training_protocol"]["invariants"]["training.max_train_steps"] = 1
    path = tmp_path / "wrong-training-protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SummaryError, match="training protocol drifted"):
        build_summary([f"E0@42={path}"])


def test_first_stage_requires_all_variants(tmp_path):
    spec = _write(tmp_path, "E0", 42, fid=30.0, is_mean=50.0)
    with pytest.raises(SummaryError, match="missing variants"):
        build_summary([spec], "first-stage")
