"""Archived tests for the completed Q-factor summary; not part of CI."""
import hashlib
import json

import pytest

from scripts.evaluate_single_stream_fid_is import EVALUATOR_RNG_CONTRACT
from scripts.image_embedder_ablation_matrix import (
    FLOW_HEAD_INVARIANTS,
    TRAINING_PROTOCOL_INVARIANTS,
    TRAINING_PROTOCOL_SCHEMA,
    training_protocol_fingerprint,
)
from scripts.image_embedder_confirmation_protocol import (
    AUGMENTATION_CONTRACT,
    EVALUATOR_RNG_CONTRACT_SHA256,
    INITIALIZATION_CONTRACT,
    TRAIN_ORDER_CONTRACT,
    canonical_sha256,
)
from scripts.image_mask_position_ablation_protocol import (
    Q_FACTOR_CONFIG_CONTRACT_SCHEMA,
    Q_FACTOR_IDS,
    Q_FACTOR_PROVENANCE_SCHEMA,
    Q_FACTOR_RUNTIME_CONTEXT_SCHEMA,
    Q_FACTOR_SEEDS,
    Q_FACTOR_VARIANTS,
    q_factor_run_slug,
)
from scripts.summarize_image_mask_position_ablation import (
    PAIRING_GATE_SCHEMA,
    SCHEMA,
    SummaryError,
    build_summary,
)


def _digest(label):
    return hashlib.sha256(str(label).encode()).hexdigest()


def _runtime_context(**updates):
    value = {
        "schema": Q_FACTOR_RUNTIME_CONTEXT_SCHEMA,
        "world_size": 8,
        "distributed_type": "DEEPSPEED",
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "torch_version": "2.7.0",
        "cuda_version": "12.8",
        "cuda_available": True,
        "cuda_device_name": "NVIDIA H100 80GB HBM3",
        "deepspeed_zero_stage": 2,
    }
    value.update(updates)
    value["runtime_context_sha256"] = canonical_sha256(value)
    return value


def _config_contract(variant_id, seed, **extra_resolved):
    variant = Q_FACTOR_VARIANTS[variant_id]
    slug = q_factor_run_slug(variant_id, seed)
    resolved = {
        "experiment": {
            "ablation_phase": "mask_position_q_factor",
            "ablation_id": variant_id,
            "parent_ablation_id": variant.parent_ablation_id,
            "project": slug,
            "name": f"q-factor-{variant_id}-{seed}",
        },
        "training": {"seed": seed, "dataloader_shuffle_seed": seed},
        "evaluation": {
            "seed": 42,
            "checkpoint": f"output/{slug}/hf_model-final-ema",
        },
        "model": {
            "image_query_stage_mode": "none",
            "image_observed_position_mode": variant.observed_position_mode,
            "image_mask_position_mode": variant.mask_position_mode,
            "image_rope_mode": "row_col_2d",
            "image_space_to_depth_factor": 1,
        },
        "optimizer": {"name": "adamw"},
    }
    resolved.update(extra_resolved)
    return {"schema": Q_FACTOR_CONFIG_CONTRACT_SCHEMA, "resolved_config": resolved}


def _input_files():
    return {
        label: {
            "path": f"public/datasets/{label}",
            "size_bytes": 1000 + index,
            "sha256": _digest(f"input-{label}"),
        }
        for index, label in enumerate(
            ("cache", "manifest", "split_manifest", "synset_mapping")
        )
    }


def _metrics(variant_id, seed, *, fid, score):
    variant = Q_FACTOR_VARIANTS[variant_id]
    slug = q_factor_run_slug(variant_id, seed)
    declaration_sha = _digest(f"declaration-{variant_id}-{seed}")
    study_sha = _digest("study")
    parent_sha = _digest("parent")
    source_sha = _digest("source")
    provenance_sha = _digest(f"provenance-{variant_id}-{seed}")
    contract = _config_contract(variant_id, seed)
    config_sha = canonical_sha256(contract)
    runtime = _runtime_context()
    input_files = _input_files()
    provenance = {
        "schema": Q_FACTOR_PROVENANCE_SCHEMA,
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "architecture": {
            "parent_ablation_id": variant.parent_ablation_id,
            "observed_position_mode": variant.observed_position_mode,
            "mask_position_mode": variant.mask_position_mode,
            "query_stage_mode": "none",
            "rope_mode": "row_col_2d",
            "space_to_depth_factor": 1,
        },
        "provenance_sha256": provenance_sha,
        "q_factor_declaration_sha256": declaration_sha,
        "study_manifest_sha256": study_sha,
        "parent_summary_sha256": parent_sha,
        "config_contract_sha256": config_sha,
        "config_contract": contract,
        "runtime_source_manifest_sha256": source_sha,
        "initial_state": {
            "contract": INITIALIZATION_CONTRACT,
            "image_modules": {
                "parameter_count": 100,
                "parameter_schema_sha256": _digest("image-schema"),
                "state_sha256": _digest(f"image-state-{seed}"),
            },
            "special_token_names_and_ids": [
                ["boi", 1],
                ["eoi", 2],
                ["image_mask", 3],
                ["mask", 4],
            ],
            "special_token_rows_sha256": _digest(f"special-rows-{seed}"),
        },
        "train_data": {
            "contract": TRAIN_ORDER_CONTRACT,
            "dataloader_shuffle_seed": seed,
            "initial_generator_state_sha256": _digest(f"generator-{seed}"),
            "dataloader_base_seed": 100_000 + seed,
            "dataset_length": 100_000,
            "epoch0_ordered_sample_identity_sha256": _digest(f"order-{seed}"),
            "augmentation_contract": AUGMENTATION_CONTRACT,
            "epoch0_augmentation_decisions_sha256": _digest(f"augment-{seed}"),
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
        "runtime_context": runtime,
    }
    flow_head = {
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
    }
    q_factor = {
        "declaration_sha256": declaration_sha,
        "study_manifest_sha256": study_sha,
        "parent_summary_sha256": parent_sha,
        "config_contract_sha256": config_sha,
        "source_manifest_sha256": source_sha,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "dataloader_shuffle_seed": seed,
        "provenance_path": f"output/{slug}/q_factor_training_provenance.json",
        "provenance_sha256": provenance_sha,
        "provenance": provenance,
    }
    wall = 1000.0
    return {
        "official_protocol": True,
        "implementation_contracts": {
            "image_stage_buffer": "fixed_sincos_nonpersistent_rebuild_v1",
            "evaluator_rng_contract": EVALUATOR_RNG_CONTRACT,
            "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
            "canonical_initial_noise_enabled": True,
            "canonical_noise_manifest_schema": "canonical_image_flow_noise_manifest_v1",
            "canonical_noise_manifest_sha256": _digest("canonical-noise"),
            "ordered_eval_sample_manifest_schema": "ordered_image_embedder_eval_samples_v1",
            "ordered_eval_sample_manifest_sha256": _digest("ordered-samples"),
            "paired_sample_count": 10_000,
        },
        "metric_protocol": {
            "fid_reducer": "symmetric_eigendecomposition",
            "is_split_assignment": "contiguous_by_global_sample_index",
            "is_std": "population",
            "is_splits": 10,
        },
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
                "checkpoint_metadata_path": f"output/{slug}/checkpoint-35920/metadata.json",
                "ema_state_path": f"output/{slug}/checkpoint-35920/ema_state.pt",
                "ema_state_size_bytes": 3_000_000_000,
                "hf_model_weights_path": f"output/{slug}/hf_model-final-ema/model.safetensors",
                "hf_model_weights_size_bytes": 3_000_000_000,
                "checkpoint_metadata_sha256": _digest(f"metadata-{variant_id}-{seed}"),
                "ema_state_sha256": _digest(f"ema-{variant_id}-{seed}"),
                "hf_model_weights_sha256": _digest(f"hf-{variant_id}-{seed}"),
                "q_factor_provenance_path": f"output/{slug}/q_factor_training_provenance.json",
                "q_factor_hf_provenance_path": (
                    f"output/{slug}/hf_model-final-ema/q_factor_training_provenance.json"
                ),
            },
            "q_factor": q_factor,
        },
        "architecture": {
            "ablation_id": variant_id,
            "parent_ablation_id": variant.parent_ablation_id,
            "q_factor_id": variant_id,
            "mask_query_position_factor": variant_id.rsplit("-", 1)[-1],
            "image_query_stage_mode": "none",
            "image_observed_position_mode": variant.observed_position_mode,
            "image_mask_position_mode": variant.mask_position_mode,
            "image_rope_mode": "row_col_2d",
            "image_space_to_depth_factor": 1,
            "image_canonical_grid_side": 16,
            "image_canonical_latent_dim": 16,
            "image_grid_side": 16,
            "image_tokens_per_img": 256,
            "image_latent_dim": 16,
            "padded_sequence_length": 320,
            "flow_head": flow_head,
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
            "autocast_enabled": False,
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
        "real_stats_metadata": {"manifest_sha256": _digest("real-stats")},
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
                "inception_score_mean": score,
                "inception_score_std": 1.0,
                "inception_score_splits": [score + index * 0.01 for index in range(10)],
                "latent_mse_to_target": 2.0,
                "latent_rms": 1.0,
                "generation_step_max": 256.0,
                "generation_wall_seconds": wall,
                "generation_samples_per_second": 10_000 / wall,
            }
        },
    }


BASE_FID = {"E2b-Q1": 25.3, "E2b-Q0": 25.1, "E2-Q1": 24.9, "E2-Q0": 25.1}
BASE_IS = {"E2b-Q1": 61.0, "E2b-Q0": 61.2, "E2-Q1": 61.1, "E2-Q0": 61.0}


def _specs(tmp_path, mutate=None):
    specs = []
    for variant_id in Q_FACTOR_IDS:
        for seed in sorted(Q_FACTOR_SEEDS):
            offset = (seed - 44) * 0.1
            payload = _metrics(
                variant_id,
                seed,
                fid=BASE_FID[variant_id] + offset,
                score=BASE_IS[variant_id] - offset,
            )
            if mutate is not None:
                mutate(variant_id, seed, payload)
            path = tmp_path / f"{variant_id}-{seed}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            specs.append(f"{variant_id}@{seed}={path}")
    return specs


def test_summary_validates_exact_matrix_effects_interaction_ranking_and_preference(tmp_path):
    summary = build_summary(_specs(tmp_path))
    assert summary["schema"] == SCHEMA
    assert summary["pairing_gate"]["schema"] == PAIRING_GATE_SCHEMA
    assert summary["pairing_gate"]["validated_runs"] == 12
    assert summary["ranking_by_fid"][0] == "E2-Q1"
    assert {item["id"] for item in summary["fid_is_pareto_frontier"]} == {
        "E2-Q1",
        "E2b-Q0",
    }
    assert summary["selection"]["selected_id"] == "E2-Q0"
    assert summary["selection"]["selected_reason"] == "simplicity_preference"

    e2b_q = summary["paired_effects"]["mask_Q0_minus_Q1_at_E2b"]
    assert e2b_q["fid_candidate_minus_reference"]["mean"] == pytest.approx(-0.2)
    assert e2b_q["fid_candidate_minus_reference"]["n"] == 3
    assert len(e2b_q["fid_candidate_minus_reference"]["ci95"]) == 2
    assert summary["paired_effects"]["mask_Q0_minus_Q1_at_E2"][
        "fid_candidate_minus_reference"
    ]["mean"] == pytest.approx(0.2)
    assert summary["interaction"]["fid"]["mean"] == pytest.approx(0.4)
    assert summary["interaction"]["fid"]["n"] == 3


def test_preference_falls_back_to_best_fid_outside_margins(tmp_path):
    def mutate(variant_id, _seed, payload):
        if variant_id == "E2-Q0":
            payload["strategies"]["spatial_halton"]["fid"] += 1.0

    summary = build_summary(_specs(tmp_path, mutate))
    assert not summary["selection"]["within_simplicity_margins"]
    assert summary["selection"]["selected_id"] == "E2-Q1"
    assert summary["selection"]["selected_reason"] == "best_mean_fid"


def test_summary_requires_exact_cartesian_matrix_and_rejects_duplicates(tmp_path):
    specs = _specs(tmp_path)
    with pytest.raises(SummaryError, match="exact 4x3 matrix"):
        build_summary(specs[:-1])
    with pytest.raises(SummaryError, match="duplicate"):
        build_summary(specs + [specs[0]])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda _id, _seed, p: p["architecture"].__setitem__(
                "image_rope_mode", "sequence_1d"
            ),
            "image_rope_mode",
        ),
        (
            lambda _id, _seed, p: p["training_protocol"]["artifacts"].__setitem__(
                "ema_state_sha256", "bad"
            ),
            "SHA256",
        ),
        (
            lambda _id, _seed, p: p["strategies"]["spatial_halton"].__setitem__(
                "fid", float("nan")
            ),
            "FID must be positive and finite",
        ),
    ],
)
def test_summary_rejects_architecture_checkpoint_hash_and_nonfinite_metrics(
    tmp_path, mutate, message
):
    def mutate_one(variant_id, seed, payload):
        if variant_id == "E2b-Q1" and seed == 43:
            mutate(variant_id, seed, payload)

    with pytest.raises(SummaryError, match=message):
        build_summary(_specs(tmp_path, mutate_one))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("image_state", "image_state_sha256"),
        ("train_order", "epoch0_ordered_sample_identity_sha256"),
        ("source", "source_manifest_sha256"),
        ("base", "base_model_manifest_sha256"),
        ("noise", "canonical_noise_manifest_sha256"),
        ("sample", "ordered_eval_sample_manifest_sha256"),
    ],
)
def test_summary_rejects_paired_and_global_evidence_drift(tmp_path, field, message):
    def mutate(variant_id, seed, payload):
        if variant_id != "E2b-Q1" or seed != 43:
            return
        provenance = payload["training_protocol"]["q_factor"]["provenance"]
        if field == "image_state":
            provenance["initial_state"]["image_modules"]["state_sha256"] = _digest(
                "wrong-image-state"
            )
        elif field == "train_order":
            provenance["train_data"]["epoch0_ordered_sample_identity_sha256"] = _digest(
                "wrong-order"
            )
        elif field == "source":
            value = _digest("wrong-source")
            payload["training_protocol"]["q_factor"]["source_manifest_sha256"] = value
            provenance["runtime_source_manifest_sha256"] = value
        elif field == "base":
            provenance["base_model_manifest_sha256"] = _digest("wrong-base")
        elif field == "noise":
            payload["implementation_contracts"]["canonical_noise_manifest_sha256"] = _digest(
                "wrong-noise"
            )
        elif field == "sample":
            payload["implementation_contracts"][
                "ordered_eval_sample_manifest_sha256"
            ] = _digest("wrong-sample")

    with pytest.raises(SummaryError, match=message):
        build_summary(_specs(tmp_path, mutate))


def test_summary_rejects_runtime_and_unapproved_config_drift(tmp_path):
    def runtime_mutate(variant_id, seed, payload):
        if variant_id == "E2b-Q1" and seed == 43:
            runtime = payload["training_protocol"]["q_factor"]["provenance"][
                "runtime_context"
            ]
            runtime.pop("runtime_context_sha256")
            runtime["world_size"] = 4
            runtime["runtime_context_sha256"] = canonical_sha256(runtime)

    with pytest.raises(SummaryError, match="world_size=8"):
        build_summary(_specs(tmp_path, runtime_mutate))

    def config_mutate(variant_id, seed, payload):
        if variant_id == "E2b-Q1" and seed == 43:
            q_factor = payload["training_protocol"]["q_factor"]
            contract = q_factor["provenance"]["config_contract"]
            contract["resolved_config"]["optimizer"]["name"] = "sgd"
            digest = canonical_sha256(contract)
            q_factor["config_contract_sha256"] = digest
            q_factor["provenance"]["config_contract_sha256"] = digest

    with pytest.raises(SummaryError, match="config_pairing_projection_sha256"):
        build_summary(_specs(tmp_path, config_mutate))
