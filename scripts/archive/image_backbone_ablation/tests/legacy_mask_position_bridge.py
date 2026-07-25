"""Archived tests for the legacy Q1 evidence bridge; not part of CI."""
import hashlib
import json
import shutil
from dataclasses import asdict

import pytest

import scripts.summarize_image_mask_position_ablation_legacy_bridge as bridge
from scripts.evaluate_single_stream_fid_is import EVALUATOR_RNG_CONTRACT
from scripts.image_embedder_ablation_matrix import (
    FLOW_HEAD_INVARIANTS,
    TRAINING_PROTOCOL_INVARIANTS,
    TRAINING_PROTOCOL_SCHEMA,
    training_protocol_fingerprint,
)
from scripts.image_embedder_confirmation_protocol import (
    AUGMENTATION_CONTRACT,
    CONFIRMATION_DECLARATION_SCHEMA,
    CONFIRMATION_PROVENANCE_SCHEMA,
    EVALUATOR_RNG_CONTRACT_SCHEMA,
    EVALUATOR_RNG_CONTRACT_SHA256,
    INITIALIZATION_CONTRACT,
    TRAIN_ORDER_CONTRACT,
    canonical_sha256,
)
from scripts.image_mask_position_ablation_protocol import (
    PARENT_CONFIRMATION_IDS,
    PARENT_PAIRING_GATE_SCHEMA,
    PARENT_SUMMARY_SCHEMA,
    Q_FACTOR_CONFIG_CONTRACT_SCHEMA,
    Q_FACTOR_DECLARATION_SCHEMA,
    Q_FACTOR_DECISION_RULE,
    Q_FACTOR_IDS,
    Q_FACTOR_PHASE,
    Q_FACTOR_PROVENANCE_SCHEMA,
    Q_FACTOR_RUNTIME_CONTEXT_SCHEMA,
    Q_FACTOR_RUNTIME_SOURCE_FILES,
    Q_FACTOR_SEEDS,
    Q_FACTOR_SOURCE_SCHEMA,
    Q_FACTOR_STUDY_SCHEMA,
    Q_FACTOR_VARIANTS,
    load_parent_summary_evidence,
    q_factor_run_slug,
)


def _digest(label):
    return hashlib.sha256(str(label).encode("utf-8")).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _flow_head():
    return {
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


def _initial_state(seed, *, drift=False):
    return {
        "contract": INITIALIZATION_CONTRACT,
        "image_modules": {
            "parameter_count": 100,
            "parameter_schema_sha256": _digest("image-schema"),
            "state_sha256": _digest(f"image-state-{seed}{'-drift' if drift else ''}"),
        },
        "special_token_names_and_ids": [
            ["boi", 1],
            ["eoi", 2],
            ["image_mask", 3],
            ["mask", 4],
        ],
        "special_token_rows_sha256": _digest(f"special-rows-{seed}"),
    }


def _train_data(seed):
    return {
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
        "input_files": _input_files(),
    }


def _file_evidence(label):
    files = [
        {
            "path": f"evidence/{label}.bin",
            "size_bytes": 123,
            "sha256": _digest(f"evidence-{label}"),
        }
    ]
    return {"files": files, "manifest_sha256": canonical_sha256(files)}


def _q_source(label="q0-source"):
    files = [
        {
            "path": relative,
            "size_bytes": index + 1,
            "sha256": _digest(f"{label}-{relative}"),
        }
        for index, relative in enumerate(Q_FACTOR_RUNTIME_SOURCE_FILES)
    ]
    return {
        "schema": Q_FACTOR_SOURCE_SCHEMA,
        "required_files": list(Q_FACTOR_RUNTIME_SOURCE_FILES),
        "files": files,
        "manifest_sha256": canonical_sha256(files),
    }


def _runtime_context():
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
    value["runtime_context_sha256"] = canonical_sha256(value)
    return value


def _hf_config(observed, *, mask_marker="absent"):
    value = {
        "image_query_stage_mode": "none",
        "image_observed_position_mode": observed,
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
    if mask_marker != "absent":
        value["image_mask_position_mode"] = mask_marker
    return value


def _metric_payload(slug, architecture, training_protocol, *, fid, score):
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
        "training_protocol": training_protocol,
        "architecture": architecture,
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
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": True,
            "float32_matmul_precision": "highest",
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


def _legacy_declaration(physical_id, seed):
    candidate_manifest = {"candidate_ids": ["E2b", "E2"]}
    declaration = {
        "schema": CONFIRMATION_DECLARATION_SCHEMA,
        "ablation_id": physical_id,
        "training_seed": seed,
        "dataloader_shuffle_seed": seed,
        "evaluation_seed": 42,
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "screen_summary_path": "output/screen.json",
        "screen_summary_sha256": _digest("screen-summary"),
        "candidate_manifest": candidate_manifest,
        "candidate_manifest_sha256": canonical_sha256(candidate_manifest),
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
    }
    declaration["declaration_sha256"] = canonical_sha256(declaration)
    return declaration


def _write_legacy_run(
    root, physical_id, seed, *, fid, score, legacy_source, base_model
):
    analysis_id = f"{physical_id}-Q1"
    slug = f"selfless-flow-image-embedder-{physical_id.lower()}-seed{seed}"
    run_root = root / "output" / slug
    declaration = _legacy_declaration(physical_id, seed)
    initial = _initial_state(seed)
    train_data = _train_data(seed)
    full = {
        "schema": CONFIRMATION_PROVENANCE_SCHEMA,
        "ablation_id": physical_id,
        "training_seed": seed,
        "space_to_depth_factor": 1,
        "confirmation_declaration": declaration,
        "confirmation_declaration_sha256": declaration["declaration_sha256"],
        "initial_state": initial,
        "train_data": train_data,
        "base_model": base_model,
        "runtime_source": legacy_source,
    }
    full["provenance_sha256"] = canonical_sha256(full)
    provenance_path = f"output/{slug}/confirmation_training_provenance.json"
    compact = {
        "schema": full["schema"],
        "ablation_id": physical_id,
        "training_seed": seed,
        "provenance_sha256": full["provenance_sha256"],
        "confirmation_declaration_sha256": declaration["declaration_sha256"],
        "space_to_depth_factor": 1,
        "initial_state": initial,
        "train_data": train_data,
        "base_model_manifest_sha256": base_model["manifest_sha256"],
        "runtime_source_manifest_sha256": legacy_source["manifest_sha256"],
    }
    confirmation = {
        "declaration_sha256": declaration["declaration_sha256"],
        "screen_summary_sha256": declaration["screen_summary_sha256"],
        "candidate_manifest_sha256": declaration["candidate_manifest_sha256"],
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "dataloader_shuffle_seed": seed,
        "provenance_path": provenance_path,
        "provenance_sha256": full["provenance_sha256"],
        "provenance": compact,
    }
    training = {
        "schema": TRAINING_PROTOCOL_SCHEMA,
        "training_seed": seed,
        "invariants": dict(TRAINING_PROTOCOL_INVARIANTS),
        "invariants_sha256": training_protocol_fingerprint(
            TRAINING_PROTOCOL_INVARIANTS
        ),
        "final_global_step": 35_920,
        "confirmation": confirmation,
        "artifacts": {
            "checkpoint_metadata_path": f"output/{slug}/checkpoint-35920/metadata.json",
            "ema_state_path": f"output/{slug}/checkpoint-35920/ema_state.pt",
            "ema_state_size_bytes": 3_000_000_000,
            "hf_model_weights_path": f"output/{slug}/hf_model-final-ema/model.safetensors",
            "hf_model_weights_size_bytes": 3_000_000_000,
            "confirmation_provenance_path": provenance_path,
            "confirmation_hf_provenance_path": (
                f"output/{slug}/hf_model-final-ema/confirmation_training_provenance.json"
            ),
        },
    }
    observed = Q_FACTOR_VARIANTS[analysis_id].observed_position_mode
    architecture = {
        "ablation_id": physical_id,
        "image_query_stage_mode": "none",
        "image_observed_position_mode": observed,
        "image_rope_mode": "row_col_2d",
        "image_space_to_depth_factor": 1,
        "image_canonical_grid_side": 16,
        "image_canonical_latent_dim": 16,
        "image_grid_side": 16,
        "image_tokens_per_img": 256,
        "image_latent_dim": 16,
        "padded_sequence_length": 320,
        "flow_head": _flow_head(),
    }
    payload = _metric_payload(slug, architecture, training, fid=fid, score=score)
    metrics_path = run_root / "fid_is_selected_cfg3p5_ema/metrics.json"
    _write_json(metrics_path, payload)
    _write_json(run_root / "confirmation_training_provenance.json", full)
    _write_json(
        run_root / "hf_model-final-ema/confirmation_training_provenance.json",
        full,
    )
    checkpoint_model = _hf_config(observed)
    _write_json(
        run_root / "checkpoint-35920/metadata.json",
        {
            "global_step": 35_920,
            "model_config": checkpoint_model,
            "confirmation_provenance": {
                "path": provenance_path,
                "sha256": full["provenance_sha256"],
                "declaration_sha256": declaration["declaration_sha256"],
            },
        },
    )
    (run_root / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (run_root / "config.yaml").write_text(
        f"experiment:\n  ablation_id: {physical_id}\n", encoding="utf-8"
    )
    _write_json(run_root / "hf_model-final-ema/config.json", _hf_config(observed))
    return analysis_id, metrics_path, payload


def _parent_row(physical_id, seed, metrics_path, payload):
    result = payload["strategies"]["spatial_halton"]
    parameters = payload["parameters"]
    distributed = payload["distributed"]
    return {
        "id": physical_id,
        "training_seed": seed,
        "purpose": "fixture",
        "fid": result["fid"],
        "is_mean": result["inception_score_mean"],
        "is_std": result["inception_score_std"],
        "sampling_wall_seconds": result["generation_wall_seconds"],
        "sampling_samples_per_second": result["generation_samples_per_second"],
        "latent_mse_to_target": result["latent_mse_to_target"],
        "latent_rms": result["latent_rms"],
        "generation_step_max": result["generation_step_max"],
        "peak_cuda_allocated_mib": distributed["peak_cuda_allocated_mib"],
        "peak_cuda_reserved_mib": distributed["peak_cuda_reserved_mib"],
        **parameters,
        "metrics_path": str(metrics_path.resolve()),
        "training_protocol_sha256": payload["training_protocol"]["invariants_sha256"],
    }


def _write_parent_summary(root, legacy_runs):
    lookup = {
        (physical_id, seed): _parent_row(physical_id, seed, path, payload)
        for physical_id, seed, path, payload in legacy_runs
    }
    runs = []
    for candidate_id in PARENT_CONFIRMATION_IDS:
        for seed in sorted(Q_FACTOR_SEEDS):
            if (candidate_id, seed) in lookup:
                runs.append(lookup[(candidate_id, seed)])
            else:
                runs.append({"id": candidate_id, "training_seed": seed})
    aggregate_values = {
        "E0": (26.3, 59.0),
        "E1": (26.5, 58.9),
        "E2b": (25.3, 61.2),
        "E2": (25.0, 61.1),
        "E4b": (24.8, 61.4),
        "E4": (25.2, 61.0),
    }
    aggregates = [
        {
            "id": candidate_id,
            "seeds": sorted(Q_FACTOR_SEEDS),
            "runs": 3,
            "fid_mean": aggregate_values[candidate_id][0],
            "fid_sample_std": 0.1,
            "is_mean": aggregate_values[candidate_id][1],
            "is_sample_std": 0.1,
            "sampling_samples_per_second_mean": 10.0,
        }
        for candidate_id in PARENT_CONFIRMATION_IDS
    ]
    pairing_gate = {
        "schema": PARENT_PAIRING_GATE_SCHEMA,
        "validated_runs": 18,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
    }
    payload = {
        "schema": PARENT_SUMMARY_SCHEMA,
        "expected": "confirmation",
        "runs": runs,
        "aggregates": aggregates,
        "confirmation_scope_manifest": {
            "candidate_ids": list(PARENT_CONFIRMATION_IDS),
            "required_space_to_depth_factor": 1,
        },
        "confirmation_pairing_gate": pairing_gate,
    }
    path = root / bridge.PARENT_SUMMARY_RELATIVE_PATH
    _write_json(path, payload)
    return path


def _q_config_contract(variant_id, seed):
    variant = Q_FACTOR_VARIANTS[variant_id]
    slug = q_factor_run_slug(variant_id, seed)
    return {
        "schema": Q_FACTOR_CONFIG_CONTRACT_SCHEMA,
        "resolved_config": {
            "experiment": {
                "ablation_phase": Q_FACTOR_PHASE,
                "ablation_id": variant_id,
                "parent_ablation_id": variant.parent_ablation_id,
                "project": slug,
                "name": f"fixture-{variant_id}-{seed}",
            },
            "training": {"seed": seed, "dataloader_shuffle_seed": seed},
            "evaluation": {
                "seed": 42,
                "checkpoint": f"output/{slug}/hf_model-final-ema",
            },
            "model": {
                "image_query_stage_mode": "none",
                "image_observed_position_mode": variant.observed_position_mode,
                "image_mask_position_mode": "none",
                "image_rope_mode": "row_col_2d",
                "image_space_to_depth_factor": 1,
            },
            "optimizer": {"name": "adamw"},
        },
    }


def _q_study(parent_path, q_source):
    parent = load_parent_summary_evidence(parent_path)
    parent["path"] = bridge.PARENT_SUMMARY_RELATIVE_PATH.as_posix()
    study = {
        "schema": Q_FACTOR_STUDY_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_ids": list(Q_FACTOR_IDS),
        "training_seeds": sorted(Q_FACTOR_SEEDS),
        "evaluation_seed": 42,
        "matrix": {
            variant_id: asdict(variant)
            for variant_id, variant in Q_FACTOR_VARIANTS.items()
        },
        "parent_summary": parent,
        "runtime_source": q_source,
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
        "decision_rule": Q_FACTOR_DECISION_RULE,
    }
    study["study_manifest_sha256"] = canonical_sha256(study)
    return study


def _write_q0_run(
    root,
    variant_id,
    seed,
    *,
    fid,
    score,
    q_source,
    study,
    base_model,
    initial_drift=False,
):
    variant = Q_FACTOR_VARIANTS[variant_id]
    slug = q_factor_run_slug(variant_id, seed)
    run_root = root / "output" / slug
    contract = _q_config_contract(variant_id, seed)
    config_sha = canonical_sha256(contract)
    declaration = {
        "schema": Q_FACTOR_DECLARATION_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "dataloader_shuffle_seed": seed,
        "evaluation_seed": 42,
        "run_slug": slug,
        "architecture": asdict(variant),
        "config_contract": contract,
        "config_contract_sha256": config_sha,
        "study_manifest": study,
        "study_manifest_sha256": study["study_manifest_sha256"],
        "parent_summary_sha256": study["parent_summary"]["sha256"],
        "runtime_source_manifest_sha256": q_source["manifest_sha256"],
        "source_manifest_sha256": q_source["manifest_sha256"],
        "evaluator_rng_contract_schema": EVALUATOR_RNG_CONTRACT_SCHEMA,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "initialization_contract": INITIALIZATION_CONTRACT,
        "train_order_contract": TRAIN_ORDER_CONTRACT,
        "augmentation_contract": AUGMENTATION_CONTRACT,
    }
    declaration["declaration_sha256"] = canonical_sha256(declaration)
    initial = _initial_state(seed, drift=initial_drift)
    train_data = _train_data(seed)
    runtime = _runtime_context()
    full = {
        "schema": Q_FACTOR_PROVENANCE_SCHEMA,
        "phase": Q_FACTOR_PHASE,
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "architecture": asdict(variant),
        "q_factor_declaration": declaration,
        "q_factor_declaration_sha256": declaration["declaration_sha256"],
        "study_manifest_sha256": study["study_manifest_sha256"],
        "parent_summary_sha256": study["parent_summary"]["sha256"],
        "config_contract_sha256": config_sha,
        "runtime_source_manifest_sha256": q_source["manifest_sha256"],
        "source_manifest_sha256": q_source["manifest_sha256"],
        "initial_state": initial,
        "train_data": train_data,
        "base_model": base_model,
        "runtime_source": q_source,
        "runtime_context": runtime,
    }
    full["provenance_sha256"] = canonical_sha256(full)
    compact = {
        "schema": full["schema"],
        "q_factor_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "training_seed": seed,
        "architecture": asdict(variant),
        "provenance_sha256": full["provenance_sha256"],
        "q_factor_declaration_sha256": declaration["declaration_sha256"],
        "study_manifest_sha256": study["study_manifest_sha256"],
        "parent_summary_sha256": study["parent_summary"]["sha256"],
        "config_contract_sha256": config_sha,
        "config_contract": contract,
        "runtime_source_manifest_sha256": q_source["manifest_sha256"],
        "initial_state": initial,
        "train_data": train_data,
        "base_model_manifest_sha256": base_model["manifest_sha256"],
        "runtime_context": runtime,
    }
    provenance_rel = f"output/{slug}/q_factor_training_provenance.json"
    q_factor = {
        "declaration_sha256": declaration["declaration_sha256"],
        "study_manifest_sha256": study["study_manifest_sha256"],
        "parent_summary_sha256": study["parent_summary"]["sha256"],
        "config_contract_sha256": config_sha,
        "source_manifest_sha256": q_source["manifest_sha256"],
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
        "dataloader_shuffle_seed": seed,
        "provenance_path": provenance_rel,
        "provenance_sha256": full["provenance_sha256"],
        "provenance": compact,
    }
    checkpoint_path = run_root / "checkpoint-35920/metadata.json"
    _write_json(
        checkpoint_path,
        {
            "global_step": 35_920,
            "model_config": _hf_config(
                variant.observed_position_mode, mask_marker="none"
            ),
            "q_factor_provenance": {
                "path": provenance_rel,
                "sha256": full["provenance_sha256"],
                "declaration_sha256": declaration["declaration_sha256"],
                "study_manifest_sha256": study["study_manifest_sha256"],
                "config_contract_sha256": config_sha,
                "source_manifest_sha256": q_source["manifest_sha256"],
            },
        },
    )
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    training = {
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
            "checkpoint_metadata_sha256": checkpoint_sha,
            "ema_state_sha256": _digest(f"ema-{variant_id}-{seed}"),
            "hf_model_weights_sha256": _digest(f"hf-{variant_id}-{seed}"),
            "q_factor_provenance_path": provenance_rel,
            "q_factor_hf_provenance_path": (
                f"output/{slug}/hf_model-final-ema/q_factor_training_provenance.json"
            ),
        },
        "q_factor": q_factor,
    }
    architecture = {
        "ablation_id": variant_id,
        "parent_ablation_id": variant.parent_ablation_id,
        "q_factor_id": variant_id,
        "mask_query_position_factor": "Q0",
        "image_query_stage_mode": "none",
        "image_observed_position_mode": variant.observed_position_mode,
        "image_mask_position_mode": "none",
        "image_rope_mode": "row_col_2d",
        "image_space_to_depth_factor": 1,
        "image_canonical_grid_side": 16,
        "image_canonical_latent_dim": 16,
        "image_grid_side": 16,
        "image_tokens_per_img": 256,
        "image_latent_dim": 16,
        "padded_sequence_length": 320,
        "flow_head": _flow_head(),
    }
    payload = _metric_payload(slug, architecture, training, fid=fid, score=score)
    metrics_path = run_root / "fid_is_selected_cfg3p5_ema/metrics.json"
    _write_json(metrics_path, payload)
    _write_json(run_root / "q_factor_training_provenance.json", full)
    _write_json(run_root / "hf_model-final-ema/q_factor_training_provenance.json", full)
    _write_json(
        run_root / "hf_model-final-ema/config.json",
        _hf_config(variant.observed_position_mode, mask_marker="none"),
    )
    return metrics_path


def _write_waiver(root, q_source_sha, monkeypatch):
    bytecode = b"frozen-bytecode-fixture"
    bytecode_sha = hashlib.sha256(bytecode).hexdigest()
    monkeypatch.setattr(
        bridge, "EVALUATION_WAIVER_FROZEN_BYTECODE_RAW_SHA256", bytecode_sha
    )
    bytecode_path = (
        root / "output/image_mask_position_ablation/source_drift_waiver/"
        "frozen_modeling_selfless_flow.cpython-312.pyc"
    )
    bytecode_path.parent.mkdir(parents=True, exist_ok=True)
    bytecode_path.write_bytes(bytecode)
    launcher_path = root / bridge.EVALUATION_WAIVER_LAUNCHER_RELATIVE_PATH
    launcher_path.parent.mkdir(parents=True, exist_ok=True)
    launcher_path.write_bytes(b"#!/usr/bin/env bash\n# fixture launcher\n")
    monkeypatch.setattr(
        bridge,
        "EVALUATION_WAIVER_LAUNCHER_RAW_SHA256",
        hashlib.sha256(launcher_path.read_bytes()).hexdigest(),
    )
    sitecustomize_path = root / bridge.EVALUATION_WAIVER_SITECUSTOMIZE_RELATIVE_PATH
    sitecustomize_path.parent.mkdir(parents=True, exist_ok=True)
    sitecustomize_path.write_bytes(b'"""fixture source waiver"""\n')
    monkeypatch.setattr(
        bridge,
        "EVALUATION_WAIVER_SITECUSTOMIZE_RAW_SHA256",
        hashlib.sha256(sitecustomize_path.read_bytes()).hexdigest(),
    )
    current_source_path = root / bridge.MODEL_SOURCE_RELATIVE_PATH
    current_source_path.parent.mkdir(parents=True, exist_ok=True)
    current_source_path.write_bytes(b"# current evaluation model fixture\n")
    current_source_sha = hashlib.sha256(current_source_path.read_bytes()).hexdigest()
    monkeypatch.setattr(bridge, "CURRENT_MODEL_SOURCE_SHA256", current_source_sha)
    monkeypatch.setattr(
        bridge, "CURRENT_MODEL_SOURCE_SIZE", current_source_path.stat().st_size
    )
    waiver = {
        "schema": "selfless_flow_q_factor_evaluation_source_equivalence_v1",
        "scope": {
            "q_factor_ids": ["E2-Q0"],
            "training_seeds": [44, 45],
            "evaluation_seed": 42,
        },
        "preregistered_runtime_source_manifest_sha256": q_source_sha,
        "drift": {
            "path": bridge.MODEL_SOURCE_RELATIVE_PATH.as_posix(),
            "preregistered_size_bytes": bridge.FROZEN_MODEL_SOURCE_SIZE,
            "preregistered_sha256": bridge.FROZEN_MODEL_SOURCE_SHA256,
            "evaluation_size_bytes": bridge.CURRENT_MODEL_SOURCE_SIZE,
            "evaluation_sha256": bridge.CURRENT_MODEL_SOURCE_SHA256,
            "classification": "formatting_and_unused_import_cleanup",
        },
        "frozen_bytecode_evidence": {
            "path": (
                "output/image_mask_position_ablation/source_drift_waiver/"
                "frozen_modeling_selfless_flow.cpython-312.pyc"
            ),
            "sha256": bytecode_sha,
            "cache_tag": "cpython-312",
            "header_flags": 0,
            "header_source_mtime": 1,
            "header_source_size": bridge.FROZEN_MODEL_SOURCE_SIZE,
        },
        "proof": {
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
        },
        "semantics": {
            "model_execution_patch_applied": False,
            "metric_overwrite_allowed": False,
            "reason": "fixture",
        },
    }
    canonical_path = root / bridge.EVALUATION_SOURCE_EQUIVALENCE_RELATIVE_PATH
    _write_json(canonical_path, waiver)
    raw_sha = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    for seed in (44, 45):
        sidecar = (
            root
            / "output"
            / q_factor_run_slug("E2-Q0", seed)
            / "fid_is_selected_cfg3p5_ema/evaluation_source_equivalence.json"
        )
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical_path, sidecar)
    return raw_sha


def _fake_equivalence_report(legacy_source_sha, q_source_sha):
    return {
        "historical_implementation": {
            "runtime_source_manifest_sha256": legacy_source_sha
        },
        "q0_registered_implementation": {
            "runtime_source_manifest_sha256": q_source_sha
        },
        "diff_evidence": {
            "classification": "q1_semantics_preserving_mask_mode_gate_only"
        },
        "report_sha256": bridge.EQUIVALENCE_REPORT_CANONICAL_SHA256,
    }


def _matrix(tmp_path, monkeypatch, *, initial_drift=None, source_drift=None):
    root = tmp_path
    legacy_source = _file_evidence("legacy-source")
    base_model = _file_evidence("base-model")
    legacy_runs = []
    paths = {}
    base_fid = {"E2b-Q1": 25.3, "E2-Q1": 25.0}
    base_is = {"E2b-Q1": 61.2, "E2-Q1": 61.1}
    for analysis_id, physical_id in bridge.LEGACY_ANALYSIS_TO_PHYSICAL.items():
        for seed in sorted(Q_FACTOR_SEEDS):
            offset = (seed - 44) * 0.1
            _, path, payload = _write_legacy_run(
                root,
                physical_id,
                seed,
                fid=base_fid[analysis_id] + offset,
                score=base_is[analysis_id] - offset,
                legacy_source=legacy_source,
                base_model=base_model,
            )
            paths[(analysis_id, seed)] = path
            legacy_runs.append((physical_id, seed, path, payload))
    parent_path = _write_parent_summary(root, legacy_runs)
    manifest = bridge.build_legacy_reuse_manifest(root)
    manifest_path = root / "reuse_manifest.json"
    _write_json(manifest_path, manifest)

    q_source = _q_source()
    monkeypatch.setattr(
        bridge, "REGISTERED_Q0_SOURCE_SHA256", q_source["manifest_sha256"]
    )
    monkeypatch.setattr(
        bridge,
        "_validate_legacy_equivalence_report",
        lambda: _fake_equivalence_report(
            legacy_source["manifest_sha256"], q_source["manifest_sha256"]
        ),
    )
    study = _q_study(parent_path, q_source)
    q_fid = {"E2b-Q0": 25.2, "E2-Q0": 25.1}
    q_is = {"E2b-Q0": 61.3, "E2-Q0": 61.0}
    for analysis_id in bridge.Q0_IDS:
        for seed in sorted(Q_FACTOR_SEEDS):
            current_source = (
                _q_source("drifted-source")
                if source_drift == (analysis_id, seed)
                else q_source
            )
            current_study = (
                _q_study(parent_path, current_source)
                if current_source is not q_source
                else study
            )
            offset = (seed - 44) * 0.1
            path = _write_q0_run(
                root,
                analysis_id,
                seed,
                fid=q_fid[analysis_id] + offset,
                score=q_is[analysis_id] - offset,
                q_source=current_source,
                study=current_study,
                base_model=base_model,
                initial_drift=initial_drift == (analysis_id, seed),
            )
            paths[(analysis_id, seed)] = path
    waiver_sha = _write_waiver(root, q_source["manifest_sha256"], monkeypatch)
    monkeypatch.setattr(bridge, "EVALUATION_SOURCE_EQUIVALENCE_RAW_SHA256", waiver_sha)
    q0_attestation = bridge.build_q0_metrics_attestation(root)
    q0_attestation_path = root / bridge.Q0_ATTESTATION_MANIFEST_RELATIVE_PATH
    _write_json(q0_attestation_path, q0_attestation)
    monkeypatch.setattr(
        bridge,
        "Q0_ATTESTATION_MANIFEST_RAW_SHA256",
        hashlib.sha256(q0_attestation_path.read_bytes()).hexdigest(),
    )
    specs = [
        f"{analysis_id}@{seed}={paths[(analysis_id, seed)]}"
        for analysis_id in bridge.ANALYSIS_IDS
        for seed in sorted(Q_FACTOR_SEEDS)
    ]
    return root, manifest_path, specs, paths


def _build(root, manifest_path, specs):
    return bridge.build_summary(
        specs,
        artifact_root=root,
        reuse_manifest_path=manifest_path,
        q0_attestation_manifest_path=(
            root / bridge.Q0_ATTESTATION_MANIFEST_RELATIVE_PATH
        ),
    )


def _repin_q0_metrics_entry(root, analysis_id, seed, metrics_path, monkeypatch):
    attestation_path = root / bridge.Q0_ATTESTATION_MANIFEST_RELATIVE_PATH
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in payload["runs"]
        if item["analysis_id"] == analysis_id and item["training_seed"] == seed
    )
    entry["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    body = dict(payload)
    body.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_sha256(body)
    _write_json(attestation_path, payload)
    monkeypatch.setattr(
        bridge,
        "Q0_ATTESTATION_MANIFEST_RAW_SHA256",
        hashlib.sha256(attestation_path.read_bytes()).hexdigest(),
    )


def test_bridge_accepts_exact_mixed_matrix_and_labels_cross_source(
    monkeypatch, tmp_path
):
    root, manifest_path, specs, _paths = _matrix(tmp_path, monkeypatch)
    summary = _build(root, manifest_path, specs)

    assert summary["schema"] == bridge.SCHEMA
    assert summary["analysis_design"] == "historical_control_bridge"
    assert summary["comparison_kind"] == "seed_aligned_cross_source_descriptive"
    assert summary["prospective_matrix_complete"] is False
    assert summary["formal_q_factor"] is False
    assert summary["post_hoc_amendment"] is True
    assert summary["same_source_training"] is False
    assert summary["source_revision_confounded"] is True
    assert len(summary["runs"]) == 12
    assert summary["q0_metrics_attestation"]["validated_q0_runs"] == 6
    assert summary["q0_metrics_attestation"]["manifest_raw_sha256"] == (
        bridge.Q0_ATTESTATION_MANIFEST_RAW_SHA256
    )
    assert "paired_effects" not in summary
    cross = summary["seed_aligned_cross_source_effects"]
    assert cross["mask_Q0_minus_Q1_at_E2b"]["same_source_training"] is False
    assert cross["mask_Q0_minus_Q1_at_E2b"]["comparison_design"] == (
        "seed_aligned_cross_source"
    )
    assert summary["cross_cohort_comparability_gate"]["source_revision_confounded"]
    modes = summary["cohort_gates"]["q0"]["evaluation_source_modes"]
    assert modes["E2-Q0@44"] == "frozen_bytecode_equivalence_waiver"
    assert modes["E2-Q0@45"] == "frozen_bytecode_equivalence_waiver"
    assert modes["E2-Q0@43"] == "frozen_registered_source"
    assert summary["selection"]["best_fid_id"] == "E4b"
    assert summary["selection"]["selected_id"] == "E2-Q0"
    assert summary["selection"]["selected_reason"] == "simplicity_preference"
    assert (
        len(summary["selection"]["decision_evidence"]["authoritative_candidates"]) == 10
    )
    attestation = json.loads(
        (root / bridge.Q0_ATTESTATION_MANIFEST_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )
    attested = {
        (item["analysis_id"], item["training_seed"]): item
        for item in attestation["runs"]
    }
    for seed, job_name in (
        (44, "imgemb-qf-e2-q0-s44-ev-fz"),
        (45, "imgemb-qf-e2-q0-s45-ev-fz"),
    ):
        entry = attested[("E2-Q0", seed)]
        assert entry["evaluation_job_name"] == job_name
        assert entry["evaluation_waiver"]["launcher"]["sha256"]
        assert entry["evaluation_waiver"]["sitecustomize"]["sha256"]
        assert entry["evaluation_waiver"]["frozen_source"]["sha256"]
        assert entry["evaluation_waiver"]["current_source"]["sha256"]
    for pair in bridge.EXPECTED_Q0_ATTESTATION_PAIRS - {
        ("E2-Q0", 44),
        ("E2-Q0", 45),
    }:
        assert attested[pair]["evaluation_source_mode"] == ("frozen_registered_source")
        assert attested[pair]["evaluation_waiver"] is None


def test_bridge_requires_exact_matrix_and_rejects_fresh_q1_path(monkeypatch, tmp_path):
    root, manifest_path, specs, _paths = _matrix(tmp_path, monkeypatch)
    with pytest.raises(bridge.SummaryError, match="exact 4x3"):
        _build(root, manifest_path, specs[:-1])
    with pytest.raises(bridge.SummaryError, match="duplicate"):
        _build(root, manifest_path, specs + [specs[0]])

    fresh = (
        root / "output/selfless-flow-image-embedder-qf-e2b-q1-seed43/"
        "fid_is_selected_cfg3p5_ema/metrics.json"
    )
    _write_json(fresh, {})
    replaced = list(specs)
    replaced[0] = f"E2b-Q1@43={fresh}"
    with pytest.raises(bridge.SummaryError, match="must be exactly"):
        _build(root, manifest_path, replaced)

    (root / bridge.Q0_ATTESTATION_MANIFEST_RELATIVE_PATH).unlink()
    with pytest.raises(bridge.SummaryError, match="missing Q0 metrics attestation"):
        _build(root, manifest_path, specs)


def test_bridge_rejects_legacy_reuse_hash_and_q0_nonfinite(monkeypatch, tmp_path):
    root, manifest_path, specs, paths = _matrix(tmp_path, monkeypatch)
    legacy_path = paths[("E2b-Q1", 43)]
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_payload["strategies"]["spatial_halton"]["fid"] += 0.01
    _write_json(legacy_path, legacy_payload)
    with pytest.raises(bridge.SummaryError, match="reuse manifest"):
        _build(root, manifest_path, specs)

    root2, manifest2, specs2, paths2 = _matrix(tmp_path / "nan", monkeypatch)
    q0_path = paths2[("E2b-Q0", 43)]
    q0_payload = json.loads(q0_path.read_text(encoding="utf-8"))
    q0_payload["strategies"]["spatial_halton"]["fid"] = float("nan")
    _write_json(q0_path, q0_payload)
    _repin_q0_metrics_entry(root2, "E2b-Q0", 43, q0_path, monkeypatch)
    with pytest.raises(bridge.SummaryError, match="finite|non-finite"):
        _build(root2, manifest2, specs2)

    q0_payload["strategies"]["spatial_halton"]["fid"] = 25.21
    _write_json(q0_path, q0_payload)
    with pytest.raises(bridge.SummaryError, match="Q0 attestation"):
        _build(root2, manifest2, specs2)


def test_q0_attestation_rejects_wrong_path_seed_extra_and_fresh(monkeypatch, tmp_path):
    root, manifest_path, specs, _paths = _matrix(tmp_path, monkeypatch)
    attestation_path = root / bridge.Q0_ATTESTATION_MANIFEST_RELATIVE_PATH
    original = json.loads(attestation_path.read_text(encoding="utf-8"))

    repinned_only_by_self_digest = json.loads(json.dumps(original))
    repinned_only_by_self_digest["runs"][0]["metrics_path"] = "output/fresh.json"
    body = dict(repinned_only_by_self_digest)
    body.pop("manifest_sha256")
    repinned_only_by_self_digest["manifest_sha256"] = canonical_sha256(body)
    _write_json(attestation_path, repinned_only_by_self_digest)
    with pytest.raises(bridge.SummaryError, match="raw digest differs from pin"):
        _build(root, manifest_path, specs)

    def assert_rejected(payload, match):
        body = dict(payload)
        body.pop("manifest_sha256", None)
        payload["manifest_sha256"] = canonical_sha256(body)
        _write_json(attestation_path, payload)
        monkeypatch.setattr(
            bridge,
            "Q0_ATTESTATION_MANIFEST_RAW_SHA256",
            hashlib.sha256(attestation_path.read_bytes()).hexdigest(),
        )
        with pytest.raises(bridge.SummaryError, match=match):
            _build(root, manifest_path, specs)

    wrong_path = json.loads(json.dumps(original))
    wrong_path["runs"][0]["metrics_path"] = wrong_path["runs"][1]["metrics_path"]
    assert_rejected(wrong_path, "metrics path drifted")

    wrong_seed = json.loads(json.dumps(original))
    wrong_seed["runs"][0]["training_seed"] = 46
    assert_rejected(wrong_seed, "unexpected Q0 attestation pair")

    extra = json.loads(json.dumps(original))
    extra["runs"].append(dict(extra["runs"][0]))
    assert_rejected(extra, "exactly six runs")

    fresh = json.loads(json.dumps(original))
    fresh["runs"][0]["metrics_path"] = (
        "output/selfless-flow-image-embedder-qf-e2b-q0-fresh-seed43/"
        "fid_is_selected_cfg3p5_ema/metrics.json"
    )
    assert_rejected(fresh, "metrics path drifted")


def test_bridge_rejects_full_q0_provenance_and_seed_alignment_drift(
    monkeypatch, tmp_path
):
    root, manifest_path, specs, paths = _matrix(tmp_path, monkeypatch)
    provenance_path = (
        paths[("E2b-Q0", 43)].parents[1] / "q_factor_training_provenance.json"
    )
    original_full = json.loads(provenance_path.read_text(encoding="utf-8"))
    full = dict(original_full)
    full["training_seed"] = 99
    _write_json(provenance_path, full)
    with pytest.raises(
        bridge.SummaryError, match="full Q0 provenance|digest|provenance copies"
    ):
        _build(root, manifest_path, specs)

    _write_json(provenance_path, original_full)
    q0_metrics_path = paths[("E2b-Q0", 43)]
    checkpoint_path = q0_metrics_path.parents[1] / "checkpoint-35920/metadata.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["q_factor_provenance"]["sha256"] = _digest("wrong-checkpoint-link")
    _write_json(checkpoint_path, checkpoint)
    q0_metrics = json.loads(q0_metrics_path.read_text(encoding="utf-8"))
    q0_metrics["training_protocol"]["artifacts"]["checkpoint_metadata_sha256"] = (
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    )
    _write_json(q0_metrics_path, q0_metrics)
    _repin_q0_metrics_entry(root, "E2b-Q0", 43, q0_metrics_path, monkeypatch)
    with pytest.raises(bridge.SummaryError, match="checkpoint Q0 provenance binding"):
        _build(root, manifest_path, specs)

    root2, manifest2, specs2, _ = _matrix(
        tmp_path / "drift", monkeypatch, initial_drift=("E2b-Q0", 43)
    )
    with pytest.raises(bridge.SummaryError, match="image_state_sha256"):
        _build(root2, manifest2, specs2)


def test_bridge_requires_exact_scoped_evaluation_waiver(monkeypatch, tmp_path):
    root, manifest_path, specs, paths = _matrix(tmp_path, monkeypatch)
    sidecar = paths[("E2-Q0", 44)].parent / "evaluation_source_equivalence.json"
    sidecar.unlink()
    with pytest.raises(bridge.SummaryError, match="missing metrics waiver"):
        _build(root, manifest_path, specs)

    shutil.copyfile(root / bridge.EVALUATION_SOURCE_EQUIVALENCE_RELATIVE_PATH, sidecar)
    forbidden = paths[("E2b-Q0", 43)].parent / "evaluation_source_equivalence.json"
    shutil.copyfile(
        root / bridge.EVALUATION_SOURCE_EQUIVALENCE_RELATIVE_PATH, forbidden
    )
    with pytest.raises(bridge.SummaryError, match="must not claim"):
        _build(root, manifest_path, specs)

    root2, manifest2, specs2, paths2 = _matrix(tmp_path / "symlink", monkeypatch)
    sidecar2 = paths2[("E2-Q0", 44)].parent / "evaluation_source_equivalence.json"
    sidecar2.unlink()
    sidecar2.symlink_to(root2 / bridge.EVALUATION_SOURCE_EQUIVALENCE_RELATIVE_PATH)
    with pytest.raises(bridge.SummaryError, match="must not be a symlink"):
        _build(root2, manifest2, specs2)


def test_checked_in_reuse_and_equivalence_manifests_are_self_consistent():
    manifest = bridge.load_and_validate_legacy_reuse_manifest(
        bridge.DEFAULT_REUSE_MANIFEST, enforce_production_pin=True
    )
    report = bridge._validate_legacy_equivalence_report()
    assert len(manifest["runs"]) == 6
    assert report["claims"]["formal_q_factor"] is False
    assert report["claims"]["post_hoc_amendment"] is True
