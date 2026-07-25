"""Archived tests for the completed Q-factor protocol; not part of CI."""
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from scripts.image_embedder_confirmation_protocol import (
    EVALUATOR_RNG_CONTRACT_SHA256,
    canonical_sha256,
)
from scripts.image_mask_position_ablation_protocol import (
    PARENT_CONFIRMATION_IDS,
    Q_FACTOR_CONFIG_CONTRACT_SCHEMA,
    Q_FACTOR_IDS,
    Q_FACTOR_PHASE,
    Q_FACTOR_PROVENANCE_SCHEMA,
    Q_FACTOR_SEEDS,
    Q_FACTOR_VARIANTS,
    build_q_factor_declaration,
    build_q_factor_study_manifest,
    build_q_factor_training_provenance,
    is_q_factor_config,
    load_and_validate_q_factor_training_provenance,
    normalize_q_factor_id,
    q_factor_config_contract,
    q_factor_run_slug,
    q_factor_runtime_source_evidence,
    validate_q_factor_declaration,
    validate_q_factor_study_manifest,
    validate_runtime_source_evidence,
    write_q_factor_training_provenance,
)


SOURCE_FILES = ("runtime/a.py", "runtime/b.yaml")


class _Dataset(Dataset):
    def __init__(self, seed):
        self.img_ids = torch.tensor([101, 102, 103, 104, 105, 106])
        self.source_paths = {
            int(value): f"class/image-{int(value)}.JPEG" for value in self.img_ids
        }
        self.seed = int(seed)
        self.latent_hflip_prob = 0.5

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        return index


def _write_runtime_sources(root):
    for index, relative in enumerate(SOURCE_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n", encoding="utf-8")


def _write_parent_summary(root):
    pairing_gate = {
        "schema": "selfless_flow_image_embedder_confirmation_pairing_gate_v1",
        "validated_runs": 18,
        "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
    }
    payload = {
        "schema": "selfless_flow_image_embedder_ablation_summary_v3",
        "expected": "confirmation",
        "runs": [
            {"id": variant_id, "training_seed": seed}
            for variant_id in PARENT_CONFIRMATION_IDS
            for seed in sorted(Q_FACTOR_SEEDS)
        ],
        "aggregates": [{"id": variant_id} for variant_id in PARENT_CONFIRMATION_IDS],
        "confirmation_scope_manifest": {
            "candidate_ids": list(PARENT_CONFIRMATION_IDS),
            "required_space_to_depth_factor": 1,
        },
        "confirmation_pairing_gate": pairing_gate,
    }
    path = root / "output/image_embedder_ablation/confirmation_d1_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _config(root, variant_id="E2-Q0", seed=43):
    variant = Q_FACTOR_VARIANTS[variant_id]
    model_dir = root / "base-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text('{"model_type":"qwen3"}\n')
    inputs = {}
    for name in ("cache", "manifest", "split", "synset"):
        path = root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"evidence-{name}".encode())
        inputs[name] = str(path)
    slug = q_factor_run_slug(variant_id, seed)
    return OmegaConf.create(
        {
            "experiment": {
                "ablation_phase": Q_FACTOR_PHASE,
                "ablation_id": variant_id,
                "parent_ablation_id": variant.parent_ablation_id,
                "project": slug,
                "name": f"q-factor-{variant_id}-{seed}",
                "output_dir": "output",
            },
            "model": {
                "model_path": str(model_dir),
                "image_query_stage_mode": variant.query_stage_mode,
                "image_observed_position_mode": variant.observed_position_mode,
                "image_mask_position_mode": variant.mask_position_mode,
                "image_rope_mode": variant.rope_mode,
                "image_space_to_depth_factor": variant.space_to_depth_factor,
            },
            "training": {
                "seed": seed,
                "dataloader_shuffle_seed": seed,
                "batch_size": 2,
                "total_batch_size": 4,
                "mixed_precision": "bf16",
            },
            "dataset": {
                "params": {
                    "cache_path": inputs["cache"],
                    "manifest_jsonl": inputs["manifest"],
                    "split_manifest_jsonl": inputs["split"],
                    "synset_mapping_path": inputs["synset"],
                }
            },
            "evaluation": {
                "seed": 42,
                "checkpoint": f"output/{slug}/hf_model-final-ema",
            },
        }
    )


def _declared_config(root, variant_id="E2-Q0", seed=43):
    _write_runtime_sources(root)
    parent = _write_parent_summary(root)
    config = _config(root, variant_id, seed)
    study = build_q_factor_study_manifest(
        parent.relative_to(root),
        repo_root=root,
        source_files=SOURCE_FILES,
    )
    contract = q_factor_config_contract(config)
    declaration = build_q_factor_declaration(
        variant_id=variant_id,
        seed=seed,
        config_contract=contract,
        study_manifest=study,
        repo_root=root,
        source_files=SOURCE_FILES,
    )
    config.experiment.q_factor_protocol = declaration
    return config, parent, study, declaration


def _loader(seed):
    return DataLoader(
        Subset(_Dataset(seed), list(range(6))),
        batch_size=2,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _model():
    return SimpleNamespace(
        image_flow_head=nn.Linear(3, 4),
        image_flow_condition_proj=nn.Linear(4, 4),
        image_token_embedder=nn.Linear(2, 4),
        model=SimpleNamespace(embed_tokens=nn.Embedding(16, 4)),
    )


def test_q_factor_identity_is_an_independent_four_by_three_matrix():
    assert Q_FACTOR_IDS == ("E2b-Q1", "E2b-Q0", "E2-Q1", "E2-Q0")
    assert Q_FACTOR_SEEDS == {43, 44, 45}
    assert len(
        {
            q_factor_run_slug(variant_id, seed)
            for variant_id in Q_FACTOR_IDS
            for seed in Q_FACTOR_SEEDS
        }
    ) == 12
    for variant in Q_FACTOR_VARIANTS.values():
        assert variant.query_stage_mode == "none"
        assert variant.rope_mode == "row_col_2d"
        assert variant.space_to_depth_factor == 1
    assert Q_FACTOR_VARIANTS["E2b-Q1"].parent == "E2b"
    assert Q_FACTOR_VARIANTS["E2b-Q1"].observed == "additive_2d"
    assert Q_FACTOR_VARIANTS["E2b-Q1"].mask == "additive_2d"
    assert Q_FACTOR_VARIANTS["E2-Q0"].parent == "E2"
    assert Q_FACTOR_VARIANTS["E2-Q0"].observed == "none"
    assert Q_FACTOR_VARIANTS["E2-Q0"].mask == "none"
    assert normalize_q_factor_id("e2B-q0") == "E2b-Q0"
    with pytest.raises(ValueError, match="Unknown Q-factor ID"):
        normalize_q_factor_id("E2")


def test_config_contract_ignores_only_runtime_populated_fields(tmp_path):
    config = _config(tmp_path)
    before = q_factor_config_contract(config)
    assert before["schema"] == Q_FACTOR_CONFIG_CONTRACT_SCHEMA
    config.config = "output/image_mask_position_ablation/configs/e2-q0-seed43.yaml"
    config.experiment.output_dir = "output/expanded-run-dir"
    config.experiment.q_factor_provenance_path = "output/provenance.json"
    config.model.mask_token_id = 99
    config.model.image_offset = 200_000
    assert q_factor_config_contract(config) == before

    config.model.image_mask_position_mode = "additive_2d"
    assert q_factor_config_contract(config) != before


def test_runtime_source_manifest_detects_content_drift(tmp_path):
    _write_runtime_sources(tmp_path)
    evidence = q_factor_runtime_source_evidence(
        tmp_path, source_files=SOURCE_FILES
    )
    assert evidence["required_files"] == list(SOURCE_FILES)
    validate_runtime_source_evidence(
        evidence,
        repo_root=tmp_path,
        source_files=SOURCE_FILES,
    )

    (tmp_path / SOURCE_FILES[0]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after preregistration"):
        validate_runtime_source_evidence(
            evidence,
            repo_root=tmp_path,
            source_files=SOURCE_FILES,
        )


def test_study_and_declaration_bind_parent_source_config_and_identity(tmp_path):
    config, _, study, declaration = _declared_config(tmp_path)
    assert study["q_factor_ids"] == list(Q_FACTOR_IDS)
    assert study["training_seeds"] == [43, 44, 45]
    assert declaration["study_manifest_sha256"] == study["study_manifest_sha256"]
    assert declaration["source_manifest_sha256"] == study["runtime_source"][
        "manifest_sha256"
    ]
    assert declaration["parent_summary_sha256"] == study["parent_summary"][
        "sha256"
    ]
    validate_q_factor_study_manifest(
        study,
        repo_root=tmp_path,
        source_files=SOURCE_FILES,
    )
    validated = validate_q_factor_declaration(
        declaration,
        variant_id="E2-Q0",
        seed=43,
        config_contract=q_factor_config_contract(config),
        repo_root=tmp_path,
        source_files=SOURCE_FILES,
    )
    assert validated["declaration_sha256"] == declaration["declaration_sha256"]
    assert is_q_factor_config(config)

    with pytest.raises(ValueError, match="training seed"):
        build_q_factor_declaration(
            variant_id="E2-Q0",
            seed=42,
            config_contract=q_factor_config_contract(config),
            study_manifest=study,
            repo_root=tmp_path,
            source_files=SOURCE_FILES,
        )
    with pytest.raises(ValueError, match="resolved config"):
        validate_q_factor_declaration(
            declaration,
            variant_id="E2-Q0",
            seed=43,
            config_contract=q_factor_config_contract(_config(tmp_path, "E2-Q1", 43)),
            repo_root=tmp_path,
            source_files=SOURCE_FILES,
        )


def test_declaration_rejects_digest_parent_and_source_tampering(tmp_path):
    config, parent, _, declaration = _declared_config(tmp_path)
    tampered = json.loads(json.dumps(declaration))
    tampered["architecture"]["mask_position_mode"] = "additive_2d"
    with pytest.raises(ValueError, match="declaration digest mismatch"):
        validate_q_factor_declaration(
            tampered,
            variant_id="E2-Q0",
            seed=43,
            config_contract=q_factor_config_contract(config),
            repo_root=tmp_path,
            source_files=SOURCE_FILES,
        )

    parent.write_text(parent.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parent summary changed"):
        validate_q_factor_declaration(
            declaration,
            variant_id="E2-Q0",
            seed=43,
            config_contract=q_factor_config_contract(config),
            repo_root=tmp_path,
            source_files=SOURCE_FILES,
        )


def test_training_provenance_round_trip_and_independent_schema_validation(tmp_path):
    config, _, _, declaration = _declared_config(tmp_path)
    provenance = build_q_factor_training_provenance(
        config=config,
        model=_model(),
        train_loader=_loader(43),
        special_token_ids={"mask": 1, "boi": 2, "eoi": 3, "image_mask": 4},
        repo_root=tmp_path,
        source_files=SOURCE_FILES,
    )
    assert provenance["schema"] == Q_FACTOR_PROVENANCE_SCHEMA
    assert provenance["q_factor_id"] == "E2-Q0"
    assert provenance["q_factor_declaration_sha256"] == declaration[
        "declaration_sha256"
    ]
    assert provenance["source_manifest_sha256"] == declaration[
        "source_manifest_sha256"
    ]

    path = tmp_path / "q_factor_training_provenance.json"
    digest = write_q_factor_training_provenance(path, provenance)
    loaded = load_and_validate_q_factor_training_provenance(
        path,
        expected_sha256=digest,
        variant_id="E2-Q0",
        seed=43,
        config=config,
        repo_root=tmp_path,
        source_files=SOURCE_FILES,
    )
    assert loaded["provenance_sha256"] == digest

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["schema"] = "selfless_flow_image_embedder_confirmation_training_provenance_v1"
    tampered_without_digest = dict(tampered)
    tampered_without_digest.pop("provenance_sha256")
    tampered["provenance_sha256"] = canonical_sha256(tampered_without_digest)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="schema drifted"):
        load_and_validate_q_factor_training_provenance(
            path,
            expected_sha256=tampered["provenance_sha256"],
            variant_id="E2-Q0",
            seed=43,
            config=config,
            repo_root=tmp_path,
            source_files=SOURCE_FILES,
        )


def test_training_provenance_rejects_wrong_run_identity_and_payload_tampering(tmp_path):
    config, _, _, _ = _declared_config(tmp_path)
    provenance = build_q_factor_training_provenance(
        config=config,
        model=_model(),
        train_loader=_loader(43),
        special_token_ids={"mask": 1, "boi": 2},
        repo_root=tmp_path,
        source_files=SOURCE_FILES,
    )
    path = tmp_path / "provenance.json"
    digest = write_q_factor_training_provenance(path, provenance)
    with pytest.raises(ValueError, match="q_factor_id drifted"):
        load_and_validate_q_factor_training_provenance(
            path,
            expected_sha256=digest,
            variant_id="E2-Q1",
            seed=43,
            repo_root=tmp_path,
            source_files=SOURCE_FILES,
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_data"]["dataloader_shuffle_seed"] = 44
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance digest mismatch"):
        load_and_validate_q_factor_training_provenance(
            path,
            expected_sha256=digest,
            variant_id="E2-Q0",
            seed=43,
            repo_root=tmp_path,
            source_files=SOURCE_FILES,
        )
