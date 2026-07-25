"""Archived tests for the historical confirmation protocol; not part of CI."""
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from scripts.image_embedder_confirmation_protocol import (
    canonical_sha256,
    initial_state_evidence,
    load_and_validate_training_provenance,
    train_data_evidence,
    tensor_sha256,
    write_training_provenance,
)


class _Dataset(Dataset):
    def __init__(self):
        self.img_ids = torch.tensor([101, 102, 103, 104, 105, 106])
        self.source_paths = {
            int(value): f"class/image-{int(value)}.JPEG" for value in self.img_ids
        }
        self.seed = 43
        self.latent_hflip_prob = 0.5

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        return index


def test_tensor_digest_supports_scalar_bfloat16_parameters():
    value = torch.tensor(1.25, dtype=torch.bfloat16)
    assert tensor_sha256(value) == tensor_sha256(value.clone())


def _config(tmp_path, seed=43):
    files = {}
    for name in ("cache", "manifest", "split", "synset"):
        path = tmp_path / name
        path.write_bytes(f"evidence-{name}".encode())
        files[name] = str(path)
    return OmegaConf.create(
        {
            "training": {
                "seed": seed,
                "dataloader_shuffle_seed": seed,
                "batch_size": 2,
                "total_batch_size": 4,
            },
            "dataset": {
                "params": {
                    "cache_path": files["cache"],
                    "manifest_jsonl": files["manifest"],
                    "split_manifest_jsonl": files["split"],
                    "synset_mapping_path": files["synset"],
                }
            },
        }
    )


def _loader(seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(_Dataset(), [0, 1, 2, 3, 4, 5]),
        batch_size=2,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )


def test_train_data_evidence_is_independent_of_global_rng_and_seed_sensitive(tmp_path):
    first = train_data_evidence(_loader(43), _config(tmp_path, 43))
    torch.manual_seed(9)
    torch.rand(1000)
    second = train_data_evidence(_loader(43), _config(tmp_path, 43))
    assert first["epoch0_ordered_sample_identity_sha256"] == second[
        "epoch0_ordered_sample_identity_sha256"
    ]
    assert first["epoch0_augmentation_decisions_sha256"] == second[
        "epoch0_augmentation_decisions_sha256"
    ]

    third_config = _config(tmp_path, 44)
    third = train_data_evidence(_loader(44), third_config)
    assert first["epoch0_ordered_sample_identity_sha256"] != third[
        "epoch0_ordered_sample_identity_sha256"
    ]


def test_initial_state_evidence_hashes_parameters_and_special_rows():
    model = SimpleNamespace(
        image_flow_head=nn.Linear(3, 4),
        image_flow_condition_proj=nn.Linear(4, 4),
        image_token_embedder=nn.Linear(2, 4),
        model=SimpleNamespace(embed_tokens=nn.Embedding(8, 4)),
    )
    first = initial_state_evidence(model, {"mask": 1, "boi": 2})
    second = initial_state_evidence(model, {"mask": 1, "boi": 2})
    assert first == second
    with torch.no_grad():
        model.image_flow_head.weight[0, 0] += 1
    changed = initial_state_evidence(model, {"mask": 1, "boi": 2})
    assert first["image_modules"]["state_sha256"] != changed["image_modules"][
        "state_sha256"
    ]


def test_training_provenance_digest_is_bound_and_tampering_is_rejected(tmp_path):
    payload = {
        "schema": "selfless_flow_image_embedder_confirmation_training_provenance_v1",
        "ablation_id": "E0",
        "training_seed": 43,
        "evidence": {"value": 1},
    }
    payload["provenance_sha256"] = canonical_sha256(payload)
    path = tmp_path / "provenance.json"
    digest = write_training_provenance(path, payload)
    loaded = load_and_validate_training_provenance(
        path,
        expected_sha256=digest,
        variant_id="E0",
        seed=43,
    )
    assert loaded["provenance_sha256"] == digest

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["evidence"]["value"] = 2
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_and_validate_training_provenance(
            path,
            expected_sha256=digest,
            variant_id="E0",
            seed=43,
        )
