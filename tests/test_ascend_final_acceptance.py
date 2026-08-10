from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.validate_ascend_imagenet100_assets import validate_config
from scripts.validate_ascend_imagenet100_final_run import ordered_rng_state_paths


def test_rng_shards_are_ordered_by_numeric_rank(tmp_path: Path) -> None:
    for rank in range(16):
        (tmp_path / f"random_states_{rank}.pkl").touch()

    paths = ordered_rng_state_paths(tmp_path)

    assert [path.name for path in paths] == [
        f"random_states_{rank}.pkl" for rank in range(16)
    ]


def test_rng_shards_must_cover_every_rank(tmp_path: Path) -> None:
    for rank in range(16):
        if rank != 7:
            (tmp_path / f"random_states_{rank}.pkl").touch()

    with pytest.raises(RuntimeError, match="non-contiguous RNG shard ranks"):
        ordered_rng_state_paths(tmp_path)


def _lr_experiment_config():
    config = OmegaConf.load(
        "configs/selfless/imagenet100_class_base_80ep_ascend_16npu.yaml"
    )
    config.optimizer.params.learning_rate = 2.5e-5
    config.optimizer.params.backbone_learning_rate = 5e-5
    config.optimizer.params.projector_learning_rate = 2.5e-5
    config.optimizer.params.flow_learning_rate = 2.5e-5
    config.optimizer.params.special_token_learning_rate = 5e-5
    return config


def test_validate_config_accepts_explicit_lr_experiment() -> None:
    result = validate_config(
        _lr_experiment_config(),
        16,
        Counter({"train": 115_000, "validation": 10_000}),
        expected_backbone_lr=5e-5,
        expected_flow_head_lr=2.5e-5,
    )

    assert result["backbone_learning_rate"] == 5e-5
    assert result["flow_head_learning_rate"] == 2.5e-5


def test_validate_config_rejects_inconsistent_lr_group_mapping() -> None:
    config = _lr_experiment_config()
    config.optimizer.params.projector_learning_rate = 1e-4

    with pytest.raises(RuntimeError, match="config projector_lr mismatch"):
        validate_config(
            config,
            16,
            Counter({"train": 115_000, "validation": 10_000}),
            expected_backbone_lr=5e-5,
            expected_flow_head_lr=2.5e-5,
        )
