from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scripts.precompute_imagenet_fid_stats import load_selected_paths
from scripts.validate_ascend_imagenet1k_pretraining import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml"
)


def test_formal_imagenet1k_config_contract():
    config = OmegaConf.load(CONFIG_PATH)

    report = validate_config(config, world_size=64)

    assert report["global_batch"] == 1024
    assert report["optimizer_steps_per_epoch"] == 1251
    assert report["max_optimizer_steps"] == 1_000_800
    assert report["dropped_samples_per_epoch"] == 143
    assert report["wsd_epochs"] == {
        "warmup": 5,
        "stable": 595,
        "decay": 200,
    }
    assert report["ema_half_life_epochs"] == pytest.approx(
        5.540472,
        rel=1e-6,
    )


def test_formal_imagenet1k_config_rejects_old_ema_decay():
    config = OmegaConf.load(CONFIG_PATH)
    config.training.ema_decay = 0.999

    with pytest.raises(RuntimeError, match="ema_decay mismatch"):
        validate_config(config, world_size=64)


def test_real_stats_class_root_ignores_root_level_duplicate_files(tmp_path):
    root = tmp_path / "val"
    root.mkdir()
    (root / "duplicate.JPEG").touch()
    for synset in ("n00000001", "n00000002"):
        class_dir = root / synset
        class_dir.mkdir()
        for image_index in range(2):
            (class_dir / f"{synset}_{image_index}.JPEG").touch()
    args = SimpleNamespace(
        class_image_root=str(root),
        manifest=None,
        split_manifest=None,
        imagenet_train_dir=None,
        expected_samples=4,
        expected_classes=2,
        expected_samples_per_class=2,
        split="validation",
    )

    paths, records = load_selected_paths(args)

    assert len(paths) == 4
    assert [path.parent.name for path in paths] == [
        "n00000001",
        "n00000001",
        "n00000002",
        "n00000002",
    ]
    assert {record["split"] for record in records} == {"validation"}
    assert records[0]["source_path"] == "n00000001/n00000001_0.JPEG"
