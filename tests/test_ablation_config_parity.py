from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

CONFIG_DIR = Path("configs/selfless")
BASELINE = CONFIG_DIR / "imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml"
ABLATIONS = {
    "positionwise_head": CONFIG_DIR
    / "imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_positionwise_head.yaml",
}


def _container(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _delete_path(payload: dict, path: str) -> None:
    cursor = payload
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.get(part, {})
    cursor.pop(parts[-1], None)


def test_ablation_configs_match_every_shared_baseline_setting():
    baseline = _container(BASELINE)
    common_exceptions = {
        "dataset.params.image_sigma_order",
        "evaluation.checkpoint",
        "experiment.name",
        "experiment.project",
        "model.architecture_variant",
    }
    variant_exceptions = {
        "positionwise_head": set(),
    }

    for variant, path in ABLATIONS.items():
        expected = deepcopy(baseline)
        actual = _container(path)
        for exception in common_exceptions | variant_exceptions[variant]:
            _delete_path(expected, exception)
            _delete_path(actual, exception)
        assert actual == expected
