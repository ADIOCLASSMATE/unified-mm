import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from scripts import evaluate_single_stream_fid_is as evaluator


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_evaluator_default_is_4096_global(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_single_stream_fid_is.py"],
    )
    args = evaluator.parse_args()
    assert args.device == "npu"
    assert args.batch_size == 4096
    assert not hasattr(args, "allow_nonofficial_fid")
    assert not hasattr(args, "split")
    assert not hasattr(args, "is_split_assignment")
    assert evaluator.per_rank_batch_size(args.batch_size, 16) == 256


def test_global_batch_must_be_positive_and_evenly_sharded():
    with pytest.raises(ValueError, match="positive"):
        evaluator.per_rank_batch_size(0, 8)
    with pytest.raises(ValueError, match="divisible"):
        evaluator.per_rank_batch_size(4097, 8)


def test_global_batches_are_sharded_before_dataset_collation():
    samplers = [
        evaluator.GlobalBatchStrideSampler(
            samples=20,
            global_batch_size=8,
            rank=rank,
            world_size=4,
        )
        for rank in range(4)
    ]
    batches = [list(sampler) for sampler in samplers]
    assert batches[0] == [[0, 4], [8, 12], [16]]
    assert batches[3] == [[3, 7], [11, 15], [19]]
    all_indices = [
        index
        for rank_batches in batches
        for batch in rank_batches
        for index in batch
    ]
    assert sorted(all_indices) == list(range(20))
    with pytest.raises(ValueError, match="samples divisible"):
        evaluator.GlobalBatchStrideSampler(
            samples=18,
            global_batch_size=8,
            rank=0,
            world_size=4,
        )


def test_production_config_records_final_training_contract():
    config = OmegaConf.load(
        REPO_ROOT
        / "configs"
        / "selfless"
        / "imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml"
    )
    assert config.dataset.params.conditioning_mode == "class"
    assert config.training.batch_size == 16
    assert config.training.total_batch_size == 1024
    assert config.training.total_batch_size // (config.training.batch_size * 64) == 1
    assert config.training.use_gradient_checkpointing is False
    assert config.optimizer.params.backbone_learning_rate == 30e-5
    assert config.optimizer.params.special_token_learning_rate == 30e-5
    assert config.optimizer.params.flow_learning_rate == 4e-5
    assert config.optimizer.params.projector_learning_rate == 4e-5
    assert config.training.num_train_epochs == 800
    assert config.training.max_train_steps == 1_000_800
    assert config.training.ema_decay == 0.9999
    assert config.evaluation.batch_size == 4096
    assert config.evaluation.batch_size_per_npu == 256


def _write_real_stats(path, dtype=torch.float32, image_size=256):
    torch.save(
        {
            "schema": "imagenet_val_inception_feature_moments_v2",
            "stats": {
                "count": 50_000,
                "sum": torch.tensor([1.000000001, 0.0], dtype=dtype),
                "outer_sum": torch.eye(2, dtype=dtype),
            },
            "metadata": {
                "source": {
                    "split": "val",
                    "classes": 1_000,
                    "samples_per_class": 50,
                },
                "feature": {
                    "feature": 2,
                    "extractor": "torch-fidelity-inception-v3-compat",
                    "accumulation_dtype": str(dtype),
                },
                "image_transform": {
                    "resize": image_size,
                    "interpolation": "bicubic",
                    "center_crop": image_size,
                    "color_mode": "RGB",
                },
            },
        },
        path,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("image_size", [256, 512])
def test_real_stats_loader_requires_canonical_imagenet_val_contract(tmp_path, dtype, image_size):
    path = tmp_path / "stats.pt"
    _write_real_stats(path, dtype=dtype, image_size=image_size)
    payload = evaluator.load_shared_original_real_stats(
        str(path),
        fid_feature=2,
        image_size=image_size,
    )
    assert payload["stats"]["count"] == 50_000
    assert payload["stats"]["sum"].dtype == dtype
    moments = evaluator.shared_feature_moments(payload, feature=2, device="cpu")
    assert torch.equal(moments.sum, payload["stats"]["sum"].double())
    if dtype == torch.float64:
        assert moments.sum[0] > 1.0


@pytest.mark.parametrize("field", ["sum", "outer_sum"])
def test_real_stats_loader_rejects_misreported_precision(tmp_path, field):
    path = tmp_path / "stats.pt"
    _write_real_stats(path, dtype=torch.float64, image_size=512)
    payload = torch.load(path, weights_only=True)
    payload["stats"][field] = payload["stats"][field].float()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="dtype does not match"):
        evaluator.load_shared_original_real_stats(str(path), fid_feature=2, image_size=512)


def test_real_stats_loader_still_rejects_wrong_resolution(tmp_path):
    path = tmp_path / "stats.pt"
    _write_real_stats(path, dtype=torch.float64, image_size=512)
    with pytest.raises(ValueError, match="resize"):
        evaluator.load_shared_original_real_stats(str(path), fid_feature=2, image_size=256)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_stats_loader_rejects_low_precision_accumulation(tmp_path, dtype):
    path = tmp_path / "stats.pt"
    _write_real_stats(path, dtype=dtype, image_size=512)
    with pytest.raises(ValueError, match="unsupported.*accumulation_dtype"):
        evaluator.load_shared_original_real_stats(str(path), fid_feature=2, image_size=512)
