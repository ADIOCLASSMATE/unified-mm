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
    assert args.batch_size == 4096
    assert evaluator.per_rank_batch_size(args.batch_size, 8) == 512


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


@pytest.mark.parametrize(
    "config_name",
    [
        "imagenet_flow_full_from_qwen3base.yaml",
        "imagenet_flow_caption_from_qwen3base.yaml",
    ],
)
def test_configs_record_h100_batch_contract(config_name):
    config = OmegaConf.load(
        REPO_ROOT / "configs" / "selfless" / config_name
    )
    assert config.evaluation.batch_size == 4096
    assert config.evaluation.batch_size_per_h100 == 512


def test_real_stats_loader_is_not_bound_to_an_ablation_split(tmp_path):
    path = tmp_path / "stats.pt"
    torch.save(
        {
            "stats": {
                "count": 10,
                "sum": torch.zeros(2, dtype=torch.float64),
                "outer_sum": torch.eye(2, dtype=torch.float64),
            },
            "metadata": {"feature": {"feature": 2}},
        },
        path,
    )
    payload = evaluator.load_shared_original_real_stats(
        str(path),
        config=OmegaConf.create({"dataset": {"params": {}}}),
        fid_feature=2,
        real_image_size=256,
        inception_weights_path="",
    )
    assert payload["stats"]["count"] == 10
