from typing import ClassVar

import torch
from accelerate.data_loader import BatchSamplerShard
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler

from utils.dataset_imagenet_flow_cache import (
    POSTERIOR_CACHE_FORMAT,
    POSTERIOR_STATS_LAYOUT,
    ImageNetFlowCacheDataset,
    collate_imagenet_flow_cache,
)
from utils.imagenet_flow_dataloaders import (
    _build_dataset_subsets,
    training_samples_per_epoch,
)


class _Tokenizer:
    eos_token_id = 14
    _token_ids: ClassVar[dict[str, int]] = {
        "Generate": 100,
        "an": 101,
        "image": 104,
        "matching": 105,
        "this": 106,
        "description:": 107,
        "test": 102,
        "caption": 103,
    }

    def encode(self, text, add_special_tokens=False):
        return [self._token_ids[word] for word in text.split()]


def _make_dataset(tmp_path):
    means = torch.zeros((1, 4, 1), dtype=torch.float16)
    stds = torch.ones_like(means)
    cache_path = tmp_path / "posterior.pt"
    manifest_path = tmp_path / "manifest.jsonl"
    captions_path = tmp_path / "captions.jsonl"
    torch.save(
        {
            "posterior_stats": torch.cat((means, stds), dim=-1),
            "img_ids": torch.tensor([1]),
            "metadata": {
                "format": POSTERIOR_CACHE_FORMAT,
                "stats_layout": POSTERIOR_STATS_LAYOUT,
            },
        },
        cache_path,
    )
    manifest_path.write_text(
        '{"img_id": 1, "source_path": '
        '"/data/train/n00000001/n00000001_1.JPEG"}\n'
    )
    captions_path.write_text(
        '{"path": "n00000001/n00000001_1.JPEG", '
        '"recaption_short": "test caption"}\n'
    )
    return ImageNetFlowCacheDataset(
        cache_path=str(cache_path),
        tokenizer=_Tokenizer(),
        boi_token_id=11,
        eoi_token_id=12,
        mask_token_id=13,
        eos_token_id=14,
        image_tokens_per_img=4,
        image_latent_dim=1,
        manifest_jsonl=str(manifest_path),
        conditioning_mode="caption",
        caption_jsonl=str(captions_path),
        seed=2,
        max_seq_length=16,
    )


def test_epoch_updates_reach_persistent_workers_for_posterior_sampling(tmp_path):
    dataset = _make_dataset(tmp_path)
    dataset.set_training_indices([0])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )

    try:
        expected_epoch_zero = dataset[0]["image_latents"]
        epoch_zero = next(iter(loader))
        dataset.set_epoch(1)
        expected_epoch_one = dataset[0]["image_latents"]
        epoch_one = next(iter(loader))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()

    assert dataset.epoch == 1
    assert torch.equal(epoch_zero["image_latents"][0], expected_epoch_zero)
    assert torch.equal(epoch_one["image_latents"][0], expected_epoch_one)
    assert not torch.equal(expected_epoch_zero, expected_epoch_one)
    expected_generator = torch.Generator().manual_seed(
        dataset._stable_sample_seed(0, 1, "vae_posterior")
    )
    expected_noise = torch.randn(
        (4, 1), generator=expected_generator, dtype=torch.float32
    ).to(torch.float16)
    assert torch.equal(expected_epoch_one, expected_noise)
    assert epoch_zero["input_ids"][0, 0].item() == 100
    assert epoch_one["input_ids"][0, 0].item() == 100


def test_validation_posterior_sample_is_fixed_across_epochs(tmp_path):
    dataset = _make_dataset(tmp_path)
    dataset.set_training_indices([])
    epoch_zero = dataset[0]
    dataset.set_epoch(9)
    epoch_nine = dataset[0]

    assert torch.equal(epoch_zero["image_latents"], epoch_nine["image_latents"])
    assert epoch_zero["augmentation_sha256"] == epoch_nine["augmentation_sha256"]


def test_validation_reveal_order_is_fixed_across_rng_and_epochs(tmp_path):
    dataset = _make_dataset(tmp_path)
    dataset.set_training_indices([])

    torch.manual_seed(11)
    first = collate_imagenet_flow_cache([dataset[0]])
    dataset.set_epoch(9)
    torch.manual_seed(999)
    second = collate_imagenet_flow_cache([dataset[0]])

    assert torch.equal(first["sigma"], second["sigma"])


def test_overlapping_validation_view_remains_deterministic(tmp_path):
    dataset = _make_dataset(tmp_path)
    train_dataset, val_dataset = _build_dataset_subsets(
        dataset,
        train_indices=[],
        val_indices=[0],
        validation_overlap_train=True,
    )

    first_train = train_dataset[0]
    first_val = val_dataset[0]
    train_dataset.dataset.set_epoch(9)
    second_train = train_dataset[0]
    second_val = val_dataset[0]

    assert train_dataset.dataset._is_training_index(0) is True
    assert val_dataset.dataset._is_training_index(0) is False
    assert not torch.equal(
        first_train["image_latents"], second_train["image_latents"]
    )
    assert torch.equal(first_val["image_latents"], second_val["image_latents"])


def test_training_reveal_order_changes_with_epoch(tmp_path):
    dataset = _make_dataset(tmp_path)
    dataset.set_training_indices([0])

    first = collate_imagenet_flow_cache([dataset[0]])
    dataset.set_epoch(1)
    second = collate_imagenet_flow_cache([dataset[0]])

    assert not torch.equal(first["sigma"], second["sigma"])


def test_exact_epoch_budget_has_no_partial_gradient_accumulation_step():
    dataset = range(115_000)
    config = OmegaConf.create(
        {
            "training": {
                "total_batch_size": 512,
                "samples_per_epoch": 114_688,
            }
        }
    )
    budget = training_samples_per_epoch(config, len(dataset))
    sampler = RandomSampler(
        dataset,
        replacement=False,
        num_samples=budget,
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(
        dataset,
        batch_size=16,
        sampler=sampler,
        drop_last=True,
    )
    rank_batches = BatchSamplerShard(
        loader.batch_sampler,
        num_processes=16,
        process_index=0,
        split_batches=False,
        even_batches=True,
    )

    assert len(loader) == 7_168
    assert len(rank_batches) == 448
    assert len(rank_batches) % 2 == 0
    assert len(rank_batches) // 2 == 224


def test_unbounded_epoch_would_reproduce_the_partial_ga_tail():
    dataset = range(115_000)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
    )
    rank_batches = BatchSamplerShard(
        loader.batch_sampler,
        num_processes=16,
        process_index=0,
        split_batches=False,
        even_batches=True,
    )

    assert len(loader) == 7_187
    assert len(rank_batches) == 449
    assert len(rank_batches) % 2 == 1
