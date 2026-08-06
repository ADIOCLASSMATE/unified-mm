import torch
from torch.utils.data import DataLoader

from utils.dataset_imagenet_flow_cache import (
    ImageNetFlowCacheDataset,
    POSTERIOR_CACHE_FORMAT,
    POSTERIOR_STATS_LAYOUT,
    collate_imagenet_flow_cache,
)


class _Tokenizer:
    eos_token_id = 14
    _token_ids = {
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


def test_training_reveal_order_changes_with_epoch(tmp_path):
    dataset = _make_dataset(tmp_path)
    dataset.set_training_indices([0])

    first = collate_imagenet_flow_cache([dataset[0]])
    dataset.set_epoch(1)
    second = collate_imagenet_flow_cache([dataset[0]])

    assert not torch.equal(first["sigma"], second["sigma"])
