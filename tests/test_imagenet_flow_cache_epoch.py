import json
import struct
from typing import ClassVar

import pytest
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
from utils.imagenet_synthetic_text_index import INDEX_SCHEMA


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
        "Describe": 108,
        "in": 109,
        "one": 110,
        "detailed": 111,
        "caption:": 112,
        "synthetic": 113,
        "second": 114,
    }

    def encode(self, text, add_special_tokens=False):
        return [self._token_ids[word] for word in text.split()]


def _make_dataset(tmp_path, **overrides):
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
    arguments = dict(
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
    arguments.update(overrides)
    return ImageNetFlowCacheDataset(**arguments)


def _write_offset_jsonl(path, rows):
    offsets_path = path.with_suffix(path.suffix + ".offsets.u64")
    position = 0
    with path.open("wb") as output, offsets_path.open("wb") as offsets:
        offsets.write(struct.pack("<Q", 0))
        for row in rows:
            encoded = (json.dumps(row) + "\n").encode()
            output.write(encoded)
            position += len(encoded)
            offsets.write(struct.pack("<Q", position))
    return offsets_path


def _make_synthetic_text_index(tmp_path):
    caption_path = tmp_path / "indexed-captions.jsonl"
    caption_offsets = _write_offset_jsonl(
        caption_path,
        [
            {
                "manifest_index": 0,
                "img_id": 1,
                "path": "n00000001/n00000001_1.JPEG",
                "captions": [
                    {"source": "original", "text": "test caption"},
                    {"source": "local_qwen", "text": "synthetic caption"},
                    {"source": "api_distilled", "text": "second caption"},
                ],
            }
        ],
    )
    t2i_path = tmp_path / "indexed-t2i.jsonl"
    t2i_offsets = _write_offset_jsonl(
        t2i_path,
        [
            {
                "image_id": "train/n00000001_1",
                "model_result": {
                    "prompts": [
                        {"prompt": "test caption"},
                        {"prompt": "test synthetic"},
                    ]
                },
            }
        ],
    )
    mapping_path = tmp_path / "mapping.bi"
    mapping_path.write_bytes(struct.pack("<BI", 0, 0))
    manifest_path = tmp_path / "synthetic-index.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": INDEX_SCHEMA,
                "split": "train",
                "records": 1,
                "caption": {
                    "path": caption_path.name,
                    "offsets_path": caption_offsets.name,
                },
                "t2i": {
                    "shards": [
                        {
                            "shard_index": 0,
                            "path": t2i_path.name,
                            "offsets_path": t2i_offsets.name,
                            "records": 1,
                        }
                    ]
                },
                "mapping": {"path": mapping_path.name},
            }
        )
    )
    return caption_path, manifest_path


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


def test_joint_caption_tasks_use_disjoint_text_and_image_targets(tmp_path):
    dataset = _make_dataset(
        tmp_path,
        caption_sequence_modes=["t2i", "i2t"],
    )
    dataset.set_training_indices([0])

    first = dataset[0]
    dataset.set_epoch(1)
    second = dataset[0]
    by_task = {first["task_mode"]: first, second["task_mode"]: second}

    assert set(by_task) == {"t2i", "i2t"}
    t2i = by_task["t2i"]
    i2t = by_task["i2t"]
    assert bool((t2i["labels"] == -100).all())
    assert int(t2i["image_loss_mask"].sum()) == dataset.image_tokens_per_img
    assert not bool(i2t["image_loss_mask"].any())

    suffix_len = int(i2t["suffix_len"])
    image_start = int(i2t["image_start"])
    suffix_start = image_start + dataset.image_tokens_per_img + 1
    assert suffix_len == 2
    assert i2t["labels"][suffix_start : suffix_start + suffix_len].tolist() == [
        102,
        103,
    ]
    assert int(i2t["labels"][-1]) == _Tokenizer.eos_token_id
    assert bool((i2t["labels"][:suffix_start] == -100).all())

    i2t_batch = collate_imagenet_flow_cache([i2t])
    sigma = i2t_batch["sigma"][0, : i2t["input_ids"].numel()]
    image_end = image_start + dataset.image_tokens_per_img
    assert bool(
        (
            sigma[image_start:image_end].unsqueeze(0)
            < sigma[suffix_start : suffix_start + suffix_len].unsqueeze(1)
        ).all()
    )
    assert torch.equal(
        i2t_batch["image_loss_mask"][0, : i2t["input_ids"].numel()],
        i2t["image_loss_mask"],
    )


def test_joint_index_uses_distinct_caption_and_t2i_text_sources(tmp_path):
    caption_path, index_manifest = _make_synthetic_text_index(tmp_path)
    dataset = _make_dataset(
        tmp_path,
        caption_jsonl=str(caption_path),
        synthetic_text_index_manifest=str(index_manifest),
        caption_include_original=False,
        caption_sequence_modes=["t2i", "i2t"],
    )
    dataset.set_training_indices([0])

    observed = {"t2i": [], "i2t": []}
    for epoch in range(8):
        dataset.set_epoch(epoch)
        item = dataset[0]
        observed[item["task_mode"]].append(item)

    assert len(observed["t2i"]) == 4
    assert len(observed["i2t"]) == 4
    assert {int(item["caption_count"]) for item in observed["t2i"]} == {2}
    assert {int(item["caption_count"]) for item in observed["i2t"]} == {2}
    assert all(102 in item["input_ids"].tolist() for item in observed["t2i"])
    assert all(102 not in item["labels"].tolist() for item in observed["i2t"])
    assert {int(item["caption_index"]) for item in observed["t2i"]} == {0, 1}
    assert {int(item["caption_index"]) for item in observed["i2t"]} == {0, 1}


def test_caption_manifest_can_exclude_published_original_caption(tmp_path):
    published = tmp_path / "published.jsonl"
    published.write_text(
        '{"img_id": 1, "path": "n00000001/n00000001_1.JPEG", '
        '"captions": ['
        '{"source": "original", "caption_slot": -1, "text": "test caption"}, '
        '{"source": "local_qwen", "caption_slot": 0, "text": "synthetic caption"}, '
        '{"source": "local_qwen", "caption_slot": 1, "text": "second caption"}'
        ']}\n'
    )

    dataset = _make_dataset(
        tmp_path,
        caption_jsonl=str(published),
        caption_include_original=False,
    )

    assert dataset.captions[1] == ("synthetic caption", "second caption")
    assert int(dataset[0]["caption_count"]) == 2


def test_validation_joint_task_is_fixed_across_epochs(tmp_path):
    dataset = _make_dataset(
        tmp_path,
        caption_sequence_modes=["t2i", "i2t"],
    )
    dataset.set_training_indices([])
    first = dataset[0]
    dataset.set_epoch(7)
    second = dataset[0]

    assert first["task_mode"] == second["task_mode"]
    assert torch.equal(first["labels"], second["labels"])
    assert torch.equal(first["image_loss_mask"], second["image_loss_mask"])


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


def test_sequential_image_sigma_is_strict_and_keeps_eoi_visible(tmp_path):
    dataset = _make_dataset(tmp_path, image_sigma_order="sequential")
    item = dataset[0]
    batch = collate_imagenet_flow_cache([item])

    image_start = item["image_start"].item()
    image_end = image_start + dataset.image_tokens_per_img
    eoi_position = image_end
    sigma = batch["sigma"][0, : item["input_ids"].numel()]

    assert torch.equal(
        sigma[image_start:image_end],
        torch.arange(
            sigma[eoi_position].item() + 1,
            sigma[eoi_position].item() + 1 + dataset.image_tokens_per_img,
        ),
    )
    allowed = sigma.unsqueeze(0) < sigma.unsqueeze(1)
    assert not bool(torch.diagonal(allowed).any())
    assert bool(allowed[image_start:image_end, eoi_position].all())
    assert torch.equal(
        allowed[image_start:image_end, image_start:image_end],
        torch.tril(
            torch.ones(
                dataset.image_tokens_per_img,
                dataset.image_tokens_per_img,
                dtype=torch.bool,
            ),
            diagonal=-1,
        ),
    )


def test_image_sigma_order_rejects_unknown_strategy(tmp_path):
    with pytest.raises(ValueError, match="expected 'random' or 'sequential'"):
        _make_dataset(tmp_path, image_sigma_order="diagonal")


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
