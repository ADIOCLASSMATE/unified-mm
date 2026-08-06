import json

import pytest
import torch
from omegaconf import OmegaConf

from models.modeling_model.modeling_selfless_flow import ImageTokenEmbedder
from utils.dataset_imagenet_flow_cache import (
    DEFAULT_CAPTION_PREFIX,
    ImageNetFlowCacheDataset,
    POSTERIOR_CACHE_FORMAT,
    POSTERIOR_STATS_LAYOUT,
    build_imagenet_flow_cache_dataloaders,
    collate_imagenet_flow_cache,
)
from utils.dataset_utils import get_dataloaders


class WordTokenizer:
    eos_token_id = 14

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = 100 + len(self.vocab)
            ids.append(self.vocab[word])
        return ids


def _write_data(tmp_path, captions=("first caption", "second caption")):
    cache = tmp_path / "latents.pt"
    manifest = tmp_path / "manifest.jsonl"
    mapping = tmp_path / "mapping.txt"
    caption_manifest = tmp_path / "captions.jsonl"
    means = torch.arange(
        len(captions) * 8,
        dtype=torch.float16,
    ).view(len(captions), 4, 2)
    torch.save(
        {
            "posterior_stats": torch.cat(
                (means, torch.zeros_like(means)), dim=-1
            ),
            "img_ids": torch.arange(1, len(captions) + 1),
            "metadata": {
                "format": POSTERIOR_CACHE_FORMAT,
                "stats_layout": POSTERIOR_STATS_LAYOUT,
            },
        },
        cache,
    )
    manifest_rows = []
    caption_rows = []
    for index, caption in enumerate(captions, start=1):
        image_id = f"n00000001_{index}"
        relative_path = f"n00000001/{image_id}.JPEG"
        manifest_rows.append(
            {
                "img_id": index,
                "source_path": f"/imagenet/train/{relative_path}",
                "synset": "n00000001",
            }
        )
        caption_rows.append(
            {
                "path": relative_path,
                "id": image_id,
                "recaption_short": caption,
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    caption_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in caption_rows),
        encoding="utf-8",
    )
    mapping.write_text("n00000001 example class, alternate\n", encoding="utf-8")
    return cache, manifest, mapping, caption_manifest


def _dataset(tmp_path, mode, **overrides):
    cache, manifest, mapping, captions = _write_data(tmp_path)
    arguments = {
        "cache_path": str(cache),
        "tokenizer": WordTokenizer(),
        "boi_token_id": 11,
        "eoi_token_id": 12,
        "mask_token_id": 13,
        "eos_token_id": 14,
        "image_tokens_per_img": 4,
        "image_latent_dim": 2,
        "manifest_jsonl": str(manifest),
        "synset_mapping_path": str(mapping),
        "conditioning_mode": mode,
    }
    if mode == "caption":
        arguments["caption_jsonl"] = str(captions)
    arguments.update(overrides)
    return ImageNetFlowCacheDataset(**arguments)


def test_frozen_latent_cache_is_explicitly_rejected(tmp_path):
    cache, manifest, mapping, _ = _write_data(tmp_path)
    torch.save(
        {
            "latents": torch.zeros((2, 4, 2), dtype=torch.float16),
            "img_ids": torch.tensor([1, 2]),
        },
        cache,
    )
    with pytest.raises(ValueError, match="no longer supported"):
        ImageNetFlowCacheDataset(
            cache_path=str(cache),
            tokenizer=WordTokenizer(),
            boi_token_id=11,
            eoi_token_id=12,
            mask_token_id=13,
            eos_token_id=14,
            image_tokens_per_img=4,
            image_latent_dim=2,
            manifest_jsonl=str(manifest),
            synset_mapping_path=str(mapping),
            conditioning_mode="class",
        )


def test_class_mode_serializes_class_name_only(tmp_path):
    dataset = _dataset(tmp_path, "class")
    item = dataset[0]
    tokenizer = dataset.tokenizer
    prompt = tokenizer.encode("example class")

    assert item["input_ids"][: len(prompt)].tolist() == prompt
    assert item["input_ids"][len(prompt)].item() == 11
    assert item["token_types"].eq(1).sum().item() == 4
    assert torch.count_nonzero(item["labels"] + 100) == 0
    assert dataset.sequence_cache


def test_caption_mode_serializes_fixed_prefix_and_full_caption(tmp_path):
    long_caption = " ".join(f"word{index}" for index in range(100))
    cache, manifest, mapping, captions = _write_data(
        tmp_path,
        captions=(long_caption,),
    )
    tokenizer = WordTokenizer()
    dataset = ImageNetFlowCacheDataset(
        cache_path=str(cache),
        tokenizer=tokenizer,
        boi_token_id=11,
        eoi_token_id=12,
        mask_token_id=13,
        eos_token_id=14,
        image_tokens_per_img=4,
        image_latent_dim=2,
        manifest_jsonl=str(manifest),
        synset_mapping_path=str(mapping),
        conditioning_mode="caption",
        caption_jsonl=str(captions),
        model_context_length=512,
    )
    item = dataset[0]
    expected = tokenizer.encode(f"{DEFAULT_CAPTION_PREFIX} {long_caption}")
    assert item["input_ids"][: len(expected)].tolist() == expected
    assert item["serialized_length"].item() == len(expected) + 7
    assert dataset.sequence_cache == {}


def test_caption_mode_requires_complete_one_to_one_membership(tmp_path):
    cache, manifest, mapping, captions = _write_data(tmp_path)
    rows = captions.read_text(encoding="utf-8").splitlines()
    captions.write_text(rows[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing 1 captions"):
        ImageNetFlowCacheDataset(
            cache_path=str(cache),
            tokenizer=WordTokenizer(),
            boi_token_id=11,
            eoi_token_id=12,
            mask_token_id=13,
            eos_token_id=14,
            image_tokens_per_img=4,
            image_latent_dim=2,
            manifest_jsonl=str(manifest),
            synset_mapping_path=str(mapping),
            conditioning_mode="caption",
            caption_jsonl=str(captions),
        )


def test_multicaption_rows_cycle_by_epoch_and_fix_validation_caption(tmp_path):
    cache, manifest, mapping, captions = _write_data(
        tmp_path,
        captions=("legacy caption",),
    )
    captions.write_text(
        json.dumps(
            {
                "path": "n00000001/n00000001_1.JPEG",
                "id": "n00000001_1",
                "recaption_short": "legacy caption",
                "captions": [
                    {"text": "original caption", "source": "original"},
                    {"text": "teacher caption alpha", "source": "api"},
                    {"text": "teacher caption beta", "source": "api"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ImageNetFlowCacheDataset(
        cache_path=str(cache),
        tokenizer=WordTokenizer(),
        boi_token_id=11,
        eoi_token_id=12,
        mask_token_id=13,
        eos_token_id=14,
        image_tokens_per_img=4,
        image_latent_dim=2,
        manifest_jsonl=str(manifest),
        synset_mapping_path=str(mapping),
        conditioning_mode="caption",
        caption_jsonl=str(captions),
        caption_list_key="captions",
    )

    dataset.set_training_indices([0])
    selected = []
    token_hashes = []
    for epoch in range(3):
        dataset.set_epoch(epoch)
        item = dataset[0]
        selected.append(item["caption_index"].item())
        token_hashes.append(item["token_ids_sha256"])
        assert item["caption_count"].item() == 3
    assert set(selected) == {0, 1, 2}
    assert len(set(token_hashes)) == 3

    dataset.set_training_indices([])
    dataset.set_epoch(0)
    validation_zero = dataset[0]
    dataset.set_epoch(11)
    validation_later = dataset[0]
    assert validation_zero["caption_index"].item() == 0
    assert validation_later["caption_index"].item() == 0
    assert (
        validation_zero["token_ids_sha256"]
        == validation_later["token_ids_sha256"]
    )


def test_multicaption_rows_reject_duplicate_text(tmp_path):
    cache, manifest, mapping, captions = _write_data(
        tmp_path,
        captions=("legacy caption",),
    )
    captions.write_text(
        json.dumps(
            {
                "path": "n00000001/n00000001_1.JPEG",
                "captions": ["same caption", {"text": "same caption"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate captions"):
        ImageNetFlowCacheDataset(
            cache_path=str(cache),
            tokenizer=WordTokenizer(),
            boi_token_id=11,
            eoi_token_id=12,
            mask_token_id=13,
            eos_token_id=14,
            image_tokens_per_img=4,
            image_latent_dim=2,
            manifest_jsonl=str(manifest),
            synset_mapping_path=str(mapping),
            conditioning_mode="caption",
            caption_jsonl=str(captions),
        )


def test_caption_img_id_must_match_manifest_path(tmp_path):
    cache, manifest, mapping, captions = _write_data(
        tmp_path,
        captions=("first caption", "second caption"),
    )
    captions.write_text(
        json.dumps(
            {
                "img_id": 1,
                "id": "n00000001_2",
                "path": "n00000001/n00000001_2.JPEG",
                "recaption_short": "misaligned caption",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="img_id/path mismatch"):
        ImageNetFlowCacheDataset(
            cache_path=str(cache),
            tokenizer=WordTokenizer(),
            boi_token_id=11,
            eoi_token_id=12,
            mask_token_id=13,
            eos_token_id=14,
            image_tokens_per_img=4,
            image_latent_dim=2,
            manifest_jsonl=str(manifest),
            synset_mapping_path=str(mapping),
            conditioning_mode="caption",
            caption_jsonl=str(captions),
        )


@pytest.mark.parametrize(
    "mode",
    ["class_image", "caption_image", "prompt_image", "image_only"],
)
def test_retired_conditioning_modes_are_rejected(tmp_path, mode):
    cache, manifest, mapping, captions = _write_data(tmp_path)
    with pytest.raises(ValueError, match="expected 'class' or 'caption'"):
        ImageNetFlowCacheDataset(
            cache_path=str(cache),
            tokenizer=WordTokenizer(),
            boi_token_id=11,
            eoi_token_id=12,
            mask_token_id=13,
            eos_token_id=14,
            image_tokens_per_img=4,
            image_latent_dim=2,
            manifest_jsonl=str(manifest),
            synset_mapping_path=str(mapping),
            conditioning_mode=mode,
            caption_jsonl=str(captions),
        )


def test_collate_places_latents_and_uses_strict_reveal_order(tmp_path):
    dataset = _dataset(tmp_path, "class")
    batch = collate_imagenet_flow_cache(
        [dataset[0], dataset[1]],
        pad_to_multiple_of=8,
    )
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    for row in range(2):
        image_positions = batch["token_types"][row] == 1
        eoi_position = (
            (batch["input_ids"][row] == 12).nonzero(as_tuple=True)[0].item()
        )
        assert torch.all(
            batch["sigma"][row, image_positions]
            > batch["sigma"][row, eoi_position]
        )
        assert torch.equal(
            batch["image_latents"][row, image_positions],
            dataset[row]["image_latents"],
        )


def test_loader_api_accepts_only_imagenet_flow_cache_dataset():
    config = OmegaConf.create(
        {"dataset": {"class_name": "TextArrowDataset"}}
    )
    with pytest.raises(ValueError, match="supports only"):
        get_dataloaders(config, WordTokenizer())


def test_train_packing_is_enabled_without_packing_validation(tmp_path):
    cache, manifest, mapping, captions = _write_data(
        tmp_path,
        captions=tuple(f"caption {index}" for index in range(8)),
    )
    config = OmegaConf.create(
        {
            "model": {
                "boi_token_id": 11,
                "eoi_token_id": 12,
                "mask_token_id": 13,
                "image_tokens_per_img": 4,
                "image_latent_dim": 2,
                "image_uncond_prob": 0.1,
            },
            "dataset": {
                "params": {
                    "cache_path": str(cache),
                    "manifest_jsonl": str(manifest),
                    "synset_mapping_path": str(mapping),
                    "conditioning_mode": "caption",
                    "caption_jsonl": str(captions),
                    "image_tokens_per_img": 4,
                    "image_latent_dim": 2,
                    "val_ratio": 0.25,
                    "split_strategy": "random",
                    "split_seed": 43,
                    "packing": {
                        "enabled": True,
                        "nominal_capacity": 64,
                    },
                },
                "preprocessing": {"max_seq_length": 4096},
            },
            "training": {
                "batch_size": 4,
                "dataloader_workers": 0,
                "seed": 43,
            },
        }
    )
    train_loader, val_loader = build_imagenet_flow_cache_dataloaders(
        config,
        WordTokenizer(),
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert "segment_ids" in train_batch
    assert "segment_ids" not in val_batch


def test_image_token_embedder_casts_inputs_to_weight_dtype():
    embedder = ImageTokenEmbedder(4, 8, image_tokens_per_img=4).to(
        dtype=torch.bfloat16
    )
    output = embedder(torch.randn(5, 4, dtype=torch.float32))
    assert output.shape == (5, 8)
    assert output.dtype == torch.bfloat16
