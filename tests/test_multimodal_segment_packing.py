import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3Model
from utils.dataset_imagenet_flow_cache import (
    DEFAULT_CAPTION_PREFIX,
    ImageNetFlowCacheDataset,
    POSTERIOR_CACHE_FORMAT,
    POSTERIOR_STATS_LAYOUT,
    build_imagenet_flow_cache_dataloaders,
    collate_imagenet_flow_cache,
)
from utils.multimodal_segment_packing import (
    collate_segment_packed,
    deterministic_best_fit_decreasing,
    row_used_length,
)
from utils.utils import get_selfless_mask


class CharacterTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10 + ord(character) for character in text]


def _write_dataset_files(tmp_path: Path, captions: list[str]):
    cache = tmp_path / "latents.pt"
    means = torch.arange(
        len(captions) * 4 * 2, dtype=torch.float32
    ).reshape(len(captions), 4, 2)
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
    manifest = tmp_path / "manifest.jsonl"
    caption_path = tmp_path / "captions.jsonl"
    manifest_rows = []
    caption_rows = []
    for index, caption in enumerate(captions, start=1):
        image_name = f"n00000001_{index}.JPEG"
        relative_path = f"n00000001/{image_name}"
        manifest_rows.append(
            {
                "img_id": index,
                "source_path": f"/dataset/train/{relative_path}",
                "synset": "n00000001",
            }
        )
        caption_rows.append(
            {
                "path": relative_path,
                "id": Path(image_name).stem,
                "recaption_short": caption,
            }
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    caption_path.write_text(
        "".join(json.dumps(row) + "\n" for row in caption_rows),
        encoding="utf-8",
    )
    mapping = tmp_path / "synsets.txt"
    mapping.write_text("n00000001 example class\n", encoding="utf-8")
    return cache, manifest, caption_path, mapping


def _dataset(tmp_path: Path, captions: list[str]):
    cache, manifest, caption_path, mapping = _write_dataset_files(
        tmp_path, captions
    )
    return ImageNetFlowCacheDataset(
        cache_path=str(cache),
        tokenizer=CharacterTokenizer(),
        boi_token_id=3,
        eoi_token_id=4,
        mask_token_id=5,
        eos_token_id=2,
        image_tokens_per_img=4,
        image_latent_dim=2,
        manifest_jsonl=str(manifest),
        synset_mapping_path=str(mapping),
        conditioning_mode="caption",
        caption_jsonl=str(caption_path),
        model_context_length=4096,
        seed=43,
    )


def test_best_fit_decreasing_and_overflow_are_deterministic():
    lengths = [7, 6, 5, 21]
    img_ids = [40, 10, 30, 20]
    first = deterministic_best_fit_decreasing(
        lengths,
        img_ids,
        nominal_capacity=16,
    )
    second = deterministic_best_fit_decreasing(
        lengths,
        img_ids,
        nominal_capacity=16,
    )
    assert first == second
    assert sorted(index for row in first for index in row.sample_indices) == [
        0,
        1,
        2,
        3,
    ]
    assert all(
        row.overflow or row_used_length(row, lengths) <= 16 for row in first
    )
    overflow = [row for row in first if row.overflow]
    assert len(overflow) == 1
    assert overflow[0].sample_indices == (3,)
    assert overflow[0].padded_length == 32


def test_packing_rejects_non_power_of_two_row_length():
    with pytest.raises(ValueError, match="power of two"):
        deterministic_best_fit_decreasing(
            [7, 6],
            [1, 2],
            nominal_capacity=12,
        )


def test_prompt_payload_is_not_truncated_and_overflow_keeps_hash(tmp_path):
    caption = "x" * 2100
    dataset = _dataset(tmp_path, [caption])
    item = dataset[0]
    prompt = f"{DEFAULT_CAPTION_PREFIX} {caption}"
    expected_text = CharacterTokenizer().encode(prompt)
    assert item["input_ids"][: len(expected_text)].tolist() == expected_text
    assert item["serialized_length"].item() > 2048

    packed = collate_segment_packed(
        [item],
        nominal_capacity=2048,
        image_uncond_prob=0.1,
    )
    assert packed["image_count"] == 1
    assert packed["pack_details"][-1] == 1
    assert packed["sample_token_sha256"] == [item["token_ids_sha256"]]
    valid = packed["segment_ids"][0] >= 0
    assert packed["input_ids"][0, valid].tolist() == item["input_ids"].tolist()


def test_packed_mask_and_positions_are_segment_isolated(tmp_path):
    dataset = _dataset(tmp_path, ["small caption", "another caption"])
    packed = collate_segment_packed(
        [dataset[0], dataset[1]],
        nominal_capacity=256,
        image_uncond_prob=0.0,
    )
    assert packed["image_count"] == 2
    assert packed["image_span_table"].shape == (2, 5)

    segment_ids = packed["segment_ids"]
    row = int(packed["image_span_table"][0, 0].item())
    assert int(packed["image_span_table"][1, 0].item()) == row
    first_segment = int(packed["image_span_table"][0, 1].item())
    second_segment = int(packed["image_span_table"][1, 1].item())
    first_start = int((segment_ids[row] == first_segment).nonzero()[0].item())
    second_start = int(
        (segment_ids[row] == second_segment).nonzero()[0].item()
    )
    assert packed["position_ids"][:, row, first_start].tolist() == [0, 0]
    assert packed["position_ids"][:, row, second_start].tolist() == [0, 0]

    mask = get_selfless_mask(
        sigma=packed["sigma"],
        seq_len=packed["sigma"].shape[1],
        device="cpu",
        token_types=packed["token_types"],
        segment_ids=packed["segment_ids"],
        image_uncond_mask=packed["image_uncond_mask"],
    )

    def allowed(q_idx, kv_idx):
        value = mask.mask_mod(
            torch.tensor(row),
            torch.tensor(0),
            torch.tensor(q_idx),
            torch.tensor(kv_idx),
        )
        return bool(value.item() if torch.is_tensor(value) else value)

    second_image_start = int(packed["image_span_table"][1, 2].item())
    assert not allowed(second_image_start, first_start)
    assert not allowed(first_start, second_start)
    assert all(
        segment_ids[row, key].item() == second_segment
        for key in range(packed["sigma"].shape[1])
        if allowed(second_image_start, key)
    )


def test_packed_mask_matches_multimodal_visibility_truth_table(tmp_path):
    dataset = _dataset(tmp_path, ["small caption", "another caption"])
    packed = collate_segment_packed(
        [dataset[0], dataset[1]],
        nominal_capacity=256,
        image_uncond_prob=0.0,
    )
    mask = get_selfless_mask(
        sigma=packed["sigma"],
        seq_len=packed["sigma"].shape[1],
        device="cpu",
        token_types=packed["token_types"],
        segment_ids=packed["segment_ids"],
        image_uncond_mask=packed["image_uncond_mask"],
    )

    def allowed(row_idx, q_idx, kv_idx):
        value = mask.mask_mod(
            torch.tensor(row_idx),
            torch.tensor(0),
            torch.tensor(q_idx),
            torch.tensor(kv_idx),
        )
        return bool(value.item() if torch.is_tensor(value) else value)

    # Exhaustively prove the low-level rule for every real and padded token:
    # strict sigma order, same non-padding segment, and no diagonal.
    row_count, physical_length = packed["sigma"].shape
    for row_idx in range(row_count):
        for q_idx in range(physical_length):
            for kv_idx in range(physical_length):
                q_segment = int(packed["segment_ids"][row_idx, q_idx])
                kv_segment = int(packed["segment_ids"][row_idx, kv_idx])
                expected = (
                    q_segment >= 0
                    and q_segment == kv_segment
                    and int(packed["sigma"][row_idx, kv_idx])
                    < int(packed["sigma"][row_idx, q_idx])
                )
                assert allowed(row_idx, q_idx, kv_idx) is expected

    # Prove the multimodal role-level contract for every packed sample.
    for row_idx, segment_id, image_start, image_end, _ in packed[
        "image_span_table"
    ].tolist():
        segment_positions = (
            packed["segment_ids"][row_idx] == segment_id
        ).nonzero(as_tuple=True)[0].tolist()
        boi_pos = image_start - 1
        eoi_pos = image_end
        eos_pos = segment_positions[-1]
        prefix_positions = [
            position for position in segment_positions if position < boi_pos
        ]
        suffix_positions = [
            position
            for position in segment_positions
            if eoi_pos < position < eos_pos
        ]
        image_positions = list(range(image_start, image_end))

        def visible_keys(q_idx):
            return {
                kv_idx
                for kv_idx in range(physical_length)
                if allowed(row_idx, q_idx, kv_idx)
            }

        # BOI is strict autoregressive text: it sees prior text, not itself.
        assert visible_keys(boi_pos) == set(prefix_positions)
        # EOI is ordered before all image tokens in sigma space, so it cannot
        # attend to any image token even though it follows them physically.
        assert visible_keys(eoi_pos) == set(prefix_positions) | {boi_pos}
        assert not (visible_keys(eoi_pos) & set(image_positions))

        for image_pos in image_positions:
            earlier_image_positions = {
                candidate
                for candidate in image_positions
                if packed["sigma"][row_idx, candidate]
                < packed["sigma"][row_idx, image_pos]
            }
            assert visible_keys(image_pos) == (
                set(prefix_positions)
                | {boi_pos, eoi_pos}
                | earlier_image_positions
            )

        # Any future suffix text and EOS remain strict sigma-autoregressive.
        for text_pos in prefix_positions + suffix_positions + [eos_pos]:
            assert text_pos not in visible_keys(text_pos)


def test_packed_cfg_dropout_removes_only_image_text_conditioning(tmp_path):
    item = _dataset(tmp_path, ["small caption"])[0]
    packed = collate_segment_packed(
        [item],
        nominal_capacity=64,
        image_uncond_prob=1.0,
    )
    mask = get_selfless_mask(
        sigma=packed["sigma"],
        seq_len=packed["sigma"].shape[1],
        device="cpu",
        token_types=packed["token_types"],
        segment_ids=packed["segment_ids"],
        image_uncond_mask=packed["image_uncond_mask"],
    )
    row_idx, _, image_start, image_end, _ = packed[
        "image_span_table"
    ][0].tolist()

    def visible_keys(q_idx):
        keys = set()
        for kv_idx in range(packed["sigma"].shape[1]):
            value = mask.mask_mod(
                torch.tensor(row_idx),
                torch.tensor(0),
                torch.tensor(q_idx),
                torch.tensor(kv_idx),
            )
            if bool(value.item() if torch.is_tensor(value) else value):
                keys.add(kv_idx)
        return keys

    image_positions = list(range(image_start, image_end))
    assert packed["image_uncond_mask"][
        row_idx, image_start:image_end
    ].all()
    for image_pos in image_positions:
        assert visible_keys(image_pos) == {
            candidate
            for candidate in image_positions
            if packed["sigma"][row_idx, candidate]
            < packed["sigma"][row_idx, image_pos]
        }


def test_repeated_packing_reuses_identical_pack_manifest(tmp_path):
    dataset = _dataset(tmp_path, ["a", "bb", "ccc"])
    items = [dataset[index] for index in range(3)]
    first = collate_segment_packed(items, nominal_capacity=64)
    second = collate_segment_packed(items, nominal_capacity=64)
    assert first["pack_manifest_sha256"] == second["pack_manifest_sha256"]
    assert torch.equal(first["sigma"], second["sigma"])
    assert first["augmentation_sha256"] == second["augmentation_sha256"]


def test_packed_and_unpacked_batches_preserve_future_ce_labels(tmp_path):
    dataset = _dataset(tmp_path, ["a", "bb"])
    items = [dataset[0], dataset[1]]
    for item_index, item in enumerate(items):
        item["labels"] = torch.arange(
            item["input_ids"].numel(), dtype=torch.long
        ) + 1000 * item_index

    unpacked = collate_imagenet_flow_cache(items)
    for row_index, item in enumerate(items):
        length = item["labels"].numel()
        assert torch.equal(
            unpacked["labels"][row_index, :length], item["labels"]
        )

    packed = collate_segment_packed(items, nominal_capacity=64)
    for row_index, segment_id, _, _, img_id in packed[
        "image_span_table"
    ].tolist():
        item = next(
            candidate
            for candidate in items
            if int(candidate["img_id"].item()) == img_id
        )
        segment_positions = (
            packed["segment_ids"][row_index] == segment_id
        ).nonzero(as_tuple=True)[0]
        assert torch.equal(
            packed["labels"][row_index, segment_positions],
            item["labels"],
        )


def test_packing_is_train_only_and_validation_rows_stay_independent(tmp_path):
    cache, manifest, caption_path, mapping = _write_dataset_files(
        tmp_path,
        [f"caption {index}" for index in range(8)],
    )
    config = OmegaConf.create(
        {
            "model": {
                "boi_token_id": 3,
                "eoi_token_id": 4,
                "mask_token_id": 5,
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
                    "caption_jsonl": str(caption_path),
                    "model_context_length": 4096,
                    "image_tokens_per_img": 4,
                    "image_latent_dim": 2,
                    "val_ratio": 0.25,
                    "split_strategy": "random",
                    "split_seed": 43,
                    "packing": {
                        "enabled": True,
                        "algorithm": "deterministic_best_fit_decreasing",
                        "nominal_capacity": 128,
                        "overflow_policy": "dedicated_next_power_of_two",
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
        CharacterTokenizer(),
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert "segment_ids" in train_batch
    assert train_batch["image_count"] == 4
    assert train_batch["input_ids"].shape[0] < 4
    assert "segment_ids" not in val_batch
    assert "position_ids" in val_batch
    assert "image_local_positions" in val_batch
    assert val_batch["image_span_table"].shape == (2, 5)
    assert val_batch["input_ids"].shape[0] == 2
    assert (val_batch["token_types"] == 1).sum(dim=1).tolist() == [4, 4]


def test_packed_and_unpacked_backbone_hidden_states_are_equivalent(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("FlexAttention equivalence is a CUDA integration test")
    device = torch.device("cuda")
    dataset = _dataset(tmp_path, ["red object", "blue background"])
    items = [dataset[0], dataset[1]]
    packed = collate_segment_packed(items, nominal_capacity=128)
    for key in (
        "input_ids",
        "position_ids",
        "token_types",
        "sigma",
        "segment_ids",
        "image_latents",
        "image_span_table",
    ):
        packed[key] = packed[key].to(device)

    config = Qwen3Config(
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    config.mask_token_id = 5
    config.image_mask_token_id = 6
    config.image_latent_dim = 2
    config.image_tokens_per_img = 4
    config.image_input_noise_strength = 0.0
    model = Qwen3Model(config).to(device).eval()

    packed_mask = get_selfless_mask(
        sigma=packed["sigma"],
        seq_len=packed["sigma"].shape[1],
        device=device,
        token_types=packed["token_types"],
        segment_ids=packed["segment_ids"],
    )
    packed_hidden = model(
        X0_input_ids=packed["input_ids"],
        attention_mask=packed_mask,
        position_ids=packed["position_ids"],
        token_types=packed["token_types"],
        image_latents=packed["image_latents"],
        calculate_likelihood=False,
    ).last_hidden_state

    for item in items:
        img_id = int(item["img_id"].item())
        matching_spans = packed["image_span_table"][
            packed["image_span_table"][:, 4] == img_id
        ]
        assert matching_spans.shape[0] == 1
        row, segment_id, _, _, _ = matching_spans[0].tolist()
        positions = (
            packed["segment_ids"][row] == segment_id
        ).nonzero(as_tuple=True)[0]
        start = int(positions[0].item())
        end = int(positions[-1].item()) + 1
        length = int(item["input_ids"].shape[0])
        assert end - start == length

        single_input_ids = packed["input_ids"][row, start:end].unsqueeze(0)
        single_token_types = packed["token_types"][
            row, start:end
        ].unsqueeze(0)
        single_sigma = packed["sigma"][row, start:end].unsqueeze(0)
        single_position_ids = packed["position_ids"][
            :, row, start:end
        ].unsqueeze(1)
        single_latents = packed["image_latents"][
            row, start:end
        ].unsqueeze(0)
        single_mask = get_selfless_mask(
            sigma=single_sigma,
            seq_len=length,
            device=device,
            token_types=single_token_types,
        )
        single_hidden = model(
            X0_input_ids=single_input_ids,
            attention_mask=single_mask,
            position_ids=single_position_ids,
            token_types=single_token_types,
            image_latents=single_latents,
            calculate_likelihood=False,
        ).last_hidden_state
        assert torch.allclose(
            packed_hidden[row, start:end],
            single_hidden[0],
            atol=2e-5,
            rtol=2e-5,
        )
