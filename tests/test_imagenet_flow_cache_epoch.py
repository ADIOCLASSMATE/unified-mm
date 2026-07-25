import torch
from torch.utils.data import DataLoader

from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset


class _Tokenizer:
    eos_token_id = 14
    _token_ids = {
        "epochzero": 100,
        "epochone": 101,
        "test": 102,
        "caption": 103,
    }

    def encode(self, text, add_special_tokens=False):
        return [self._token_ids[word] for word in text.split()]


def test_epoch_updates_reach_persistent_workers_for_augmentation_and_caption(tmp_path):
    latents = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0]]],
        dtype=torch.float16,
    )
    cache_path = tmp_path / "latents.pt"
    manifest_path = tmp_path / "manifest.jsonl"
    captions_path = tmp_path / "captions.jsonl"
    torch.save({"latents": latents, "img_ids": torch.tensor([1])}, cache_path)
    manifest_path.write_text(
        '{"img_id": 1, "source_path": '
        '"/data/train/n00000001/n00000001_1.JPEG"}\n'
    )
    captions_path.write_text(
        '{"path": "n00000001/n00000001_1.JPEG", '
        '"recaption_short": "test caption"}\n'
    )

    dataset = ImageNetFlowCacheDataset(
        cache_path=str(cache_path),
        tokenizer=_Tokenizer(),
        boi_token_id=11,
        eoi_token_id=12,
        mask_token_id=13,
        eos_token_id=14,
        image_tokens_per_img=4,
        image_latent_dim=1,
        manifest_jsonl=str(manifest_path),
        conditioning_mode="caption_image",
        caption_jsonl=str(captions_path),
        caption_sequence_modes=["t2i"],
        caption_t2i_prefixes=["epochzero", "epochone"],
        latent_hflip_prob=0.5,
        seed=2,
        max_seq_length=16,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )

    try:
        epoch_zero = next(iter(loader))
        dataset.set_epoch(1)
        epoch_one = next(iter(loader))
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()

    expected_flipped = torch.tensor(
        [[2.0], [1.0], [4.0], [3.0]],
        dtype=torch.float16,
    )
    assert dataset.epoch == 1
    assert torch.equal(epoch_zero["image_latents"][0], expected_flipped)
    assert torch.equal(epoch_one["image_latents"][0], latents[0])
    assert epoch_zero["input_ids"][0, 0].item() == 100
    assert epoch_one["input_ids"][0, 0].item() == 101
