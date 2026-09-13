import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from models.modeling_model.image_backbone import validate_image_data_layout
from scripts.evaluate_single_stream_fid_is import build_canonical_initial_noise_bank
from utils.kl16_layout import kl16_layout
from utils.sharded_posterior import load_sharded_posterior


ROOT = Path(__file__).resolve().parents[1]


def test_layout_and_unified_config():
    assert kl16_layout(256) == (16, 256)
    assert kl16_layout(512) == (32, 1024)
    with pytest.raises(ValueError):
        kl16_layout(510)
    cfg = OmegaConf.load(ROOT / "configs/selfless/unified_b_x0_images512_v1_ascend64.yaml")
    assert validate_image_data_layout(cfg) == (1024, 16)
    assert cfg.dataset.params.image.max_seq_length == cfg.dataset.params.image.pad_to_length == 2048
    original = OmegaConf.load(ROOT / "configs/selfless/unified_baseline_100b_ascend_64npu.yaml")
    assert cfg.dataset.params.sources.climbmix == original.dataset.params.sources.climbmix
    cfg.dataset.params.image.image_tokens_per_img = 256
    with pytest.raises(ValueError, match="must match"):
        validate_image_data_layout(cfg)


def test_512_noise_is_partition_independent():
    full, _ = build_canonical_initial_noise_bank([0, 1, 2], evaluation_seed=42, grid_side=32)
    subset, _ = build_canonical_initial_noise_bank([2], evaluation_seed=42, grid_side=32)
    assert full.shape == (3, 1024, 16)
    torch.testing.assert_close(full[2], subset[0], rtol=0, atol=0)


def test_index_merge_retains_interleaved_512_shard_identity(tmp_path):
    checkpoint = tmp_path / "vae.ckpt"
    checkpoint.touch()
    for shard, ids in enumerate(([1, 3], [2, 4])):
        stats = torch.stack([torch.full((1024, 32), float(value), dtype=torch.float16) for value in ids])
        torch.save({"posterior_stats": stats, "img_ids": torch.tensor(ids), "metadata": {
            "format": "imagenet_kl16_scaled_posterior_v1", "stats_layout": "scaled_mean_then_scaled_std",
            "image_size": 512, "num_shards": 2, "shard_index": shard,
            "scaling_factor": 0.2325, "vae_checkpoint": str(checkpoint), "runtime_hashing_enabled": False,
        }}, tmp_path / f"shard-{shard:05d}-of-00002.pt")
    manifest = tmp_path / "images.jsonl"
    manifest.write_text("".join(json.dumps({"img_id": i}) + "\n" for i in range(1, 5)))
    result = subprocess.run([sys.executable, str(ROOT / "pretrain/merge_flow_latent_shards.py"),
        "--shard_dir", str(tmp_path), "--output_path", str(tmp_path / "index.json"),
        "--row_index_path", str(tmp_path / "image-pool" / "posterior.rows.pt"),
        "--manifest_jsonl", str(manifest), "--index_only", "--no_hash"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cache = load_sharded_posterior(tmp_path / "index.json")
    assert cache["metadata"]["image_tokens_per_img"] == 1024
    index = json.loads((tmp_path / "index.json").read_text())
    assert Path(index["row_index"]).is_absolute()
    assert not (tmp_path / "index.rows.pt").exists()
    assert cache["img_ids"].tolist() == [1, 2, 3, 4]
    for index in range(4):
        torch.testing.assert_close(cache["posterior_stats"][index], torch.full((1024, 32), index + 1., dtype=torch.float16))
    assert cache["posterior_stats"][:2].shape == (2, 1024, 32)
    shard_path = tmp_path / "shard-00000-of-00002.pt"
    changed = torch.load(shard_path, weights_only=True)
    changed["img_ids"][0] = 999
    replacement = tmp_path / "replacement.pt"
    torch.save(changed, replacement)
    replacement.replace(shard_path)
    with pytest.raises(ValueError, match="image identity changed"):
        load_sharded_posterior(tmp_path / "index.json")["posterior_stats"][0]
