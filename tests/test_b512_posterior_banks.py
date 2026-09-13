import hashlib
import json
from pathlib import Path
import select
import subprocess
import sys

from PIL import Image
import pytest
import torch

from scripts.compose_b512_posterior_index import compose_index
from scripts.imagenet_encode_kl16_vae import ImagePathDataset, exclusive_shard_writer
from utils.sharded_posterior import SCHEMA, load_sharded_posterior


def make_bank(root, values):
    root.mkdir()
    rows = [{"img_id": i + 1, "view_sha256": hashlib.sha256(str(v).encode()).hexdigest()}
            for i, v in enumerate(values)]
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    ids = torch.arange(1, len(values) + 1)
    stats = torch.stack([torch.full((1024, 32), float(v), dtype=torch.float16) for v in values])
    shard = root / "shard.pt"
    torch.save({"img_ids": ids, "posterior_stats": stats}, shard)
    torch.save({"img_ids": ids, "shard_rows": torch.tensor([[0, i] for i in range(len(ids))])}, root / "rows.pt")
    index = root / "index.json"
    index.write_text(json.dumps({"schema": SCHEMA, "row_index": "rows.pt", "shards": [str(shard)],
        "token_shape": [1024, 32], "metadata": {"image_size": 512, "frozen_views": True,
            "format": "imagenet_kl16_scaled_posterior_v1", "stats_layout": "scaled_mean_then_scaled_std",
            "stats_are_scaled": True, "storage_dtype": "float16",
            "source_view_hashes_verified": True, "manifest_jsonl": str(manifest),
            "manifest_sha256": digest, "source_manifest_sha256": digest,
            "vae_checkpoint_sha256": "same-checkpoint", "scaling_factor": 0.2325}}))
    return index, rows


def make_publication(root, hashes):
    root.mkdir()
    (root / "manifest.jsonl").write_text("".join(json.dumps({"img_id": i + 1, "key": str(i),
        "split": "train", "view_sha256": h}) + "\n" for i, h in enumerate(hashes)))
    (root / "publication.json").write_text(json.dumps({"records": len(hashes)}))
    return root


def test_banks_reindex_by_pixels_across_colliding_source_ids(tmp_path):
    a, a_rows = make_bank(tmp_path / "a", [10, 20])
    b, b_rows = make_bank(tmp_path / "b", [30])
    dataset = make_publication(tmp_path / "published", [a_rows[1]["view_sha256"], b_rows[0]["view_sha256"], a_rows[0]["view_sha256"]])
    output, row_map = dataset / "posterior_index.json", tmp_path / "images" / "rows.pt"
    report = compose_index(dataset, [a, b], output, row_map)
    assert report["posterior_tensors_copied"] == 0
    payload = load_sharded_posterior(output)
    assert payload["img_ids"].tolist() == [1, 2, 3]
    assert payload["posterior_stats"].storage_img_ids.tolist() == [2, 1, 1]
    for i, value in enumerate([20., 30., 10.]):
        torch.testing.assert_close(payload["posterior_stats"][i], torch.full((1024, 32), value, dtype=torch.float16))
    torch.testing.assert_close(payload["posterior_stats"][:1][0], payload["posterior_stats"][0])
    assert list(dataset.glob("*.pt")) == []
    assert json.loads(output.read_text())["metadata"]["identity_mapping"] == "frozen_view_sha256"
    shard = a.parent / "shard.pt"
    changed = torch.load(shard, weights_only=True)
    changed["img_ids"][1] = 999
    temporary = shard.with_suffix(".new")
    torch.save(changed, temporary)
    temporary.replace(shard)
    with pytest.raises(ValueError, match="image identity changed"):
        load_sharded_posterior(output)["posterior_stats"][0]


def test_composition_requires_unchanged_source_manifest_and_all_images(tmp_path):
    bank, rows = make_bank(tmp_path / "bank", [10])
    dataset = make_publication(tmp_path / "published", ["missing-view"])
    output, row_map = dataset / "posterior_index.json", tmp_path / "images" / "rows.pt"
    with pytest.raises(ValueError, match="missing posterior"):
        compose_index(dataset, [bank], output, row_map)
    assert not output.exists() and not row_map.exists()
    (bank.parent / "manifest.jsonl").write_text(json.dumps(rows[0]) + "\n\n")
    with pytest.raises(ValueError, match="source manifest changed"):
        compose_index(dataset, [bank], output, row_map)


def test_frozen_vae_input_checks_pixels_before_encoding(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (512, 512), "red").save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = ImagePathDataset([(1, path, None)], 512, True, {1: digest})
    assert tuple(dataset[0][1].shape) == (3, 512, 512)
    Image.new("RGB", (512, 512), "blue").save(path)
    with pytest.raises(ValueError, match="SHA256 changed"):
        dataset[0]


def test_inprocess_cached_bank_rejects_wrong_resolution_before_publication(tmp_path):
    from scripts.encode_b512_supply import finalize_cached_bank

    destination, work = tmp_path / "cache", tmp_path / "work"
    (destination / "shards").mkdir(parents=True)
    work.mkdir()
    image = tmp_path / "image.png"
    Image.new("RGB", (512, 512), "red").save(image)
    manifest = work / "manifest.jsonl"
    manifest.write_text(json.dumps({"img_id": 1, "source_path": str(image)}) + "\n")
    bank = {"records": 1, "manifest_jsonl": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    assert not finalize_cached_bank(bank, destination, work, ("checkpoint", "module"))
    torch.save({"img_ids": torch.tensor([1]), "posterior_stats": torch.zeros((1, 256, 32), dtype=torch.float16)},
               destination / "shards/shard-00000-of-00001.pt")
    with pytest.raises(RuntimeError, match="existing shard cannot be reused"):
        finalize_cached_bank(bank, destination, work, ("checkpoint", "module"))
    assert not (destination / "posterior_index.json").exists()


@pytest.mark.parametrize("fail_first_writer", [False, True])
def test_independent_encoders_wait_for_shard_and_release_on_failure(tmp_path, fail_first_writer):
    path = tmp_path / "shard.pt"
    script = """
from pathlib import Path
import sys
from scripts.imagenet_encode_kl16_vae import exclusive_shard_writer
print('ready', flush=True)
with exclusive_shard_writer(Path(sys.argv[1])):
    print('entered', flush=True)
"""
    process = None
    try:
        try:
            with exclusive_shard_writer(path):
                process = subprocess.Popen([sys.executable, "-c", script, str(path)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                assert select.select([process.stdout], [], [], 30)[0], "second encoder did not start"
                assert process.stdout.readline().strip() == "ready"
                assert not select.select([process.stdout], [], [], 0.2)[0], "two writers entered one shard"
                if fail_first_writer:
                    raise RuntimeError("first encoder failed")
        except RuntimeError:
            assert fail_first_writer
        output, errors = process.communicate(timeout=30)
        assert process.returncode == 0, errors
        assert output.strip() == "entered"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def test_helper_encoder_skips_live_writer_without_loading_vae(tmp_path):
    path = tmp_path / "shard-00000-of-00001.pt"
    with exclusive_shard_writer(path):
        result = subprocess.run([sys.executable, "scripts/imagenet_encode_kl16_vae.py",
            "--cache_shard_dir", str(tmp_path), "--skip_locked",
            "--vae_path", str(tmp_path / "deliberately_missing.ckpt")],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert "Skipped busy posterior shard:" in result.stdout
        assert not path.exists()
    result = subprocess.run([sys.executable, "scripts/imagenet_encode_kl16_vae.py",
        "--cache_shard_dir", str(tmp_path), "--skip_locked",
        "--vae_path", str(tmp_path / "deliberately_missing.ckpt")],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Missing KL16 checkpoint" in result.stderr


def test_npu_helper_requires_explicit_cpu_handoff(tmp_path):
    from argparse import Namespace
    from scripts.accelerate_b512_posteriors import run

    (tmp_path / "plan.json").write_text("{}")
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps({"state": "cpu_writers_paused", "controllers": [], "writers": []}))
    with pytest.raises(ValueError, match="pause and record CPU controllers"):
        run(Namespace(root=tmp_path, cpu_handoff=handoff))
