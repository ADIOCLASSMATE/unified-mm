"""Check every row of a completed B512 release with actual I2T/T2I loaders."""

import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.imagenet_flow_dataloaders import _build_cache_dataset


REPO = Path(__file__).resolve().parents[1]


def audit_training_loaders(dataset, publication_audit, output, config_path):
    if not __debug__:
        raise RuntimeError("Run the loader audit without Python optimization; assertions must remain enabled")
    started = time.monotonic()
    torch.set_num_threads(1)
    release = Path(dataset).resolve()
    config_path = Path(config_path).resolve()
    config = OmegaConf.load(config_path)
    audit = json.loads(Path(publication_audit).read_text())
    assert Path(audit["dataset"]).resolve() == release, "publication audit belongs to another release"
    assert audit["posterior"]["all_rows_verified"], "full posterior audit must finish first"
    assert audit["posterior"]["source_view_hashes_verified_at_encoding"]
    assert Path(audit["posterior"]["index"]).resolve() == release / "posterior_index.json"
    assert config.model.image_tokens_per_img == 1024
    assert config.model.image_latent_dim == 16
    assert config.dataset.params.image.pad_to_length == 2048
    assert config.dataset.params.image.max_seq_length == 2048
    batch_sizes = {
        mode: int(config.dataset.params.sources[mode].micro_batch_size)
        for mode in ("i2t", "t2i")
    }
    assert all(size > 0 for size in batch_sizes.values())
    manifest_sha = hashlib.sha256((release / "manifest.jsonl").read_bytes()).hexdigest()
    assert manifest_sha == audit["manifest_sha256"]
    count = int(audit["verified_images"])
    assert count > 0
    assert audit["posterior"]["shape"] == [count, 1024, 32]
    tokenizer = AutoTokenizer.from_pretrained(
        str(REPO / config.model.model_path), local_files_only=True
    )
    datasets = {}
    stats = {}
    pending = {}
    for mode in ("i2t", "t2i"):
        params = copy.deepcopy(config.dataset.params.image)
        for key, filename in {
            "cache_path": "posterior_index.json",
            "manifest_jsonl": "manifest.jsonl",
            "caption_jsonl": "captions.jsonl",
            "synthetic_text_index_manifest": "text_index.json",
        }.items():
            params[key] = str(release / filename)
        params.expected_records = count
        params.caption_sequence_modes = [mode]
        datasets[mode] = _build_cache_dataset(config, params, tokenizer)
        datasets[mode].set_epoch(3)
        assert len(datasets[mode]) == count
        stats[mode] = {"rows": 0, "max_serialized_length": 0, "batches": 0}
        pending[mode] = []

    def flush(mode):
        rows = pending[mode]
        batch = collate_imagenet_flow_cache(rows, pad_to_length=2048, pad_to_multiple_of=64)
        assert list(batch["input_ids"].shape) == [len(rows), 2048]
        for j, item in enumerate(rows):
            length = item["input_ids"].numel()
            assert torch.equal(batch["input_ids"][j, :length], item["input_ids"])
            assert torch.equal(batch["labels"][j, :length], item["labels"])
            assert torch.equal(batch["image_loss_mask"][j, :length], item["image_loss_mask"])
            assert torch.all(batch["labels"][j, length:] == -100)
            assert not batch["image_loss_mask"][j, length:].any()
            start = int(item["image_start"])
            assert torch.equal(batch["image_latents"][j, start:start + 1024], item["image_latents"])
        stats[mode]["batches"] += 1
        pending[mode] = []

    for index in range(count):
        items = {mode: dataset[index] for mode, dataset in datasets.items()}
        assert torch.equal(items["i2t"]["image_latents"], items["t2i"]["image_latents"])
        assert items["i2t"]["posterior_seed"] == items["t2i"]["posterior_seed"]
        for mode, item in items.items():
            assert item["task_mode"] == mode
            assert int(item["img_id"]) == index + 1
            assert list(item["image_latents"].shape) == [1024, 16]
            assert torch.isfinite(item["image_latents"]).all()
            length = item["input_ids"].numel()
            assert length <= 2048
            start = int(item["image_start"])
            assert int(item["input_ids"][start - 1]) == config.model.boi_token_id
            assert int(item["input_ids"][start + 1024]) == config.model.eoi_token_id
            assert (item["labels"][start:start + 1024] == -100).all()
            if mode == "i2t":
                suffix_start = start + 1025
                assert (item["labels"][:suffix_start] == -100).all()
                assert torch.equal(item["labels"][suffix_start:], item["input_ids"][suffix_start:])
                assert int((item["labels"] != -100).sum()) == int(item["suffix_len"]) + 1
                assert not item["image_loss_mask"].any()
            else:
                assert (item["labels"] == -100).all()
                assert item["image_loss_mask"][start:start + 1024].all()
                assert int(item["image_loss_mask"].sum()) == 1024
            stats[mode]["rows"] += 1
            stats[mode]["max_serialized_length"] = max(stats[mode]["max_serialized_length"], length)
            pending[mode].append(item)
            if len(pending[mode]) == batch_sizes[mode]:
                flush(mode)
        if (index + 1) % 512 == 0:
            print(json.dumps({"verified_pairs": index + 1}), flush=True)
    for mode in datasets:
        if pending[mode]:
            flush(mode)
    old_latents = datasets["i2t"][0]["image_latents"].clone()
    for dataset in datasets.values():
        dataset.set_epoch(4)
    new_i2t, new_t2i = datasets["i2t"][0], datasets["t2i"][0]
    assert torch.equal(new_i2t["image_latents"], new_t2i["image_latents"])
    assert not torch.equal(old_latents, new_i2t["image_latents"])
    assert hashlib.sha256((release / "manifest.jsonl").read_bytes()).hexdigest() == manifest_sha
    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(release.resolve()),
        "manifest_sha256": manifest_sha,
        "training_configuration": str(config_path),
        "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "publication_audit": str(Path(publication_audit).resolve()),
        "training_config_mutated": False,
        "tokenizer": str((REPO / config.model.model_path).resolve()),
        "posterior_shape": [count, 1024, 32],
        "loaders": stats,
        "all_rows_share_same_latent_between_tasks": True,
        "all_loss_masks_and_collated_padding_verified": True,
        "epoch_resampling_checked_on_first_row": True,
        "pad_to_length": 2048,
        "micro_batch_sizes": batch_sizes,
        "includes_model_forward": False,
        "seconds": time.monotonic() - started,
    }
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--publication-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(REPO / "configs/selfless/unified_b_x0_images512_v1_ascend64.yaml"))
    args = parser.parse_args()
    audit_training_loaders(args.dataset, args.publication_audit, args.output, args.config)


if __name__ == "__main__":
    main()
