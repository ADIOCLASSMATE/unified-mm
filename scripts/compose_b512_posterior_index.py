"""Join approved publication rows to pre-encoded banks by frozen image SHA256."""

import argparse
import glob
import json
from pathlib import Path
import sqlite3
import tempfile

import torch

from pretrain.merge_flow_latent_shards import POSTERIOR_CACHE_FORMAT, POSTERIOR_STATS_LAYOUT, sha256_file
from utils.sharded_posterior import SCHEMA, load_sharded_posterior


def compose_index(dataset, banks, output, row_index_path):
    dataset, output, row_index_path = map(lambda p: Path(p).resolve(), (dataset, output, row_index_path))
    if output.exists() or row_index_path.exists():
        raise FileExistsError("posterior index publication is immutable")
    if output.suffix != ".json" or row_index_path.suffix != ".pt":
        raise ValueError("use separate .json and .pt index paths")
    if row_index_path.is_relative_to(dataset):
        raise ValueError("binary row index belongs in the image pool")
    row_index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dataset / "manifest.jsonl"
    publication = json.loads((dataset / "publication.json").read_text())
    fields = ("format", "stats_layout", "stats_are_scaled", "image_size", "frozen_views",
              "vae_checkpoint_sha256", "vae_module_sha256", "scaling_factor", "vae_dtype", "storage_dtype")
    reference, paths, source_banks = None, [], []
    with tempfile.TemporaryDirectory(prefix="posterior-join-", dir=row_index_path.parent) as scratch:
        db = sqlite3.connect(Path(scratch) / "views.sqlite3")
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE views (sha TEXT PRIMARY KEY, shard INTEGER, row INTEGER, storage_id INTEGER)")
            for bank in banks:
                bank = Path(bank).resolve()
                payload = load_sharded_posterior(bank)
                meta, stats, ids = payload["metadata"], payload["posterior_stats"], payload["img_ids"]
                if (meta.get("format") != POSTERIOR_CACHE_FORMAT or meta.get("stats_layout") != POSTERIOR_STATS_LAYOUT
                        or meta.get("stats_are_scaled") is not True or meta.get("storage_dtype") != "float16"):
                    raise ValueError(f"unsupported posterior cache contract: {bank}")
                if (not meta.get("source_view_hashes_verified") or not meta.get("frozen_views")
                        or meta.get("image_size") != 512 or stats.shape[1:] != (1024, 32)):
                    raise ValueError(f"bank lacks verified frozen 512px views: {bank}")
                if reference is None:
                    reference = dict(meta)
                elif any(meta.get(field) != reference.get(field) for field in fields):
                    raise ValueError(f"incompatible VAE bank: {bank}")
                if not torch.equal(ids, torch.arange(1, len(ids) + 1)):
                    raise ValueError("bank image IDs must be contiguous from 1")
                source_manifest = Path(meta["manifest_jsonl"])
                digest = sha256_file(source_manifest)
                if digest != meta.get("manifest_sha256") or digest != meta.get("source_manifest_sha256"):
                    raise ValueError(f"bank source manifest changed: {source_manifest}")
                remap = {}
                for shard, path in enumerate(stats.paths):
                    if path not in paths:
                        paths.append(path)
                    remap[shard] = paths.index(path)
                seen = set()
                with source_manifest.open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        img_id = int(row["img_id"])
                        if img_id in seen or not 1 <= img_id <= len(ids):
                            raise ValueError("invalid bank manifest image IDs")
                        seen.add(img_id)
                        shard, position = map(int, stats.shard_rows[img_id - 1])
                        storage_id = int(stats.storage_img_ids[img_id - 1])
                        db.execute("INSERT OR IGNORE INTO views VALUES(?,?,?,?)",
                                   (row["view_sha256"], remap[shard], position, storage_id))
                if len(seen) != len(ids):
                    raise ValueError("bank manifest/cache length mismatch")
                source_banks.append({"path": str(bank), "sha256": sha256_file(bank),
                                     "manifest_sha256": digest, "records": len(ids)})
            if reference is None:
                raise ValueError("no posterior banks supplied")
            db.commit()
            global_ids, shard_rows, storage_ids = [], [], []
            with manifest.open() as handle:
                for offset, line in enumerate(handle):
                    row = json.loads(line)
                    if row["img_id"] != offset + 1 or row["split"] != "train":
                        raise ValueError("publication image IDs must be contiguous from 1")
                    match = db.execute("SELECT shard,row,storage_id FROM views WHERE sha=?",
                                       (row["view_sha256"],)).fetchone()
                    if match is None:
                        raise ValueError(f"missing posterior for published view {row['key']}")
                    global_ids.append(row["img_id"])
                    shard_rows.append(match[:2])
                    storage_ids.append(match[2])
            if len(global_ids) != publication["records"] or not global_ids:
                raise ValueError("publication/cache count mismatch")
            index = {"img_ids": torch.tensor(global_ids, dtype=torch.int64),
                     "shard_rows": torch.tensor(shard_rows, dtype=torch.int64),
                     "storage_img_ids": torch.tensor(storage_ids, dtype=torch.int64)}
            digest = sha256_file(manifest)
            metadata = {**reference, "num_images": len(global_ids), "manifest_jsonl": str(manifest),
                        "manifest_sha256": digest, "source_manifest_sha256": digest,
                        "source_banks": source_banks, "source_shards": paths,
                        "identity_mapping": "frozen_view_sha256"}
            metadata.pop("source_shard_dir", None)
            metadata.pop("source_image_root", None)
            # Validate the actual selected rows before exposing the index.
            from utils.sharded_posterior import ShardedPosterior
            selected = ShardedPosterior(paths, index["shard_rows"], [1024, 32], index["img_ids"],
                                        storage_img_ids=index["storage_img_ids"])
            # Validate in physical shard order, avoiding a mmap reopen per row
            # when source encoding used many interleaved shards.
            for i in torch.argsort(index["shard_rows"][:, 0], stable=True).tolist():
                value = selected[i]
                if not bool(torch.isfinite(value).all()) or bool((value[..., 16:] < 0).any()):
                    raise ValueError(f"invalid posterior for published image {global_ids[i]}")
            rows_temp = row_index_path.with_suffix(".pt.tmp")
            torch.save(index, rows_temp)
            output.parent.mkdir(parents=True, exist_ok=True)
            json_temp = output.with_suffix(".json.tmp")
            json_temp.write_text(json.dumps({"schema": SCHEMA, "row_index": str(row_index_path),
                                            "shards": paths, "token_shape": [1024, 32],
                                            "metadata": metadata}, indent=2) + "\n")
            rows_temp.replace(row_index_path)
            json_temp.replace(output)
            return {"records": len(global_ids), "banks": len(source_banks), "shards": len(paths),
                    "output": str(output), "row_index": str(row_index_path), "posterior_tensors_copied": 0}
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bank", required=True, action="append")
    parser.add_argument("--bank-glob", action="append", default=[], help="Resolve completed rolling bank indexes at publication time.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--row-index-path", required=True)
    args = parser.parse_args()
    banks = list(args.bank)
    for pattern in args.bank_glob:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise ValueError(f"no completed posterior bank matches {pattern}")
        banks.extend(matched)
    print(json.dumps(compose_index(args.dataset, banks, args.output, args.row_index_path)))


if __name__ == "__main__":
    main()
