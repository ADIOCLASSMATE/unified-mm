"""Freeze image-only manifests for VAE encoding independently of text synthesis."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from data_synthesis.io import atomic_json, digest_file
from data_synthesis.integrity import check_file_size, hashing_enabled, view_reference


def prepare_bank(run, output, compute_hashes=None):
    if compute_hashes is None:
        compute_hashes = hashing_enabled()
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    marker = output / "bank.json"
    if marker.exists():
        contract = json.loads(marker.read_text())
        if contract["source_run"] != str(run) or (compute_hashes and digest_file(manifest) != contract["manifest_sha256"]):
            raise ValueError("posterior bank manifest contract changed")
        check_file_size(manifest, contract.get("manifest_bytes"))
        return contract
    if manifest.exists():
        raise FileExistsError("incomplete posterior bank publication; inspect the existing manifest")
    db = sqlite3.connect(f"file:{run / 'state.sqlite3'}?mode=ro", uri=True)
    try:
        modern = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='items'").fetchone()
        if modern:
            frozen = db.execute("SELECT value FROM metadata WHERE key='frozen'").fetchone()
            if not frozen or not json.loads(frozen[0]):
                raise ValueError("freeze the image pool before encoding")
            contract_hash = hashlib.sha256(db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0].encode()).hexdigest() if compute_hashes else None
            query = "SELECT key,row_json,view_json FROM items ORDER BY key"
        else:
            contract_hash = hashlib.sha256((run / "run.json").read_bytes()).hexdigest() if compute_hashes else None
            query = "SELECT key,row,view FROM tasks WHERE view IS NOT NULL AND status NOT IN ('excluded','duplicate') ORDER BY key"
        count = 0
        temporary = manifest.with_suffix(".jsonl.tmp")
        with temporary.open("w") as handle:
            for key, raw_row, raw_view in db.execute(query):
                row, view = json.loads(raw_row), json.loads(raw_view)
                if not compute_hashes:
                    view.update(source_sha256=None, view_sha256=None, hashes_computed=False,
                                view_id=view_reference(view))
                    view.pop("perceptual_hashes", None)
                if view.get("image_size", 512) != 512:
                    raise ValueError("bank requires frozen 512px images")
                count += 1
                handle.write(json.dumps({**view, "img_id": count, "key": key,
                                         "source": row["source"], "source_id": row["source_id"],
                                         "split": "train"}) + "\n")
        if not count:
            raise ValueError("no frozen images in source run")
        contract = {"schema": "b512_frozen_image_bank_v1", "source_run": str(run),
                    "source_run_contract_sha256": contract_hash,
                    "records": count, "manifest_sha256": digest_file(temporary) if compute_hashes else None,
                    "manifest_bytes": check_file_size(temporary), "compute_hashes": compute_hashes,
                    "manifest_jsonl": str(manifest),
                    "note": "Image-only encoding manifest; labels and final publication IDs are assigned separately."}
        temporary.replace(manifest)
        atomic_json(marker, contract)
        return contract
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_bank(args.run, args.output)))


if __name__ == "__main__":
    main()
