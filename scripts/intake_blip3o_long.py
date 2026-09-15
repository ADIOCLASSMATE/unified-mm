"""Feed verified Long tars to the independent 512px service without copying originals."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import time

from data_synthesis.blip3o import archive_image_rows
from data_synthesis.config import DEFAULT_CONFIG, load_config
from data_synthesis.io import atomic_json, dumps, file_sha, pin_cohort_id
from data_synthesis.integrity import hashing_enabled


def verified_receipt(row, compute_hashes=True):
    path = Path(row["path"])
    receipt = path.with_name(path.name + ".v3-verified.json")
    if not path.is_file() or not receipt.is_file():
        return None
    value = json.loads(receipt.read_text())
    stat = path.stat()
    if (value.get("id") != row["id"] or value.get("revision") != row["revision"]
            or (compute_hashes and value.get("sha256") != row["sha256"]) or stat.st_size != row["bytes"]
            or value.get("size") != stat.st_size or value.get("mtime_ns") != stat.st_mtime_ns):
        raise ValueError("Long archive receipt does not match pinned source bytes")
    return value


def run(args):
    config = load_config(args.config)
    compute_hashes = hashing_enabled(config)
    root = args.root or Path(config["preparation_root"]) / "supply/blip3o_long_v1"
    catalogue_path = args.catalogue or Path(config["preparation_root"]) / "downloads/blip3o_long_v1/archive_catalogue.json"
    catalogue = json.loads(catalogue_path.read_text())
    raw = root / "raw_batches"
    raw.mkdir(parents=True, exist_ok=True)
    cohort = pin_cohort_id(root)
    with (root / "intake.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(root / "intake.sqlite3")
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE IF NOT EXISTS archives(id TEXT PRIMARY KEY,status TEXT,rows_done INTEGER);
        """)
        contract_data = {"batch_size": args.batch_size}
        if compute_hashes:
            contract_data["catalogue_sha256"] = file_sha(catalogue_path)
        else:
            contract_data.update(compute_hashes=False, catalogue_files=sorted(
                [[r["id"], r["revision"], r["bytes"], str(Path(r["path"]).resolve())]
                 for r in catalogue["files"]]))
        contract = dumps(contract_data)
        previous = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if previous and previous[0] != contract:
            old = json.loads(previous[0])
            ids = {r["id"] for r in catalogue["files"]}
            if (compute_hashes or "catalogue_files" in old or old.get("batch_size") != args.batch_size
                    or not {r[0] for r in db.execute("SELECT id FROM archives")} <= ids):
                raise ValueError("Long intake scope/batch size changed")
            atomic_json(root / "hash_policy_transition.json", {"previous_contract": old,
                "compute_hashes": False, "changed_at": time.time(),
                "catalogue_rehashed": False, "preserved_archive_ids": True})
            db.execute("UPDATE metadata SET value=? WHERE key='contract'", (contract,))
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract',?)", (contract,))
        db.execute("INSERT OR IGNORE INTO metadata VALUES ('serial','0')")
        db.commit()
        serial = int(db.execute("SELECT value FROM metadata WHERE key='serial'").fetchone()[0])

        def report(state):
            value = {"state": state, "batches": serial,
                     "archives": dict(db.execute("SELECT status,count(*) FROM archives GROUP BY status")),
                     "image_records": db.execute("SELECT coalesce(sum(rows_done),0) FROM archives").fetchone()[0],
                     "original_images_copied": 0, "bulk_synthesis_started": False, "updated_at": time.time(),
                     "compute_hashes": compute_hashes}
            atomic_json(root / "intake_status.json", value)
            return value

        def publish(rows, row, consumed):
            nonlocal serial
            while True:
                status = root / "prepare_status.json"
                completed = json.loads(status.read_text()).get("batches", 0) if status.exists() else 0
                if serial - completed < args.max_pending_batches:
                    break
                report("waiting_for_512_preparation")
                time.sleep(5)
            name = f"supply-{cohort}-{serial:06d}"
            path = raw / (name + ".json")
            payload = {"batch_id": name, "records": len(rows), "rows": rows,
                       "source_archive": row["id"], "source_archive_sha256": row["sha256"]}
            if path.exists():
                if json.loads(path.read_text()) != payload:
                    raise ValueError("Long raw batch changed on resume")
            else:
                atomic_json(path, payload)
            serial += 1
            db.execute("UPDATE metadata SET value=? WHERE key='serial'", (str(serial),))
            db.execute("UPDATE archives SET rows_done=? WHERE id=?", (consumed, row["id"]))
            db.commit()
            report("intaking")

        try:
            while True:
                statuses = {i: (state, n) for i, state, n in db.execute("SELECT id,status,rows_done FROM archives")}
                # Finish an interrupted archive before assigning the next serial.
                ordered = sorted(catalogue["files"], key=lambda r: (statuses.get(r["id"], ("",))[0] != "active", r["id"]))
                worked = False
                for row in ordered:
                    state, skip = statuses.get(row["id"], ("pending", 0))
                    if state == "completed":
                        continue
                    receipt = verified_receipt(row, compute_hashes=compute_hashes)
                    if receipt is None:
                        continue
                    worked = True
                    db.execute("INSERT OR IGNORE INTO archives VALUES (?,'active',0)", (row["id"],))
                    db.commit()
                    rows, consumed = [], 0
                    for consumed, entry in enumerate(archive_image_rows(row["path"], receipt,
                                                        compute_hashes=compute_hashes), 1):
                        if consumed <= skip:
                            continue
                        rows.append(entry)
                        if len(rows) == args.batch_size:
                            publish(rows, row, consumed)
                            rows = []
                    if rows:
                        publish(rows, row, consumed)
                    if consumed < skip:
                        raise ValueError("Long archive row coverage changed")
                    db.execute("UPDATE archives SET status='completed',rows_done=? WHERE id=?", (consumed, row["id"]))
                    db.commit()
                count = db.execute("SELECT count(*) FROM archives WHERE status='completed'").fetchone()[0]
                if count == len(catalogue["files"]):
                    atomic_json(root / "download.closed.json", {"batches": serial, "completed_at": time.time(),
                        "catalogue_sha256": file_sha(catalogue_path) if compute_hashes else None,
                        "compute_hashes": compute_hashes, "verified_archives": count})
                    print(dumps(report("completed")))
                    return
                report("waiting_for_verified_archives")
                if not args.watch:
                    return
                if not worked:
                    time.sleep(5)
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-pending-batches", type=int, default=64)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_pending_batches < 1:
        parser.error("batch size and backlog limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
