import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile
from types import SimpleNamespace

from PIL import Image
import pytest

from data_synthesis.blip3o import (
    EXPECTED_GROUPS, EXPECTED_TAR_BYTES, REPO_ID, REVISION,
    archive_image_rows, build_long_catalogue,
)
from data_synthesis.config import DEFAULT_CONFIG, load_config
from data_synthesis.io import cohort_id, file_sha, pin_cohort_id
from scripts.intake_blip3o_long import run as intake, verified_receipt
from scripts.supply_b512_images import canonicalize_download_paths
from utils.image_shard_io import read_image_bytes


def test_cohort_identity_survives_copy_and_compatibility_symlink(tmp_path):
    old, new = tmp_path / "global" / "supply", tmp_path / "project" / "supply"
    old.mkdir(parents=True)
    before = cohort_id(old)
    assert pin_cohort_id(old) == before
    shutil.copytree(old, new)
    shutil.rmtree(old)
    old.symlink_to(new, target_is_directory=True)
    assert cohort_id(new) == cohort_id(old) == before
    marker = new / "cohort_identity.json"
    marker.write_text('{"schema":"b512_cohort_identity_v1","id":"../bad"}')
    with pytest.raises(ValueError, match="invalid persisted"):
        cohort_id(old)


def full_metadata():
    siblings = []
    for prefix, group in [("sa_", "sa1b"), ("webdataset_shard_", "cc12m"), ("webdataset_JDB_", "journeydb")]:
        siblings.extend({"rfilename": f"{prefix}{i}.tar", "size": 1,
                         "lfs": {"size": 1, "sha256": "a" * 64}} for i in range(EXPECTED_GROUPS[group]))
    size = EXPECTED_TAR_BYTES - len(siblings) + 1
    siblings[-1]["size"] = siblings[-1]["lfs"]["size"] = size
    return {"id": REPO_ID, "sha": REVISION, "siblings": siblings}


def test_long_catalogue_rejects_partial_scope_or_changed_identity(tmp_path):
    metadata = full_metadata()
    result = build_long_catalogue(metadata, tmp_path)
    assert len(result["files"]) == 2891 and result["declared_bytes"] == EXPECTED_TAR_BYTES
    assert all(Path(r["path"]).is_relative_to(tmp_path) and REVISION in r["url"] for r in result["files"])
    metadata["siblings"].pop()
    with pytest.raises(ValueError, match="complete pinned"):
        build_long_catalogue(metadata, tmp_path)
    metadata["sha"] = "main"
    with pytest.raises(ValueError, match="revision changed"):
        build_long_catalogue(metadata, tmp_path)


def archive_fixture(tmp_path):
    path = tmp_path / "sa_000000.tar"
    image = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(image, format="PNG")
    with tarfile.open(path, "w") as handle:
        for name, data in [("item.txt", b"A red square."), ("item.png", image.getvalue())]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
    digest = file_sha(path)
    row = {"id": "blip3o_long:sa_000000.tar", "path": str(path), "bytes": path.stat().st_size,
           "revision": REVISION, "sha256": digest}
    receipt = {"id": row["id"], "revision": REVISION, "size": row["bytes"],
               "mtime_ns": path.stat().st_mtime_ns, "sha256": digest}
    path.with_name(path.name + ".v3-verified.json").write_text(json.dumps(receipt))
    return path, row, receipt, image.getvalue()


def test_tar_intake_preserves_image_text_pair_without_image_copy_and_resumes(tmp_path):
    path, row, receipt, pixels = archive_fixture(tmp_path)
    parsed = list(archive_image_rows(path, receipt))
    assert len(parsed) == 1
    assert read_image_bytes(parsed[0]["reference"]) == pixels
    assert parsed[0]["row"]["caption_candidates"][0]["text"] == "A red square."
    assert parsed[0]["key"] == hashlib.sha256(b"blip3o_long:sa_000000.tar/item").hexdigest()
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(json.dumps({"files": [row]}))
    args = SimpleNamespace(config=DEFAULT_CONFIG, root=tmp_path / "supply", catalogue=catalogue,
                           batch_size=256, max_pending_batches=64, watch=False)
    intake(args)
    first = list((args.root / "raw_batches").glob("*.json"))
    before = {p.name: p.read_bytes() for p in first}
    intake(args)
    assert {p.name: p.read_bytes() for p in (args.root / "raw_batches").glob("*.json")} == before
    assert json.loads((args.root / "download.closed.json").read_text())["batches"] == 1
    assert not (args.root / "images").exists()
    row["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="pinned source bytes"):
        verified_receipt(row)


def test_runtime_keeps_image_text_state_on_project_and_tensor_cache_separate():
    config = load_config()
    project = Path(config["project_root"])
    for name in ("image_root", "text_root", "preparation_root", "state_root"):
        assert Path(config[name]).is_relative_to(project)
    assert not Path(config["posterior_root"]).is_relative_to(project)
    assert config["export_shard_records"] * 256 >= 38881167


def test_relocated_download_does_not_double_count_imported_batches(tmp_path):
    old, new = tmp_path / "global", tmp_path / "project"
    new.mkdir()
    old.symlink_to(new, target_is_directory=True)
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE TABLE files(path TEXT PRIMARY KEY,sha TEXT);"
                    "CREATE TABLE recovery_imports(path TEXT PRIMARY KEY,sha256 TEXT);")
    legacy = str(old / "candidates/a.jsonl")
    canonical = str(new / "candidates/a.jsonl")
    db.executemany("INSERT INTO files VALUES (?,?)", [(legacy, "digest"), (canonical, "digest")])
    db.execute("INSERT INTO recovery_imports VALUES (?,?)", (str(old / "recovery_inbox/a.json"), "recovery-digest"))
    canonicalize_download_paths(db, new)
    assert list(db.execute("SELECT * FROM files")) == [(canonical, "digest")]
    assert list(db.execute("SELECT * FROM recovery_imports")) == [(str(new / "recovery_inbox/a.json"), "recovery-digest")]
    db.execute("INSERT INTO files VALUES (?,?)", (legacy, "conflicting-digest"))
    with pytest.raises(ValueError, match="hashes conflict"):
        canonicalize_download_paths(db, new)
    db.close()
