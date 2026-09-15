import asyncio
import hashlib
import json
import sqlite3
import ssl
from types import SimpleNamespace

import anyio
import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_synthesis.io import dumps, file_sha, sha
from data_synthesis.pixmo_points import point_rows
from data_synthesis.sources import ingest_candidates, seal_candidates
from scripts import download_b512_corners_v3 as archives
from scripts.supply_b512_images import prepare_one
from scripts import supply_b512_images as supply


def point(url, digest, label):
    return {"image_url": url, "image_sha256": digest, "label": label,
            "points": [{"x": 29.583, "y": 11.834}], "count": 1, "collection_method": "counting"}


def shard(tmp_path, name, rows):
    path = tmp_path / name
    pq.write_table(pa.Table.from_pylist(rows), path)
    return {"path": str(path), "sha256": file_sha(path), "revision": "pinned-fixture",
            "upstream_path": "data/train-" + name}


def test_points_quarantine_late_cross_shard_conflict_preserving_all_annotations(tmp_path):
    good, bad = "https://example.com/good.jpg", "https://example.com/changed.jpg"
    inputs = [shard(tmp_path, "a.parquet", [point(good, "a" * 64, "cup"), point(bad, "b" * 64, "cat")]),
              shard(tmp_path, "b.parquet", [point(good, "a" * 64, "handle"), point(bad, "c" * 64, "dog"),
                                             point(bad, "b" * 64, "cat face")])]
    audit = tmp_path / "audit"
    rows = list(point_rows(inputs, set(), audit))
    assert len(rows) == 2 and {r["url"] for r in rows} == {good}
    assert all(r["expected_source_sha256"] == "a" * 64 for r in rows)
    assert rows[0]["annotations"][0]["points"] == [{"x": 29.583, "y": 11.834}]
    assert rows[0]["annotations"][0]["verified"] is False
    assert rows[0]["annotations"][0]["coordinate_system_verified"] is False
    rejected = [json.loads(line) for line in (audit / "quarantined_annotations.jsonl").read_text().splitlines()]
    assert [r["annotation"]["label"] for r in rejected] == ["cat", "dog", "cat face"]
    report = json.loads((audit / "normalization_summary.json").read_text())
    assert report["candidate_images"] == 1 and report["quarantined_annotation_rows"] == 3
    assert report["quarantined_urls"] == 1
    manifest = tmp_path / "normalized.jsonl"
    manifest.write_text("".join(dumps(row) + "\n" for row in rows))
    supply = tmp_path / "supply"
    assert ingest_candidates(manifest, supply, "points")["records"] == 1
    assert seal_candidates(supply)["records"] == 1
    joined = json.loads((supply / "candidates/points-00000.jsonl").read_text())
    assert {a["label"] for a in joined["annotations"]} == {"cup", "handle"}
    changed = tmp_path / "changed.jpg"
    changed.write_bytes(b"wrong version")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_one(str(changed), joined)


def test_points_respects_exclusions_and_rejects_changed_metadata(tmp_path):
    url = "https://example.com/excluded.jpg"
    obj = shard(tmp_path, "a.parquet", [point(url, "a" * 64, "cup")])
    assert list(point_rows([obj], {"pixmo_points:" + sha(url.encode())}, tmp_path / "audit")) == []
    obj["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="metadata SHA256"):
        list(point_rows([obj], set(), tmp_path / "changed-audit"))


class InterruptedStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"abc"
        raise anyio.EndOfStream


def archive_row(tmp_path):
    return {"id": "fixture:one", "revision": "pinned-fixture", "path": str(tmp_path / "archive.bin"),
            "bytes": 6, "sha256": hashlib.sha256(b"abcdef").hexdigest(), "url": "https://example.com/archive"}


async def no_delay(_):
    pass


def test_image_download_starts_before_full_annotation_corpus_is_imported(tmp_path, monkeypatch):
    root = tmp_path / "supply"
    candidates = root / "candidates"
    candidates.mkdir(parents=True)
    for i in range(5):
        row = {"source": "fixture", "source_id": str(i), "split": "train",
               "url": f"https://example.com/{i}.jpg", "capabilities": ["counting"]}
        (candidates / f"{i}.jsonl").write_text(dumps(row) + "\n")
    seal_candidates(root)
    observations = []

    class StopProbe(BaseException):
        pass

    class Downloader:
        def __init__(self, *_):
            self.host_retry_at = {}

        async def get(self, row):
            with sqlite3.connect(root / "download.sqlite3") as db:
                observations.append(db.execute("SELECT count(*) FROM files").fetchone()[0])
            raise StopProbe

        async def close(self):
            pass

    monkeypatch.setattr(supply, "DirectDownloader", Downloader)
    monkeypatch.setattr(supply, "check_direct_routes", lambda: None)
    args = SimpleNamespace(root=root, image_root=tmp_path / "images", workers=1, per_host=1)
    with pytest.raises(StopProbe):
        asyncio.run(supply.download_service(args))
    assert len(observations) == 1 and 0 < observations[0] < 5


def test_archive_interrupted_stream_resumes_exact_bytes_and_verifies(tmp_path, monkeypatch):
    requests = []
    row = archive_row(tmp_path)

    def handle(request):
        requests.append(request.headers["range"])
        if len(requests) == 1:
            return httpx.Response(206, headers={"Content-Range": "bytes 0-5/6"}, stream=InterruptedStream())
        return httpx.Response(206, headers={"Content-Range": "bytes 3-5/6"}, stream=httpx.ByteStream(b"def"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            progress = {"network_bytes": 0}
            receipt = await archives.fetch_file(client, row, 2, progress)
            assert progress["network_bytes"] == 6
            assert await archives.fetch_file(client, row, 2, progress) == receipt
            return receipt

    monkeypatch.setattr(archives.asyncio, "sleep", no_delay)
    receipt = asyncio.run(run())
    assert requests == ["bytes=0-5", "bytes=3-5"]
    assert (tmp_path / "archive.bin").read_bytes() == b"abcdef"
    assert receipt["sha256"] == row["sha256"]


@pytest.mark.parametrize("failure", ["wrong_range", "wrong_checksum", "transport"])
def test_archive_failure_never_publishes_bad_bytes_and_retries_are_bounded(tmp_path, monkeypatch, failure):
    requests = []
    row = archive_row(tmp_path)

    def handle(request):
        requests.append(request.headers["range"])
        if failure == "transport":
            raise anyio.EndOfStream
        return httpx.Response(206, headers={"Content-Range": "bytes 1-6/7" if failure == "wrong_range" else "bytes 0-5/6"},
                              stream=httpx.ByteStream(b"xxxxxx"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            await archives.fetch_file(client, row, 1, {"network_bytes": 0})

    monkeypatch.setattr(archives.asyncio, "sleep", no_delay)
    with pytest.raises(ValueError if failure == "wrong_checksum" else ExceptionGroup) as error:
        asyncio.run(run())
    if failure == "wrong_checksum":
        assert "SHA256 verification" in str(error.value)
    elif failure == "wrong_range":
        assert "exact byte range" in str(error.value.exceptions[0])
    else:
        assert isinstance(error.value.exceptions[0], anyio.EndOfStream)
    assert len(requests) == (6 if failure == "transport" else 1)
    assert not (tmp_path / "archive.bin").exists()
    assert not (tmp_path / "archive.bin.v3-verified.json").exists()


def test_archive_restart_prioritizes_failures_and_preserves_direct_transport_and_counters(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    failed = {**archive_row(tmp_path), "reuse_existing": False}
    pending = {**failed, "id": "fixture:small", "path": str(tmp_path / "small.bin"),
               "url": "https://example.com/small", "bytes": 3, "sha256": sha(b"xyz")}
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(dumps({"files": [pending, failed], "declared_bytes": 9}))
    with sqlite3.connect(root / "archive_download.sqlite3") as db:
        db.execute("CREATE TABLE objects(id TEXT PRIMARY KEY,status TEXT,receipt TEXT,error TEXT)")
        db.execute("INSERT INTO objects VALUES (?,'failed',NULL,'old timeout')", (failed["id"],))
        db.execute("INSERT INTO objects VALUES (?,'downloading',NULL,NULL)", (pending["id"],))
    (root / "archive_download_status.json").write_text(dumps({"network_bytes": 11}))
    requests, snapshots = [], []
    real_client, real_atomic = httpx.AsyncClient, archives.atomic_json

    def handle(request):
        requests.append(str(request.url))
        payload = b"abcdef" if str(request.url) == failed["url"] else b"xyz"
        return httpx.Response(206, headers={"Content-Range": f"bytes 0-{len(payload)-1}/{len(payload)}"},
                              stream=httpx.ByteStream(payload))

    def client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["proxy"] is None and kwargs["http2"] is False
        assert kwargs["verify"].check_hostname and kwargs["verify"].verify_mode == ssl.CERT_REQUIRED
        assert kwargs["verify"].minimum_version == kwargs["verify"].maximum_version == ssl.TLSVersion.TLSv1_2
        return real_client(**kwargs, transport=httpx.MockTransport(handle))

    def capture(path, value):
        if path.name == "archive_download_status.json":
            snapshots.append(value)
        real_atomic(path, value)

    monkeypatch.setattr(archives.httpx, "AsyncClient", client)
    monkeypatch.setattr(archives, "atomic_json", capture)
    monkeypatch.setattr(archives, "check_direct_routes", lambda: None)
    args = SimpleNamespace(root=root, catalogue=catalogue, command="download", files=1, ranges=1)
    asyncio.run(archives.run(args))
    assert requests == [failed["url"], pending["url"]]
    assert snapshots[0]["counts"] == {"pending": 2}
    assert snapshots[-1]["state"] == "completed" and snapshots[-1]["network_bytes"] == 20
    asyncio.run(archives.run(args))
    assert len(requests) == 2 and snapshots[-1]["network_bytes"] == 20
