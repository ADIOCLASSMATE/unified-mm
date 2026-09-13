import argparse
import asyncio
from collections import Counter
import gzip
import json
import sqlite3

import httpx
from PIL import Image
import pytest

from scripts import recover_b512_pixmo_images as mirror
from scripts import supply_b512_images as supply
from data_synthesis.io import atomic_json, dumps, file_sha, sha


def fixture_image(tmp_path):
    pool = tmp_path / "images"
    pool.mkdir()
    image = pool / "mirror.jpg"
    Image.new("RGB", (512, 512), "blue").save(image)
    row = {"source": "pixmo_cap", "source_id": "original-id", "url": "https://i.redd.it/test.jpg",
           "capabilities": ["counting"]}
    key = sha(b"pixmo_cap:original-id")
    item = {"key": key, "source_id": row["source_id"], "original_url": row["url"],
            "candidate_sha256": sha(dumps(row).encode()), "reference": str(image), "sha256": file_sha(image),
            "mirror": {"proxy": False, "rendition": "huggingface_viewer_full_size_jpeg_or_png"}}
    return pool, row, item


def test_asset_must_match_original_metadata_url_and_mirror_revision():
    row = {"image_url": "https://i.redd.it/a.jpg", "caption": "Two blue squares.", "transcripts": ["two squares"]}
    target = {"row": {"url": row["image_url"]}, "metadata_sha256": mirror.metadata_hash(row)}
    asset = {"src": f"https://datasets-server.huggingface.co/cached-assets/{mirror.REPO}/--/{mirror.REVISION}/--/default/train/3/image/image.jpg?Expires=1",
             "width": 512, "height": 512}
    entry = {"row_idx": 3, "truncated_cells": [], "row": {**row, "image": asset}}
    assert mirror.validate_asset(entry, target) == asset
    changed = json.loads(json.dumps(entry))
    changed["row"]["caption"] = "Three squares."
    with pytest.raises(ValueError, match="metadata"):
        mirror.validate_asset(changed, target)
    changed = json.loads(json.dumps(entry))
    changed["row"]["image"]["src"] = asset["src"].replace(mirror.REVISION, "unfrozen")
    with pytest.raises(ValueError, match="revision"):
        mirror.validate_asset(changed, target)
    changed = json.loads(json.dumps(entry))
    changed["row"]["image_url"] = "https://i.redd.it/different.jpg"
    with pytest.raises(ValueError, match="URL"):
        mirror.validate_asset(changed, target)


def test_recovery_rejects_unknown_identity_changed_hash_and_outside_pool(tmp_path):
    pool, row, item = fixture_image(tmp_path)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE tasks(key,row_json,status,attempts)")
    db.execute("INSERT INTO tasks VALUES (?,?,?,?)", (item["key"], dumps(row), "failed", 7))
    manifest = tmp_path / "recovery.json"
    atomic_json(manifest, {"schema": "b512.image_recovery.v1", "rows": [item]})
    result = supply.validate_recovered_images(db, manifest, pool)
    assert result[0]["previous_attempts"] == 7
    assert result[0]["row"]["url"] == row["url"]
    for field, value, message in [("source_id", "other", "identity"), ("sha256", "bad", "hash"),
                                   ("reference", str(tmp_path / "outside.jpg"), "outside")]:
        atomic_json(manifest, {"schema": "b512.image_recovery.v1", "rows": [{**item, field: value}]})
        with pytest.raises(ValueError, match=message):
            supply.validate_recovered_images(db, manifest, pool)


def test_duplicate_url_uses_exact_metadata_and_records_alternate_response(tmp_path):
    original = {"image_url": "https://i.redd.it/a.jpg", "caption": "Two blue squares.", "transcripts": ["two squares"]}
    target = {"row": {"url": original["image_url"]}, "metadata_sha256": mirror.metadata_hash(original),
              "alternate_row_indices": [7]}
    asset = {"src": f"https://datasets-server.huggingface.co/cached-assets/{mirror.REPO}/--/{mirror.REVISION}/--/default/train/7/image/image.jpg",
             "width": 512, "height": 512}
    wrong = {"row_idx": 3, "row": {**original, "caption": "A different annotation", "image": asset}}
    correct = {"row_idx": 7, "row": {**original, "image": asset}, "truncated_cells": []}
    class Network:
        async def get(self, url, **kwargs):
            assert kwargs["params"]["offset"] == 7
            return json.dumps({"partial": False, "num_rows_total": 8, "rows": [correct]}).encode(), {}, 200
    result = asyncio.run(mirror.resolve_matching_asset(Network(), wrong, target, tmp_path / "old.json.gz", "old", 8))
    assert result[0] == asset and result[1] == 7
    assert result[2].exists() and result[3] != "old"
    import gzip
    assert sha(gzip.decompress(result[2].read_bytes())) == result[3]
    target["metadata_sha256"] = "no matching original annotation"
    with pytest.raises(ValueError, match="metadata"):
        asyncio.run(mirror.resolve_matching_asset(Network(), wrong, target, tmp_path / "old.json.gz", "old", 8))


def test_direct_range_requires_exact_206_and_client_ignores_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://invalid-proxy.example:9")
    async def exercise():
        network = mirror.DirectHTTP(2)
        assert network.client._trust_env is False
        calls = []
        async def response(url, **kwargs):
            calls.append(url)
            return b"abc", {"content-range": "bytes 0-2/9"}, 206
        network.get = response
        assert await network.range("https://example.com/file", 0, 2, 9) == b"abc"
        network.resolved_files["https://example.com/file"] = "https://cdn.example.com/signed-file"
        assert await network.range("https://example.com/file", 0, 2, 9) == b"abc"
        assert calls[-1] == "https://cdn.example.com/signed-file"
        with pytest.raises(ValueError, match="exact byte range"):
            await network.range("https://example.com/file", 1, 3, 9)
        await network.client.aclose()
    asyncio.run(exercise())


@pytest.mark.parametrize("status", [500, 503, 429])
def test_server_errors_do_not_stall_healthy_requests_but_rate_limits_do(status):
    async def exercise():
        network = mirror.DirectHTTP(2)
        await network.client.aclose()
        failed, healthy = asyncio.Event(), asyncio.Event()
        async def respond(request):
            if request.url.path == "/broken":
                failed.set()
                return httpx.Response(status, headers={"Retry-After": "60"})
            healthy.set()
            return httpx.Response(200, content=b"image bytes")
        network.client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
        broken_task = asyncio.create_task(network.get("https://example.com/broken"))
        good_task = None
        try:
            await asyncio.wait_for(failed.wait(), 1)
            good_task = asyncio.create_task(network.get("https://example.com/healthy"))
            if status == 429:
                assert network.retry_until > mirror.time.monotonic()
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(good_task, .05)
                assert not healthy.is_set()
            else:
                data, _, code = await asyncio.wait_for(good_task, 1)
                assert data == b"image bytes" and code == 200
                assert healthy.is_set() and network.retry_until == 0
        finally:
            tasks = [task for task in (broken_task, good_task) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await network.client.aclose()
    asyncio.run(exercise())


def test_request_already_queued_before_429_waits_without_holding_a_connection():
    async def exercise():
        network = mirror.DirectHTTP(1)
        await network.client.aclose()
        entered, release, healthy = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def respond(request):
            if request.url.path == "/broken":
                entered.set()
                await release.wait()
                return httpx.Response(429, headers={"Retry-After": "60"})
            healthy.set()
            return httpx.Response(200, content=b"image")
        network.client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
        tasks = [asyncio.create_task(network.get("https://example.com/broken"))]
        try:
            await asyncio.wait_for(entered.wait(), 1)
            tasks.append(asyncio.create_task(network.get("https://example.com/healthy")))
            await asyncio.sleep(.01)
            assert network.request_tasks == 2
            release.set()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(healthy.wait(), .05)
            assert not healthy.is_set()
            assert network.active_requests == 0 and network.semaphore._value == 1
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert network.request_tasks == 0 and network.active_requests == 0
            await network.client.aclose()
    asyncio.run(exercise())


def test_viewer_metadata_rate_limit_allows_image_requests_to_continue():
    async def exercise():
        network = mirror.DirectHTTP(2)
        await network.client.aclose()
        limited = asyncio.Event()
        async def respond(request):
            if request.url.path == "/rows":
                limited.set()
                return httpx.Response(429, headers={"Retry-After": "60"})
            return httpx.Response(200, content=b"full image")
        network.client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
        task = asyncio.create_task(network.get("https://datasets-server.huggingface.co/rows"))
        try:
            await asyncio.wait_for(limited.wait(), 1)
            data, _, code = await asyncio.wait_for(
                network.get("https://datasets-server.huggingface.co/cached-assets/example/image.jpg"), 1)
            assert code == 200 and data == b"full image"
            status = network.snapshot()
            assert set(status["cooldown_until"]) == {"viewer_metadata"}
            assert status["events"] == {"viewer_metadata:http_429": 1, "viewer_images:http_200": 1}
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await network.client.aclose()
    asyncio.run(exercise())


def test_metadata_requests_are_evenly_spaced_across_concurrent_pages():
    async def exercise():
        network = mirror.DirectHTTP(4, metadata_rps=50)
        await network.client.aclose()
        started = []
        async def respond(request):
            started.append(mirror.time.monotonic())
            return httpx.Response(200, content=b"{}")
        network.client = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
        try:
            await asyncio.wait_for(asyncio.gather(*(
                network.get("https://datasets-server.huggingface.co/rows", params={"offset": i})
                for i in range(4))), 2)
            assert len(started) == 4
            assert all(b - a >= .017 for a, b in zip(started, started[1:]))
            assert network.snapshot()["request_tasks"] == 0
        finally:
            await network.client.aclose()
    asyncio.run(exercise())


def test_cooldown_survives_restart_without_becoming_a_global_image_pause():
    async def exercise():
        network = mirror.DirectHTTP(2)
        network.cooldowns["viewer_metadata"] = mirror.time.monotonic() + 60
        snapshot = network.snapshot()
        replacement = mirror.DirectHTTP(2)
        try:
            replacement.restore_cooldowns(snapshot["cooldown_until"])
            assert 59 < replacement.cooldowns["viewer_metadata"] - mirror.time.monotonic() <= 60
            assert "viewer_images" not in replacement.cooldowns
        finally:
            await network.client.aclose()
            await replacement.client.aclose()
    asyncio.run(exercise())


@pytest.mark.parametrize("change", [None, "partial", "total", "index", "missing", "extra"])
def test_individual_row_fallback_keeps_exact_identity_and_response_proof(tmp_path, change):
    row = {"image_url": "https://i.redd.it/a.jpg", "caption": "Two blue squares.", "transcripts": ["two squares"]}
    target = {"row": {"url": row["image_url"]}, "metadata_sha256": mirror.metadata_hash(row)}
    asset = {"src": f"https://datasets-server.huggingface.co/cached-assets/{mirror.REPO}/--/{mirror.REVISION}/--/default/train/3/image/image.jpg",
             "width": 512, "height": 512}
    entry = {"row_idx": 3, "truncated_cells": [], "row": {**row, "image": asset}}
    payload = {"partial": False, "num_rows_total": 10, "rows": [entry]}
    if change == "partial":
        payload["partial"] = True
    elif change == "total":
        payload["num_rows_total"] = 11
    elif change == "index":
        entry["row_idx"] = 4
    elif change == "missing":
        payload["rows"] = []
    elif change == "extra":
        payload["rows"] = [entry, entry]
    data = json.dumps(payload).encode()
    class Network:
        async def get(self, url, **kwargs):
            assert kwargs["params"] == {"dataset": mirror.REPO, "config": "default", "split": "train", "offset": 3, "length": 1}
            return data, {}, 200
    if change:
        with pytest.raises(ValueError, match="individual mirror row"):
            asyncio.run(mirror.single_mirror_row(Network(), 3, 10, tmp_path))
        assert not list(tmp_path.glob("responses/*"))
    else:
        actual, path, digest = asyncio.run(mirror.single_mirror_row(Network(), 3, 10, tmp_path))
        assert mirror.validate_asset(actual, target) == asset
        assert gzip.decompress(path.read_bytes()) == data
        assert digest == sha(data)


def test_download_service_imports_recovery_once_preserving_attempt_history(tmp_path, monkeypatch):
    pool, row, item = fixture_image(tmp_path)
    root = tmp_path / "supply"
    root.mkdir()
    db = sqlite3.connect(root / "download.sqlite3")
    db.execute("""CREATE TABLE tasks(key TEXT PRIMARY KEY,row_json TEXT,status TEXT,attempts INTEGER,
                  retry_at REAL DEFAULT 0,error TEXT,reference TEXT,batch_id TEXT,family TEXT)""")
    db.execute("INSERT INTO tasks(key,row_json,status,attempts,error) VALUES (?,?,?,?,?)",
               (item["key"], dumps(row), "failed", 7, "original transport failure"))
    db.commit()
    db.close()
    (root / "candidates").mkdir()
    (root / "candidates" / "first.jsonl").write_text(dumps(row) + "\n")
    atomic_json(root / "candidates.closed.json", {"files": 1})
    atomic_json(root / "recovery_inbox" / "first.json", {"schema": "b512.image_recovery.v1", "rows": [item]})
    class NoNetwork:
        def __init__(self, *args):
            self.host_retry_at = {}
        async def get(self, row):
            raise AssertionError("already recovered image was downloaded again")
        async def close(self):
            pass
    monkeypatch.setattr(supply, "DirectDownloader", NoNetwork)
    monkeypatch.setattr(supply, "check_direct_routes", lambda: None)
    args = argparse.Namespace(root=str(root), image_root=str(pool), workers=2, per_host=1,
                              request_timeout=1, direct_dns_host=[], direct_dns_rps=2)
    asyncio.run(supply.download_service(args))
    asyncio.run(supply.download_service(args))
    db = sqlite3.connect(root / "download.sqlite3")
    assert db.execute("SELECT status,attempts,error FROM tasks").fetchone() == ("downloaded", 7, "original transport failure")
    assert db.execute("SELECT count(*) FROM recovery_imports").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM batches").fetchone()[0] == 1
    batch = json.loads(next((root / "raw_batches").glob("*.json")).read_text())
    assert batch["rows"][0]["row"]["url"] == row["url"]
    assert batch["rows"][0]["row"]["download_transport"] == "direct_huggingface_dataset_mirror"


def test_fast_image_is_durable_and_importable_while_another_image_is_pending(tmp_path):
    pool, row, item = fixture_image(tmp_path)
    target = {"key": item["key"], "row": row, "row_sha256": item["candidate_sha256"],
              "metadata_sha256": "frozen-metadata"}
    slow_target = {**target, "key": "slow"}
    root = tmp_path / "supply"
    complete, errors, counts = {}, {}, Counter()

    async def exercise():
        published, release = asyncio.Event(), asyncio.Event()
        async def fast():
            return target, (pool / "mirror.jpg").read_bytes(), [512, 512], "/asset", 7, tmp_path / "proof.json.gz", "proof-sha"
        async def slow():
            await release.wait()
            raise ValueError("deliberately unavailable image")
        slow_task = asyncio.create_task(slow())
        saver = asyncio.create_task(mirror.persist_image_results(
            {asyncio.create_task(fast()): target, slow_task: slow_target}, pool / "archive", root,
            0, complete, errors, counts, published.set, flush_seconds=.02))
        await asyncio.wait_for(published.wait(), 2)
        assert not saver.done() and not slow_task.done()
        manifest = next((root / "recovery_inbox").glob("*.json"))
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE tasks(key,row_json,status,attempts)")
        db.execute("INSERT INTO tasks VALUES (?,?,?,?)", (item["key"], dumps(row), "failed", 7))
        recovered = supply.validate_recovered_images(db, manifest, pool)
        assert len(recovered) == 1 and recovered[0]["previous_attempts"] == 7
        assert complete[item["key"]]["mirror"]["row_idx"] == 7
        assert counts["downloaded"] == 1
        db.close()
        release.set()
        await saver
        assert "deliberately unavailable" in errors["slow"]
        assert list(complete) == [item["key"]]
        assert len(list((root / "recovery_inbox").glob("*.json"))) == 1
    asyncio.run(exercise())


def test_cancellation_flushes_buffered_images_and_cancels_pending_downloads(tmp_path):
    pool, row, item = fixture_image(tmp_path)
    target = {"key": item["key"], "row": row, "row_sha256": item["candidate_sha256"],
              "metadata_sha256": "frozen-metadata"}
    root, archive = tmp_path / "supply", pool / "archive"
    complete, errors, counts = {}, {}, Counter()

    async def exercise():
        async def fast():
            return target, (pool / "mirror.jpg").read_bytes(), [512, 512], "/asset", 7, tmp_path / "proof.json.gz", "proof-sha"
        async def slow():
            await asyncio.Event().wait()
        slow_task = asyncio.create_task(slow())
        saver = asyncio.create_task(mirror.persist_image_results(
            {asyncio.create_task(fast()): target, slow_task: {**target, "key": "slow"}}, archive,
            root, 0, complete, errors, counts, lambda: None, flush_seconds=60))
        for _ in range(200):
            if list(archive.glob("images/page-*/*.tar")):
                break
            await asyncio.sleep(.01)
        assert list(archive.glob("images/page-*/*.tar"))
        assert not list((root / "recovery_inbox").glob("*.json"))
        saver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await saver
        assert slow_task.cancelled()
        assert list(complete) == [item["key"]] and counts["downloaded"] == 1
        assert len(list((root / "recovery_inbox").glob("*.json"))) == 1
    asyncio.run(exercise())
