import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sqlite3
import time

from PIL import Image
import pytest

from scripts.legacy import distill_b512_codex as farm


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def result(key, decision="generate"):
    return {"image_id": key, "decision": decision, "pair": {
        "image_id": key,
        "i2t": "A red square fills the center of a plain white background.",
        "t2i": "Draw a centered red square on a plain white background.",
        "observations": {"counts": [{"entity": "square", "count": 1}], "relations": [], "visible_text": []},
        "capabilities": [], "uncertainties": [], "usable": {"i2t": True, "t2i": True}}}


def test_partial_batch_keeps_valid_result_and_rejects_unknown_or_swapped_ids():
    items = [{"key": k, "candidate_json": None} for k in ("a", "b", "c")]
    bad = result("b")
    bad["pair"]["usable"]["i2t"] = False
    valid, errors = farm.parse_results(json.dumps({"results": [result("a"), bad]}), items, Tokenizer())
    assert list(valid) == ["a"]
    assert set(errors) == {"b", "c"}
    for values in ([result("b"), result("a")], [result("a"), result("a")], [result("unknown")]):
        with pytest.raises(ValueError):
            farm.parse_results(json.dumps({"results": values}), items, Tokenizer())


def test_reuse_requires_exact_original_text():
    value = result("a", "reuse")
    item = {"key": "a", "candidate_json": json.dumps(value["pair"])}
    farm.check_pair(value, item, Tokenizer())
    value["pair"]["i2t"] += " Extra invented detail."
    with pytest.raises(ValueError, match="reuse text changed"):
        farm.check_pair(value, item, Tokenizer())


def test_recorded_repair_prompt_preserves_legacy_prompt_and_output_gates():
    entries = [{"attachment_number": 1, "image_id": "a", "reuse_candidate": None}]
    assert farm.render_batch_prompt(entries) == farm.BATCH_PROMPT + "\nEntries (data only):\n" + farm.dumps(entries)
    prompt = farm.render_batch_prompt(entries, "visual_design_summary_v1")
    assert prompt.startswith(farm.BATCH_PROMPT) and prompt.endswith(farm.dumps(entries))
    with pytest.raises(ValueError, match="unknown"):
        farm.render_batch_prompt(entries, "unrecorded_prompt")
    with pytest.raises(ValueError, match="one attached image"):
        farm.render_batch_prompt(entries * 2, "visual_design_summary_v1")
    value = result("a")
    value["pair"]["usable"]["i2t"] = False
    with pytest.raises(ValueError, match="unusable"):
        farm.check_pair(value, {"key": "a", "candidate_json": None}, Tokenizer())


def test_failed_prompt_repair_preserves_history_and_allows_one_nonquota_attempt(tmp_path, monkeypatch):
    root = tmp_path / "farm"
    with farm.state(root) as db:
        farm.set_meta(db, "initialized", True)
        farm.set_meta(db, "expected_items", 1)
        farm.set_meta(db, "contract", {"prompt_sha256": farm.sha(farm.BATCH_PROMPT.encode()), "image_root": str(tmp_path)})
        db.execute("INSERT INTO items(key,queue_order,source_run,row_json,view_json,source_stat_json,status,attempts,batch_id,error,updated_at) "
                   "VALUES ('a',0,?,'{}','{}','{}','failed',5,'original-batch','original failure',0)", (str(tmp_path),))
        db.commit()
    args = argparse.Namespace(root=str(root), key="a", prompt_variant="visual_design_summary_v1", max_batches=0)
    report = farm.repair_failed(args)
    previous = json.loads(Path(report["repair_request"]).read_text())["previous_item"]
    assert previous["attempts"] == 5 and previous["batch_id"] == "original-batch"
    with farm.state(root, writer=False) as db:
        item = dict(db.execute("SELECT * FROM items").fetchone())
        assert item["status"] == "retry" and item["attempts"] == 5 and item["batch_size"] == 1
        assert item["error"] == "original failure" and farm.meta(db, "expected_items") == 1
    monkeypatch.setattr(farm.subprocess, "check_output", lambda *a, **k: "codex fixture")
    calls = []

    async def worker(_root, _contract, items, _batch_id, _version):
        calls.append(items)
        return {"transport_healthy": False, "raw": "", "error": "transport failure", "provenance": {}}

    assert asyncio.run(farm.run(args, worker=worker, tokenizer=Tokenizer())) == "needs_repair"
    assert len(calls) == 1
    with farm.state(root, writer=False) as db:
        assert tuple(db.execute("SELECT status,attempts FROM items").fetchone()) == ("failed", 6)
    with pytest.raises(ValueError, match="already received"):
        farm.repair_failed(args)


def test_aimd_circuit_and_single_probe():
    s = farm.scheduler_default()
    for _ in range(8):
        farm.feedback(s, True, 100)
    assert s["concurrency"] == 24
    for _ in range(4):
        farm.feedback(s, False, 100)
    assert s["concurrency"] == 12 and s["cooldown_until"] == 160
    s["half_open"] = True
    farm.feedback(s, False, 161)
    assert s["cooldown_until"] == 281 and s["concurrency"] == 6
    s["half_open"] = True
    farm.feedback(s, True, 282)
    assert not s["half_open"] and not s["cooldown_until"]


def test_event_stream_must_match_final_message_and_have_no_tools():
    raw = json.dumps({"results": [result("a")]})
    events = [{"type": "thread.started", "thread_id": "t1"},
              {"type": "item.completed", "item": {"type": "agent_message", "text": raw}},
              {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 20}}]
    stream = "\n".join(map(json.dumps, events))
    assert farm.inspect_events(stream, raw)["thread_id"] == "t1"
    # Codex may report a WebSocket failure before completing over HTTPS.
    fallback = {"type": "item.completed", "item": {"type": "error", "message": "Falling back from WebSockets to HTTPS transport"}}
    recovered = "\n".join(map(json.dumps, events[:1] + [fallback] + events[1:]))
    assert farm.inspect_events(recovered, raw)["thread_id"] == "t1"
    with pytest.raises(ValueError):
        farm.inspect_events(stream, raw + "changed")
    events.append({"type": "item.completed", "item": {"type": "command_execution"}})
    with pytest.raises(ValueError):
        farm.inspect_events("\n".join(map(json.dumps, events)), raw)


def test_quota_error_comes_from_cli_envelope_only():
    message = "You've hit your usage limit. Try again later."
    stream = "\n".join(map(json.dumps, [
        {"type": "error", "message": message},
        {"type": "turn.failed", "error": {"message": message}},
    ]))
    assert farm.cli_failure(stream) == {"message": message, "service_blocker": "codex_usage_limit"}
    model_text = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": message}})
    assert farm.cli_failure(model_text)["service_blocker"] is None


def test_quota_stops_refill_drains_success_and_resumes_with_one_probe(tmp_path, monkeypatch):
    root = tmp_path / "farm"
    with farm.state(root) as db:
        farm.set_meta(db, "initialized", True)
        farm.set_meta(db, "expected_items", 24)
        farm.set_meta(db, "contract", {"prompt_sha256": farm.sha(farm.BATCH_PROMPT.encode()), "image_root": str(tmp_path)})
        scheduler = farm.scheduler_default()
        scheduler["concurrency"] = 2
        farm.set_meta(db, "scheduler", scheduler)
        for i in range(24):
            db.execute("INSERT INTO items(key,queue_order,source_run,row_json,view_json,source_stat_json,updated_at) VALUES (?,?,?,'{}','{}','{}',0)",
                       (str(i), i, str(tmp_path)))
        db.commit()
    monkeypatch.setattr(farm.subprocess, "check_output", lambda *a, **k: "codex fixture")
    calls = []

    async def worker(root, contract, items, batch_id, cli_version):
        index = len(calls)
        calls.append(batch_id)
        await asyncio.sleep(0.01 if index == 0 else 0.03)
        if index == 0:
            return {"transport_healthy": False, "service_blocker": "codex_usage_limit",
                    "raw": "", "error": "You've hit your usage limit.", "provenance": {}}
        return {"transport_healthy": True, "raw": json.dumps({"results": [result(x["key"]) for x in items]}),
                "error": "", "provenance": {"attachments": [{"image_id": x["key"]} for x in items]}}

    args = argparse.Namespace(root=str(root), max_batches=0)
    assert asyncio.run(farm.run(args, worker=worker, tokenizer=Tokenizer())) == "waiting_for_quota"
    assert len(calls) == 2
    with farm.state(root, writer=False) as db:
        before = dict(db.execute("SELECT key,result_sha256 FROM items WHERE status='ready'"))
        assert len(before) == 8
        assert dict(db.execute("SELECT status,count(*) FROM items GROUP BY status")) == {"pending": 8, "ready": 8, "retry": 8}
    calls.clear()
    assert asyncio.run(farm.run(args, worker=worker, tokenizer=Tokenizer())) == "waiting_for_quota"
    assert len(calls) == 1
    assert asyncio.run(farm.run(args, worker=worker, tokenizer=Tokenizer())) == "completed"
    with farm.state(root, writer=False) as db:
        after = dict(db.execute("SELECT key,result_sha256 FROM items WHERE status='ready'"))
        assert len(after) == 24 and all(after[k] == v for k, v in before.items())
        assert farm.meta(db, "service_blocker") is None


def test_interrupt_drains_completed_calls_and_resume_preserves_results(tmp_path, monkeypatch):
    source, images, root = (tmp_path / x for x in ("source", "images", "state"))
    source.mkdir()
    images.mkdir()
    c = sqlite3.connect(source / "state.sqlite3")
    c.execute("CREATE TABLE tasks (key TEXT,row TEXT,view TEXT,result TEXT,status TEXT)")
    for i in range(32):
        path = images / f"{i}.png"
        Image.new("RGB", (512, 512), (i, 0, 0)).save(path)
        digest = farm.file_sha(path)
        c.execute("INSERT INTO tasks VALUES (?,?,?,NULL,'prepared')", (
            str(i), json.dumps({"source": "fixture", "source_id": str(i), "split": "train"}),
            json.dumps({"source_sha256": digest, "view_sha256": digest, "image_size": 512,
                        "source_path": str(path), "original_ref": str(path), "extension": "png"})))
    c.commit()
    c.close()
    args = argparse.Namespace(root=str(root), source_run=[str(source)], image_root=str(images), tokenizer="fixture", max_batches=0)
    farm.initialize(args)
    with farm.state(root) as db:
        s = farm.scheduler_default()
        s["concurrency"] = 2
        farm.set_meta(db, "scheduler", s)
        db.commit()
    monkeypatch.setattr(farm.subprocess, "check_output", lambda *a, **k: "codex fixture")
    interrupted = False

    async def worker(root, contract, items, batch_id, cli_version):
        nonlocal interrupted
        await asyncio.sleep(0.01)
        if not interrupted:
            interrupted = True
            os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(0.03)
        return {"transport_healthy": True, "raw": json.dumps({"results": [result(x["key"]) for x in items]}),
                "error": "", "provenance": {"attachments": [{"image_id": x["key"]} for x in items]}}

    assert asyncio.run(farm.run(args, worker=worker, tokenizer=Tokenizer())) == "paused"
    with farm.state(root, writer=False) as db:
        before = {r[0]: r[1] for r in db.execute("SELECT key,result_sha256 FROM items WHERE status='ready'")}
        assert len(before) == 16
        assert db.execute("SELECT count(*) FROM items WHERE status='running'").fetchone()[0] == 0
    assert asyncio.run(farm.run(args, worker=worker, tokenizer=Tokenizer())) == "completed"
    with farm.state(root, writer=False) as db:
        after = dict(db.execute("SELECT key,result_sha256 FROM items WHERE status='ready'"))
        assert len(after) == 32
        assert all(after[k] == v for k, v in before.items())


def test_full_publication_refuses_pending_scope(tmp_path):
    root = tmp_path / "state"
    with farm.state(root) as db:
        farm.set_meta(db, "expected_items", 2)
        farm.set_meta(db, "contract", {})
        farm.set_meta(db, "scope", {})
        db.commit()
    with pytest.raises(ValueError, match="strict full publication"):
        farm.export(argparse.Namespace(root=str(root), output=str(tmp_path / "export"), snapshot=False), tokenizer=Tokenizer())


def test_encoder_rejects_overwriting_frozen_source_manifest(tmp_path, monkeypatch):
    from scripts.imagenet_encode_kl16_vae import main
    path = tmp_path / "manifest.jsonl"
    original = '{"view_sha256":"keep-this-source-hash"}\n'
    path.write_text(original)
    monkeypatch.setattr("sys.argv", ["encoder", "--source_mode", "manifest_jsonl",
        "--source_manifest_jsonl", str(path), "--manifest_jsonl", str(path),
        "--cache_shard_dir", str(tmp_path / "cache")])
    with pytest.raises(ValueError, match="must not overwrite"):
        main()
    assert path.read_text() == original


def test_rolling_batch_is_frozen_deduplicated_and_prioritized(tmp_path):
    images, source, root = (tmp_path / n for n in ("images", "source", "farm"))
    images.mkdir()
    source.mkdir()
    c = sqlite3.connect(source / "state.sqlite3")
    c.execute("CREATE TABLE tasks(key TEXT,row TEXT,view TEXT,result TEXT,status TEXT)")

    def row(i, color):
        path = images / f"{i}.png"
        Image.new("RGB", (512, 512), color).save(path)
        key = farm.sha(f"fixture:{i}".encode())
        return (key, json.dumps({"source": "fixture", "source_id": str(i), "split": "train"}),
                json.dumps({"source_sha256": farm.file_sha(path), "view_sha256": farm.file_sha(path),
                            "image_size": 512, "source_path": str(path), "original_ref": str(path), "extension": "png"}))

    c.execute("INSERT INTO tasks VALUES (?,?,?,NULL,'prepared')", row(0, "red"))
    c.commit()
    c.close()
    farm.initialize(argparse.Namespace(root=str(root), source_run=[str(source)], image_root=str(images), tokenizer="fixture"))
    new = tmp_path / "new"
    new.mkdir()
    c = sqlite3.connect(new / "state.sqlite3")
    c.execute("CREATE TABLE tasks(key TEXT,row TEXT,view TEXT,status TEXT)")
    for i, color in ((1, "red"), (2, "blue"), (3, "green")):
        c.execute("INSERT INTO tasks VALUES (?,?,?,'prepared')", row(i, color))
    c.commit()
    c.close()
    marker = new / "batch.json"
    farm.atomic_json(marker, {"batch_id": "test", "records": 3})
    inbox = root / "input_queue"
    farm.atomic_json(inbox / "test.json", {"batch_id": "test", "source_run": str(new), "records": 3,
        "state_sha256": farm.file_sha(new / "state.sqlite3"), "batch_manifest": str(marker),
        "batch_manifest_sha256": farm.file_sha(marker)})
    with farm.state(root) as db:
        supply = farm.InputBatches(db, root)
        assert not supply.closed()
        assert supply.poll(force=True) == 2
        assert farm.meta(db, "expected_items") == 3
        assert supply.poll(force=True) == 0
        assert len(farm.queue_rows(db, time.time(), 1, size=8)) == 2
        assert len(farm.queue_rows(db, time.time(), 0, size=8)) == 1
        farm.atomic_json(inbox / "closed.json", {})
        assert supply.closed()
        digest = farm.hashlib.sha256()
        for item in db.execute("SELECT key,view_json FROM items ORDER BY queue_order"):
            digest.update((item["key"] + ":" + json.loads(item["view_json"])["view_sha256"] + "\n").encode())
        assert digest.hexdigest() == farm.meta(db, "scope")["scope_sha256"]


def test_full_frame_preparation_keeps_objects_at_both_edges():
    import io
    from utils.image_text_preprocessing import prepare_view
    image = Image.new("RGB", (1024, 512), "white")
    image.paste("red", (0, 0, 32, 512))
    image.paste("blue", (992, 0, 1024, 512))
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    prepared, view = prepare_view(raw.getvalue(), {"capabilities": ["ocr", "counting"], "view_policy": "fit_pad"})
    with Image.open(io.BytesIO(prepared)) as result:
        assert result.size == (512, 512)
        assert result.getpixel((4, 256)) == (255, 0, 0)
        assert result.getpixel((507, 256)) == (0, 0, 255)
    assert view["crop"] == [0, 0, 1024, 512]
    assert view["content_box"] == [0, 128, 512, 384]


def test_slow_origin_cannot_occupy_slots_for_other_hosts(tmp_path, monkeypatch):
    from scripts import supply_b512_images as supply

    root, images = tmp_path / "supply", tmp_path / "images"
    (root / "candidates").mkdir(parents=True)
    images.mkdir()
    ids = sorted(range(5), key=lambda i: farm.sha(f"fixture:{i}".encode()))
    rows = [{"source": "fixture", "source_id": str(i), "split": "train", "capabilities": ["counting"],
             "url": f"https://{'a.slow.test' if i != ids[-1] else 'z.fast.test'}/{i}"} for i in ids]
    (root / "candidates/test.jsonl").write_text("".join(farm.dumps(row) + "\n" for row in rows))
    farm.atomic_json(root / "candidates.closed.json", {"files": 1})
    monkeypatch.setattr(supply, "check_direct_routes", lambda: None)

    async def scenario():
        release, fast_started = asyncio.Event(), asyncio.Event()
        active = {}

        class Client:
            host_retry_at = {}

            def __init__(self, *_args):
                pass

            async def get(self, row):
                host = supply.download_host(row)
                active[host] = active.get(host, 0) + 1
                assert active[host] == 1
                try:
                    if host == "a.slow.test":
                        await release.wait()
                    else:
                        fast_started.set()
                    return b"fixture image bytes"
                finally:
                    active[host] -= 1

            async def close(self):
                pass

        monkeypatch.setattr(supply, "DirectDownloader", Client)
        args = argparse.Namespace(root=root, image_root=images, workers=2, per_host=1)
        task = asyncio.create_task(supply.download_service(args))
        try:
            await asyncio.wait_for(fast_started.wait(), timeout=5)
            assert not release.is_set(), "fast host had to wait for the slow host"
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=10)
        db = sqlite3.connect(root / "download.sqlite3")
        assert db.execute("SELECT count(*) FROM tasks WHERE status='downloaded'").fetchone()[0] == 5
        db.close()

    asyncio.run(scenario())


def test_download_deadline_remains_retryable():
    from scripts.supply_b512_images import bounded_download

    class Client:
        failures = []

        async def get(self, _row):
            await asyncio.Event().wait()

        def mark_host_failure(self, host):
            self.failures.append(host)

    client = Client()
    with pytest.raises(ValueError, match="transport.*deadline"):
        asyncio.run(bounded_download(client, {"url": "https://slow.test/image"}, 0.01))
    assert client.failures == ["slow.test"]


def test_download_recovery_preserves_permanent_http_failures():
    from scripts.supply_b512_images import retryable_download_error

    for error in ("direct image transport failed after retries", "direct image host temporarily unavailable; retry later",
                  "HTTP 429", "HTTP 522 after retries", "HTTP 503 after retries"):
        assert retryable_download_error(error)
    for error in ("HTTP 403", "HTTP 404", "HTTP 410", "source SHA mismatch"):
        assert not retryable_download_error(error)


def test_direct_dns_download_keeps_image_connection_and_parent_proxy_separate(monkeypatch):
    from types import SimpleNamespace
    from scripts import supply_b512_images as supply

    monkeypatch.setenv("HTTPS_PROXY", "http://parent-proxy.invalid:8080")
    captured = {}

    class Process:
        returncode = 0

        async def communicate(self):
            return b"pixels\n__B512_DIRECT_IMAGE_STATUS__200", b""

    async def spawn(*argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return Process()

    monkeypatch.setattr(supply.asyncio, "create_subprocess_exec", spawn)
    client = SimpleNamespace(max_bytes=6, host_failures={}, host_retry_at={})
    assert asyncio.run(supply.direct_dns_download(client, {"url": "https://live.staticflickr.com/image"})) == b"pixels"
    argv = captured["argv"]
    assert argv[argv.index("--proxy") + 1] == ""
    assert argv[argv.index("--noproxy") + 1] == "*"
    assert argv[argv.index("--doh-url") + 1] == "https://dns.alidns.com/dns-query"
    assert "HTTPS_PROXY" not in captured["env"]
    assert os.environ["HTTPS_PROXY"] == "http://parent-proxy.invalid:8080"


def test_direct_dns_origin_is_paced_before_claiming_tasks(tmp_path, monkeypatch):
    from scripts import supply_b512_images as supply

    root, images = tmp_path / "supply", tmp_path / "images"
    (root / "candidates").mkdir(parents=True)
    images.mkdir()
    rows = [{"source": "fixture", "source_id": str(i), "capabilities": ["counting"],
             "url": f"https://paced.test/{i}"} for i in range(5)]
    (root / "candidates/test.jsonl").write_text("".join(farm.dumps(row) + "\n" for row in rows))
    farm.atomic_json(root / "candidates.closed.json", {"files": 1})
    monkeypatch.setattr(supply, "check_direct_routes", lambda: None)
    started = []

    async def get(_client, _row):
        started.append(time.monotonic())
        return b"fixture image bytes"

    monkeypatch.setattr(supply, "direct_dns_download", get)
    args = argparse.Namespace(root=root, image_root=images, workers=4, per_host=4,
                              direct_dns_host=["paced.test"], direct_dns_rps=10)
    asyncio.run(supply.download_service(args))
    assert len(started) == 5
    assert all(b - a >= 0.08 for a, b in zip(started, started[1:]))
    db = sqlite3.connect(root / "download.sqlite3")
    assert list(db.execute("SELECT DISTINCT status,attempts FROM tasks")) == [("downloaded", 1)]
    db.close()


def test_independent_download_prepare_and_seal(tmp_path, monkeypatch):
    from scripts.supply_b512_images import download_service, prepare_service
    from utils.image_near_duplicates import write_index
    import io
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    image = Image.new("RGB", (1024, 512), "white")
    image.paste("red", (0, 0, 32, 512))
    image.paste("blue", (992, 0, 1024, 512))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    pixels = buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = pixels if self.path == "/valid" else b"not an image"
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root, images = tmp_path / "farm" / "source_supply", tmp_path / "images"
    (root / "candidates").mkdir(parents=True)
    images.mkdir()
    exclude = root / "exclude.txt"
    exclude.write_text("")
    benchmark = root / "benchmark"
    write_index(benchmark, [], [], [])
    rows = [{"source": "fixture", "source_id": name, "split": "train", "capabilities": ["counting", "ocr"],
             "url": f"http://127.0.0.1:{server.server_port}/{name}"} for name in ("valid", "invalid")]
    (root / "candidates" / "test.jsonl").write_text("".join(farm.dumps(row) + "\n" for row in rows))
    farm.atomic_json(root / "candidates.closed.json", {"files": 1})
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    args = argparse.Namespace(root=str(root), image_root=str(images), workers=2, per_host=2,
                              farm_root=str(root.parent), exclude=str(exclude), near_exclude_index=str(benchmark))

    async def pipeline():
        await asyncio.wait_for(asyncio.gather(download_service(args), prepare_service(args)), timeout=40)

    try:
        asyncio.run(pipeline())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert json.loads((root / "download.closed.json").read_text())["batches"] == 1
    assert json.loads((root.parent / "input_queue" / "closed.json").read_text())["batches"] == 1
    marker = next((root / "prepared_batches").glob("*/batch.json"))
    assert json.loads(marker.read_text())["records"] == 1
    assert len(json.loads(marker.read_text())["rejections"]) == 1
    c = sqlite3.connect(marker.parent / "state.sqlite3")
    view = json.loads(c.execute("SELECT view FROM tasks WHERE status='prepared'").fetchone()[0])
    c.close()
    assert view["crop"] == [0, 0, 1024, 512]
    assert view["content_box"] == [0, 128, 512, 384]
