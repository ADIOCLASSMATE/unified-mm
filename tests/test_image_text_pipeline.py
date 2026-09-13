import argparse
import asyncio
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from pathlib import Path

from PIL import Image
import pytest

from scripts.synthesize_image_text import (
    DirectDownloader, ImageArchives, export_training, run_pipeline, validate_pair,
)
from utils.image_shard_io import read_image_bytes
from utils.image_text_preprocessing import prepare_view
from utils.imagenet_synthetic_text_index import ImageNetSyntheticTextIndex
from utils.image_text_teacher import (
    PRIMARY_MODEL, FINAL_MODEL, QwenGenerator, QwenSettings, CodexFinalTeacher,
    load_qwen_settings, parse_judgement,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def pair(key):
    return {"image_id": key, "i2t": "A red square.", "t2i": "A red square on a red background.",
            "observations": {"counts": [], "relations": [], "visible_text": []},
            "capabilities": [], "uncertainties": [], "usable": {"i2t": True, "t2i": True}}


class Generator:
    def __init__(self, bad_json=False, error=None):
        self.calls = []
        self.bad_json, self.error = bad_json, error

    async def generate(self, key, pixels, extension):
        self.calls.append((key, pixels))
        if self.error:
            raise self.error
        with Image.open(io.BytesIO(pixels)) as image:
            assert image.size == (512, 512)
        return {"status": "completed", "output_text": "invalid-json" if self.bad_json else json.dumps(pair(key)),
                "response": {"test_fixture": True}}

    async def close(self):
        pass


class Teacher:
    def __init__(self, decision="accept"):
        self.calls = []
        self.decision = decision

    async def evaluate_batch(self, samples):
        self.calls.append(samples)
        results = [{"image_id": s["key"], "decision": self.decision, "issues": [],
                    "replacement": pair(s["key"]) if self.decision == "replace" else None} for s in samples]
        return {"status": "completed", "output_text": json.dumps({"results": results}),
                "response": {"test_fixture": True}}

    async def close(self):
        pass


def pixels(size=(768, 512), color="red"):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def test_square_crop_protects_required_objects():
    with pytest.raises(ValueError, match="retain all required"):
        prepare_view(pixels(), {"required_boxes": [[0, 0, 40, 40], [700, 450, 768, 512]]})
    output, metadata = prepare_view(pixels(), {"required_boxes": [[700, 450, 768, 512]]})
    assert metadata["crop"] == [256, 0, 768, 512]
    normalized, normalized_metadata = prepare_view(pixels(), {
        "required_boxes_normalized": [[700 / 768, 450 / 512, 1, 1]]})
    assert normalized == output and normalized_metadata == metadata
    assert Image.open(io.BytesIO(output)).size == (512, 512)
    with pytest.raises(ValueError, match="needs required_boxes"):
        prepare_view(pixels(), {"capabilities": ["counting"]})


def test_invalid_icc_profile_is_a_recoverable_image_rejection():
    output = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(output, format="PNG", icc_profile=b"invalid profile")
    with pytest.raises(ValueError, match="ICC profile"):
        prepare_view(output.getvalue(), {})


def test_openimages_selection_preserves_both_regions_and_unique_images(tmp_path):
    from scripts.prepare_b512_candidates import openimages_candidates
    source = tmp_path / "relations.csv"
    source.write_text("ImageID,LabelName1,LabelName2,XMin1,XMax1,YMin1,YMax1,XMin2,XMax2,YMin2,YMax2,RelationshipLabel\n"
        "0000000000000001,a,b,0.1,0.3,0.2,0.4,0.5,0.7,0.6,0.8,holds\n"
        "0000000000000001,a,b,0.1,0.3,0.2,0.4,0.5,0.7,0.6,0.8,on\n"
        "0000000000000002,a,b,0.1,0.3,0.2,0.4,0.5,0.7,0.6,0.8,holds\n")
    rows = list(openimages_candidates(source, 2, 42))
    assert len({r["source_id"] for r in rows}) == 2
    for row in rows:
        assert row["required_boxes_normalized"] == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        assert row["url"].endswith("/train/" + row["source_id"] + ".jpg")


def test_wikiart_extracts_distinct_native_resolution_images_to_archives(tmp_path):
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.prepare_b512_candidates import wikiart_candidates
    red, green, small = pixels((512, 512)), pixels((640, 600), "green"), pixels((128, 128), "blue")
    path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"image": {"bytes": data, "path": f"{i}.png"}, "style": style, "artist": 0, "genre": 1}
        for i, (data, style) in enumerate([(red, 12), (green, 4), (red, 12), (small, 3)])
    ]), path)
    rows = list(wikiart_candidates(path, tmp_path / "originals", 10, "fixture-revision", 42))
    assert len(rows) == 2
    assert {r["selection_annotation"]["style_id"] for r in rows} == {4, 12}
    assert {read_image_bytes(r["local_path"]) for r in rows} == {red, green}
    for row in rows:
        assert row["expected_source_sha256"] == hashlib.sha256(read_image_bytes(row["local_path"])).hexdigest()
        assert row["min_short_side"] == 512 and row["local_path"].startswith("tar:")


def test_archive_frozen_view_roundtrip(tmp_path):
    data = pixels()
    archives = ImageArchives(tmp_path, max_bytes=10)
    first = archives.add("first.png", data)
    second = archives.add("second.png", data)
    archives.close()
    assert read_image_bytes(first) == read_image_bytes(second) == data
    assert len(list(tmp_path.glob("*.tar"))) == 2
    async def check_local_archive():
        downloader = DirectDownloader(1, 1, max_bytes=len(data))
        try:
            assert await downloader.get({"local_path": first}) == data
            downloader.max_bytes -= 1
            with pytest.raises(ValueError, match="byte limit"):
                await downloader.get({"local_path": first})
        finally:
            await downloader.close()
    asyncio.run(check_local_archive())


def test_download_ignores_all_proxy_environment(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"direct-image-response")

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")

    async def check():
        downloader = DirectDownloader(2, 1)
        try:
            return await downloader.get({"url": f"http://127.0.0.1:{server.server_port}/image"})
        finally:
            await downloader.close()
    try:
        assert asyncio.run(check()) == b"direct-image-response"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_archive_read_survives_file_cache_eviction(tmp_path, monkeypatch):
    from utils import image_shard_io
    data = pixels()
    archives = ImageArchives(tmp_path)
    reference = archives.add("view.png", data)
    archives.close()
    original_pread = image_shard_io.os.pread
    def read_after_eviction(descriptor, size, offset):
        image_shard_io._open.cache_clear()
        return original_pread(descriptor, size, offset)
    monkeypatch.setattr(image_shard_io.os, "pread", read_after_eviction)
    assert read_image_bytes(reference) == data


def args_for(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(pixels((512, 512)))
    manifest = tmp_path / "candidate.jsonl"
    rows = [{"source": "unit", "source_id": "red", "local_path": str(source)},
            {"source": "unit", "source_id": "heldout", "local_path": str(source)},
            {"source": "unit", "source_id": "duplicate", "local_path": str(source)}]
    rows.append(dict(rows[0]))  # Duplicate input identity must not duplicate API work.
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("unit:heldout\n")
    example = tmp_path / "test_api.py"
    example.write_text('from anthropic import Anthropic\n'
        'client = Anthropic(base_url="http://127.0.0.1:9", api_key="test-key")\n'
        'client.messages.create(model="qwen3.8-27b", max_tokens=3200, '
        'thinking={"type":"enabled", "budget_tokens":1600}, messages=[])\n')
    return argparse.Namespace(manifest=str(manifest), exclude=str(exclude), output=str(tmp_path / "run"),
        tokenizer=str(tmp_path / "tokenizer"), partition=0, partitions=1,
        download_workers=1, cpu_workers=1, qwen_workers=1, teacher_workers=1, per_host=1, rpm=120, tpm=600000,
        judge_batch_size=4, judge_flush_seconds=0.01, qwen_example=str(example), codex_bin="codex", codex_timeout=30,
        prepare_only=True, retry_failed=False, regenerate_failed=False, min_short_side=512)


def test_prepare_resume_dedup_and_export_share_exact_teacher_pixels(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: ["test-loopback"])
    args = args_for(tmp_path)
    first = asyncio.run(run_pipeline(args))
    assert first == {"prepared": 1, "excluded": 1, "duplicate": 1}
    # Resume must use the saved exact view, even if the original URL/file disappears.
    (tmp_path / "source.png").unlink()
    args.prepare_only = False
    teacher = Teacher()
    generator = Generator()
    second = asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))
    assert second == {"ready": 1, "excluded": 1, "duplicate": 1}
    asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))
    assert len(teacher.calls) == len(generator.calls) == 1
    destination = tmp_path / "published"
    assert export_training(args.output, destination) == 1
    row = json.loads((destination / "manifest.jsonl").read_text())
    assert read_image_bytes(row["source_path"]) == generator.calls[0][1] == teacher.calls[0][0]["pixels"]
    # The VAE reader must consume the same frozen tar bytes as the teacher.
    from scripts.imagenet_encode_kl16_vae import ImagePathDataset
    _, image_tensor, _, _ = ImagePathDataset([(1, row["source_path"], None)], 512, True)[0]
    assert tuple(image_tensor.shape) == (3, 512, 512)
    with pytest.raises(FileExistsError):
        export_training(args.output, destination)
    index = ImageNetSyntheticTextIndex(destination / "text_index.json")
    assert index.read_caption(0)["img_id"] == 1
    assert index.read_t2i(0)["image_id"] == "train/" + generator.calls[0][0]
    assert index.read_caption(0)["captions"][0]["source"] == PRIMARY_MODEL
    assert index.read_caption(0)["provenance"]["judge_model"] == FINAL_MODEL
    index.close()


def test_received_response_survives_parser_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    args.prepare_only = False
    teacher = Teacher()
    generator = Generator()
    # A recoverable parser error also leaves the raw response durably recorded.
    def fail_parse(*_):
        raise ValueError("bad parser")
    monkeypatch.setattr("scripts.synthesize_image_text.parse_judgement", fail_parse)
    assert asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))["failed"] == 1
    import sqlite3
    db = sqlite3.connect(tmp_path / "run/state.sqlite3")
    status, raw, judgement = db.execute("SELECT status,raw,judge_raw FROM tasks WHERE raw IS NOT NULL").fetchone()
    assert status == "failed" and json.loads(raw)["status"] == "completed" and judgement
    db.close()
    # Retry reparses received results without another model call by default.
    args.retry_failed = True
    monkeypatch.setattr("scripts.synthesize_image_text.parse_judgement", parse_judgement)
    assert asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))["ready"] == 1
    assert len(teacher.calls) == len(generator.calls) == 1


@pytest.mark.parametrize("bad_json", [False, True])
def test_rejected_qwen_pair_is_replaced_in_the_same_final_call(tmp_path, monkeypatch, bad_json):
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    args.prepare_only = False
    generator, teacher = Generator(bad_json=bad_json), Teacher("replace")
    assert asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))["ready"] == 1
    assert len(generator.calls) == len(teacher.calls) == 1
    assert bool(teacher.calls[0][0]["primary_errors"]) == bad_json
    destination = tmp_path / "publication"
    assert export_training(args.output, destination) == 1
    index = ImageNetSyntheticTextIndex(destination / "text_index.json")
    assert index.read_caption(0)["captions"][0]["source"] == FINAL_MODEL
    assert index.read_caption(0)["provenance"]["decision"] == "replace"
    index.close()


@pytest.mark.parametrize("always_invalid", [False, True])
def test_bad_teacher_replacement_retries_only_that_image_once(tmp_path, monkeypatch, always_invalid):
    import sqlite3
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    rows = []
    for color in ["red", "green"]:
        path = tmp_path / f"{color}.png"
        path.write_bytes(pixels((512, 512), color))
        rows.append({"source": "unit", "source_id": color, "local_path": str(path)})
    Path(args.manifest).write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.prepare_only = False
    args.judge_flush_seconds = 1.0

    class WrongReplacementID(Teacher):
        async def evaluate_batch(self, samples):
            raw = await super().evaluate_batch(samples)
            if len(self.calls) == 1 or always_invalid:
                value = json.loads(raw["output_text"])
                value["results"][0]["replacement"]["image_id"] = "wrong-image"
                raw["output_text"] = json.dumps(value)
            return raw

    teacher, generator = WrongReplacementID("replace"), Generator()
    counts = asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))
    assert counts == ({"failed": 1, "ready": 1} if always_invalid else {"ready": 2})
    assert len(generator.calls) == 2
    assert [len(batch) for batch in teacher.calls] == [2, 1]
    retry_key = teacher.calls[0][0]["key"]
    assert teacher.calls[1][0]["key"] == retry_key
    with sqlite3.connect(Path(args.output) / "state.sqlite3") as db:
        batches = list(db.execute("SELECT id,raw,image_ids FROM judge_batches ORDER BY rowid"))
        assert len(batches) == 2
        assert json.loads(batches[1][1])["retry_of_batch_id"] == batches[0][0]
        assert json.loads(batches[1][2]) == [retry_key]
        # The original malformed response is retained for inspection.
        assert json.loads(json.loads(batches[0][1])["output_text"])["results"][0]["replacement"]["image_id"] == "wrong-image"
        other_key = teacher.calls[0][1]["key"]
        result = json.loads(db.execute("SELECT result FROM tasks WHERE key=?", (other_key,)).fetchone()[0])
        assert result["provenance"]["judge_batch_id"] == batches[0][0]


def test_teacher_timeout_retry_keeps_received_qwen_response(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    args.prepare_only = False

    class TimeoutOnce(Teacher):
        async def evaluate_batch(self, samples):
            if not self.calls:
                self.calls.append(samples)
                raise TimeoutError("teacher did not finish")
            return await super().evaluate_batch(samples)

    teacher, generator = TimeoutOnce(), Generator()
    assert asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))["failed"] == 1
    key = generator.calls[0][0]
    with sqlite3.connect(Path(args.output) / "state.sqlite3") as db:
        before, judgement = db.execute("SELECT raw,judge_raw FROM tasks WHERE key=?", (key,)).fetchone()
        assert before is not None and judgement is None
    args.retry_failed = args.regenerate_failed = True
    args.retry_phase = "sol"
    assert asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))["ready"] == 1
    assert len(teacher.calls) == 2 and len(generator.calls) == 1
    with sqlite3.connect(Path(args.output) / "state.sqlite3") as db:
        after = db.execute("SELECT raw FROM tasks WHERE key=?", (key,)).fetchone()[0]
        assert before == after


def test_final_rejection_and_network_failure_never_bypass_teacher(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    args.prepare_only = False
    generator, teacher = Generator(error=OSError("network unavailable")), Teacher()
    assert asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))["failed"] == 1
    assert not teacher.calls
    args.retry_failed = True
    generator, teacher = Generator(), Teacher("reject")
    assert asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))["review"] == 1
    with pytest.raises(ValueError, match="no approved"):
        export_training(args.output, tmp_path / "unapproved")


@pytest.mark.parametrize("qwen_recovers", [False, True])
def test_legacy_reuse_does_not_hide_a_qwen_service_outage(tmp_path, monkeypatch, qwen_recovers):
    import hashlib
    import sqlite3

    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    args.prepare_only = False
    args.judge_batch_size = 1
    rows, legacy_indices = [], {}
    for index in range(33):
        for legacy in (False, True):
            source_id = f"{'legacy' if legacy else 'fresh'}-{index}"
            path = tmp_path / f"{source_id}.png"
            path.write_bytes(pixels((512, 512), (index * 7, 128 if legacy else 0, 120)))
            row = {"source": "unit", "source_id": source_id, "local_path": str(path)}
            if legacy:
                row["reuse_pair"] = {
                    "i2t": "A colored square.", "t2i": "A colored square on a flat background.",
                    "i2t_source": "local_qwen", "t2i_style": "faithful_photo",
                    "i2t_model": "legacy-qwen", "t2i_model": "legacy-luna",
                    "source_dataset": "existing-synthetic-v1",
                }
                legacy_indices[hashlib.sha256(f"unit:{source_id}".encode()).hexdigest()] = index
            rows.append(row)
    Path(args.manifest).write_text("".join(json.dumps(row) + "\n" for row in rows))

    async def exercise():
        legacy_ready = [asyncio.Event() for _ in range(33)]

        class InterleavedTeacher(Teacher):
            async def evaluate_batch(self, samples):
                result = await super().evaluate_batch(samples)
                for sample in samples:
                    if sample["key"] in legacy_indices:
                        legacy_ready[legacy_indices[sample["key"]]].set()
                return result

        class InterleavedGenerator(Generator):
            async def generate(self, key, pixels, extension):
                index = len(self.calls)
                self.calls.append((key, pixels))
                # Every failing API call is preceded by a successful reuse.
                await asyncio.wait_for(legacy_ready[index].wait(), timeout=10)
                if qwen_recovers and index == 16:
                    return {"status": "completed", "output_text": json.dumps(pair(key))}
                raise OSError("simulated SII service outage")

        generator, teacher = InterleavedGenerator(), InterleavedTeacher()
        if qwen_recovers:
            counts = await run_pipeline(args, teacher, Tokenizer(), generator)
            assert counts == {"failed": 32, "ready": 34}
            assert len(generator.calls) == 33
        else:
            with pytest.raises(ExceptionGroup) as raised:
                await run_pipeline(args, teacher, Tokenizer(), generator)

            def leaf_errors(error):
                if isinstance(error, BaseExceptionGroup):
                    return [leaf for child in error.exceptions for leaf in leaf_errors(child)]
                return [error]

            assert any("qwen service/configuration failed" in str(error)
                       for error in leaf_errors(raised.value))
            assert len(generator.calls) == 32
        # Service failures never receive a teacher-generated replacement.
        called_keys = {sample["key"] for batch in teacher.calls for sample in batch}
        failed_keys = {key for index, (key, _) in enumerate(generator.calls)
                       if not (qwen_recovers and index == 16)}
        assert not called_keys & failed_keys

    asyncio.run(exercise())
    with sqlite3.connect(Path(args.output) / "state.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM tasks WHERE status='failed' AND error LIKE 'qwen:%'").fetchone()[0] == 32
        assert db.execute("SELECT count(*) FROM tasks WHERE status='ready' AND json_extract(result,'$.reused_unchanged')=1").fetchone()[0] >= 32


def test_example_is_read_without_execution_or_secret_publication(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.setenv("EXAMPLE_KEY", "test-secret")
    example = tmp_path / "example.py"
    example.write_text('import os\nfrom anthropic import Anthropic\n'
        'client = Anthropic(base_url="https://example.invalid/", api_key=os.getenv("EXAMPLE_KEY"))\n'
        'message = client.messages.create(model="qwen3.8-27b", max_tokens=3200, '
        'thinking={"type": "enabled", "budget_tokens": 1600})\n'
        'raise AssertionError("this example must never be executed")\n')
    settings = load_qwen_settings(example)
    assert settings.api_key == "test-secret"
    assert settings.max_tokens == 3200 and settings.thinking["budget_tokens"] == 1600
    assert "test-secret" not in repr(settings) + json.dumps(settings.public_contract())


def test_qwen_messages_image_request_ignores_proxy_environment(monkeypatch):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/messages"
            assert self.headers["x-api-key"] == "test-key"
            received.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            message = {"id": "msg-test", "type": "message", "role": "assistant",
                "model": PRIMARY_MODEL, "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0}}
            events = [
                {"type": "message_start", "message": message},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": json.dumps(pair("probe"))}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"},
            ]
            body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    async def check():
        gen = QwenGenerator(QwenSettings(f"http://127.0.0.1:{server.server_port}", "test-key"))
        try:
            return await gen.generate("probe", pixels(), "png")
        finally:
            await gen.close()
    try:
        result = asyncio.run(check())
        assert result["status"] == "completed"
        assert json.loads(result["output_text"]) == pair("probe")
        assert result["transport"] == "sii_direct_native_curl_http2_stream"
        assert received[0]["stream"] is True
        assert received[0]["model"] == PRIMARY_MODEL
        assert received[0]["thinking"] == {"type": "enabled", "budget_tokens": 1600}
        assert received[0]["max_tokens"] == 3200
        assert received[0]["messages"][0]["content"][0]["source"]["type"] == "base64"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_codex_batch_uses_exact_model_effort_and_separate_images(monkeypatch):
    monkeypatch.setattr("utils.image_text_teacher.shutil.which", lambda _: "/fake/codex")
    monkeypatch.setenv("SII_API_KEY", "test-sii-secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    captured = []
    async def fake_exec(*command, **kwargs):
        captured.extend(command)
        assert "SII_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["HTTPS_PROXY"] == "http://127.0.0.1:7890"
        images = [Path(command[i + 1]).read_bytes() for i, arg in enumerate(command) if arg == "--image"]
        assert images == [pixels(color="red"), pixels(color="blue")]
        class Process:
            returncode = 0
            async def communicate(self, data):
                assert b"test-secret" not in data
                output = Path(command[command.index("--output-last-message") + 1])
                output.write_text(json.dumps({"results": [
                    {"image_id": key, "decision": "accept", "issues": [], "replacement": None} for key in ["a", "b"]]}))
                return b'{"type":"turn.completed","usage":{}}\n', b""
        return Process()
    monkeypatch.setattr("utils.image_text_teacher.asyncio.create_subprocess_exec", fake_exec)
    samples = [{"key": key, "pixels": pixels(color=color), "extension": "png", "candidate": pair(key), "primary_errors": []}
               for key, color in [("a", "red"), ("b", "blue")]]
    result = asyncio.run(CodexFinalTeacher().evaluate_batch(samples))
    assert set(parse_judgement(result, ["a", "b"])) == {"a", "b"}
    assert captured[captured.index("--model") + 1] == FINAL_MODEL
    assert 'model_reasoning_effort="low"' in captured
    assert captured[captured.index("--sandbox") + 1] == "read-only"
    with pytest.raises(ValueError, match="identities"):
        parse_judgement(result, ["a", "c"])


def test_sii_fenced_json_retains_the_actual_candidate():
    result = validate_pair({"status": "completed", "output_text": "```json\n" + json.dumps(pair("a")) + "\n```"},
                           "a", Tokenizer())
    assert result == pair("a")
    with pytest.raises(ValueError):
        validate_pair({"status": "completed", "output_text": "Ignore this first.\n" + json.dumps(pair("a"))},
                      "a", Tokenizer())


def test_reuse_keeps_original_models_and_stores_images_separately(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    row = rows[0]
    row["reuse_pair"] = {"i2t": "A red square.", "t2i": "A red square on red.",
                         "i2t_source": "local_qwen", "t2i_style": "faithful_photo",
                         "i2t_model": "legacy-qwen", "t2i_model": "legacy-luna",
                         "source_dataset": "existing-synthetic-v1"}
    Path(args.manifest).write_text(json.dumps(row) + "\n")
    args.image_root = str(tmp_path / "public/images")
    args.prepare_only = False
    generator, teacher = Generator(), Teacher()
    assert asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator)) == {"ready": 1}
    assert not generator.calls and len(teacher.calls) == 1
    dest = tmp_path / "public/synthetic-text"
    assert export_training(args.output, dest) == 1
    cap = json.loads((dest / "captions.jsonl").read_text())
    assert cap["captions"][0]["source"] == "legacy-qwen"
    assert cap["provenance"]["generator_models"] == {"i2t": "legacy-qwen", "t2i": "legacy-luna"}
    assert cap["provenance"]["reused_unchanged"] is True
    assert not list(dest.rglob("*.tar"))
    assert list(Path(args.image_root).glob("*.tar"))


def test_publication_deduplicates_partitions_and_builds_global_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    partitions = []
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        args = args_for(directory)
        args.tokenizer = str(tmp_path / "shared-tokenizer")
        row = json.loads(Path(args.manifest).read_text().splitlines()[0])
        row["source_id"] = "red-" + name
        rows = [row]
        if name == "b":
            blue = directory / "blue.png"
            blue.write_bytes(pixels((512, 512), "blue"))
            rows.append({"source": "unit", "source_id": "blue", "local_path": str(blue)})
        Path(args.manifest).write_text("".join(json.dumps(r) + "\n" for r in rows))
        args.prepare_only = False
        asyncio.run(run_pipeline(args, Teacher(), Tokenizer(), Generator()))
        partitions.append(args.output)
    output = tmp_path / "combined"
    assert export_training(partitions, output) == 2
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert [r["img_id"] for r in rows] == [1, 2]
    index = ImageNetSyntheticTextIndex(output / "text_index.json")
    for offset, row in enumerate(rows):
        assert index.read_caption(offset)["img_id"] == row["img_id"]
        assert index.read_t2i(offset)["image_id"] == "train/" + row["key"]
    assert json.loads((output / "publication.json").read_text())["duplicate_images_dropped"] == 1
    index.close()
    from scripts.audit_image_text_publication import audit_publication
    audit = audit_publication(output, image_root=tmp_path, tokenizer=Tokenizer())
    assert audit["verified_images"] == 2 and audit["all_sol_low_approved"]
    rows[0]["view_sha256"] = "0" * 64
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="view_sha256 mismatch"):
        audit_publication(output, image_root=tmp_path, tokenizer=Tokenizer())


def test_parallel_teachers_share_one_batch_assembler(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    rows = []
    for color in ["red", "green", "blue", "yellow"]:
        path = tmp_path / f"{color}.png"
        path.write_bytes(pixels((512, 512), color))
        rows.append({"source": "unit", "source_id": color, "local_path": str(path)})
    Path(args.manifest).write_text("".join(json.dumps(row) + "\n" for row in rows))
    args.prepare_only = False
    args.teacher_workers = 4
    args.judge_flush_seconds = 1.0
    class StaggeredGenerator(Generator):
        async def generate(self, key, pixels, extension):
            await asyncio.sleep(0.02)
            return await super().generate(key, pixels, extension)
    teacher = Teacher()
    counts = asyncio.run(run_pipeline(args, teacher, Tokenizer(), StaggeredGenerator()))
    assert counts == {"ready": 4}
    assert len(teacher.calls) == 1 and len(teacher.calls[0]) == 4


def test_benchmark_near_duplicates_are_excluded_before_model_calls(tmp_path, monkeypatch):
    from utils.image_near_duplicates import perceptual_hashes, write_index
    import sqlite3
    monkeypatch.setattr("scripts.synthesize_image_text.check_direct_routes", lambda: [])
    args = args_for(tmp_path)
    benchmark = tmp_path / "benchmark-index"
    values = perceptual_hashes(pixels((1024, 1024)))
    write_index(benchmark, values, [0] * len(values), ["heldout.jpg"])
    args.near_exclude_index = str(benchmark)
    args.prepare_only = False
    generator, teacher = Generator(), Teacher()
    counts = asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))
    assert counts == {"excluded": 3}
    assert not generator.calls and not teacher.calls
    db = sqlite3.connect(Path(args.output) / "state.sqlite3")
    assert db.execute('SELECT count(*) FROM tasks WHERE view IS NOT NULL').fetchone()[0] == 0
    db.close()
    # Changing the screening index cannot silently resume the same run.
    (benchmark / "sources.json").write_text('["another-heldout.jpg"]\n')
    with pytest.raises(ValueError, match="contract changed"):
        asyncio.run(run_pipeline(args, teacher, Tokenizer(), generator))
