from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import subprocess
import time
from base64 import b64decode
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from caption_farm.config import CONFIG_SCHEMA, prepare_run
from caption_farm.controller import (
    CONTROLLER_TUNING_SCHEMA,
    Controller,
    acknowledge_attention,
    supervise_controller,
    wait_for_controller,
)
from caption_farm.io import canonical_json, load_json
from caption_farm.model import validate_model_snapshot
from caption_farm.inspire_submit import _validate_request
from caption_farm.platform import InspireClient, InspireError, LiveTarget
from caption_farm.publish import audit_run, publish_run
from caption_farm.queue import TaskStore
from caption_farm.worker import VllmOpenAIEngine, WORKER_TUNING_SCHEMA, Worker


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _make_config(
    tmp_path: Path,
    *,
    rows: int = 8,
    captions_per_image: int = 2,
    platform_enabled: bool = False,
    lease_seconds: float = 0.6,
    heartbeat_seconds: float = 0.1,
) -> tuple[Path, Path]:
    source_manifest = tmp_path / "source-manifest.jsonl"
    originals = tmp_path / "originals.jsonl"
    manifest_rows = []
    original_rows = []
    for index in range(rows):
        image_id = f"n00000001_{index + 1}"
        path = f"n00000001/{image_id}.JPEG"
        manifest_rows.append(
            {
                "img_id": index + 1,
                "source_path": f"/old/mount/train/{path}",
                "synset": "n00000001",
            }
        )
        original_rows.append(
            {
                "img_id": index + 1,
                "id": image_id,
                "path": path,
                "synset": "n00000001",
                "recaption_short": f"Original caption for test image number {index + 1}.",
            }
        )
    _write_jsonl(source_manifest, manifest_rows)
    _write_jsonl(originals, original_rows)
    config = {
        "schema": CONFIG_SCHEMA,
        "dataset": {
            "source_manifest": str(source_manifest),
            "original_captions": str(originals),
            "dataset_id": "imagenet",
            "version_id": "v1",
            "platform_path": "rclone-worker-1/imagenet/v1",
            "container_path": "/inspire/dataset/imagenet/v1",
            "compatibility_manifests": [],
        },
        "model": {"engine": "fake", "repository": "fake-caption-model-v1"},
        "caption": {
            "captions_per_image": captions_per_image,
            "prompt_version": "test-prompt-v1",
            "system_prompt": "Return one English caption.",
            "prompt_variants": [
                "Describe visible details in {target_min_words}-{target_max_words} words for slot {caption_slot}."
            ],
            "target_min_words": 32,
            "target_max_words": 60,
            "max_output_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
        },
        "queue": {
            "lease_seconds": lease_seconds,
            "heartbeat_seconds": heartbeat_seconds,
            "claim_batch_size": 3,
            "max_attempts": 3,
            "lock_timeout_seconds": 10,
            "lock_stale_seconds": 1,
            "idle_exit_seconds": 0.2,
        },
        "platform": {
            "enabled": platform_enabled,
            "workspace": "分布式训练空间",
            "image": "docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1",
            "image_readiness_notebook": "dev-wjx",
            "priority": 1,
            "gpu_type": "H100",
            "quota": "1,20,200",
            "nodes": 1,
            "repository_path": str(Path.cwd()),
            "worker_script": "scripts/imagenet_qwen_caption_farm.py",
            "runtime_python": str(Path.cwd() / ".venv/bin/python"),
            "projects": [
                {"name": "project-a", "weight": 2, "max_active_jobs": 2},
                {"name": "project-b", "weight": 1, "max_active_jobs": 2},
            ],
            "targets": [
                {
                    "group": "开发区-H100-cuda12.8版本-183核",
                    "quota": "1,20,200",
                    "weight": 1,
                    "max_active_jobs": 4,
                }
            ],
            "shm_size_gib": 64,
            "max_time_hours": 24,
        },
        "controller": {
            "global_max_active_jobs": 2,
            "target_active_jobs": 1,
            "submission_burst": 1,
            "min_submit_interval_seconds": 0,
            "reconcile_interval_seconds": 0.1,
            "rejection_backoff_base_seconds": 0.01,
            "rejection_backoff_max_seconds": 1,
            "circuit_breaker_rejections": 2,
            "circuit_breaker_recovery_seconds": 1,
            "no_progress_attention_seconds": 60,
            "zero_active_attention_seconds": 60,
        },
        "output": {
            "version": "test-output-v1",
            "published_jsonl": str(tmp_path / "published" / "captions.jsonl"),
            "verify_images_on_publish": False,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run_dir = tmp_path / "run"
    prepare_run(config_path, run_dir)
    return config_path, run_dir


def _run_fake_worker(run_dir: str, worker_id: str, max_tasks: int | None = None) -> None:
    Worker(Path(run_dir), worker_id=worker_id, max_tasks=max_tasks).run_forever()


def _claim_then_wait(run_dir: str, ready_path: str) -> None:
    store = TaskStore(Path(run_dir))
    lease = store.claim("kill-me", 2)
    assert lease is not None
    Path(ready_path).write_text(lease.lease_id, encoding="utf-8")
    while True:
        time.sleep(1)


def test_model_snapshot_validation_checks_fp8_and_all_shards(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    required = {
        "config.json": {"architectures": ["VisionModel"], "model_type": "vl", "quantization_config": {"quant_method": "fp8"}},
        "tokenizer.json": {},
        "tokenizer_config.json": {"chat_template": "x"},
        "preprocessor_config.json": {},
        "chat_template.jinja": "{{ messages }}",
        "model.safetensors.index.json": {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}},
    }
    for name, value in required.items():
        if isinstance(value, str):
            (snapshot / name).write_text(value, encoding="utf-8")
        else:
            (snapshot / name).write_text(json.dumps(value), encoding="utf-8")
    (snapshot / "model-1.safetensors").write_bytes(b"x" * 16)
    (snapshot / "model-2.safetensors").write_bytes(b"y" * 24)
    info = validate_model_snapshot(snapshot)
    assert info["quant_method"] == "fp8"
    assert info["shard_count"] == 2
    assert info["weight_bytes"] == 40
    assert len(info["fingerprint"]) == 64
    (snapshot / "model-2.safetensors").unlink()
    with pytest.raises(ValueError, match="missing shard"):
        validate_model_snapshot(snapshot)


def test_official_dataset_submit_adapter_rejects_drift():
    payload = {
        "task_priority": 1,
        "framework_config": [
            {
                "gpu_count": 1,
                "instance_count": 1,
                "image": "docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1",
            }
        ],
        "dataset_info": [
            {
                "dataset_id": "imagenet",
                "version_id": "v1",
                "path": "rclone-worker-1/imagenet/v1",
            }
        ],
    }
    request = {
        "action": "create-training-job",
        "payload": payload,
        "expected": {
            "image": "docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1",
            "dataset_id": "imagenet",
            "version_id": "v1",
            "dataset_path": "rclone-worker-1/imagenet/v1",
        },
    }
    assert _validate_request(request) is payload
    bad = json.loads(json.dumps(request))
    bad["payload"]["task_priority"] = 2
    with pytest.raises(ValueError, match="task_priority=1"):
        _validate_request(bad)
    bad = json.loads(json.dumps(request))
    bad["payload"]["dataset_info"] = []
    with pytest.raises(ValueError, match="dataset_info"):
        _validate_request(bad)


def test_dry_run_payload_extraction_requires_create_kwargs():
    payload = {"task_priority": 1, "framework_config": [{"gpu_count": 1}]}
    assert InspireClient._create_payload_from_dry_run({"create_kwargs": payload}) == payload
    assert InspireClient._create_payload_from_dry_run(
        {"data": {"create_kwargs": payload}}
    ) == payload
    with pytest.raises(InspireError, match="create_kwargs"):
        InspireClient._create_payload_from_dry_run({"dry_run": True})


def test_inspire_client_retries_transient_connection_error(tmp_path: Path, monkeypatch):
    _, run_dir = _make_config(tmp_path, rows=1, captions_per_image=1)
    client = InspireClient(load_json(run_dir / "run.json"))
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout=json.dumps(
                    {
                        "success": False,
                        "error": {
                            "message": "APIRequestContext.get: connect ECONNREFUSED 10.0.0.1:443"
                        },
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"success": True, "data": {"ok": True}}),
            stderr="",
        )

    monkeypatch.setattr("caption_farm.platform.subprocess.run", fake_run)
    monkeypatch.setattr("caption_farm.platform.time.sleep", lambda _: None)
    assert client._run(["project", "list"])["data"] == {"ok": True}
    assert calls == 2


def test_private_fixed_image_is_verified_from_stopped_notebook_identity(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, rows=1, captions_per_image=1)
    run = load_json(run_dir / "run.json")
    client = InspireClient(run)
    expected = run["platform"]["image"]

    def fake_run(args, **kwargs):
        if args[:2] == ["image", "list"]:
            return {"data": {"images": []}}
        return {
            "data": {
                "status": "STOPPED",
                "start_config": {"mirror_url": expected},
                "image": {
                    "address": "inspire-studio/dev-wjx:v-2.1",
                    "source": "SOURCE_PRIVATE",
                },
            }
        }

    client._run = fake_run  # type: ignore[method-assign]
    evidence = client.verify_image_ready()
    assert evidence["status"] == "READY"
    assert evidence["source"] == "notebook-image-identity"
    assert evidence["evidence"]["notebook_status"] == "STOPPED"

    assert InspireClient._status_has_fixed_image(
        {
            "framework_config": [
                {"image": expected, "gpu_count": 1, "instance_count": 1}
            ]
        },
        expected,
    )
    assert not InspireClient._status_has_fixed_image(
        {
            "framework_config": [
                {"image": expected + "-wrong", "gpu_count": 1, "instance_count": 1}
            ]
        },
        expected,
    )


def test_dataset_submit_retries_429_idempotently(tmp_path: Path, monkeypatch):
    _, run_dir = _make_config(tmp_path, rows=1, captions_per_image=1)
    client = InspireClient(load_json(run_dir / "run.json"))
    client.platform.update(
        {
            "submission_rate_limit_retries": 1,
            "submission_rate_limit_base_seconds": 0,
        }
    )
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout=json.dumps(
                    {"success": False, "error": "ValueError: API returned 429 Too Many Requests"}
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"success": True, "data": {"name": "job-a"}}),
            stderr="",
        )

    monkeypatch.setattr("caption_farm.platform.subprocess.run", fake_run)
    monkeypatch.setattr("caption_farm.platform.time.sleep", lambda _: None)
    monkeypatch.setattr(client, "job_status", lambda name: {})
    dry_run = {"create_kwargs": {"task_priority": 1}}
    result = client._submit_with_dataset(
        dry_run,
        name="job-a",
        command="true",
        target=LiveTarget("project-a", 1, 1, "group-a", "1,20,200", 1, 1, 0, 0),
    )
    assert result == {"name": "job-a"}
    assert calls == 2


def test_dataset_submit_429_recovers_already_created_job(tmp_path: Path, monkeypatch):
    _, run_dir = _make_config(tmp_path, rows=1, captions_per_image=1)
    client = InspireClient(load_json(run_dir / "run.json"))
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=json.dumps(
            {"success": False, "error": "ValueError: API returned 429 Too Many Requests"}
        ),
        stderr="",
    )
    monkeypatch.setattr("caption_farm.platform.subprocess.run", lambda *a, **k: completed)
    monkeypatch.setattr(client, "job_status", lambda name: {"name": name})
    result = client._submit_with_dataset(
        {"create_kwargs": {"task_priority": 1}},
        name="job-a",
        command="true",
        target=LiveTarget("project-a", 1, 1, "group-a", "1,20,200", 1, 1, 0, 0),
    )
    assert result["recovered_after_transient_submit_error"] is True
    assert result["job"]["name"] == "job-a"


def test_worker_tuning_is_bound_to_run_and_cli_overrides_win(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, rows=1, captions_per_image=1)
    run = load_json(run_dir / "run.json")
    tuning = {
        "schema": WORKER_TUNING_SCHEMA,
        "run_fingerprint": run["run_fingerprint"],
        "model_fingerprint": run["model"]["fingerprint"],
        "request_concurrency": 4,
        "max_num_seqs": 8,
        "claim_batch_size": 6,
        "request_timeout_seconds": 60,
        "max_tasks_per_worker": 16000,
    }
    (run_dir / "worker_tuning.json").write_text(json.dumps(tuning), encoding="utf-8")
    worker = Worker(
        run_dir,
        worker_id="tuned",
        request_concurrency=3,
        max_num_seqs=5,
        claim_batch_size=2,
    )
    assert worker.run["model"]["inference"]["request_concurrency"] == 3
    assert worker.run["model"]["inference"]["max_num_seqs"] == 5
    assert worker.run["model"]["inference"]["request_timeout_seconds"] == 60
    assert worker.claim_batch_size == 2
    assert worker.max_tasks == 16000
    worker.store.close()
    tuning["run_fingerprint"] = "wrong"
    (run_dir / "worker_tuning.json").write_text(json.dumps(tuning), encoding="utf-8")
    with pytest.raises(ValueError, match="another run"):
        Worker(run_dir, worker_id="wrong-tuning")


class _TimeoutEngine:
    def __init__(self) -> None:
        self.load_seconds = 0.0
        self.calls: list[int] = []
        self.closed = False

    def start(self, health_task: dict) -> None:
        return None

    def caption(self, task: dict) -> dict:
        ordinal = int(task["ordinal"])
        self.calls.append(ordinal)
        if ordinal == 0:
            raise TimeoutError("stalled local inference request")
        return {
            "caption": "A complete caption returned alongside the simulated transport failure.",
            "latency_seconds": 0.01,
        }

    def close(self) -> None:
        self.closed = True


def test_transport_timeout_requeues_batch_and_retires_worker(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, rows=3, captions_per_image=1)
    engine = _TimeoutEngine()
    worker = Worker(
        run_dir,
        worker_id="timeout-worker",
        engine=engine,
        request_concurrency=2,
        claim_batch_size=3,
    )
    assert worker.run_forever() == 0
    assert sorted(engine.calls) == [0, 1]
    assert engine.closed is True

    store = TaskStore(run_dir)
    snapshot = store.snapshot()
    assert snapshot["COMPLETE"] == 1
    assert snapshot["PENDING"] == 2
    assert snapshot["LEASED"] == 0
    assert snapshot["FAILED"] == 0
    store.close()
    events = [json.loads(line) for line in worker.worker_log.read_text().splitlines()]
    retirements = [event for event in events if event["event"] == "worker_retiring_unhealthy"]
    assert len(retirements) == 1
    assert retirements[0]["deferred_tasks"] == 2


def test_claim_heartbeat_expire_reclaim_and_idempotent_commit(tmp_path: Path):
    _, run_dir = _make_config(
        tmp_path, rows=2, captions_per_image=1, lease_seconds=0.3, heartbeat_seconds=0.05
    )
    store = TaskStore(run_dir)
    first = store.claim("worker-a", 1)
    assert first is not None
    old_expiry = first.expires_at
    assert store.heartbeat(first.lease_id, first.owner) > old_expiry
    time.sleep(0.35)
    second = store.claim("worker-b", 1)
    assert second is not None
    assert second.tasks[0]["ordinal"] == first.tasks[0]["ordinal"]
    task = second.tasks[0]
    record = {"caption": "A complete non-empty caption.", "caption_sha256": "x", "word_count": 4}
    assert store.commit(second, task, record) is True
    assert store.commit(second, task, record) is False
    snapshot = store.snapshot()
    assert snapshot["COMPLETE"] == 1
    assert snapshot["PENDING"] == 1
    assert snapshot["LEASED"] == 0
    store.close()


def test_batch_commit_resolves_one_lease_atomically(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, rows=2, captions_per_image=1)
    store = TaskStore(run_dir)
    lease = store.claim("batch-worker", 2)
    assert lease is not None
    items = [
        (
            task,
            {
                "caption": f"A complete batch caption for {task['task_key']}.",
                "caption_sha256": "test",
                "word_count": 6,
            },
        )
        for task in lease.tasks
    ]
    outcomes = store.commit_many(lease, items)
    assert outcomes == {task["task_key"]: True for task in lease.tasks}
    assert store.snapshot()["COMPLETE"] == 2
    assert store.snapshot()["active_leases"] == 0
    assert store.commit_many(lease, items) == {
        task["task_key"]: False for task in lease.tasks
    }
    assert store.snapshot()["stats"]["commits_created"] == 2
    assert store.snapshot()["stats"]["commits_reused"] == 0
    store.close()


def test_legacy_lock_settings_get_a_recoverable_runtime_window(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, rows=1, captions_per_image=1)
    run = load_json(run_dir / "run.json")
    run["queue"].update(
        {
            "lease_seconds": 900,
            "lock_timeout_seconds": 120,
            "lock_stale_seconds": 300,
        }
    )
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    store = TaskStore(run_dir)
    assert store.lock_timeout_seconds == 300
    assert store.lock_stale_seconds == 60
    store.close()


def test_failed_tasks_are_archived_and_atomically_requeued_for_repair(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, rows=1, captions_per_image=1)
    store = TaskStore(run_dir)
    for attempt in range(1, 4):
        lease = store.claim(f"worker-{attempt}", 1)
        assert lease is not None
        assert lease.tasks[0]["attempt_count"] == attempt
        outcome = store.nack(lease, lease.tasks[0], "permanent test error")
    assert outcome == "failed"
    assert store.snapshot()["FAILED"] == 1
    repair = store.repair_failed("fixed test decoder")
    assert repair["repaired"] == 1
    assert len(list(Path(repair["archive"]).glob("*/*.json"))) == 1
    snapshot = store.snapshot()
    assert snapshot["FAILED"] == 0
    assert snapshot["PENDING"] == 1
    assert snapshot["stats"]["repaired_failures"] == 1
    reclaimed = store.claim("fixed-worker", 1)
    assert reclaimed is not None
    assert reclaimed.tasks[0]["attempt_count"] == 1
    store.release(reclaimed, reclaimed.tasks, "test cleanup")
    store.close()


def test_oversized_images_are_resized_in_memory_for_short_vision_context(tmp_path: Path):
    image_path = tmp_path / "large.JPEG"
    Image.new("RGB", (4288, 2848), color=(30, 60, 90)).save(image_path, format="JPEG")
    run = {
        "model": {
            "inference": {
                "host": "127.0.0.1",
                "port": 18080,
                "max_image_side": 1024,
                "max_image_pixels": 1024 * 1024,
            }
        }
    }
    engine = VllmOpenAIEngine(run, "resize-test", tmp_path / "server.log")
    url = engine._image_data_url(image_path)
    header, encoded = url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    with Image.open(BytesIO(b64decode(encoded))) as resized:
        assert max(resized.size) == 1024
        assert resized.width * resized.height <= 1024 * 1024


def test_multiprocess_workers_compete_without_duplicate_visible_keys(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, rows=24, captions_per_image=2)
    processes = [
        mp.Process(target=_run_fake_worker, args=(str(run_dir), f"worker-{index}"))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    report = audit_run(run_dir)
    assert report["status"] == "passed"
    assert report["canonical_result_count"] == 48
    assert report["duplicate_keys"] == 0


def test_kill9_worker_lease_is_reclaimed(tmp_path: Path):
    _, run_dir = _make_config(
        tmp_path, rows=3, captions_per_image=1, lease_seconds=0.25, heartbeat_seconds=0.05
    )
    ready = tmp_path / "claimed"
    process = mp.Process(target=_claim_then_wait, args=(str(run_dir), str(ready)))
    process.start()
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.exists()
    os.kill(process.pid, signal.SIGKILL)
    process.join(timeout=5)
    assert process.exitcode == -signal.SIGKILL
    time.sleep(0.3)
    store = TaskStore(run_dir)
    reclaimed = store.claim("replacement", 2)
    assert reclaimed is not None
    assert {task["ordinal"] for task in reclaimed.tasks} == {0, 1}
    assert store.snapshot()["stats"]["lease_expirations"] == 1
    store.release(reclaimed, reclaimed.tasks, "test cleanup")
    store.close()


class _FakePlatform:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.submissions: list[str] = []
        self.jobs: list[dict] = []
        self.target = LiveTarget(
            project="project-a",
            project_weight=1,
            project_max_active_jobs=2,
            group="开发区-H100-cuda12.8版本-183核",
            quota="1,20,200",
            target_weight=1,
            target_max_active_jobs=2,
            available_gpus=1,
            low_priority_gpus=0,
        )

    def verify_image_ready(self):
        return {"status": "READY"}

    def discover_targets(self):
        return [self.target], {"targets": [self.target.key]}

    def list_jobs(self, *, active=True, keyword=None):
        return list(self.jobs)

    def dry_run_job(self, name, command, target):
        return {"task_priority": 1, "gpu_count": 1}

    def submit_job(self, name, command, target):
        if self.reject:
            raise InspireError("quota rejected")
        self.submissions.append(name)
        self.jobs.append(
            {
                "name": name,
                "status": "job_queuing",
                "project_name": target.project,
                "compute_group_name": target.group,
                "gpu_count": 1,
                "priority": 1,
            }
        )
        return {"dry_run": {"task_priority": 1}, "status": {"dataset_info": [{"path": "rclone-worker-1/imagenet/v1"}]}}

    def stop_job(self, name, *, check=True):
        return {"success": True}

    def diagnose_job(self, name):
        return {"name": name}


class _TransientPlatform(_FakePlatform):
    def __init__(self) -> None:
        super().__init__()
        self.discovery_calls = 0

    def discover_targets(self):
        self.discovery_calls += 1
        if self.discovery_calls == 1:
            raise InspireError("connect ECONNREFUSED 10.0.0.1:443")
        return super().discover_targets()


def test_controller_tuning_is_bound_and_overrides_active_limits(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    run = load_json(run_dir / "run.json")
    tuning = {
        "schema": CONTROLLER_TUNING_SCHEMA,
        "run_fingerprint": run["run_fingerprint"],
        "model_fingerprint": run["model"]["fingerprint"],
        "global_max_active_jobs": 16,
        "target_active_jobs": 16,
        "submission_burst": 1,
        "min_submit_interval_seconds": 30,
        "project_max_active_jobs": {"project-a": 16, "project-b": 16},
        "target_max_active_jobs": {
            "开发区-H100-cuda12.8版本-183核": 16,
        },
    }
    (run_dir / "controller_tuning.json").write_text(
        json.dumps(tuning), encoding="utf-8"
    )
    controller = Controller(run_dir, client=_FakePlatform(), once=True)
    assert controller.config["global_max_active_jobs"] == 16
    assert controller.config["target_active_jobs"] == 16
    assert controller.config["submission_burst"] == 1
    assert controller.config["min_submit_interval_seconds"] == 30
    assert {item["max_active_jobs"] for item in controller.platform["projects"]} == {16}
    assert {item["max_active_jobs"] for item in controller.platform["targets"]} == {16}
    controller.queue.close()

    tuning["run_fingerprint"] = "wrong"
    (run_dir / "controller_tuning.json").write_text(
        json.dumps(tuning), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="another run"):
        Controller(run_dir, client=_FakePlatform(), once=True)


def test_controller_reconciles_existing_jobs_after_restart(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _FakePlatform()
    first = Controller(run_dir, client=platform, once=True)
    assert first.run_once() == -1
    first.queue.close()
    assert len(platform.submissions) == 1
    second = Controller(run_dir, client=platform, once=True)
    assert second.state["restart_count"] == 1
    assert second.run_once() == -1
    second.queue.close()
    assert len(platform.submissions) == 1


def test_first_worker_after_idle_resets_stale_progress_timer(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _FakePlatform()
    controller = Controller(run_dir, client=platform, once=True)
    controller.state["last_progress_unix"] = 0
    controller.state["last_complete_count"] = 0
    started = time.time()
    assert controller.run_once() == -1
    assert controller.state["last_progress_unix"] >= started
    assert controller.state["last_complete_count"] == 0
    controller.queue.close()


def test_controller_backs_off_after_mocked_quota_rejection(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _FakePlatform(reject=True)
    controller = Controller(run_dir, client=platform, once=True)
    assert controller.run_once() == -1
    target_state = controller.state["target_state"][platform.target.key]
    assert target_state["consecutive_rejections"] == 1
    assert target_state["next_allowed_unix"] > time.time()
    controller.queue.close()


def test_controller_recovers_from_transient_platform_outage(tmp_path: Path, monkeypatch):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _TransientPlatform()
    monkeypatch.setattr("caption_farm.controller.time.sleep", lambda _: None)
    controller = Controller(run_dir, client=platform, once=True)
    assert controller.run_forever() == 0
    assert platform.discovery_calls == 2
    assert len(platform.submissions) == 1
    assert controller.state["consecutive_platform_transient_failures"] == 0
    assert not (run_dir / "NEEDS_ATTENTION.json").exists()


def test_controller_recovers_from_stale_queue_lock_timeout(tmp_path: Path, monkeypatch):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    controller = Controller(run_dir, client=_FakePlatform(), once=True)
    calls = 0

    def run_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("timed out acquiring shared lock /tmp/claim.lock")
        return -1

    monkeypatch.setattr(controller, "run_once", run_once)
    monkeypatch.setattr("caption_farm.controller.time.sleep", lambda _: None)
    assert controller.run_forever() == 0
    assert calls == 2
    assert controller.state["consecutive_queue_lock_timeouts"] == 0
    assert not (run_dir / "NEEDS_ATTENTION.json").exists()


def test_controller_exception_alert_survives_queue_snapshot_failure(
    tmp_path: Path, monkeypatch
):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    controller = Controller(run_dir, client=_FakePlatform(), once=True)
    controller._write_status(controller.queue.snapshot(), [], None)
    monkeypatch.setattr(
        controller, "run_once", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        controller.queue,
        "snapshot",
        lambda: (_ for _ in ()).throw(
            TimeoutError("timed out acquiring shared lock /tmp/claim.lock")
        ),
    )
    assert controller.run_forever() == 2
    attention = load_json(run_dir / "NEEDS_ATTENTION.json")
    assert attention["reason"] == "controller_exception"
    assert "boom" in attention["details"]["error"]


class _DiagnosingPlatform(_FakePlatform):
    def __init__(self, diagnosis: dict) -> None:
        super().__init__()
        self.diagnosis = diagnosis

    def diagnose_job(self, name):
        return self.diagnosis


def test_non_preemption_job_failure_needs_attention(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _DiagnosingPlatform({"logs": "Traceback: worker crashed; exit code 1"})
    controller = Controller(run_dir, client=platform, once=True)
    controller.config["worker_failure_attention_threshold"] = 1
    assert controller.run_once() == -1
    platform.jobs[0]["status"] = "job_failed"
    assert controller.run_once() == 2
    attention = load_json(run_dir / "NEEDS_ATTENTION.json")
    assert attention["reason"] == "non_preemption_job_failure"
    assert attention["details"]["jobs"][0]["outcome"] == "worker_failure"
    controller.queue.close()


def test_isolated_worker_failure_is_replaced_without_stopping_farm(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _DiagnosingPlatform({"logs": "Traceback: worker crashed; exit code 1"})
    controller = Controller(run_dir, client=platform, once=True)
    controller.config["worker_failure_attention_threshold"] = 3
    assert controller.run_once() == -1
    platform.jobs[0]["status"] = "job_failed"
    assert controller.run_once() == -1
    assert not (run_dir / "NEEDS_ATTENTION.json").exists()
    assert len(platform.submissions) == 2
    controller.queue.close()


def test_transient_diagnosis_failure_is_deferred_and_retried(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _DiagnosingPlatform(
        {
            "events": {"success": False, "error": "API returned 429 Too Many Requests"},
            "logs": {"error": "API returned 429 Too Many Requests"},
        }
    )
    controller = Controller(run_dir, client=platform, once=True)
    controller.config["diagnosis_retry_seconds"] = 0
    assert controller.run_once() == -1
    failed_name = platform.jobs[0]["name"]
    platform.jobs[0]["status"] = "job_failed"
    assert controller.run_once() == -1
    record = controller.state["jobs"][failed_name]
    assert record["terminal_outcome"] == "diagnosis_unavailable"
    assert record["diagnosed_at"] is None
    assert not (run_dir / "NEEDS_ATTENTION.json").exists()

    platform.diagnosis = {"events": "worker was preempted by higher priority"}
    assert controller.run_once() == -1
    assert record["terminal_outcome"] == "preemption"
    assert record["diagnosed_at"] is not None
    controller.queue.close()


def test_preemption_is_reported_but_replaced_without_attention(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _DiagnosingPlatform({"events": "worker was preempted by higher priority"})
    controller = Controller(run_dir, client=platform, once=True)
    assert controller.run_once() == -1
    platform.jobs[0]["status"] = "job_failed"
    assert controller.run_once() == -1
    assert not (run_dir / "NEEDS_ATTENTION.json").exists()
    status = load_json(run_dir / "status.json")
    assert status["platform"]["terminal_outcomes"]["counts"]["preemption"] == 1
    controller.queue.close()


def test_queue_lock_worker_failure_is_reported_and_replaced(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _DiagnosingPlatform(
        {"logs": "TimeoutError: timed out acquiring shared lock /run/state/claim.lock"}
    )
    controller = Controller(run_dir, client=platform, once=True)
    assert controller.run_once() == -1
    platform.jobs[0]["status"] = "job_failed"
    assert controller.run_once() == -1
    assert not (run_dir / "NEEDS_ATTENTION.json").exists()
    status = load_json(run_dir / "status.json")
    assert (
        status["platform"]["terminal_outcomes"]["counts"]["queue_lock_contention"]
        == 1
    )
    controller.queue.close()


def test_zero_active_jobs_needs_attention_after_grace_period(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _FakePlatform()
    controller = Controller(run_dir, client=platform, once=True)
    controller.config["zero_active_attention_seconds"] = 0
    controller.state["zero_active_since_unix"] = time.time() - 1
    assert controller.run_once() == 2
    attention = load_json(run_dir / "NEEDS_ATTENTION.json")
    assert attention["reason"] == "zero_active_jobs"
    controller.queue.close()


def test_controller_wait_detects_dead_controller_process(tmp_path: Path, monkeypatch):
    _, run_dir = _make_config(tmp_path)
    (run_dir / "controller.pid").write_text("999999999\n", encoding="utf-8")
    monkeypatch.setattr("caption_farm.controller.time.sleep", lambda _: None)
    assert wait_for_controller(run_dir, interval_seconds=0.01) == 2
    attention = load_json(run_dir / "NEEDS_ATTENTION.json")
    assert attention["reason"] == "controller_process_missing"


def test_foreground_supervisor_returns_compact_attention_signal(
    tmp_path: Path, capsys
):
    _, run_dir = _make_config(tmp_path)
    controller = Controller(run_dir, once=True)
    status = controller._write_status(controller.queue.snapshot(), [], None)
    assert controller._needs_attention("test_attention", status, sample="evidence") == 2
    controller.queue.close()
    assert supervise_controller(run_dir) == 2
    signal = json.loads(capsys.readouterr().err)
    assert signal["event"] == "NEEDS_ATTENTION"
    assert signal["reason"] == "test_attention"
    assert signal["details"] == {"sample": "evidence"}
    assert "jobs" not in signal
    assert not (run_dir / "controller.pid").exists()


def test_controller_clears_stale_fixed_image_circuit_after_recovery(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _FakePlatform()
    controller = Controller(run_dir, client=platform, once=True)
    target_state = controller._target_state(platform.target.key)
    target_state.update(
        {
            "consecutive_rejections": 7,
            "next_allowed_unix": time.time() + 600,
            "circuit_open_until_unix": time.time() + 3600,
            "last_error": (
                "fixed image is not visible as READY/SUCCESS and no live "
                "running-notebook evidence matched"
            ),
        }
    )
    controller._recover_fixed_image_backoffs()
    assert target_state["consecutive_rejections"] == 0
    assert target_state["next_allowed_unix"] == 0
    assert target_state["circuit_open_until_unix"] == 0
    assert target_state["last_error"] is None
    controller.queue.close()


def test_controller_does_not_treat_low_priority_queueing_as_business_stall(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, platform_enabled=True)
    platform = _FakePlatform()
    platform.jobs.append(
        {
            "name": "qcf-placeholder",
            "status": "job_queuing",
            "project_name": platform.target.project,
            "compute_group_name": platform.target.group,
            "gpu_count": 1,
        }
    )
    controller = Controller(run_dir, client=platform, once=True)
    platform.jobs[0]["name"] = controller.job_prefix + "queued"
    controller.config["no_progress_attention_seconds"] = 0
    controller.state["last_progress_unix"] = 0
    assert controller.run_once() == -1
    assert not (run_dir / "NEEDS_ATTENTION.json").exists()
    controller.queue.close()


def test_attention_requires_paused_drained_acknowledgement_and_is_archived(tmp_path: Path):
    _, run_dir = _make_config(tmp_path)
    controller = Controller(run_dir, once=True)
    status = controller._write_status(controller.queue.snapshot(), [], None)
    assert controller._needs_attention("business_no_progress", status) == 2
    controller.queue.close()
    with pytest.raises(RuntimeError, match="paused"):
        acknowledge_attention(run_dir, "test repair")
    (run_dir / "PAUSE").touch()
    result = acknowledge_attention(run_dir, "test repair")
    assert result["status"] == "acknowledged"
    assert Path(result["archive"]).is_file()
    assert not (run_dir / "NEEDS_ATTENTION.json").exists()
    recovered_state = load_json(run_dir / "controller_state.json")
    assert recovered_state["phase"] == "RECOVERING"
    assert recovered_state["last_complete_count"] == status["queue"]["COMPLETE"]
    assert recovered_state["last_progress_unix"] > 0


def test_zero_active_acknowledgement_resets_submission_circuits(tmp_path: Path):
    _, run_dir = _make_config(tmp_path)
    controller = Controller(run_dir, once=True)
    target = controller._target_state("project-a|group-a|1,20,200")
    target.update(
        {
            "consecutive_rejections": 4,
            "next_allowed_unix": time.time() + 600,
            "circuit_open_until_unix": time.time() + 3600,
            "last_error": "429 Too Many Requests",
        }
    )
    status = controller._write_status(controller.queue.snapshot(), [], None)
    assert controller._needs_attention("zero_active_jobs", status) == 2
    controller.queue.close()
    (run_dir / "PAUSE").touch()
    acknowledge_attention(run_dir, "429 retry logic repaired")
    recovered = load_json(run_dir / "controller_state.json")
    recovered_target = recovered["target_state"]["project-a|group-a|1,20,200"]
    assert recovered["zero_active_since_unix"] > 0
    assert recovered_target["consecutive_rejections"] == 0
    assert recovered_target["next_allowed_unix"] == 0
    assert recovered_target["circuit_open_until_unix"] == 0
    assert recovered_target["last_error"] is None


def test_cpu_fake_model_end_to_end_publishes_atomically(tmp_path: Path):
    _, run_dir = _make_config(tmp_path, rows=5, captions_per_image=3)
    assert Worker(run_dir, worker_id="fake-e2e").run_forever() == 0
    metadata = publish_run(run_dir)
    published = Path(metadata["path"])
    rows = [json.loads(line) for line in published.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert all(row["caption_count"] == 4 for row in rows)
    assert all(row["captions"][0]["source"] == "original" for row in rows)
    assert metadata["rows"] == 5
    assert not list(published.parent.glob(".*.tmp"))
