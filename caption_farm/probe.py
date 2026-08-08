from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any

from .io import atomic_write_json, load_json, utc_now
from .platform import InspireClient
from .queue import TaskStore


PROBE_SCHEMA = "imagenet_caption_farm_formal_probe_v1"


def _worker_command(run: dict[str, Any], run_dir: Path, name: str, max_tasks: int) -> str:
    platform = run["platform"]
    python = str(platform.get("runtime_python") or sys.executable)
    script = str(platform["worker_script"])
    return (
        f"cd {shlex.quote(str(platform['repository_path']))} && "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
        f"{shlex.quote(python)} {shlex.quote(script)} worker "
        f"--run-dir {shlex.quote(str(run_dir))} --worker-id {shlex.quote(name)} "
        f"--max-tasks {int(max_tasks)}"
    )


def submit_probe(
    run_dir: Path,
    *,
    max_tasks: int = 16,
    client: InspireClient | None = None,
) -> dict[str, Any]:
    if max_tasks < 1 or max_tasks > 64:
        raise ValueError("formal probe max_tasks must be in [1, 64]")
    probe_path = run_dir / "formal_probe.json"
    if probe_path.exists():
        existing = load_json(probe_path)
        if existing.get("phase") in {"SUBMITTED", "PASSED"}:
            return existing
        raise RuntimeError(f"existing formal probe needs explicit diagnosis: {probe_path}")
    run = load_json(run_dir / "run.json")
    store = TaskStore(run_dir)
    try:
        before = store.snapshot()
    finally:
        store.close()
    platform_client = client or InspireClient(run)
    targets, discovery = platform_client.discover_targets()
    target = sorted(targets, key=lambda item: (-item.weight, item.key))[0]
    compact_time = utc_now().replace("-", "").replace(":", "").replace("+00:00", "Z")
    name = f"qcp-{run['run_slug'][:8]}-{compact_time[4:15].lower()}"
    command = _worker_command(run, run_dir, name, max_tasks)
    evidence = platform_client.submit_job(name, command, target)
    report = {
        "schema": PROBE_SCHEMA,
        "phase": "SUBMITTED",
        "submitted_at": utc_now(),
        "run_fingerprint": run["run_fingerprint"],
        "model_fingerprint": run["model"]["fingerprint"],
        "job_name": name,
        "workspace": run["platform"]["workspace"],
        "max_tasks": max_tasks,
        "queue_before": before,
        "target": target.__dict__ | {"key": target.key},
        "discovery": discovery,
        "submission_evidence": evidence,
    }
    atomic_write_json(probe_path, report)
    return report


def _read_worker_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if isinstance(value, dict):
                events.append(value)
    return events


def verify_probe(
    run_dir: Path,
    *,
    client: InspireClient | None = None,
) -> dict[str, Any]:
    probe_path = run_dir / "formal_probe.json"
    report = load_json(probe_path)
    if report.get("schema") != PROBE_SCHEMA:
        raise ValueError("unexpected formal probe schema")
    if report.get("phase") == "PASSED":
        return report
    run = load_json(run_dir / "run.json")
    if report.get("run_fingerprint") != run["run_fingerprint"]:
        raise ValueError("formal probe belongs to another run")
    name = str(report["job_name"])
    platform_client = client or InspireClient(run)
    status = platform_client.job_status(name)
    status_name = str(status.get("status") or "").lower()
    if not any(marker in status_name for marker in ("succeed", "success", "completed")):
        diagnosis = platform_client.diagnose_job(name)
        report.update(
            {
                "phase": "FAILED",
                "verified_at": utc_now(),
                "job_status": status,
                "diagnosis": diagnosis,
                "error": f"formal probe did not succeed: {status_name or '<missing>'}",
            }
        )
        atomic_write_json(probe_path, report)
        raise RuntimeError(report["error"])
    dataset_info = status.get("dataset_info") or []
    expected_path = run["dataset"]["platform_path"]
    if not any(
        isinstance(item, dict) and item.get("path") == expected_path for item in dataset_info
    ):
        raise RuntimeError("formal probe status lost the official ImageNet dataset_info")
    priority_ok = (
        str(status.get("priority_name") or "") == "1"
        or int(status.get("task_priority") or 0) == 1
    )
    if not priority_ok:
        raise RuntimeError("formal probe did not run at priority=1")
    store = TaskStore(run_dir)
    try:
        after = store.snapshot()
    finally:
        store.close()
    before = report["queue_before"]
    delta = int(after["COMPLETE"]) - int(before["COMPLETE"])
    expected = int(report["max_tasks"])
    if delta != expected:
        raise RuntimeError(f"formal probe committed {delta} tasks, expected exactly {expected}")
    if int(after["FAILED"]) != int(before["FAILED"]) or int(after["LEASED"]) != 0:
        raise RuntimeError("formal probe left failed or leased queue entries")
    events_path = run_dir / "workers" / f"{name}.jsonl"
    events = _read_worker_events(events_path)
    ready = [event for event in events if event.get("event") == "model_ready"]
    commits = [event for event in events if event.get("event") == "caption_committed"]
    completed = [event for event in events if event.get("event") == "worker_completed"]
    failed = [event for event in events if event.get("event") in {"worker_failed", "caption_failed"}]
    if len(ready) != 1 or len(commits) != expected or len(completed) != 1 or failed:
        raise RuntimeError(
            "formal probe worker evidence is incomplete: "
            f"ready={len(ready)} commits={len(commits)} completed={len(completed)} failed={len(failed)}"
        )
    report.update(
        {
            "phase": "PASSED",
            "verified_at": utc_now(),
            "job_status": status,
            "queue_after": after,
            "complete_delta": delta,
            "worker_events": str(events_path),
            "worker_model_ready": ready[0],
            "worker_completed": completed[-1],
        }
    )
    atomic_write_json(probe_path, report)
    return report
