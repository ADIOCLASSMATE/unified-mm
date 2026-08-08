from __future__ import annotations

import json
import math
import os
import random
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from .io import append_jsonl, atomic_write_json, load_json, utc_now
from .platform import (
    DatasetMountError,
    InspireClient,
    InspireError,
    LiveTarget,
    is_transient_platform_error,
)
from .publish import publish_run
from .queue import TaskStore


CONTROLLER_STATE_SCHEMA = "imagenet_caption_farm_controller_state_v1"
CONTROLLER_TUNING_SCHEMA = "imagenet_caption_farm_controller_tuning_v1"
ACTIVE_STATUS_MARKERS = ("running", "queuing", "pending", "creating", "preparing")


class PlatformClient(Protocol):
    def verify_image_ready(self) -> dict[str, Any]: ...
    def discover_targets(self) -> tuple[list[LiveTarget], dict[str, Any]]: ...
    def list_jobs(self, *, active: bool = True, keyword: str | None = None) -> list[dict[str, Any]]: ...
    def dry_run_job(self, name: str, command: str, target: LiveTarget) -> dict[str, Any]: ...
    def submit_job(self, name: str, command: str, target: LiveTarget) -> dict[str, Any]: ...
    def stop_job(self, name: str, *, check: bool = True) -> dict[str, Any]: ...
    def diagnose_job(self, name: str) -> dict[str, Any]: ...


def _is_active(status: str) -> bool:
    lowered = status.lower()
    return any(marker in lowered for marker in ACTIVE_STATUS_MARKERS)


def _is_preemption(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("preempt", "evict", "抢占", "驱逐"))


def _is_terminal_failure(status: str) -> bool:
    lowered = status.lower()
    return any(
        marker in lowered
        for marker in ("fail", "error", "oom", "cancel", "stopped", "terminated")
    )


def _classify_terminal_outcome(status: str, diagnosis_text: str) -> str:
    lowered_status = status.lower()
    lowered = diagnosis_text.lower()
    if "stopped" in lowered_status:
        return "operator_stopped"
    if _is_preemption(lowered):
        return "preemption"
    if "timed out acquiring shared lock" in lowered:
        return "queue_lock_contention"
    if "traceback" in lowered or "exit code 1" in lowered:
        return "worker_failure"
    if any(
        marker in lowered
        for marker in (
            "unschedulable",
            "insufficient nvidia.com/gpu",
            "insufficient memory",
            "no nodes are available",
        )
    ):
        return "capacity"
    if is_transient_platform_error(diagnosis_text) and any(
        marker in lowered
        for marker in ('"success": false', "apierror", '"error"')
    ):
        return "diagnosis_unavailable"
    return "worker_failure"


def _classify_rejection(text: str) -> str:
    lowered = text.lower()
    if "429" in lowered or "too many requests" in lowered:
        return "rate_limit"
    if any(marker in lowered for marker in ("quota", "配额", "budget", "余额")):
        return "quota"
    if any(marker in lowered for marker in ("capacity", "resource", "insufficient", "资源不足", "no node")):
        return "capacity"
    if any(marker in lowered for marker in ("priority", "policy", "permission", "优先级", "策略", "权限")):
        return "policy"
    if "dataset_info" in lowered or "dataset" in lowered:
        return "dataset_mount"
    return "other"


def load_controller_tuning(
    run_dir: Path, run: dict[str, Any]
) -> dict[str, Any] | None:
    path = run_dir / "controller_tuning.json"
    if not path.is_file():
        return None
    tuning = load_json(path)
    if tuning.get("schema") != CONTROLLER_TUNING_SCHEMA:
        raise ValueError(f"unexpected controller tuning schema in {path}")
    if tuning.get("run_fingerprint") != run["run_fingerprint"]:
        raise ValueError("controller tuning belongs to another run")
    if tuning.get("model_fingerprint") != run["model"]["fingerprint"]:
        raise ValueError("controller tuning belongs to another model snapshot")

    global_max = int(tuning.get("global_max_active_jobs") or 0)
    target = int(tuning.get("target_active_jobs") or 0)
    burst = int(tuning.get("submission_burst") or 0)
    if not (1 <= target <= global_max):
        raise ValueError("controller tuning target must be within the global limit")
    if not (1 <= burst <= global_max):
        raise ValueError("controller tuning submission_burst must be within the global limit")
    if float(tuning.get("min_submit_interval_seconds", -1)) < 0:
        raise ValueError("controller tuning min_submit_interval_seconds cannot be negative")
    if "reconcile_interval_seconds" in tuning and float(
        tuning["reconcile_interval_seconds"]
    ) <= 0:
        raise ValueError("controller tuning reconcile_interval_seconds must be positive")
    if "zero_active_attention_seconds" in tuning and float(
        tuning["zero_active_attention_seconds"]
    ) <= 0:
        raise ValueError("controller tuning zero_active_attention_seconds must be positive")

    expected_projects = {str(item["name"]) for item in run["platform"]["projects"]}
    project_limits = tuning.get("project_max_active_jobs")
    if not isinstance(project_limits, dict) or set(project_limits) != expected_projects:
        raise ValueError("controller tuning project limits must exactly match the run whitelist")
    expected_targets = {str(item["group"]) for item in run["platform"]["targets"]}
    target_limits = tuning.get("target_max_active_jobs")
    if not isinstance(target_limits, dict) or set(target_limits) != expected_targets:
        raise ValueError("controller tuning target limits must exactly match the run whitelist")
    for label, limits in (("project", project_limits), ("target", target_limits)):
        if any(int(value) < 1 for value in limits.values()):
            raise ValueError(f"controller tuning {label} limits must be positive")
    return tuning


class Controller:
    def __init__(
        self,
        run_dir: Path,
        *,
        client: PlatformClient | None = None,
        once: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.run = load_json(run_dir / "run.json")
        self.queue = TaskStore(run_dir)
        self.tuning = load_controller_tuning(run_dir, self.run)
        if self.tuning is not None:
            controller = self.run["controller"]
            for key in (
                "global_max_active_jobs",
                "target_active_jobs",
                "submission_burst",
                "min_submit_interval_seconds",
            ):
                controller[key] = self.tuning[key]
            for key in (
                "reconcile_interval_seconds",
                "zero_active_attention_seconds",
            ):
                if key in self.tuning:
                    controller[key] = self.tuning[key]
            project_limits = self.tuning["project_max_active_jobs"]
            for project in self.run["platform"]["projects"]:
                project["max_active_jobs"] = int(project_limits[project["name"]])
            target_limits = self.tuning["target_max_active_jobs"]
            for target in self.run["platform"]["targets"]:
                target["max_active_jobs"] = int(target_limits[target["group"]])
        self.config = self.run["controller"]
        self.platform = self.run["platform"]
        self.client = client or InspireClient(self.run)
        self.once = once
        self.dry_run = dry_run
        self.state_path = run_dir / "controller_state.json"
        self.status_path = run_dir / "status.json"
        self.events_path = run_dir / "events.jsonl"
        self.attention_path = run_dir / "NEEDS_ATTENTION.json"
        self.completed_path = run_dir / "COMPLETED.json"
        self.stop_requested = False
        self._previous_handlers: dict[int, Any] = {}
        self.state = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = load_json(self.state_path)
            if state.get("run_fingerprint") != self.run["run_fingerprint"]:
                raise ValueError("controller state belongs to another run")
            state["restart_count"] = int(state.get("restart_count", 0)) + 1
            state["last_restart_at"] = utc_now()
            atomic_write_json(self.state_path, state)
            return state
        state = {
            "schema": CONTROLLER_STATE_SCHEMA,
            "run_fingerprint": self.run["run_fingerprint"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "restart_count": 0,
            "next_job_sequence": 1,
            "last_submit_unix": 0.0,
            "last_progress_unix": time.time(),
            "last_complete_count": 0,
            "consecutive_platform_transient_failures": 0,
            "consecutive_queue_lock_timeouts": 0,
            "zero_active_since_unix": None,
            "target_state": {},
            "jobs": {},
            "smooth_weights": {},
            "phase": "RUNNING",
        }
        atomic_write_json(self.state_path, state)
        return state

    def _save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.state_path, self.state)

    def _emit(self, event: str, **fields: Any) -> None:
        append_jsonl(
            self.events_path,
            {
                "timestamp": utc_now(),
                "event": event,
                "component": "controller",
                "run_fingerprint": self.run["run_fingerprint"],
                **fields,
            },
        )

    def request_stop(self, signum: int | None = None, frame: Any = None) -> None:
        self.stop_requested = True
        self._emit("controller_signal", signal=signum)

    @property
    def job_prefix(self) -> str:
        return f"qcf-{self.run['run_slug'][:8]}-{self.run['model']['fingerprint'][:6]}-"

    def _job_name(self) -> str:
        sequence = int(self.state["next_job_sequence"])
        return f"{self.job_prefix}{sequence:05d}"

    def _worker_command(self, name: str) -> str:
        python = str(self.platform.get("runtime_python") or sys.executable)
        script = str(self.platform["worker_script"])
        return (
            f"cd {shlex.quote(str(self.platform['repository_path']))} && "
            f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            f"{shlex.quote(python)} {shlex.quote(script)} worker "
            f"--run-dir {shlex.quote(str(self.run_dir))} --worker-id {shlex.quote(name)}"
        )

    def _target_state(self, key: str) -> dict[str, Any]:
        return self.state["target_state"].setdefault(
            key,
            {
                "consecutive_rejections": 0,
                "next_allowed_unix": 0.0,
                "circuit_open_until_unix": 0.0,
                "last_error": None,
                "last_success_at": None,
            },
        )

    def _record_rejection(self, target: LiveTarget, error: str) -> None:
        now = time.time()
        state = self._target_state(target.key)
        state["consecutive_rejections"] += 1
        state["last_error"] = error[-4000:]
        base = float(self.config["rejection_backoff_base_seconds"])
        maximum = float(self.config["rejection_backoff_max_seconds"])
        delay = min(maximum, base * 2 ** (state["consecutive_rejections"] - 1))
        delay *= random.uniform(0.8, 1.2)
        state["next_allowed_unix"] = now + delay
        threshold = int(self.config["circuit_breaker_rejections"])
        if state["consecutive_rejections"] >= threshold:
            state["circuit_open_until_unix"] = now + float(
                self.config["circuit_breaker_recovery_seconds"]
            )
        self._emit(
            "submission_rejected",
            target=target.key,
            classification=_classify_rejection(error),
            consecutive_rejections=state["consecutive_rejections"],
            next_allowed_unix=state["next_allowed_unix"],
            circuit_open_until_unix=state["circuit_open_until_unix"],
            error=error[-4000:],
        )

    def _record_success(self, target: LiveTarget) -> None:
        state = self._target_state(target.key)
        state["consecutive_rejections"] = 0
        state["next_allowed_unix"] = 0.0
        state["circuit_open_until_unix"] = 0.0
        state["last_error"] = None
        state["last_success_at"] = utc_now()

    def _recover_fixed_image_backoffs(self) -> None:
        affected = [
            (key, state)
            for key, state in self.state["target_state"].items()
            if str(state.get("last_error") or "").startswith(
                "fixed image is not visible as READY/SUCCESS"
            )
        ]
        if not affected:
            return
        try:
            evidence = self.client.verify_image_ready()
        except InspireError:
            return
        for _, state in affected:
            state["consecutive_rejections"] = 0
            state["next_allowed_unix"] = 0.0
            state["circuit_open_until_unix"] = 0.0
            state["last_error"] = None
        self._save_state()
        self._emit(
            "fixed_image_readiness_recovered",
            targets=[key for key, _ in affected],
            evidence=evidence,
        )

    def _eligible_targets(
        self, targets: list[LiveTarget], jobs: list[dict[str, Any]]
    ) -> list[LiveTarget]:
        now = time.time()
        project_counts = Counter()
        group_counts = Counter()
        target_counts = Counter()
        for job in jobs:
            if not str(job.get("name") or "").startswith(self.job_prefix):
                continue
            if not _is_active(str(job.get("status") or "")):
                continue
            project = str(job.get("project_name") or "")
            group = str(job.get("compute_group_name") or "")
            project_counts[project] += 1
            group_counts[group] += 1
            target_counts[f"{project}|{group}|{self.platform['quota']}"] += 1
        eligible: list[LiveTarget] = []
        for target in targets:
            state = self._target_state(target.key)
            if now < float(state["next_allowed_unix"]) or now < float(
                state["circuit_open_until_unix"]
            ):
                continue
            if project_counts[target.project] >= target.project_max_active_jobs:
                continue
            if group_counts[target.group] >= target.target_max_active_jobs:
                continue
            if target_counts[target.key] >= min(
                target.project_max_active_jobs, target.target_max_active_jobs
            ):
                continue
            eligible.append(target)
        return eligible

    def _choose_target(self, targets: list[LiveTarget]) -> LiveTarget:
        total_weight = sum(target.weight for target in targets)
        best: LiveTarget | None = None
        best_value = -math.inf
        for target in targets:
            value = float(self.state["smooth_weights"].get(target.key, 0)) + target.weight
            self.state["smooth_weights"][target.key] = value
            if value > best_value:
                best = target
                best_value = value
        assert best is not None
        self.state["smooth_weights"][best.key] -= total_weight
        return best

    def _reconcile_jobs(self, live_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        farm_jobs = [
            job for job in live_jobs if str(job.get("name") or "").startswith(self.job_prefix)
        ]
        state_changed = False
        for job in farm_jobs:
            name = str(job["name"])
            record = self.state["jobs"].setdefault(name, {})
            previous = record.get("status")
            record.update(
                {
                    "status": job.get("status"),
                    "project": job.get("project_name"),
                    "group": job.get("compute_group_name"),
                    "gpu_count": job.get("gpu_count"),
                    "priority": job.get("priority"),
                    "last_seen_at": utc_now(),
                }
            )
            if previous != record["status"]:
                state_changed = True
                self._emit(
                    "job_status_changed",
                    job=name,
                    previous_status=previous,
                    status=record["status"],
                )
            was_owned_and_active = previous == "SUBMITTED" or (
                previous is not None and _is_active(str(previous))
            )
            diagnosis_retry_due = (
                record.get("terminal_outcome") == "diagnosis_unavailable"
                and time.time() >= float(record.get("diagnosis_retry_after_unix", 0))
            )
            if (
                (was_owned_and_active or diagnosis_retry_due)
                and _is_terminal_failure(str(record["status"] or ""))
                and (not record.get("diagnosed_at") or diagnosis_retry_due)
            ):
                diagnosis = self.client.diagnose_job(name)
                diagnosis_path = self.run_dir / "diagnostics" / f"{name}.json"
                atomic_write_json(diagnosis_path, diagnosis)
                diagnosis_text = json.dumps(diagnosis, ensure_ascii=False, sort_keys=True)
                record["diagnosis_path"] = str(diagnosis_path)
                record["preemption"] = _is_preemption(diagnosis_text)
                record["terminal_outcome"] = _classify_terminal_outcome(
                    str(record["status"] or ""), diagnosis_text
                )
                record["last_diagnosis_attempt_at"] = utc_now()
                record["oom"] = any(
                    marker in diagnosis_text.lower()
                    for marker in ("out of memory", "cuda oom", "outofmemoryerror")
                )
                if record["terminal_outcome"] == "diagnosis_unavailable":
                    record["diagnosed_at"] = None
                    record["diagnosis_retry_after_unix"] = time.time() + float(
                        self.config.get("diagnosis_retry_seconds", 300)
                    )
                    self._emit(
                        "job_terminal_diagnosis_deferred",
                        job=name,
                        status=record["status"],
                        retry_after_unix=record["diagnosis_retry_after_unix"],
                        diagnosis_path=str(diagnosis_path),
                    )
                else:
                    record["diagnosed_at"] = utc_now()
                    record.pop("diagnosis_retry_after_unix", None)
                self._emit(
                    "job_terminal_diagnosed",
                    job=name,
                    status=record["status"],
                    preemption=record["preemption"],
                    terminal_outcome=record["terminal_outcome"],
                    oom=record["oom"],
                    diagnosis_path=str(diagnosis_path),
                )
                if record["terminal_outcome"] == "worker_failure":
                    now = time.time()
                    record["worker_failure_unix"] = now
                    window = float(
                        self.config.get("worker_failure_attention_window_seconds", 900)
                    )
                    recent_failures = [
                        item
                        for item in self.state.get("jobs", {}).values()
                        if now - float(item.get("worker_failure_unix", 0)) <= window
                    ]
                    threshold = int(
                        self.config.get("worker_failure_attention_threshold", 3)
                    )
                    if len(recent_failures) >= threshold:
                        record["failure_attention_pending"] = True
                state_changed = True
            elif (
                _is_terminal_failure(str(record.get("status") or ""))
                and record.get("diagnosed_at")
            ):
                # Backfill classifications for runs created before outcome
                # classes existed or whose classifier has since become more
                # precise. Historical failures are surfaced in status, but only
                # newly observed worker failures trigger a fresh halt.
                diagnosis_path = Path(str(record.get("diagnosis_path") or ""))
                if diagnosis_path.is_file():
                    diagnosis_text = json.dumps(
                        load_json(diagnosis_path), ensure_ascii=False, sort_keys=True
                    )
                    outcome = _classify_terminal_outcome(
                        str(record.get("status") or ""), diagnosis_text
                    )
                    if record.get("terminal_outcome") != outcome:
                        record["terminal_outcome"] = outcome
                        state_changed = True
        if state_changed:
            self._save_state()
        return farm_jobs

    def _update_zero_active_state(
        self, queue_snapshot: dict[str, Any], jobs: list[dict[str, Any]]
    ) -> float:
        active_count = sum(
            1 for job in jobs if _is_active(str(job.get("status") or ""))
        )
        should_track = (
            int(queue_snapshot["PENDING"]) > 0
            and active_count == 0
            and not (self.run_dir / "PAUSE").exists()
            and not (self.run_dir / "DRAIN").exists()
        )
        previous = self.state.get("zero_active_since_unix")
        if should_track:
            if previous is None:
                self.state["zero_active_since_unix"] = time.time()
                self._save_state()
            return max(0.0, time.time() - float(self.state["zero_active_since_unix"]))
        if previous is not None:
            self.state["zero_active_since_unix"] = None
            self._save_state()
        return 0.0

    def _terminal_outcome_summary(self) -> dict[str, Any]:
        outcomes = Counter()
        recent: list[dict[str, Any]] = []
        for name, record in self.state.get("jobs", {}).items():
            if not _is_terminal_failure(str(record.get("status") or "")):
                continue
            outcome = record.get("terminal_outcome")
            if not outcome:
                if record.get("preemption") is True:
                    outcome = "preemption"
                elif "stopped" in str(record.get("status") or "").lower():
                    outcome = "operator_stopped"
                else:
                    outcome = "unclassified"
            outcomes[str(outcome)] += 1
            recent.append(
                {
                    "name": name,
                    "status": record.get("status"),
                    "outcome": outcome,
                    "diagnosed_at": record.get("diagnosed_at"),
                    "diagnosis_path": record.get("diagnosis_path"),
                }
            )
        recent.sort(key=lambda item: str(item.get("diagnosed_at") or ""), reverse=True)
        return {"counts": dict(outcomes), "recent": recent[:10]}

    def _write_status(
        self,
        queue_snapshot: dict[str, Any],
        jobs: list[dict[str, Any]],
        discovery: dict[str, Any] | None,
    ) -> dict[str, Any]:
        job_counts = Counter(str(job.get("status") or "UNKNOWN") for job in jobs)
        now = time.time()
        complete = int(queue_snapshot["COMPLETE"])
        if complete > int(self.state.get("last_complete_count", 0)):
            elapsed = max(1.0, now - float(self.state.get("last_progress_unix", now)))
            captions_per_minute = (complete - int(self.state.get("last_complete_count", 0))) * 60 / elapsed
            self.state["last_progress_unix"] = now
            self.state["last_complete_count"] = complete
        else:
            captions_per_minute = 0.0
        status = {
            "schema": "imagenet_caption_farm_status_v1",
            "timestamp": utc_now(),
            "run_fingerprint": self.run["run_fingerprint"],
            "phase": self.state["phase"],
            "queue": queue_snapshot,
            "platform": {
                "job_counts": dict(job_counts),
                "active_jobs": sum(1 for job in jobs if _is_active(str(job.get("status") or ""))),
                "jobs": jobs,
                "discovery": discovery,
                "terminal_outcomes": self._terminal_outcome_summary(),
            },
            "business": {
                "captions_per_minute_since_last_progress": captions_per_minute,
                "consecutive_no_progress_seconds": now
                - float(self.state.get("last_progress_unix", now)),
                "zero_active_seconds": (
                    0.0
                    if self.state.get("zero_active_since_unix") is None
                    else max(0.0, now - float(self.state["zero_active_since_unix"]))
                ),
            },
            "controller": {
                "pid": os.getpid(),
                "restart_count": self.state["restart_count"],
                "paused": (self.run_dir / "PAUSE").exists(),
                "draining": (self.run_dir / "DRAIN").exists(),
                "tuning": self.tuning,
                "target_state": self.state["target_state"],
            },
        }
        atomic_write_json(self.status_path, status)
        return status

    def _best_effort_exception_status(self, exc: BaseException) -> dict[str, Any]:
        status_error: str | None = None
        try:
            return self._write_status(self.queue.snapshot(), [], None)
        except Exception as status_exc:
            status_error = f"{type(status_exc).__name__}: {status_exc}"
        if self.status_path.is_file():
            try:
                status = load_json(self.status_path)
            except Exception:
                status = {}
        else:
            status = {}
        status.update(
            {
                "schema": status.get("schema", "imagenet_caption_farm_status_v1"),
                "timestamp": utc_now(),
                "run_fingerprint": self.run["run_fingerprint"],
                "phase": self.state.get("phase", "RUNNING"),
                "controller_exception": f"{type(exc).__name__}: {exc}",
                "status_refresh_error": status_error,
            }
        )
        atomic_write_json(self.status_path, status)
        return status

    def _needs_attention(self, reason: str, status: dict[str, Any], **details: Any) -> int:
        self.state["phase"] = "NEEDS_ATTENTION"
        self._save_state()
        payload = {
            "schema": "imagenet_caption_farm_attention_v1",
            "timestamp": utc_now(),
            "run_fingerprint": self.run["run_fingerprint"],
            "reason": reason,
            "details": details,
            "status_path": str(self.status_path),
            "events_path": str(self.events_path),
            "status": status,
        }
        atomic_write_json(self.attention_path, payload)
        self._emit("needs_attention", reason=reason, details=details)
        return 2

    def _finish(self, jobs: list[dict[str, Any]], status: dict[str, Any]) -> int:
        self.state["phase"] = "DRAINING"
        (self.run_dir / "DRAIN").touch()
        self._save_state()
        active = [job for job in jobs if _is_active(str(job.get("status") or ""))]
        if active:
            self._emit("drain_waiting", active_jobs=[job.get("name") for job in active])
            return -1
        self.state["phase"] = "PUBLISHING"
        self._save_state()
        metadata = publish_run(
            self.run_dir,
            verify_images=bool(self.run["output"].get("verify_images_on_publish", True)),
        )
        self.state["phase"] = "COMPLETED"
        self._save_state()
        completed = {
            "schema": "imagenet_caption_farm_completed_v1",
            "timestamp": utc_now(),
            "run_fingerprint": self.run["run_fingerprint"],
            "published": metadata,
            "status_path": str(self.status_path),
            "events_path": str(self.events_path),
        }
        atomic_write_json(self.completed_path, completed)
        self.attention_path.unlink(missing_ok=True)
        self._emit("farm_completed", published=metadata["path"], sha256=metadata["sha256"])
        return 0

    def run_once(self) -> int:
        queue_snapshot = self.queue.snapshot()
        if int(queue_snapshot["FAILED"]) > 0:
            status = self._write_status(queue_snapshot, [], None)
            return self._needs_attention("unrecoverable_failed_tasks", status)

        platform_enabled = bool(self.platform.get("enabled", True))
        discovery: dict[str, Any] | None = None
        targets: list[LiveTarget] = []
        live_jobs: list[dict[str, Any]] = []
        if platform_enabled:
            targets, discovery = self.client.discover_targets()
            all_jobs = self.client.list_jobs(active=False, keyword=self.job_prefix)
            live_jobs = self._reconcile_jobs(all_jobs)
            self._recover_fixed_image_backoffs()
        self._update_zero_active_state(queue_snapshot, live_jobs)
        status = self._write_status(queue_snapshot, live_jobs, discovery)

        if queue_snapshot["COMPLETE"] == queue_snapshot["total"]:
            return self._finish(live_jobs, status)
        if (self.run_dir / "STOP").exists() or self.stop_requested:
            self.state["phase"] = "STOPPED"
            (self.run_dir / "DRAIN").touch()
            for job in live_jobs:
                if _is_active(str(job.get("status") or "")):
                    self.client.stop_job(str(job["name"]), check=False)
            self._save_state()
            return 3
        if (self.run_dir / "PAUSE").exists():
            self.state["phase"] = "PAUSED"
            self._save_state()
            return -1
        if self.state.get("phase") != "RUNNING":
            previous_phase = self.state.get("phase")
            self.state["phase"] = "RUNNING"
            self._save_state()
            self._emit("controller_resumed", previous_phase=previous_phase)

        pending_failures = [
            {"name": name, **record}
            for name, record in self.state.get("jobs", {}).items()
            if record.get("failure_attention_pending")
        ]
        if pending_failures:
            for failure in pending_failures:
                self.state["jobs"][failure["name"]]["failure_attention_pending"] = False
                self.state["jobs"][failure["name"]]["failure_attention_reported_at"] = utc_now()
            self._save_state()
            return self._needs_attention(
                "non_preemption_job_failure",
                status,
                jobs=[
                    {
                        "name": failure["name"],
                        "status": failure.get("status"),
                        "outcome": failure.get("terminal_outcome"),
                        "diagnosis_path": failure.get("diagnosis_path"),
                    }
                    for failure in pending_failures
                ],
            )

        no_progress = status["business"]["consecutive_no_progress_seconds"]
        active_count = status["platform"]["active_jobs"]
        running_count = sum(
            1
            for job in live_jobs
            if "running" in str(job.get("status") or "").lower()
        )
        if running_count and no_progress > float(self.config["no_progress_attention_seconds"]):
            return self._needs_attention("business_no_progress", status, seconds=no_progress)
        zero_active = status["business"]["zero_active_seconds"]
        if zero_active > float(self.config.get("zero_active_attention_seconds", 300)):
            return self._needs_attention(
                "zero_active_jobs",
                status,
                seconds=zero_active,
                pending_tasks=int(queue_snapshot["PENDING"]),
            )

        if not platform_enabled:
            return -1
        target_active = int(self.config["target_active_jobs"])
        global_max = int(self.config["global_max_active_jobs"])
        desired = min(target_active, global_max) - active_count
        if desired <= 0:
            return -1
        min_interval = float(self.config["min_submit_interval_seconds"])
        if time.time() - float(self.state["last_submit_unix"]) < min_interval:
            return -1
        burst = min(desired, int(self.config["submission_burst"]))
        submitted = 0
        for _ in range(burst):
            eligible = self._eligible_targets(targets, live_jobs)
            if not eligible:
                break
            target = self._choose_target(eligible)
            name = self._job_name()
            command = self._worker_command(name)
            try:
                if self.dry_run:
                    evidence = self.client.dry_run_job(name, command, target)
                    self._emit("submission_dry_run_passed", job=name, target=target.key)
                else:
                    evidence = self.client.submit_job(name, command, target)
                    self._record_success(target)
                    if active_count == 0 and submitted == 0:
                        # A newly submitted worker after a zero-active interval
                        # needs a fresh model-startup window. Do not carry an
                        # unrelated platform outage into the business-stall
                        # timer and immediately condemn the first RUNNING Job.
                        self.state["last_progress_unix"] = time.time()
                        self.state["last_complete_count"] = int(
                            queue_snapshot["COMPLETE"]
                        )
                        self.state["zero_active_since_unix"] = None
                        self._emit(
                            "business_progress_timer_reset",
                            reason="first_worker_submitted_after_idle",
                            complete_count=int(queue_snapshot["COMPLETE"]),
                        )
                    self.state["jobs"][name] = {
                        "status": "SUBMITTED",
                        "project": target.project,
                        "group": target.group,
                        "target": target.key,
                        "submitted_at": utc_now(),
                        "evidence": evidence,
                    }
                    live_jobs.append(
                        {
                            "name": name,
                            "status": "job_queuing",
                            "project_name": target.project,
                            "compute_group_name": target.group,
                            "gpu_count": 1,
                        }
                    )
                    self._emit("job_submitted", job=name, target=target.key)
                self.state["next_job_sequence"] += 1
                self.state["last_submit_unix"] = time.time()
                submitted += 1
                self._save_state()
            except DatasetMountError as exc:
                self._record_rejection(target, str(exc))
                self._save_state()
                status = self._write_status(self.queue.snapshot(), live_jobs, discovery)
                return self._needs_attention("official_dataset_mount_missing", status, error=str(exc))
            except InspireError as exc:
                self._record_rejection(target, str(exc))
                self._save_state()
        if self.dry_run and submitted:
            return 0
        return -1

    def run_forever(self) -> int:
        self._previous_handlers = {
            signum: signal.signal(signum, self.request_stop)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }
        self._emit("controller_started", pid=os.getpid(), restart_count=self.state["restart_count"])
        try:
            while True:
                try:
                    result = self.run_once()
                    transient_failures = int(
                        self.state.get("consecutive_platform_transient_failures", 0)
                    )
                    if transient_failures:
                        self.state["consecutive_platform_transient_failures"] = 0
                        self._save_state()
                        self._emit(
                            "platform_api_recovered",
                            previous_consecutive_failures=transient_failures,
                        )
                    lock_timeouts = int(self.state.get("consecutive_queue_lock_timeouts", 0))
                    if lock_timeouts:
                        self.state["consecutive_queue_lock_timeouts"] = 0
                        self._save_state()
                        self._emit(
                            "queue_lock_recovered",
                            previous_consecutive_timeouts=lock_timeouts,
                        )
                except TimeoutError as exc:
                    if "shared lock" not in str(exc).lower():
                        status = self._best_effort_exception_status(exc)
                        return self._needs_attention(
                            "controller_exception",
                            status,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    failures = int(self.state.get("consecutive_queue_lock_timeouts", 0)) + 1
                    self.state["consecutive_queue_lock_timeouts"] = failures
                    self._save_state()
                    delay = min(
                        float(self.config["reconcile_interval_seconds"]),
                        max(1.0, float(self.queue.lock_stale_seconds) / 4),
                    )
                    self._emit(
                        "queue_lock_contention",
                        consecutive_timeouts=failures,
                        retry_in_seconds=delay,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    time.sleep(delay)
                    continue
                except InspireError as exc:
                    if not is_transient_platform_error(exc):
                        status = self._best_effort_exception_status(exc)
                        return self._needs_attention(
                            "controller_exception",
                            status,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    failures = int(
                        self.state.get("consecutive_platform_transient_failures", 0)
                    ) + 1
                    self.state["consecutive_platform_transient_failures"] = failures
                    self._save_state()
                    base = float(self.config["rejection_backoff_base_seconds"])
                    maximum = float(self.config["rejection_backoff_max_seconds"])
                    delay = min(maximum, base * 2 ** (failures - 1))
                    delay *= random.uniform(0.8, 1.2)
                    self._emit(
                        "platform_api_transient_failure",
                        consecutive_failures=failures,
                        retry_in_seconds=delay,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    time.sleep(delay)
                    continue
                except Exception as exc:
                    status = self._best_effort_exception_status(exc)
                    return self._needs_attention(
                        "controller_exception",
                        status,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if result >= 0:
                    return result
                if self.once:
                    return 0
                interval = float(self.config["reconcile_interval_seconds"])
                time.sleep(interval * random.uniform(0.9, 1.1))
        finally:
            self._emit("controller_exited", phase=self.state["phase"])
            self.queue.close()
            for signum, handler in self._previous_handlers.items():
                signal.signal(signum, handler)


def start_controller_daemon(run_dir: Path, script_path: Path) -> dict[str, Any]:
    pid_path = run_dir / "controller.pid"
    if pid_path.exists():
        try:
            existing = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(existing, 0)
            raise RuntimeError(f"controller is already running as pid {existing}")
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
    log_path = run_dir / "controller.log"
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        [sys.executable, str(script_path), "controller", "run", "--run-dir", str(run_dir)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    return {"pid": process.pid, "log": str(log_path), "pid_file": str(pid_path)}


def _attention_signal(run_dir: Path) -> dict[str, Any]:
    attention = load_json(run_dir / "NEEDS_ATTENTION.json")
    status = attention.get("status") or {}
    platform = status.get("platform") or {}
    return {
        "schema": "imagenet_caption_farm_supervisor_signal_v1",
        "event": "NEEDS_ATTENTION",
        "timestamp": attention.get("timestamp"),
        "reason": attention.get("reason"),
        "details": attention.get("details") or {},
        "status_path": attention.get("status_path"),
        "events_path": attention.get("events_path"),
        "queue": status.get("queue") or {},
        "active_jobs": platform.get("active_jobs"),
        "business": status.get("business") or {},
    }


def supervise_controller(run_dir: Path) -> int:
    """Own one Controller and stay silent until completion or attention.

    This is the operator-facing foreground entry point. Unlike a detached
    Controller plus a second wait process, the process that observes the alert
    is the process running the Controller, so a tool call cannot lose the
    notification while the Controller continues elsewhere.
    """

    pid_path = run_dir / "controller.pid"
    if pid_path.is_file():
        try:
            existing = int(pid_path.read_text(encoding="utf-8").strip())
            if existing != os.getpid():
                os.kill(existing, 0)
                raise RuntimeError(f"controller is already running as pid {existing}")
        except ProcessLookupError:
            pass
        except ValueError:
            pass
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        if (run_dir / "COMPLETED.json").is_file():
            print(
                json.dumps(load_json(run_dir / "COMPLETED.json"), ensure_ascii=False, indent=2)
            )
            return 0
        if (run_dir / "NEEDS_ATTENTION.json").is_file():
            print(
                json.dumps(_attention_signal(run_dir), ensure_ascii=False, indent=2),
                file=sys.stderr,
            )
            return 2

        result = Controller(run_dir).run_forever()
        if result == 2 and (run_dir / "NEEDS_ATTENTION.json").is_file():
            print(
                json.dumps(_attention_signal(run_dir), ensure_ascii=False, indent=2),
                file=sys.stderr,
            )
        elif result == 0 and (run_dir / "COMPLETED.json").is_file():
            print(
                json.dumps(load_json(run_dir / "COMPLETED.json"), ensure_ascii=False, indent=2)
            )
        elif result == 3:
            print(
                json.dumps(
                    {
                        "schema": "imagenet_caption_farm_supervisor_signal_v1",
                        "event": "STOPPED",
                        "timestamp": utc_now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return result
    finally:
        try:
            if int(pid_path.read_text(encoding="utf-8").strip()) == os.getpid():
                pid_path.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError):
            pass


def acknowledge_attention(run_dir: Path, reason: str) -> dict[str, Any]:
    reason = " ".join(str(reason).split())
    if not reason:
        raise ValueError("a non-empty acknowledgement reason is required")
    attention_path = run_dir / "NEEDS_ATTENTION.json"
    if not attention_path.is_file():
        return {"status": "no_attention", "reason": reason}
    if not (run_dir / "PAUSE").is_file():
        raise RuntimeError("attention can only be acknowledged while the farm is paused")
    store = TaskStore(run_dir)
    try:
        snapshot = store.snapshot()
    finally:
        store.close()
    if int(snapshot["FAILED"]) or int(snapshot["LEASED"]):
        raise RuntimeError("repair and drain FAILED/LEASED tasks before acknowledging attention")
    attention = load_json(attention_path)
    acknowledged_at = utc_now()
    archive = run_dir / "attention_history" / (
        acknowledged_at.replace(":", "").replace("+00:00", "Z") + ".json"
    )
    atomic_write_json(
        archive,
        {
            **attention,
            "acknowledged_at": acknowledged_at,
            "acknowledgement_reason": reason,
            "queue_at_acknowledgement": snapshot,
        },
    )
    attention_path.unlink()
    state_path = run_dir / "controller_state.json"
    if state_path.is_file():
        state = load_json(state_path)
        state["phase"] = "RECOVERING"
        state["last_attention_acknowledged_at"] = acknowledged_at
        state["last_attention_archive"] = str(archive)
        if attention.get("reason") == "business_no_progress":
            state["last_progress_unix"] = time.time()
            state["last_complete_count"] = int(snapshot["COMPLETE"])
        if attention.get("reason") == "zero_active_jobs":
            state["zero_active_since_unix"] = time.time()
            for target in state.get("target_state", {}).values():
                target["consecutive_rejections"] = 0
                target["next_allowed_unix"] = 0.0
                target["circuit_open_until_unix"] = 0.0
                target["last_error"] = None
        atomic_write_json(state_path, state)
    append_jsonl(
        run_dir / "events.jsonl",
        {
            "timestamp": acknowledged_at,
            "event": "attention_acknowledged",
            "component": "controller",
            "run_fingerprint": attention["run_fingerprint"],
            "reason": reason,
            "archive": str(archive),
        },
    )
    return {"status": "acknowledged", "reason": reason, "archive": str(archive)}


def wait_for_controller(run_dir: Path, *, interval_seconds: float = 10.0) -> int:
    attention = run_dir / "NEEDS_ATTENTION.json"
    completed = run_dir / "COMPLETED.json"
    pid_path = run_dir / "controller.pid"
    consecutive_missing = 0
    while True:
        if completed.is_file():
            payload = load_json(completed)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if attention.is_file():
            payload = load_json(attention)
            print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        alive = False
        pid: int | None = None
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            alive = True
            cmdline = Path(f"/proc/{pid}/cmdline")
            if cmdline.is_file():
                command = cmdline.read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
                alive = (
                    "controller run" in command
                    and str(run_dir) in command
                )
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            alive = False
        consecutive_missing = 0 if alive else consecutive_missing + 1
        if consecutive_missing >= 2:
            run = load_json(run_dir / "run.json")
            status = load_json(run_dir / "status.json") if (run_dir / "status.json").is_file() else {}
            payload = {
                "schema": "imagenet_caption_farm_attention_v1",
                "timestamp": utc_now(),
                "run_fingerprint": run["run_fingerprint"],
                "reason": "controller_process_missing",
                "details": {"pid": pid, "pid_file": str(pid_path)},
                "status_path": str(run_dir / "status.json"),
                "events_path": str(run_dir / "events.jsonl"),
                "status": status,
            }
            atomic_write_json(attention, payload)
            append_jsonl(
                run_dir / "events.jsonl",
                {
                    "timestamp": utc_now(),
                    "event": "needs_attention",
                    "component": "controller_wait",
                    "run_fingerprint": run["run_fingerprint"],
                    "reason": "controller_process_missing",
                    "details": payload["details"],
                },
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        time.sleep(interval_seconds)
