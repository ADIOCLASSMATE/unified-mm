from __future__ import annotations

import os
import re
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io import (
    DirectoryLock,
    append_jsonl,
    atomic_link_json,
    atomic_write_json,
    load_json,
    unix_now,
    utc_now,
)
from .manifest import ManifestReader


QUEUE_SCHEMA = "imagenet_caption_farm_queue_v1"
LEASE_SCHEMA = "imagenet_caption_farm_lease_v1"
RESULT_SCHEMA = "imagenet_local_qwen_caption_result_v1"
_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class Lease:
    lease_id: str
    owner: str
    expires_at: float
    tasks: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Lease":
        return cls(
            lease_id=str(value["lease_id"]),
            owner=str(value["owner"]),
            expires_at=float(value["lease_expires_at"]),
            tasks=tuple(value.get("tasks") or ()),
        )


def default_owner() -> str:
    return f"{socket.gethostname()}-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"


class TaskStore:
    """Shared-filesystem task state with atomic claims and expiring leases."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run = load_json(run_dir / "run.json")
        self.state_dir = run_dir / "state"
        self.leases_dir = self.state_dir / "leases"
        self.failed_dir = self.state_dir / "failed"
        self.results_dir = run_dir / "staging" / "results"
        self.temporary_dir = run_dir / "staging" / "tmp"
        self.events_path = run_dir / "events.jsonl"
        self.allocator_path = self.state_dir / "allocator.json"
        self.repair_journal_path = self.state_dir / "repair_failed.json"
        self.lock_path = self.state_dir / "claim.lock"
        self.manifest = ManifestReader(
            Path(self.run["manifest"]["path"]),
            Path(self.run["manifest"]["offsets_path"]),
        )
        queue = self.run["queue"]
        self.captions_per_image = int(self.run["caption"]["captions_per_image"])
        self.total_tasks = int(self.run["manifest"]["row_count"]) * self.captions_per_image
        self.lease_seconds = float(queue["lease_seconds"])
        self.max_attempts = int(queue["max_attempts"])
        configured_timeout = float(queue.get("lock_timeout_seconds", 60))
        configured_stale = float(queue.get("lock_stale_seconds", 120))
        # Older runs could configure a timeout shorter than the stale-lock
        # horizon. A process killed while holding the lock then made every
        # waiter fail before any of them was allowed to recover the lock.
        # Preserve run compatibility while guaranteeing a recovery window.
        if configured_stale >= configured_timeout:
            configured_stale = min(60.0, max(1.0, configured_timeout * 0.5))
        self.lock_stale_seconds = configured_stale
        self.lock_timeout_seconds = max(
            configured_timeout,
            configured_stale * 2,
            min(self.lease_seconds * 0.5, 300.0),
        )

    @classmethod
    def initialize(cls, run_dir: Path, run: dict[str, Any]) -> "TaskStore":
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_dir / "run.json", run)
        state_dir = run_dir / "state"
        for path in (
            state_dir / "leases",
            state_dir / "failed",
            run_dir / "staging" / "results",
            run_dir / "staging" / "tmp",
            run_dir / "workers",
        ):
            path.mkdir(parents=True, exist_ok=True)
        total_tasks = int(run["manifest"]["row_count"]) * int(
            run["caption"]["captions_per_image"]
        )
        allocator = {
            "schema": QUEUE_SCHEMA,
            "run_fingerprint": run["run_fingerprint"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "total_tasks": total_tasks,
            "next_ordinal": 0,
            "requeue": [],
            "failed_count": 0,
            "stats": {
                "claims": 0,
                "heartbeats": 0,
                "commits_created": 0,
                "commits_reused": 0,
                "lease_expirations": 0,
                "reclaims": 0,
                "failures": 0,
                "repaired_failures": 0,
            },
            "applied_repairs": [],
        }
        atomic_write_json(state_dir / "allocator.json", allocator)
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "timestamp": utc_now(),
                "event": "queue_initialized",
                "run_fingerprint": run["run_fingerprint"],
                "total_tasks": total_tasks,
            },
        )
        return cls(run_dir)

    def close(self) -> None:
        self.manifest.close()

    def _lock(self) -> DirectoryLock:
        return DirectoryLock(
            self.lock_path,
            timeout_seconds=self.lock_timeout_seconds,
            stale_seconds=self.lock_stale_seconds,
        )

    def _load_state(self) -> dict[str, Any]:
        state = load_json(self.allocator_path)
        if state.get("run_fingerprint") != self.run["run_fingerprint"]:
            raise ValueError("allocator/run fingerprint mismatch")
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        atomic_write_json(self.allocator_path, state)

    def _recover_failed_repair_locked(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.repair_journal_path.is_file():
            return state
        journal = load_json(self.repair_journal_path)
        if journal.get("run_fingerprint") != self.run["run_fingerprint"]:
            raise ValueError("failed-task repair journal belongs to another run")
        repair_id = str(journal["repair_id"])
        items = journal.get("items") or []
        archive_root = self.state_dir / "repaired_failures" / repair_id
        for item in items:
            failed = self.failed_dir / str(item["relative_path"])
            archive = archive_root / str(item["relative_path"])
            archive.parent.mkdir(parents=True, exist_ok=True)
            if failed.is_file() and not archive.exists():
                os.link(failed, archive)
            if not archive.is_file():
                raise RuntimeError(f"repair evidence is missing for {item['task_key']}")

        applied = list(state.get("applied_repairs") or [])
        if repair_id not in applied:
            state["requeue"] = self._dedupe_requeue(
                [
                    *(state.get("requeue") or []),
                    *(
                        {
                            "ordinal": int(item["ordinal"]),
                            "attempt_count": 0,
                            "last_error": f"repaired failure: {journal['reason']}",
                            "repair_id": repair_id,
                        }
                        for item in items
                    ),
                ]
            )
            repaired = len(items)
            current_failed = int(state.get("failed_count", 0))
            if current_failed < repaired:
                raise RuntimeError("allocator failed_count is smaller than repair journal")
            state["failed_count"] = current_failed - repaired
            state.setdefault("stats", {}).setdefault("repaired_failures", 0)
            state["stats"]["repaired_failures"] += repaired
            state["applied_repairs"] = [*applied, repair_id]
            self._save_state(state)

        for item in items:
            failed = self.failed_dir / str(item["relative_path"])
            failed.unlink(missing_ok=True)
        self.repair_journal_path.unlink(missing_ok=True)
        append_jsonl(
            self.events_path,
            {
                "timestamp": utc_now(),
                "event": "failed_tasks_repaired",
                "repair_id": repair_id,
                "task_count": len(items),
                "reason": journal["reason"],
                "archive": str(archive_root),
            },
        )
        return state

    def repair_failed(self, reason: str) -> dict[str, Any]:
        reason = " ".join(str(reason).split())
        if not reason:
            raise ValueError("a non-empty repair reason is required")
        with self._lock():
            state = self._recover_failed_repair_locked(self._load_state())
            paths = sorted(self.failed_dir.glob("*/*.json"))
            if not paths:
                return {"repaired": 0, "reason": reason, "archive": None}
            if int(state.get("failed_count", 0)) != len(paths):
                raise RuntimeError(
                    "failed_count/file-count mismatch; refusing an unaudited repair"
                )
            repair_id = f"{int(unix_now())}-{uuid.uuid4().hex[:12]}"
            items: list[dict[str, Any]] = []
            for path in paths:
                record = load_json(path)
                ordinal = int(record["ordinal"])
                expected = self._task_from_ordinal(ordinal)
                if record.get("schema") != "imagenet_caption_farm_failed_v1":
                    raise ValueError(f"unexpected failure schema in {path}")
                if record.get("task_key") != expected["task_key"]:
                    raise ValueError(f"failure identity mismatch in {path}")
                if self.result_path(expected).is_file():
                    raise ValueError(f"failure already has a visible result: {path}")
                items.append(
                    {
                        "ordinal": ordinal,
                        "task_key": expected["task_key"],
                        "relative_path": str(path.relative_to(self.failed_dir)),
                    }
                )
            journal = {
                "schema": "imagenet_caption_farm_failed_repair_v1",
                "run_fingerprint": self.run["run_fingerprint"],
                "repair_id": repair_id,
                "created_at": utc_now(),
                "reason": reason,
                "items": items,
            }
            atomic_write_json(self.repair_journal_path, journal)
            self._recover_failed_repair_locked(state)
            return {
                "repaired": len(items),
                "reason": reason,
                "repair_id": repair_id,
                "archive": str(self.state_dir / "repaired_failures" / repair_id),
            }

    def _lease_path(self, lease_id: str) -> Path:
        return self.leases_dir / f"{lease_id}.json"

    def _task_key(self, manifest_index: int, image_id: str, caption_slot: int) -> str:
        return (
            f"{manifest_index:07d}:{image_id}:"
            f"{self.run['model']['fingerprint']}:{caption_slot:03d}"
        )

    def _task_from_ordinal(self, ordinal: int, attempt_count: int = 0) -> dict[str, Any]:
        manifest_index, caption_slot = divmod(ordinal, self.captions_per_image)
        identity = self.manifest.read(manifest_index)
        image_id = str(identity["image_id"])
        return {
            "ordinal": ordinal,
            "manifest_index": manifest_index,
            "img_id": int(identity["img_id"]),
            "image_id": image_id,
            "id": image_id,
            "path": identity["path"],
            "imagenet_relative_path": identity["imagenet_relative_path"],
            "source_path": identity["source_path"],
            "synset": identity["synset"],
            "original_caption_key": identity["original_caption_key"],
            "original_caption_sha256": identity["original_caption_sha256"],
            "caption_slot": caption_slot,
            "attempt_count": attempt_count,
            "task_key": self._task_key(manifest_index, image_id, caption_slot),
            "model_fingerprint": self.run["model"]["fingerprint"],
            "prompt_fingerprint": self.run["caption"]["prompt_fingerprint"],
            "output_version": self.run["output"]["version"],
        }

    @staticmethod
    def _safe_image_id(value: str) -> str:
        return _SAFE_KEY.sub("_", value)

    def result_path(self, task: dict[str, Any]) -> Path:
        shard = int(task["manifest_index"]) % 4096
        filename = f"{int(task['manifest_index']):07d}-{self._safe_image_id(str(task['image_id']))}-s{int(task['caption_slot']):03d}.json"
        return self.results_dir / f"{shard:04x}" / filename

    def failed_path(self, task: dict[str, Any]) -> Path:
        shard = int(task["manifest_index"]) % 4096
        filename = f"{int(task['manifest_index']):07d}-{self._safe_image_id(str(task['image_id']))}-s{int(task['caption_slot']):03d}.json"
        return self.failed_dir / f"{shard:04x}" / filename

    def _read_leases_locked(self) -> list[dict[str, Any]]:
        leases: list[dict[str, Any]] = []
        for path in sorted(self.leases_dir.glob("*.json")):
            try:
                lease = load_json(path)
            except (FileNotFoundError, ValueError):
                continue
            if lease.get("run_fingerprint") != self.run["run_fingerprint"]:
                raise ValueError(f"foreign lease in run directory: {path}")
            leases.append(lease)
        return leases

    def _dedupe_requeue(self, values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        by_ordinal: dict[int, dict[str, Any]] = {}
        for value in values:
            ordinal = int(value["ordinal"])
            current = by_ordinal.get(ordinal)
            if current is None or int(value.get("attempt_count", 0)) > int(
                current.get("attempt_count", 0)
            ):
                by_ordinal[ordinal] = value
        return [by_ordinal[key] for key in sorted(by_ordinal)]

    def _reap_expired_locked(
        self, state: dict[str, Any], leases: list[dict[str, Any]], now: float
    ) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        requeue = list(state.get("requeue") or [])
        for lease in leases:
            if float(lease["lease_expires_at"]) > now:
                active.append(lease)
                continue
            reclaimed = 0
            for task in lease.get("tasks") or []:
                if self.result_path(task).is_file() or self.failed_path(task).is_file():
                    continue
                requeue.append(
                    {
                        "ordinal": int(task["ordinal"]),
                        "attempt_count": int(task.get("attempt_count", 1)),
                        "last_error": "lease expired before commit",
                    }
                )
                reclaimed += 1
            self._lease_path(str(lease["lease_id"])).unlink(missing_ok=True)
            state["stats"]["lease_expirations"] += 1
            state["stats"]["reclaims"] += reclaimed
            append_jsonl(
                self.events_path,
                {
                    "timestamp": utc_now(),
                    "event": "lease_expired",
                    "lease_id": lease["lease_id"],
                    "owner": lease["owner"],
                    "reclaimed_tasks": reclaimed,
                },
            )
        active_ordinals = {
            int(task["ordinal"]) for lease in active for task in lease.get("tasks") or []
        }
        state["requeue"] = [
            item
            for item in self._dedupe_requeue(requeue)
            if int(item["ordinal"]) not in active_ordinals
        ]
        return active

    def claim(self, owner: str, limit: int) -> Lease | None:
        if limit < 1:
            raise ValueError("claim limit must be positive")
        if (self.run_dir / "PAUSE").exists() or (self.run_dir / "DRAIN").exists():
            return None
        now = unix_now()
        with self._lock():
            state = self._recover_failed_repair_locked(self._load_state())
            active = self._reap_expired_locked(state, self._read_leases_locked(), now)
            active_ordinals = {
                int(task["ordinal"])
                for lease in active
                for task in lease.get("tasks") or []
            }
            tasks: list[dict[str, Any]] = []
            remaining_requeue: list[dict[str, Any]] = []
            for item in state.get("requeue") or []:
                ordinal = int(item["ordinal"])
                if len(tasks) >= limit or ordinal in active_ordinals:
                    remaining_requeue.append(item)
                    continue
                task = self._task_from_ordinal(
                    ordinal, attempt_count=int(item.get("attempt_count", 0)) + 1
                )
                if self.result_path(task).is_file() or self.failed_path(task).is_file():
                    continue
                tasks.append(task)
                active_ordinals.add(ordinal)
            state["requeue"] = remaining_requeue

            while len(tasks) < limit and int(state["next_ordinal"]) < self.total_tasks:
                ordinal = int(state["next_ordinal"])
                state["next_ordinal"] = ordinal + 1
                if ordinal in active_ordinals:
                    continue
                task = self._task_from_ordinal(ordinal, attempt_count=1)
                if self.result_path(task).is_file() or self.failed_path(task).is_file():
                    continue
                tasks.append(task)
                active_ordinals.add(ordinal)

            if not tasks:
                self._save_state(state)
                return None
            lease_id = uuid.uuid4().hex
            lease_record = {
                "schema": LEASE_SCHEMA,
                "run_fingerprint": self.run["run_fingerprint"],
                "lease_id": lease_id,
                "owner": owner,
                "created_at": utc_now(),
                "heartbeat_at": now,
                "lease_expires_at": now + self.lease_seconds,
                "tasks": tasks,
            }
            # Lease first: after a lock-holder crash, the next claimant observes it
            # and advances a stale allocator cursor without duplicating the claim.
            atomic_write_json(self._lease_path(lease_id), lease_record)
            state["stats"]["claims"] += len(tasks)
            self._save_state(state)
            append_jsonl(
                self.events_path,
                {
                    "timestamp": utc_now(),
                    "event": "tasks_claimed",
                    "lease_id": lease_id,
                    "owner": owner,
                    "task_count": len(tasks),
                    "first_ordinal": min(int(task["ordinal"]) for task in tasks),
                    "last_ordinal": max(int(task["ordinal"]) for task in tasks),
                },
            )
            return Lease.from_dict(lease_record)

    def _load_owned_lease_locked(self, lease_id: str, owner: str) -> dict[str, Any]:
        path = self._lease_path(lease_id)
        lease = load_json(path)
        if str(lease["owner"]) != owner:
            raise PermissionError(f"lease {lease_id} belongs to another worker")
        return lease

    def heartbeat(self, lease_id: str, owner: str) -> float:
        now = unix_now()
        with self._lock():
            lease = self._load_owned_lease_locked(lease_id, owner)
            if float(lease["lease_expires_at"]) <= now:
                raise TimeoutError(f"lease {lease_id} has already expired")
            lease["heartbeat_at"] = now
            lease["lease_expires_at"] = now + self.lease_seconds
            atomic_write_json(self._lease_path(lease_id), lease)
            state = self._load_state()
            state["stats"]["heartbeats"] += 1
            self._save_state(state)
            return float(lease["lease_expires_at"])

    def _validate_visible_result(self, task: dict[str, Any], record: dict[str, Any]) -> None:
        expected = {
            "task_key": task["task_key"],
            "image_id": task["image_id"],
            "model_fingerprint": self.run["model"]["fingerprint"],
            "prompt_fingerprint": self.run["caption"]["prompt_fingerprint"],
            "caption_slot": int(task["caption_slot"]),
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"visible result {key} mismatch for {task['task_key']}")
        if not str(record.get("caption") or "").strip():
            raise ValueError(f"visible result has an empty caption: {task['task_key']}")

    def _prepare_result(
        self, lease: Lease, task: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any]:
        if task["task_key"] not in {item["task_key"] for item in lease.tasks}:
            raise ValueError("task is not part of the supplied lease")
        result = {
            **record,
            "schema": RESULT_SCHEMA,
            "run_fingerprint": self.run["run_fingerprint"],
            "task_key": task["task_key"],
            "manifest_index": int(task["manifest_index"]),
            "img_id": int(task["img_id"]),
            "image_id": task["image_id"],
            "id": task["id"],
            "path": task["path"],
            "imagenet_relative_path": task["imagenet_relative_path"],
            "source_path": task["source_path"],
            "synset": task["synset"],
            "original_caption_key": task["original_caption_key"],
            "original_caption_sha256": task["original_caption_sha256"],
            "caption_slot": int(task["caption_slot"]),
            "attempt_count": int(task["attempt_count"]),
            "model_fingerprint": self.run["model"]["fingerprint"],
            "prompt_fingerprint": self.run["caption"]["prompt_fingerprint"],
            "output_version": self.run["output"]["version"],
            "committed_at": utc_now(),
            "lease_owner": lease.owner,
            "lease_id": lease.lease_id,
        }
        self._validate_visible_result(task, result)
        return result

    def commit_many(
        self,
        lease: Lease,
        items: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    ) -> dict[str, bool]:
        prepared = [
            (task, self._prepare_result(lease, task, record))
            for task, record in items
        ]
        if not prepared:
            return {}
        task_keys = [str(task["task_key"]) for task, _ in prepared]
        if len(task_keys) != len(set(task_keys)):
            raise ValueError("commit batch contains duplicate task keys")

        outcomes: dict[str, bool] = {}
        with self._lock() as lock:
            try:
                current = self._load_owned_lease_locked(lease.lease_id, lease.owner)
            except FileNotFoundError:
                for task, _ in prepared:
                    path = self.result_path(task)
                    if not path.is_file():
                        raise
                    self._validate_visible_result(task, load_json(path))
                    outcomes[str(task["task_key"])] = False
                return outcomes

            current_task_keys = {
                str(item["task_key"]) for item in current.get("tasks") or []
            }
            if float(current["lease_expires_at"]) <= unix_now():
                missing = [
                    str(task["task_key"])
                    for task, _ in prepared
                    if not self.result_path(task).is_file()
                ]
                if missing:
                    raise TimeoutError(
                        f"lease {lease.lease_id} expired before batch commit"
                    )

            for task, result in prepared:
                task_key = str(task["task_key"])
                path = self.result_path(task)
                if path.is_file():
                    self._validate_visible_result(task, load_json(path))
                    outcomes[task_key] = False
                else:
                    if task_key not in current_task_keys:
                        raise KeyError(f"task {task_key} is no longer in lease {lease.lease_id}")
                    created = atomic_link_json(path, result, self.temporary_dir)
                    if not created:
                        self._validate_visible_result(task, load_json(path))
                    outcomes[task_key] = created
                lock.refresh()

            resolved = set(outcomes)
            remaining = [
                item
                for item in current.get("tasks") or []
                if str(item["task_key"]) not in resolved
            ]
            if remaining:
                current["tasks"] = remaining
                atomic_write_json(self._lease_path(lease.lease_id), current)
            else:
                self._lease_path(lease.lease_id).unlink(missing_ok=True)
            state = self._load_state()
            state["stats"]["commits_created"] += sum(outcomes.values())
            state["stats"]["commits_reused"] += len(outcomes) - sum(outcomes.values())
            self._save_state(state)
        append_jsonl(
            self.events_path,
            {
                "timestamp": utc_now(),
                "event": "results_committed_batch",
                "lease_id": lease.lease_id,
                "owner": lease.owner,
                "task_count": len(outcomes),
                "created": sum(outcomes.values()),
                "reused": len(outcomes) - sum(outcomes.values()),
                "task_keys": list(outcomes),
            },
        )
        return outcomes

    def commit(self, lease: Lease, task: dict[str, Any], record: dict[str, Any]) -> bool:
        return self.commit_many(lease, [(task, record)])[str(task["task_key"])]

    def nack(self, lease: Lease, task: dict[str, Any], error: str) -> str:
        error = error[-4000:]
        with self._lock():
            current = self._load_owned_lease_locked(lease.lease_id, lease.owner)
            current_tasks = current.get("tasks") or []
            if task["task_key"] not in {item["task_key"] for item in current_tasks}:
                return "already_resolved"
            state = self._load_state()
            attempt_count = int(task.get("attempt_count", 1))
            if attempt_count >= self.max_attempts:
                failed = {
                    **task,
                    "schema": "imagenet_caption_farm_failed_v1",
                    "last_error": error,
                    "failed_at": utc_now(),
                    "lease_id": lease.lease_id,
                    "lease_owner": lease.owner,
                }
                atomic_write_json(self.failed_path(task), failed)
                state["failed_count"] += 1
                state["stats"]["failures"] += 1
                outcome = "failed"
            else:
                state["requeue"] = self._dedupe_requeue(
                    [
                        *(state.get("requeue") or []),
                        {
                            "ordinal": int(task["ordinal"]),
                            "attempt_count": attempt_count,
                            "last_error": error,
                        },
                    ]
                )
                outcome = "requeued"
            self._save_state(state)
            remaining = [
                item for item in current_tasks if item["task_key"] != task["task_key"]
            ]
            if remaining:
                current["tasks"] = remaining
                atomic_write_json(self._lease_path(lease.lease_id), current)
            else:
                self._lease_path(lease.lease_id).unlink(missing_ok=True)
            append_jsonl(
                self.events_path,
                {
                    "timestamp": utc_now(),
                    "event": f"task_{outcome}",
                    "task_key": task["task_key"],
                    "attempt_count": attempt_count,
                    "error": error,
                },
            )
            return outcome

    def release(self, lease: Lease, tasks: Iterable[dict[str, Any]], reason: str) -> int:
        task_keys = {task["task_key"] for task in tasks}
        if not task_keys:
            return 0
        with self._lock():
            current = self._load_owned_lease_locked(lease.lease_id, lease.owner)
            released = [
                task
                for task in current.get("tasks") or []
                if task["task_key"] in task_keys
                and not self.result_path(task).is_file()
                and not self.failed_path(task).is_file()
            ]
            state = self._load_state()
            state["requeue"] = self._dedupe_requeue(
                [
                    *(state.get("requeue") or []),
                    *(
                        {
                            "ordinal": int(task["ordinal"]),
                            "attempt_count": int(task.get("attempt_count", 1)),
                            "last_error": reason,
                        }
                        for task in released
                    ),
                ]
            )
            self._save_state(state)
            remaining = [
                task
                for task in current.get("tasks") or []
                if task["task_key"] not in task_keys
            ]
            if remaining:
                current["tasks"] = remaining
                atomic_write_json(self._lease_path(lease.lease_id), current)
            else:
                self._lease_path(lease.lease_id).unlink(missing_ok=True)
            append_jsonl(
                self.events_path,
                {
                    "timestamp": utc_now(),
                    "event": "tasks_released",
                    "lease_id": lease.lease_id,
                    "owner": lease.owner,
                    "task_count": len(released),
                    "reason": reason,
                },
            )
            return len(released)

    def snapshot(self, *, reap_expired: bool = True) -> dict[str, Any]:
        now = unix_now()
        with self._lock():
            state = self._recover_failed_repair_locked(self._load_state())
            leases = self._read_leases_locked()
            if reap_expired:
                leases = self._reap_expired_locked(state, leases, now)
                self._save_state(state)
            leased_tasks = sum(len(lease.get("tasks") or []) for lease in leases)
            pending_unallocated = self.total_tasks - int(state["next_ordinal"])
            pending_requeue = len(state.get("requeue") or [])
            failed = int(state.get("failed_count", 0))
            pending = pending_unallocated + pending_requeue
            complete = self.total_tasks - pending - leased_tasks - failed
            oldest_lease_age = 0.0
            if leases:
                oldest_lease_age = max(
                    0.0, now - min(float(lease["heartbeat_at"]) for lease in leases)
                )
            return {
                "schema": QUEUE_SCHEMA,
                "timestamp": utc_now(),
                "run_fingerprint": self.run["run_fingerprint"],
                "total": self.total_tasks,
                "COMPLETE": complete,
                "PENDING": pending,
                "LEASED": leased_tasks,
                "FAILED": failed,
                "expired_leases": sum(
                    1 for lease in leases if float(lease["lease_expires_at"]) <= now
                ),
                "active_leases": len(leases),
                "oldest_lease_age_seconds": oldest_lease_age,
                "next_ordinal": int(state["next_ordinal"]),
                "requeue_count": pending_requeue,
                "stats": state["stats"],
                "workers": [
                    {
                        "owner": lease["owner"],
                        "lease_id": lease["lease_id"],
                        "heartbeat_at": lease["heartbeat_at"],
                        "lease_expires_at": lease["lease_expires_at"],
                        "task_count": len(lease.get("tasks") or []),
                    }
                    for lease in leases
                ],
            }

    def exhausted(self) -> bool:
        snapshot = self.snapshot()
        return (
            snapshot["PENDING"] == 0
            and snapshot["LEASED"] == 0
            and snapshot["FAILED"] == 0
            and snapshot["COMPLETE"] == snapshot["total"]
        )
