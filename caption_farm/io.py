from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import socket
import time
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unix_now() -> float:
    return time.time()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def append_jsonl(path: Path, value: Any, *, sync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, payload)
        if sync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_link_json(path: Path, value: Any, temporary_root: Path) -> bool:
    """Publish a fully synced JSON record without ever replacing a visible result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = temporary_root / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        fsync_directory(path.parent)
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class DirectoryLock(AbstractContextManager["DirectoryLock"]):
    """Cross-process lock using the atomic POSIX mkdir primitive."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 60.0,
        stale_seconds: float = 120.0,
        poll_seconds: float = 0.05,
    ) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self.poll_seconds = poll_seconds
        self.acquired = False

    def acquire(self) -> "DirectoryLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                os.mkdir(self.path)
                self.acquired = True
                owner = {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "acquired_at": utc_now(),
                    "acquired_unix": unix_now(),
                }
                atomic_write_json(self.path / "owner.json", owner)
                return self
            except FileExistsError:
                self._break_stale_lock()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring shared lock {self.path}")
                time.sleep(self.poll_seconds * random.uniform(0.8, 1.2))

    def _break_stale_lock(self) -> None:
        try:
            age = unix_now() - self.path.stat().st_mtime
        except FileNotFoundError:
            return
        if age <= self.stale_seconds:
            return
        stale = self.path.parent / f"{self.path.name}.stale.{uuid.uuid4().hex}"
        try:
            os.rename(self.path, stale)
        except (FileNotFoundError, FileExistsError, OSError):
            return
        shutil.rmtree(stale, ignore_errors=True)

    def refresh(self) -> None:
        """Refresh a held lock while performing a bounded batch operation."""
        if not self.acquired:
            raise RuntimeError("cannot refresh a lock that is not held")
        try:
            os.utime(self.path, None)
        except FileNotFoundError as exc:
            raise RuntimeError(f"held lock disappeared: {self.path}") from exc

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            (self.path / "owner.json").unlink(missing_ok=True)
            os.rmdir(self.path)
        finally:
            self.acquired = False

    def __enter__(self) -> "DirectoryLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()
