from __future__ import annotations

import hashlib
import fcntl
import json
import os
import random
import socket
import time
import uuid
import weakref
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


_held_locks: weakref.WeakSet = weakref.WeakSet()


def _close_inherited_locks() -> None:
    # flock belongs to an open file description. A forked worker must neither
    # keep its parent's lock alive nor unlock that shared description.
    for lock in list(_held_locks):
        lock._close()


os.register_at_fork(after_in_child=_close_inherited_locks)


class DirectoryLock(AbstractContextManager["DirectoryLock"]):
    """Single-host process-owned flock on a permanent file (legacy API name).

    Never unlink/replace the lock file or revoke a live holder by elapsed time.
    The kernel releases ownership on close/exit; a paused holder stays owner.
    Data synthesis is single-host. All local queue processes use this protocol.
    A leftover legacy
    mkdir lock requires an offline migration after every old worker is stopped.
    """

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
        self._fd: int | None = None
        self._pid: int | None = None

    def acquire(self) -> "DirectoryLock":
        if self.acquired:
            raise RuntimeError("cannot acquire an already held lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            self._fd = os.open(
                self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644
            )
            _held_locks.add(self)
        except IsADirectoryError as exc:
            raise RuntimeError(
                f"legacy directory lock at {self.path}; stop all old queue workers "
                "and controllers before removing the directory and upgrading together"
            ) from exc
        try:
            while True:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out acquiring shared lock {self.path}")
                    time.sleep(self.poll_seconds * random.uniform(0.8, 1.2))
            self.acquired = True
            self._pid = os.getpid()
            self.refresh()
            owner = {
                "protocol": "posix_flock_v1", "pid": self._pid,
                "hostname": socket.gethostname(), "acquired_at": utc_now(),
                "acquired_unix": unix_now(),
            }
            payload = (canonical_json(owner) + "\n").encode("utf-8")
            os.ftruncate(self._fd, 0)
            # Write via the locked descriptor; atomic replace would replace
            # the inode carrying the lock and permit concurrent holders.
            with os.fdopen(os.dup(self._fd), "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _held_locks.add(self)
            return self
        except BaseException:
            self._close()
            raise

    def refresh(self) -> None:
        """Check ownership and update diagnostic mtime; this is not a lease."""
        if not self.acquired or self._pid != os.getpid() or self._fd is None:
            raise RuntimeError("cannot refresh a lock that is not held")
        try:
            visible = self.path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"held lock disappeared: {self.path}") from exc
        held = os.fstat(self._fd)
        if (visible.st_dev, visible.st_ino) != (held.st_dev, held.st_ino):
            raise RuntimeError(f"held lock file was replaced: {self.path}")
        os.utime(self._fd, None)

    def _close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
        self._fd = None
        self._pid = None
        self.acquired = False
        _held_locks.discard(self)

    def release(self) -> None:
        self._close()

    def __enter__(self) -> "DirectoryLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()
