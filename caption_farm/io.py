from __future__ import annotations

import hashlib
import errno
import fcntl
import json
import os
import random
import socket
import time
import threading
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
_process_guards: dict[tuple, threading.Lock] = {}
_registry_guard = threading.Lock()


def _close_inherited_locks() -> None:
    global _registry_guard
    # POSIX record locks are not inherited. Close inherited descriptors and
    # reset Python mutexes so the child cannot mistake parent state for its own.
    for lock in list(_held_locks):
        lock._close()
    _process_guards.clear()
    _registry_guard = threading.Lock()


os.register_at_fork(after_in_child=_close_inherited_locks)


class DirectoryLock(AbstractContextManager["DirectoryLock"]):
    """Process-owned POSIX record lock on a permanent file (legacy API name).

    Use fcntl record locks: BSD flock on the deployment GPFS was verified to
    exclude local processes only. Never replace/unlink the lock inode or revoke
    a live holder by elapsed time. All queue participants must upgrade together.
    """

    def __init__(self, path: Path, *, timeout_seconds: float = 60.0,
                 stale_seconds: float = 120.0, poll_seconds: float = 0.05) -> None:
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds  # Configuration compatibility only.
        self.poll_seconds = poll_seconds
        self.acquired = False
        self._fd: int | None = None
        self._pid: int | None = None
        self._process_guard = None

    def _try_open(self) -> bool:
        # POSIX locks belong to the process: a second descriptor's close would
        # release the first descriptor's lock. Serialize same-process users
        # BEFORE open, including hard-link aliases of an existing inode.
        with _registry_guard:
            try:
                stat = self.path.stat()
                key = (stat.st_dev, stat.st_ino)
            except FileNotFoundError:
                key = (str(self.path.resolve()),)
            guard = _process_guards.setdefault(key, threading.Lock())
            if not guard.acquire(blocking=False):
                return False
            try:
                self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
                stat = os.fstat(self._fd)
                _process_guards[(stat.st_dev, stat.st_ino)] = guard
                self._process_guard = guard
                self._pid = os.getpid()
                _held_locks.add(self)
                return True
            except BaseException:
                guard.release()
                raise

    def _wait(self, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out acquiring shared lock {self.path}")
        time.sleep(self.poll_seconds * random.uniform(0.8, 1.2))

    def acquire(self) -> "DirectoryLock":
        if self.acquired or self._fd is not None:
            raise RuntimeError("cannot acquire an already held lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while not self._try_open():
                self._wait(deadline)
            while True:
                try:
                    fcntl.lockf(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EAGAIN, errno.EACCES}:
                        raise
                    self._wait(deadline)
            self.acquired = True
            self.refresh()
            # Diagnostic data has its own inode. Opening/closing a duplicate
            # descriptor of the lock file would release a POSIX record lock.
            atomic_write_json(self.path.with_name(self.path.name + ".owner.json"), {
                "protocol": "posix_record_lock_v1", "pid": self._pid,
                "hostname": socket.gethostname(), "acquired_at": utc_now(),
                "acquired_unix": unix_now(), "owner_id": uuid.uuid4().hex,
            })
            return self
        except IsADirectoryError as exc:
            self._close()
            raise RuntimeError(
                f"legacy directory lock at {self.path}; stop all old queue workers "
                "and controllers before removing the directory and upgrading together"
            ) from exc
        except BaseException:
            self._close()
            raise

    def refresh(self) -> None:
        """Check ownership; mtime is diagnostic and never transfers ownership."""
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
        if self._process_guard is not None:
            self._process_guard.release()
            self._process_guard = None
        _held_locks.discard(self)

    def release(self) -> None:
        self._close()

    def __enter__(self) -> "DirectoryLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()
