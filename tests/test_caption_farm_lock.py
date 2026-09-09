import multiprocessing as mp
import os
import signal

import pytest

from caption_farm.io import DirectoryLock


def _hold_lock(path, ready):
    with DirectoryLock(path):
        ready.set()
        signal.pause()


def _forked_child_probe(parent_lock, results):
    assert not parent_lock.acquired
    parent_lock.release()
    try:
        with DirectoryLock(parent_lock.path, timeout_seconds=0.1):
            results.put("acquired")
    except TimeoutError:
        results.put("blocked")


def test_old_mtime_never_revokes_live_holder(tmp_path):
    path = tmp_path / "claim.lock"
    with DirectoryLock(path, stale_seconds=0.01) as owner:
        os.utime(path, (0, 0))
        with pytest.raises(TimeoutError):
            DirectoryLock(path, stale_seconds=0.01, timeout_seconds=0.05).acquire()
        owner.refresh()


def test_release_is_idempotent_and_keeps_permanent_inode(tmp_path):
    path = tmp_path / "claim.lock"
    old = DirectoryLock(path).acquire()
    inode = path.stat().st_ino
    old.release()
    with DirectoryLock(path) as replacement:
        old.release()
        assert path.stat().st_ino == inode
        replacement.refresh()
        with pytest.raises(RuntimeError, match="not held"):
            old.refresh()


def test_same_process_alias_cannot_release_existing_record_lock(tmp_path):
    path = tmp_path / "claim.lock"
    alias = tmp_path / "alias.lock"
    with DirectoryLock(path):
        os.link(path, alias)
        with pytest.raises(TimeoutError):
            DirectoryLock(alias, timeout_seconds=0.05).acquire()
        context = mp.get_context("fork")
        ready = context.Event()
        process = context.Process(target=_hold_lock, args=(path, ready))
        process.start()
        try:
            assert not ready.wait(0.2)
        finally:
            process.kill()
            process.join(5)


def test_paused_holder_stays_exclusive_and_kill_releases_lock(tmp_path):
    path = tmp_path / "claim.lock"
    context = mp.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_lock, args=(path, ready))
    process.start()
    try:
        assert ready.wait(10)
        os.kill(process.pid, signal.SIGSTOP)
        os.utime(path, (0, 0))
        with pytest.raises(TimeoutError):
            DirectoryLock(path, timeout_seconds=0.1, stale_seconds=0.01).acquire()
        os.kill(process.pid, signal.SIGKILL)
        process.join(5)
        assert process.exitcode == -signal.SIGKILL
        with DirectoryLock(path, timeout_seconds=1):
            pass
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)


def test_forked_child_cannot_release_parent_lock(tmp_path):
    context = mp.get_context("fork")
    results = context.Queue()
    with DirectoryLock(tmp_path / "claim.lock") as owner:
        process = context.Process(target=_forked_child_probe, args=(owner, results))
        process.start()
        process.join(5)
        try:
            assert process.exitcode == 0
            assert results.get(timeout=1) == "blocked"
            owner.refresh()
        finally:
            if process.is_alive():
                process.kill()
                process.join(5)


def test_legacy_directory_requires_offline_upgrade(tmp_path):
    path = tmp_path / "claim.lock"
    path.mkdir()
    os.utime(path, (0, 0))
    with pytest.raises(RuntimeError, match="stop all old queue workers"):
        DirectoryLock(path, stale_seconds=0.01).acquire()
    assert path.is_dir()


def test_failed_owner_metadata_write_releases_lock(tmp_path, monkeypatch):
    path = tmp_path / "claim.lock"
    lock = DirectoryLock(path)
    with monkeypatch.context() as patch:
        def fail(_):
            raise OSError("injected fsync failure")
        patch.setattr("caption_farm.io.os.fsync", fail)
        with pytest.raises(OSError, match="injected"):
            lock.acquire()
    assert not lock.acquired
    with DirectoryLock(path, timeout_seconds=0.1):
        pass
