"""Read frozen image views from local files or uncompressed tar byte ranges."""

import os
from functools import lru_cache
from pathlib import Path


class _File:
    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDONLY)

    def __del__(self):
        os.close(self.fd)


@lru_cache(maxsize=32)
def _open(path: str) -> _File:
    return _File(path)


def read_image_bytes(path: str | Path, *, max_bytes: int | None = None) -> bytes:
    value = str(path)
    if not value.startswith("tar:"):
        with Path(value).open("rb") as handle:
            result = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
        if max_bytes is not None and len(result) > max_bytes:
            raise ValueError("local image exceeds byte limit")
        return result
    archive, span = value[4:].rsplit("::", 1)
    extent, _ = span.split("/", 1)
    offset, length = map(int, extent.split(":"))
    if offset < 0 or length <= 0:
        raise ValueError(f"invalid image byte range: {value}")
    if max_bytes is not None and length > max_bytes:
        raise ValueError("local image exceeds byte limit")
    # Hold a reference throughout pread: another reader can evict this LRU entry.
    handle = _open(archive)
    result = os.pread(handle.fd, length, offset)
    if len(result) != length:
        raise ValueError(f"short image shard read: {value}")
    return result
