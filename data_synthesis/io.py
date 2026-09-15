"""Shared bounded direct downloads, image archives and atomic metadata writes."""
import asyncio
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tarfile
import time
from urllib.parse import urlparse

import httpx

from utils.direct_network import direct_ssl_context
from utils.image_shard_io import read_image_bytes

def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)

class DirectDownloader:
    def __init__(self, connections=64, per_host=8, max_bytes=20 << 20):
        # Explicit transport disables environment HTTP(S)/ALL/SOCKS proxies.
        self.client = httpx.AsyncClient(
            trust_env=False, proxy=None, follow_redirects=False, http2=True,
            verify=direct_ssl_context(),
            timeout=httpx.Timeout(20, connect=5),
            limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections),
        )
        self.per_host, self.max_bytes = per_host, max_bytes
        self.hosts = {}
        self.host_failures, self.host_retry_at = {}, {}

    async def get(self, row):
        if row.get("local_path"):
            return await asyncio.to_thread(read_image_bytes, row["local_path"], max_bytes=self.max_bytes)
        initial_url = str(row["url"])
        origin = urlparse(initial_url).hostname
        if time.monotonic() < self.host_retry_at.get(origin, 0):
            raise ValueError("direct image host temporarily unavailable; retry later")
        for attempt in range(3):
            try:
                url = initial_url
                for _ in range(8):
                    parsed = urlparse(url)
                    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
                        raise ValueError("image URL must be an unauthenticated HTTP(S) URL")
                    semaphore = self.hosts.setdefault(parsed.hostname, asyncio.Semaphore(self.per_host))
                    async with semaphore, self.client.stream("GET", url) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            url = str(response.url.join(response.headers["location"]))
                            continue
                        response.raise_for_status()
                        if int(response.headers.get("content-length", 0)) > self.max_bytes:
                            raise ValueError("image exceeds byte limit")
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data) > self.max_bytes:
                                raise ValueError("image exceeds byte limit")
                        self.host_failures[origin] = 0
                        self.host_retry_at.pop(origin, None)
                        return bytes(data)
                raise ValueError("image redirect limit exceeded")
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code != 429 and code < 500:
                    raise ValueError(f"image HTTP {code}") from None
                if attempt == 2:
                    self.mark_host_failure(origin)
                    raise ValueError(f"image HTTP {code} after retries") from None
            except httpx.TransportError:
                if attempt == 2:
                    self.mark_host_failure(origin)
                    raise ValueError("direct image transport failed after retries") from None
            await asyncio.sleep(2 ** attempt + random.random())

    def mark_host_failure(self, host):
        self.host_failures[host] = self.host_failures.get(host, 0) + 1
        if self.host_failures[host] >= 2:
            # Do not let a dead origin occupy all image workers indefinitely.
            # These tasks stay retryable; no proxy or alternate source is used.
            self.host_retry_at[host] = time.monotonic() + 120

    async def close(self):
        await self.client.aclose()

class ImageArchives:
    """Sequential tar writes; byte-range reads do not scan tar member tables."""
    def __init__(self, root, max_bytes=512 << 20):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        existing = list(self.root.glob("images-*.tar"))
        self.index = max((int(p.stem.split("-")[-1]) for p in existing), default=-1) + 1
        self.archive = self.handle = None
        self.max_bytes = max_bytes

    def add(self, name, data):
        if self.handle is None or self.handle.tell() + len(data) > self.max_bytes:
            self.close()
            self.path = (self.root / f"images-{self.index:06d}.tar").resolve()
            self.index += 1
            self.handle = self.path.open("xb")
            self.archive = tarfile.open(fileobj=self.handle, mode="w", format=tarfile.USTAR_FORMAT)
        offset = self.handle.tell() + 512
        info = tarfile.TarInfo(name)
        info.size = len(data)
        self.archive.addfile(info, io.BytesIO(data))
        self.handle.flush()
        return f"tar:{self.path}::{offset}:{len(data)}/{name}"

    def sync(self):
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.sync()
            self.handle.close()
            self.archive = self.handle = None


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


file_sha = digest_file


def training_image_id(view):
    # Existing loaders derive T2I identity from the frozen image member name.
    # State keys identify original content and need not equal that filename.
    return "train/" + Path(view["source_path"]).stem


def cohort_id(root):
    root = Path(root).resolve()
    marker = root / "cohort_identity.json"
    if marker.exists():
        value = json.loads(marker.read_text())
        identity = value.get("id", "")
        if (value.get("schema") != "b512_cohort_identity_v1" or len(identity) != 16
                or any(c not in "0123456789abcdef" for c in identity)):
            raise ValueError("invalid persisted cohort identity")
        return identity
    return sha(str(root).encode())[:16]


def pin_cohort_id(root):
    """Keep pre-existing batch/archive IDs stable across storage relocation."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "cohort_identity.json"
    with marker.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = cohort_id(root)
        if not marker.exists():
            atomic_json(marker, {"schema": "b512_cohort_identity_v1", "id": identity,
                                 "original_root": str(root)})
    return identity
