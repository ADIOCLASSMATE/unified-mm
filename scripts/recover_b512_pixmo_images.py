"""Recover existing PixMo image candidates from a pinned public image mirror.

All HTTP clients are direct. Parquet byte ranges locate original URLs without
fetching all image columns. The Dataset Viewer provides full-size JPEG/PNG
renditions; they are recorded as mirror renditions, never as original bytes.
Only the original download service imports the resulting recovery manifests.
"""

import argparse
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import fcntl
import gzip
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import time
from urllib.parse import urlsplit

import httpx
from PIL import Image
import pyarrow.parquet as pq

from scripts.distill_b512_codex import atomic_json, dumps, file_sha, sha
from scripts.synthesize_image_text import ImageArchives
from utils.direct_network import check_direct_routes, direct_ssl_context


REPO = "anthracite-org/pixmo-cap-images"
REVISION = "3e42775eea79dd4874c41379230c7f871b10218e"
MAX_IMAGE_BYTES = 20 << 20


def metadata_hash(row):
    return sha(dumps({k: row[k] for k in ("image_url", "caption", "transcripts")}).encode())


def validate_asset(entry, target):
    row = entry["row"]
    if entry.get("truncated_cells") or row["image_url"] != target["row"]["url"]:
        raise ValueError("mirror row URL changed or was truncated")
    if metadata_hash(row) != target["metadata_sha256"]:
        raise ValueError("mirror metadata differs from the frozen original")
    asset = row["image"]
    url = urlsplit(asset["src"])
    prefix = f"/cached-assets/{REPO}/--/{REVISION}/--/default/train/{entry['row_idx']}/image/"
    if url.scheme != "https" or url.hostname != "datasets-server.huggingface.co" or not url.path.startswith(prefix):
        raise ValueError("mirror asset identity or revision changed")
    if url.path.removeprefix(prefix) not in {"image.jpg", "image.png"}:
        raise ValueError("unexpected mirror image rendition")
    return asset


async def resolve_matching_asset(network, entry, target, response_path, response_hash, total_rows):
    """A URL can have several authentic annotations; require the frozen one."""
    if entry["row"]["image_url"] != target["row"]["url"]:
        raise ValueError("mirror row URL changed")
    if metadata_hash(entry["row"]) != target["metadata_sha256"]:
        for row_index in target.get("alternate_row_indices", []):
            data, _, _ = await network.get("https://datasets-server.huggingface.co/rows", limit=4 << 20,
                params={"dataset": REPO, "config": "default", "split": "train", "offset": row_index, "length": 1})
            candidate = json.loads(data)
            if candidate.get("partial") or candidate["num_rows_total"] != total_rows or len(candidate["rows"]) != 1:
                raise ValueError("partial or changed alternate mirror row")
            alternate = candidate["rows"][0]
            if alternate["row_idx"] != row_index or alternate["row"]["image_url"] != target["row"]["url"]:
                raise ValueError("alternate mirror row identity changed")
            if metadata_hash(alternate["row"]) == target["metadata_sha256"]:
                entry = alternate
                response_hash = sha(data)
                response_path = response_path.parent / f"alternate-{row_index:09d}-{time.time_ns()}.json.gz"
                response_path.write_bytes(gzip.compress(data))
                break
    asset = validate_asset(entry, target)
    return asset, entry["row_idx"], response_path, response_hash


async def single_mirror_row(network, row_index, total_rows, work):
    data, _, _ = await network.get("https://datasets-server.huggingface.co/rows", limit=4 << 20,
        params={"dataset": REPO, "config": "default", "split": "train", "offset": row_index, "length": 1})
    response = json.loads(data)
    if (response.get("partial") or response["num_rows_total"] != total_rows
            or len(response["rows"]) != 1 or response["rows"][0]["row_idx"] != row_index):
        raise ValueError("partial or changed individual mirror row")
    path = work / "responses" / f"single-{row_index:09d}-{time.time_ns()}.json.gz"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.compress(data))
    return response["rows"][0], path, sha(data)


def freeze_targets(root, work):
    path = work / "targets.json.gz"
    if path.exists():
        payload = json.loads(gzip.decompress(path.read_bytes()))
        if sha(dumps(payload).encode()) != json.loads((work / "targets.sha256.json").read_text())["sha256"]:
            raise ValueError("frozen recovery targets changed")
        return payload
    db = sqlite3.connect(f"file:{root / 'download.sqlite3'}?mode=ro", uri=True)
    targets = {}
    for key, text, status, attempts in db.execute("SELECT key,row_json,status,attempts FROM tasks WHERE status!='downloaded'"):
        row = json.loads(text)
        if row["source"] == "pixmo_cap":
            targets[row["url"]] = {"key": key, "row": row, "row_sha256": sha(dumps(row).encode()),
                                   "original_status": status, "original_attempts": attempts}
    db.close()
    sources = {v["row"]["supply_source_file"]: v["row"]["supply_source_sha256"] for v in targets.values()}
    for source, digest in sources.items():
        if file_sha(source) != digest:
            raise ValueError("original candidate metadata file changed")
        for batch in pq.ParquetFile(source).iter_batches(batch_size=1024, columns=["image_url", "caption", "transcripts"]):
            for row in batch.to_pylist():
                if row["image_url"] in targets:
                    targets[row["image_url"]]["metadata_sha256"] = metadata_hash(row)
    if any("metadata_sha256" not in target for target in targets.values()):
        raise ValueError("an original candidate cannot be found in frozen metadata")
    payload = {"at": time.time(), "repo": REPO, "revision": REVISION, "targets": targets}
    data = dumps(payload).encode()
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(gzip.compress(data))
    temporary.replace(path)
    atomic_json(work / "targets.sha256.json", {"sha256": sha(data), "records": len(targets)})
    return payload


async def persist_image_results(tasks, archive_root, root, page_offset, complete, errors, counts,
                                on_flush, *, flush_images=8, flush_seconds=5):
    """Publish completed downloads while slower images in the page are pending."""
    pending = dict(tasks)
    archives, recovered, deadline = None, [], None

    def flush():
        nonlocal archives, recovered, deadline
        if archives is not None:
            archives.close()
            archives = None
        if recovered:
            atomic_json(root / "recovery_inbox" / f"pixmo-mirror-{page_offset:09d}-{time.time_ns()}.json",
                {"schema": "b512.image_recovery.v1", "created_at": time.time(), "rows": recovered})
            for item in recovered:
                complete[item["key"]] = item
                errors.pop(item["key"], None)
            counts["downloaded"] += len(recovered)
            recovered = []
            on_flush()
        deadline = None

    try:
        while pending:
            timeout = max(0, deadline - time.monotonic()) if deadline is not None else flush_seconds
            done, _ = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                target = pending.pop(task)
                key = target["key"]
                try:
                    result = task.result()
                except Exception as exc:
                    errors[key] = type(exc).__name__ + ": " + str(exc)[:160]
                    continue
                if result is None:
                    continue
                _, image_data, dimensions, asset_path, actual_index, proof_path, proof_hash = result
                if archives is None:
                    archives = ImageArchives(archive_root / "images" / f"page-{page_offset:09d}")
                recovered.append({"key": key, "source_id": target["row"]["source_id"],
                    "original_url": target["row"]["url"], "candidate_sha256": target["row_sha256"],
                    "sha256": sha(image_data), "reference": archives.add(key + ".mirror", image_data),
                    "dimensions": dimensions, "mirror": {"repo": REPO, "revision": REVISION,
                        "row_idx": actual_index, "asset_path": asset_path,
                        "metadata_sha256": target["metadata_sha256"], "response_path": str(proof_path),
                        "response_sha256": proof_hash, "rendition": "huggingface_viewer_full_size_jpeg_or_png",
                        "proxy": False}})
                if deadline is None:
                    deadline = time.monotonic() + flush_seconds
                if len(recovered) >= flush_images:
                    flush()
            if recovered and (not pending or time.monotonic() >= deadline):
                flush()
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        flush()


class DirectHTTP:
    def __init__(self, concurrency, metadata_rps=1):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(trust_env=False, proxy=None, http2=True,
            verify=direct_ssl_context(), follow_redirects=True,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
            timeout=httpx.Timeout(90, connect=15))
        self.bytes = 0
        self.cooldowns = {}
        self.next_metadata_request = 0
        self.metadata_interval = 1 / metadata_rps
        self.request_tasks = self.active_requests = 0
        self.events = Counter()
        self.resolved_files = {}

    @staticmethod
    def rate_scope(url):
        parsed = urlsplit(url)
        if parsed.hostname == "datasets-server.huggingface.co":
            if parsed.path == "/rows":
                return "viewer_metadata"
            if parsed.path.startswith("/cached-assets/"):
                return "viewer_images"
        return parsed.hostname or "unknown"

    @property
    def retry_until(self):
        """Latest cooldown for diagnostics; admission uses the request's scope."""
        return max(self.cooldowns.values(), default=0)

    def snapshot(self):
        now = time.monotonic()
        return {"request_tasks": self.request_tasks, "active_requests": self.active_requests,
                "events": dict(self.events), "metadata_rps": 1 / self.metadata_interval,
                "cooldown_until": {key: time.time() + deadline - now
                                   for key, deadline in self.cooldowns.items() if deadline > now}}

    def restore_cooldowns(self, value):
        for key, deadline in value.items():
            self.cooldowns[key] = max(self.cooldowns.get(key, 0),
                                      time.monotonic() + max(0, deadline - time.time()))

    @asynccontextmanager
    async def request_slot(self, scope):
        def delay():
            deadline = self.cooldowns.get(scope, 0)
            if scope == "viewer_metadata":
                deadline = max(deadline, self.next_metadata_request)
            return max(0, deadline - time.monotonic())
        while True:
            remaining = delay()
            if remaining:
                await asyncio.sleep(remaining)
                continue
            await self.semaphore.acquire()
            # A 429 may arrive while this request is already queued for a
            # connection. Recheck without holding a slot during the cooldown.
            if delay():
                self.semaphore.release()
                continue
            if scope == "viewer_metadata":
                self.next_metadata_request = time.monotonic() + self.metadata_interval
            break
        self.active_requests += 1
        try:
            yield
        finally:
            self.active_requests -= 1
            self.semaphore.release()

    async def get(self, url, *, limit=MAX_IMAGE_BYTES, params=None, headers=None):
        self.request_tasks += 1
        try:
            return await self._get(url, limit=limit, params=params, headers=headers)
        finally:
            self.request_tasks -= 1

    async def _get(self, url, *, limit, params, headers):
        request_retry_until = 0
        scope = self.rate_scope(url)
        for attempt in range(5):
            await asyncio.sleep(max(0, request_retry_until - time.monotonic()))
            try:
                async with self.request_slot(scope):
                    async with self.client.stream("GET", url, params=params, headers=headers) as response:
                        code = response.status_code
                        self.events[f"{scope}:http_{code}"] += 1
                        if code == 429 or code >= 500:
                            retry = response.headers.get("retry-after", "")
                            delay = float(retry) if retry.isdigit() else min(120, 5 * 2 ** attempt)
                            if code == 429:
                                self.cooldowns[scope] = max(self.cooldowns.get(scope, 0), time.monotonic() + delay)
                            else:
                                # A broken viewer page must not pause healthy assets.
                                request_retry_until = time.monotonic() + delay
                            raise OSError(f"mirror HTTP {code}; retry after {delay} seconds")
                        if code >= 400:
                            raise ValueError(f"mirror HTTP {code}")
                        if urlsplit(url).hostname == "huggingface.co" and "/resolve/" in urlsplit(url).path:
                            self.resolved_files[url] = str(response.url)
                        parts, size = [], 0
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > limit:
                                raise ValueError("mirror response exceeds byte limit")
                            parts.append(chunk)
                        self.bytes += size
                        return b"".join(parts), response.headers, code
            except (httpx.HTTPError, OSError) as exc:
                if isinstance(exc, httpx.HTTPError):
                    self.events[f"{scope}:{type(exc).__name__}"] += 1
                if attempt == 4:
                    raise
                await asyncio.sleep(min(2 ** attempt, 8))

    async def range(self, url, start, end, size):
        # A resolved Hub URL is a reusable, signed CDN URL. Re-resolving every
        # column range needlessly consumes the Hub resolver request allowance.
        data, headers, code = await self.get(self.resolved_files.get(url, url), limit=end - start + 1,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        if code != 206 or headers.get("content-range") != f"bytes {start}-{end}/{size}" or len(data) != end - start + 1:
            raise ValueError("mirror did not honor the exact byte range")
        return data


async def mirror_file_index(network, work, archive_root, item):
    name, size = Path(item["path"]).name, item["size"]
    index = work / "url_indexes" / (name + ".json.gz")
    if index.exists():
        data = gzip.decompress(index.read_bytes())
        if sha(data) != json.loads(index.with_suffix(".sha256.json").read_text())["sha256"]:
            raise ValueError("cached mirror URL index changed")
        return json.loads(data)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item['path']}"
    tail = await network.range(url, size - 65536, size - 1, size)
    if tail[-4:] != b"PAR1":
        raise ValueError("invalid Parquet footer")
    length = struct.unpack("<I", tail[-8:-4])[0] + 8
    if length > 8 << 20:
        raise ValueError("unexpectedly large mirror footer")
    if length > len(tail):
        tail = await network.range(url, size - length, size - 1, size)
    sparse = archive_root / "sparse_url_indexes" / name
    sparse.parent.mkdir(parents=True, exist_ok=True)
    with sparse.open("wb") as handle:
        handle.write(b"PAR1")
        handle.seek(size - len(tail))
        handle.write(tail)
    parquet = pq.ParquetFile(sparse)
    async def column_range(column):
        start = min(x for x in [column.dictionary_page_offset, column.data_page_offset] if x is not None and x >= 0)
        return start, await network.range(url, start, start + column.total_compressed_size - 1, size)
    requests = []
    for i in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(i)
        requests.extend(column_range(group.column(j)) for j in range(group.num_columns)
                        if group.column(j).path_in_schema == "image_url")
    chunks = await asyncio.gather(*requests)
    with sparse.open("r+b") as handle:
        for start, data in chunks:
            handle.seek(start)
            handle.write(data)
    urls = pq.read_table(sparse, columns=["image_url"], pre_buffer=False).column("image_url").to_pylist()
    payload = {"file": item, "repo": REPO, "revision": REVISION, "rows": len(urls), "urls": urls,
               "footer_sha256": sha(tail), "sparse_metadata_only": True}
    data = dumps(payload).encode()
    index.parent.mkdir(parents=True, exist_ok=True)
    temporary = index.with_suffix(".tmp")
    temporary.write_bytes(gzip.compress(data))
    temporary.replace(index)
    atomic_json(index.with_suffix(".sha256.json"), {"sha256": sha(data)})
    return payload


async def run(args):
    check_direct_routes()
    root, images = Path(args.root).resolve(), Path(args.image_root).resolve()
    work = root / "mirror_recovery"
    work.mkdir(exist_ok=True)
    lock = (work / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous_status = json.loads((work / "status.json").read_text()) if (work / "status.json").exists() else {}
    status = {"state": "running", "pid": os.getpid(), "started_at": time.time(), "stage": "freeze_targets",
              "proxy": False, "repo": REPO, "revision": REVISION}
    producer = root / "recovery_producers" / "pixmo_mirror.json"
    atomic_json(producer, status)
    network = DirectHTTP(args.image_workers, metadata_rps=args.metadata_rps)
    network.restore_cooldowns(previous_status.get("network", {}).get("cooldown_until", {}))
    if args.metadata_start_delay:
        network.cooldowns["viewer_metadata"] = max(network.cooldowns.get("viewer_metadata", 0),
                                                   time.monotonic() + args.metadata_start_delay)
    archive_root = images / "source_archives" / "pixmo_mirror" / REVISION
    counts = Counter()
    def update(**values):
        status.update(values, counts=dict(counts), updated_at=time.time(), network_bytes=network.bytes,
                      network=network.snapshot())
        atomic_json(work / "status.json", status)
        atomic_json(producer, status)
    async def heartbeat():
        while True:
            await asyncio.sleep(10)
            update()
    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        targets = freeze_targets(root, work)["targets"]
        metadata = json.loads(Path(args.mirror_metadata).read_text())
        if metadata["metadata"]["sha"] != REVISION or metadata["metadata"]["id"] != REPO:
            raise ValueError("mirror repository does not match pinned revision")
        files = sorted((x for x in metadata["tree"] if x["type"] == "file" and x["path"].endswith(".parquet")), key=lambda x: x["path"])
        if len(files) != 75:
            raise ValueError("incomplete mirror shard list")
        update(stage="index_original_urls", target_count=len(targets))
        file_semaphore = asyncio.Semaphore(8)
        async def index_one(item):
            async with file_semaphore:
                result = await mirror_file_index(network, work, archive_root, item)
            counts["indexed_files"] += 1
            update()
            return result
        indexes = await asyncio.gather(*(index_one(item) for item in files))
        pages, matched, offset = {}, set(), 0
        for index in indexes:
            for local, url in enumerate(index["urls"]):
                if url in targets:
                    row_index = offset + local
                    if url in matched:
                        targets[url].setdefault("alternate_row_indices", []).append(row_index)
                    else:
                        pages.setdefault(row_index // 100 * 100, {})[row_index] = targets[url]
                        matched.add(url)
            offset += index["rows"]
        if offset != 707647:
            raise ValueError("mirror row count changed")
        atomic_json(work / "coverage.json", {"targets": len(targets), "matched": len(matched), "pages": len(pages),
            "mirror_rows": offset, "missing_original_urls": sorted(set(targets) - matched), "at": time.time()})
        update(stage="download_images", matched_targets=len(matched), pages=len(pages))
        page_semaphore = asyncio.Semaphore(args.page_workers)

        async def download_page(page_offset, selected):
            done_path = work / "pages" / f"{page_offset:09d}.json"
            previous_page = json.loads(done_path.read_text()) if done_path.exists() else {}
            if previous_page.get("state") == "completed":
                counts["pages_reused"] += 1
                return
            async with page_semaphore:
                complete, errors = {}, {}
                single_row_mode = "mirror HTTP 5" in previous_page.get("errors", {}).get("page", "")
                for previous in sorted((root / "recovery_inbox").glob(f"pixmo-mirror-{page_offset:09d}-*.json")):
                    for item in json.loads(previous.read_text())["rows"]:
                        complete[item["key"]] = item
                # Sparse pages often contain only one missing image. Keep the
                # durable page identity, but avoid materializing unrelated rows.
                request_start = min(selected)
                request_length = max(selected) - request_start + 1
                for attempt in range(5):
                    try:
                        if not single_row_mode:
                            data, _, _ = await network.get("https://datasets-server.huggingface.co/rows", limit=4 << 20,
                                params={"dataset": REPO, "config": "default", "split": "train",
                                        "offset": request_start, "length": request_length})
                            response = json.loads(data)
                            if response.get("partial") or response["num_rows_total"] != offset:
                                raise ValueError("partial or changed mirror rows")
                            entries = {entry["row_idx"]: entry for entry in response["rows"]}
                            if not selected.keys() <= entries.keys():
                                raise ValueError("mirror response omitted requested rows")
                            response_hash = sha(data)
                            response_path = work / "responses" / f"{page_offset:09d}-{time.time_ns()}.json.gz"
                            response_path.parent.mkdir(exist_ok=True)
                            response_path.write_bytes(gzip.compress(data))
                        async def fetch_image(row_index, target):
                            if target["key"] in complete:
                                return None
                            if single_row_mode:
                                entry, row_path, row_hash = await single_mirror_row(network, row_index, offset, work)
                            else:
                                entry, row_path, row_hash = entries[row_index], response_path, response_hash
                            asset, actual_index, proof_path, proof_hash = await resolve_matching_asset(
                                network, entry, target, row_path, row_hash, offset)
                            image_data, _, _ = await network.get(asset["src"], limit=args.max_image_bytes)
                            with Image.open(io.BytesIO(image_data)) as image:
                                image.load()
                                if image.size != (asset["width"], asset["height"]):
                                    raise ValueError("mirror image dimensions changed")
                                dimensions = list(image.size)
                            if target["row"].get("expected_source_sha256") and sha(image_data) != target["row"]["expected_source_sha256"]:
                                raise ValueError("mirror rendition differs from required original bytes")
                            return target, image_data, dimensions, urlsplit(asset["src"]).path, actual_index, proof_path, proof_hash
                        tasks = {asyncio.create_task(fetch_image(i, t)): t for i, t in selected.items()
                                 if t["key"] not in complete}
                        await persist_image_results(tasks, archive_root, root, page_offset,
                            complete, errors, counts, update)
                        if len(complete) == len(selected):
                            errors.pop("page", None)
                            break
                    except Exception as exc:
                        errors["page"] = type(exc).__name__ + ": " + str(exc)[:160]
                        if isinstance(exc, OSError) and str(exc).startswith("mirror HTTP 5"):
                            single_row_mode = True
                    if attempt < 4:
                        await asyncio.sleep(min(30, 2 ** attempt))
                state = "completed" if len(complete) == len(selected) else "partial"
                atomic_json(done_path, {"state": state, "offset": page_offset, "expected": len(selected),
                    "downloaded": len(complete), "errors": errors, "at": time.time()})
                counts["pages_" + state] += 1
                update()

        await asyncio.gather(*(download_page(start, selected) for start, selected in sorted(pages.items())))
        update(state="completed" if not counts["pages_partial"] else "partial", stage="finished", finished_at=time.time())
    except BaseException as exc:
        update(state="interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed",
               error=type(exc).__name__ + ": " + str(exc)[:160])
        raise
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        await network.client.aclose()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--mirror-metadata", required=True)
    parser.add_argument("--page-workers", type=int, default=16)
    parser.add_argument("--image-workers", type=int, default=64)
    parser.add_argument("--metadata-rps", type=float, default=1)
    parser.add_argument("--metadata-start-delay", type=float, default=0)
    parser.add_argument("--max-image-bytes", type=int, default=MAX_IMAGE_BYTES,
                        help="Bound each image download; persisted completed pages are reused")
    args = parser.parse_args()
    if min(args.page_workers, args.image_workers) < 1:
        parser.error("worker counts must be positive")
    if not 1 <= args.max_image_bytes <= 128 << 20:
        parser.error("image byte limit must be between 1 byte and 128 MiB")
    if not math.isfinite(args.metadata_rps) or args.metadata_rps <= 0:
        parser.error("metadata request rate must be finite and positive")
    if not math.isfinite(args.metadata_start_delay) or args.metadata_start_delay < 0:
        parser.error("metadata start delay must be finite and nonnegative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
