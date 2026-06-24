"""
Convert OmniCorpus parquet shards into interleaved document JSONL and images.

This stage is designed to survive preemption:
  - completed JPEGs are reused;
  - partial JPEG downloads resume through .part files when the server supports
    HTTP Range;
  - JSONL output is appended from a checkpoint instead of overwritten.

Output JSONL schema:
    {
      "doc_id": "...",
      "source": "OpenGVLab/OmniCorpus-CC-210M",
      "img_ids": [900000000, ...],
      "segments": [
        {"type": "text", "content": "..."},
        {"type": "image", "img_idx": 0},
        ...
      ]
    }
"""

import argparse
import concurrent.futures as futures
import glob
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq
import requests
from PIL import Image
from tqdm import tqdm


PERMANENT_HTTP_STATUS = {400, 401, 403, 404, 410, 451}
CHECKPOINT_VERSION = 1
LOG_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()
PendingEntry = Tuple[Dict[str, Any], List[Tuple[str, Path]], int, int]


def log(message: str) -> None:
    with LOG_LOCK:
        print(message, flush=True)


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        THREAD_LOCAL.session = session
    return session


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "images": _to_python(row["images"]),
        "texts": _to_python(row["texts"]),
        "metadata": _to_python(row["metadata"]),
        "general_metadata": _to_python(row["general_metadata"]),
    }


def iter_parquet_rows(
    paths: List[Path],
    batch_size: int,
    start_scanned_rows: int,
) -> Iterable[Tuple[Dict[str, Any], int]]:
    """Yield rows plus the global scanned-row count after each yielded row."""
    columns = ["images", "texts", "metadata", "general_metadata"]
    scanned = 0
    skip_remaining = start_scanned_rows

    for path in paths:
        parquet_file = pq.ParquetFile(path)
        file_rows = parquet_file.metadata.num_rows
        if skip_remaining >= file_rows:
            skip_remaining -= file_rows
            scanned += file_rows
            continue

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            rows = batch.to_pylist()
            start = 0
            if skip_remaining > 0:
                if skip_remaining >= len(rows):
                    skip_remaining -= len(rows)
                    scanned += len(rows)
                    continue
                start = skip_remaining
                scanned += skip_remaining
                skip_remaining = 0

            for row in rows[start:]:
                scanned += 1
                yield normalize_row(row), scanned


def is_safe(row: Dict[str, Any], max_unsafe_prob: float) -> bool:
    general = row.get("general_metadata") or {}
    for key in ("porn_prob", "toxic_prob"):
        value = general.get(key)
        if value is not None and float(value) > max_unsafe_prob:
            return False
    for meta in row.get("metadata") or []:
        if meta and meta.get("unsafe_prob") is not None:
            if float(meta["unsafe_prob"]) > max_unsafe_prob:
                return False
    return True


def is_valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def existing_image_ok(path: Path, verify_existing_images: bool) -> bool:
    if verify_existing_images:
        return is_valid_image(path)
    return path.exists() and path.stat().st_size > 0


def stream_image_to_part(
    url: str,
    output_path: Path,
    timeout: float,
    min_bytes_per_sec: float,
    speed_check_after: float,
    chunk_size: int,
) -> Path:
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    resume_from = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OmniCorpusPrep/1.0)"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    session = get_session()
    with session.get(url, headers=headers, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        if resume_from > 0 and response.status_code != 206:
            resume_from = 0
            part_path.unlink(missing_ok=True)

        mode = "ab" if resume_from > 0 else "wb"
        start = time.monotonic()
        downloaded = resume_from
        with part_path.open(mode) as out:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                out.write(chunk)
                downloaded += len(chunk)
                elapsed = time.monotonic() - start
                new_bytes = downloaded - resume_from
                if (
                    min_bytes_per_sec > 0
                    and elapsed >= speed_check_after
                    and new_bytes / max(elapsed, 1e-6) < min_bytes_per_sec
                ):
                    raise TimeoutError(
                        f"download too slow: {new_bytes / elapsed:.1f} B/s "
                        f"< {min_bytes_per_sec:.1f} B/s"
                    )
    return part_path


def download_image(
    url: str,
    output_path: Path,
    timeout: float,
    retries: int,
    connect_timeout_retries: int,
    retry_sleep: float,
    max_retry_sleep: float,
    min_bytes_per_sec: float,
    speed_check_after: float,
    chunk_size: int,
    verify_existing_images: bool,
    log_image_failures: bool,
) -> bool:
    if existing_image_ok(output_path, verify_existing_images):
        return True
    output_path.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            part_path = stream_image_to_part(
                url=url,
                output_path=output_path,
                timeout=timeout,
                min_bytes_per_sec=min_bytes_per_sec,
                speed_check_after=speed_check_after,
                chunk_size=chunk_size,
            )
            with Image.open(part_path) as image:
                image = image.convert("RGB")
                image.save(output_path, format="JPEG", quality=95)
            part_path.unlink(missing_ok=True)
            return True
        except Exception as exc:
            if (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code in PERMANENT_HTTP_STATUS
            ):
                if log_image_failures:
                    log(
                        f"Skip image {output_path.name}: permanent HTTP "
                        f"{exc.response.status_code}"
                    )
                output_path.with_suffix(output_path.suffix + ".part").unlink(missing_ok=True)
                return False

            max_attempts = retries
            if isinstance(exc, requests.exceptions.ConnectTimeout):
                max_attempts = max(1, min(retries, connect_timeout_retries))

            if attempt >= max_attempts:
                if log_image_failures:
                    if isinstance(exc, requests.exceptions.ConnectTimeout):
                        log(f"Skip image {output_path.name}: connect timeout")
                    else:
                        log(f"Failed image {output_path.name}: {exc}")
                return False

            sleep_s = min(retry_sleep * attempt, max_retry_sleep)
            if log_image_failures:
                log(
                    f"[image retry {attempt}/{retries}] {output_path.name}: {exc}; "
                    f"sleep {sleep_s:.1f}s"
                )
            time.sleep(sleep_s)

    return False


def build_candidate(
    row: Dict[str, Any],
    next_img_id: int,
    image_dir: Path,
    max_images_per_doc: int,
) -> Tuple[Optional[Dict[str, Any]], List[Tuple[str, Path]], int]:
    images = row.get("images") or []
    texts = row.get("texts") or []
    n = max(len(images), len(texts))
    segments: List[Dict[str, Any]] = []
    img_ids: List[int] = []
    downloads: List[Tuple[str, Path]] = []

    for i in range(n):
        text = texts[i] if i < len(texts) else None
        if text:
            text = " ".join(str(text).split())
            if text:
                segments.append({"type": "text", "content": text})

        image_url = images[i] if i < len(images) else None
        if image_url and len(img_ids) < max_images_per_doc:
            img_id = next_img_id
            next_img_id += 1
            img_ids.append(img_id)
            downloads.append((str(image_url), image_dir / f"{img_id:012d}.jpg"))
            segments.append({"type": "image", "img_idx": len(img_ids) - 1})

    if not img_ids or not any(s["type"] == "text" for s in segments):
        return None, [], next_img_id

    general = row.get("general_metadata") or {}
    return {
        "doc_id": general.get("id", ""),
        "source": "OpenGVLab/OmniCorpus-CC-210M",
        "source_url": general.get("url", ""),
        "img_ids": img_ids,
        "segments": segments,
    }, downloads, next_img_id


def default_checkpoint_path(output_jsonl: Path) -> Path:
    return output_jsonl.with_suffix(output_jsonl.suffix + ".checkpoint.json")


def save_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    scanned_rows: int,
    written_docs: int,
    next_img_id: int,
    output_jsonl: Path,
) -> None:
    state = {
        "version": CHECKPOINT_VERSION,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parquet_glob": args.parquet_glob,
        "output_jsonl": str(output_jsonl),
        "image_dir": str(args.image_dir),
        "scanned_rows": scanned_rows,
        "written_docs": written_docs,
        "next_img_id": next_img_id,
        "output_bytes": output_jsonl.stat().st_size if output_jsonl.exists() else 0,
        "start_img_id": args.start_img_id,
        "max_images_per_doc": args.max_images_per_doc,
        "max_unsafe_prob": args.max_unsafe_prob,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(checkpoint_path)


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    state = json.loads(checkpoint_path.read_text())
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {state.get('version')}")
    return state


def count_lines_and_last_record(jsonl_path: Path) -> Tuple[int, Optional[Dict[str, Any]]]:
    lines = 0
    last = None
    with jsonl_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            lines += 1
            last = json.loads(line)
    return lines, last


def candidate_matches_jsonl_record(candidate: Dict[str, Any], record: Dict[str, Any]) -> bool:
    if candidate.get("doc_id") != record.get("doc_id"):
        return False
    if candidate.get("source_url") != record.get("source_url"):
        return False
    record_ids = set(int(x) for x in record.get("img_ids", []))
    candidate_ids = set(int(x) for x in candidate.get("img_ids", []))
    return bool(record_ids) and record_ids.issubset(candidate_ids)


def recover_checkpoint_from_jsonl(
    paths: List[Path],
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_jsonl: Path,
) -> Dict[str, Any]:
    written_docs, last_record = count_lines_and_last_record(output_jsonl)
    if written_docs == 0 or last_record is None:
        state = {
            "scanned_rows": 0,
            "written_docs": 0,
            "next_img_id": args.start_img_id,
            "output_bytes": 0,
        }
        save_checkpoint(
            checkpoint_path,
            args,
            state["scanned_rows"],
            state["written_docs"],
            state["next_img_id"],
            output_jsonl,
        )
        return load_checkpoint(checkpoint_path)

    log(f"Recovering checkpoint from existing JSONL tail: {output_jsonl}")
    next_img_id = args.start_img_id
    for row, scanned_rows in tqdm(
        iter_parquet_rows(paths, args.batch_size, 0),
        desc="Recover JSONL position",
    ):
        if args.max_scan_rows > 0 and scanned_rows > args.max_scan_rows:
            break
        if not is_safe(row, args.max_unsafe_prob):
            continue

        candidate, _, next_img_id = build_candidate(
            row, next_img_id, Path(args.image_dir), args.max_images_per_doc
        )
        if candidate is None:
            continue
        if candidate_matches_jsonl_record(candidate, last_record):
            save_checkpoint(
                checkpoint_path,
                args,
                scanned_rows,
                written_docs,
                next_img_id,
                output_jsonl,
            )
            return load_checkpoint(checkpoint_path)

    raise RuntimeError(
        "Could not recover checkpoint from the last JSONL record. "
        "Use --fresh to rebuild the JSONL from the beginning."
    )


def truncate_to_checkpoint(output_jsonl: Path, checkpoint: Dict[str, Any]) -> None:
    output_bytes = int(checkpoint.get("output_bytes", 0))
    if output_jsonl.exists() and output_jsonl.stat().st_size > output_bytes:
        with output_jsonl.open("r+b") as f:
            f.truncate(output_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_glob", default="public/datasets/omnicorpus/raw/data/**/*.parquet")
    parser.add_argument("--output_jsonl", default="public/datasets/omnicorpus/docs/train.jsonl")
    parser.add_argument("--image_dir", default="public/datasets/omnicorpus/images")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="Start a new JSONL/checkpoint run. Existing images are still reused.")
    parser.add_argument("--recover_from_jsonl", action="store_true",
                        help="Create a checkpoint from the tail of an existing JSONL when no checkpoint exists.")
    parser.add_argument("--max_docs", type=int, default=-1)
    parser.add_argument("--max_scan_rows", type=int, default=-1)
    parser.add_argument("--max_images_per_doc", type=int, default=4)
    parser.add_argument("--start_img_id", type=int, default=900000000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--candidate_batch_size", type=int, default=4096)
    parser.add_argument("--download_workers", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--image_retries", type=int, default=5)
    parser.add_argument("--connect_timeout_retries", type=int, default=1,
                        help="Retry count for hosts that cannot be connected. Old web image hosts are often permanently dead.")
    parser.add_argument("--image_retry_sleep", type=float, default=2.0)
    parser.add_argument("--image_max_retry_sleep", type=float, default=60.0)
    parser.add_argument("--min_bytes_per_sec", type=float, default=1024.0,
                        help="Abort and retry image downloads slower than this after --speed_check_after seconds. Use 0 to disable.")
    parser.add_argument("--speed_check_after", type=float, default=10.0)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--max_unsafe_prob", type=float, default=0.5)
    parser.add_argument("--verify_existing_images", action="store_true",
                        help="Verify cached JPEGs with PIL before skipping them. Slower, but stricter.")
    parser.add_argument("--log_image_failures", action="store_true",
                        help="Log per-image failures/retries. Off by default to avoid log bottlenecks at high concurrency.")
    args = parser.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.parquet_glob, recursive=True))]
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {args.parquet_glob}")

    output_jsonl = Path(args.output_jsonl)
    image_dir = Path(args.image_dir)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else default_checkpoint_path(output_jsonl)
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        checkpoint_path.unlink(missing_ok=True)
        output_jsonl.unlink(missing_ok=True)

    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path)
        truncate_to_checkpoint(output_jsonl, checkpoint)
        mode = "a"
        scanned = int(checkpoint["scanned_rows"])
        written = int(checkpoint["written_docs"])
        next_img_id = int(checkpoint["next_img_id"])
        log(f"Resuming from checkpoint: scanned_rows={scanned}, written_docs={written}, next_img_id={next_img_id}")
    elif output_jsonl.exists() and output_jsonl.stat().st_size > 0:
        if not args.recover_from_jsonl:
            raise FileExistsError(
                f"{output_jsonl} exists but {checkpoint_path} does not. "
                "Pass --recover_from_jsonl to continue from the JSONL tail, "
                "or --fresh to rebuild the JSONL from the beginning."
            )
        checkpoint = recover_checkpoint_from_jsonl(paths, args, checkpoint_path, output_jsonl)
        mode = "a"
        scanned = int(checkpoint["scanned_rows"])
        written = int(checkpoint["written_docs"])
        next_img_id = int(checkpoint["next_img_id"])
        log(f"Recovered checkpoint: scanned_rows={scanned}, written_docs={written}, next_img_id={next_img_id}")
    else:
        mode = "w"
        scanned = 0
        written = 0
        next_img_id = args.start_img_id

    checkpoint_scanned = scanned
    checkpoint_next_img_id = next_img_id
    pending: List[PendingEntry] = []
    download_executor = futures.ThreadPoolExecutor(max_workers=args.download_workers)

    def flush_pending(out_file, progress_bar) -> Tuple[int, int, int]:
        if not pending:
            return 0, checkpoint_scanned, checkpoint_next_img_id

        path_to_ok: Dict[Path, bool] = {}
        futs = {}
        for _, downloads, _, _ in pending:
            for url, path in downloads:
                if existing_image_ok(path, args.verify_existing_images):
                    path_to_ok[path] = True
                    continue
                futs[download_executor.submit(
                    download_image,
                    url,
                    path,
                    args.timeout,
                    args.image_retries,
                    args.connect_timeout_retries,
                    args.image_retry_sleep,
                    args.image_max_retry_sleep,
                    args.min_bytes_per_sec,
                    args.speed_check_after,
                    args.chunk_size,
                    args.verify_existing_images,
                    args.log_image_failures,
                )] = path

        for fut in futures.as_completed(futs):
            path_to_ok[futs[fut]] = bool(fut.result())

        n_written = 0
        committed_scanned = checkpoint_scanned
        committed_next_img_id = checkpoint_next_img_id
        for record, _, candidate_scanned, candidate_next_img_id in pending:
            if args.max_docs > 0 and written + n_written >= args.max_docs:
                break
            kept_ids = []
            old_to_new = {}
            for old_idx, img_id in enumerate(record["img_ids"]):
                path = image_dir / f"{img_id:012d}.jpg"
                ok = (
                    path_to_ok[path]
                    if path in path_to_ok
                    else existing_image_ok(path, args.verify_existing_images)
                )
                if ok:
                    old_to_new[old_idx] = len(kept_ids)
                    kept_ids.append(img_id)

            committed_scanned = candidate_scanned
            committed_next_img_id = candidate_next_img_id

            if not kept_ids:
                continue

            new_segments = []
            for seg in record["segments"]:
                if seg["type"] == "text":
                    new_segments.append(seg)
                else:
                    new_idx = old_to_new.get(seg["img_idx"])
                    if new_idx is not None:
                        new_segments.append({"type": "image", "img_idx": new_idx})

            record["img_ids"] = kept_ids
            record["segments"] = new_segments
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            progress_bar.update(1)

        pending.clear()
        return n_written, committed_scanned, committed_next_img_id

    total = None if args.max_docs < 0 else args.max_docs
    try:
        with output_jsonl.open(mode) as out:
            progress = tqdm(total=total, initial=written if total else 0, desc="OmniCorpus docs")
            for row, scanned_after in iter_parquet_rows(paths, args.batch_size, scanned):
                scanned = scanned_after
                if args.max_scan_rows > 0 and scanned > args.max_scan_rows:
                    break
                if args.max_docs > 0 and written >= args.max_docs:
                    break
                if not is_safe(row, args.max_unsafe_prob):
                    checkpoint_scanned = scanned
                    continue

                record, downloads, next_img_id = build_candidate(
                    row, next_img_id, image_dir, args.max_images_per_doc
                )
                if record is None:
                    checkpoint_scanned = scanned
                    checkpoint_next_img_id = next_img_id
                    continue

                pending.append((record, downloads, scanned, next_img_id))
                if len(pending) >= args.candidate_batch_size:
                    n_written, checkpoint_scanned, checkpoint_next_img_id = flush_pending(out, progress)
                    written += n_written
                    out.flush()
                    os.fsync(out.fileno())
                    save_checkpoint(
                        checkpoint_path,
                        args,
                        checkpoint_scanned,
                        written,
                        checkpoint_next_img_id,
                        output_jsonl,
                    )
                    if args.max_docs > 0 and written >= args.max_docs:
                        break

            if args.max_docs < 0 or written < args.max_docs:
                n_written, checkpoint_scanned, checkpoint_next_img_id = flush_pending(out, progress)
                written += n_written
                out.flush()
                os.fsync(out.fileno())
                save_checkpoint(
                    checkpoint_path,
                    args,
                    checkpoint_scanned,
                    written,
                    checkpoint_next_img_id,
                    output_jsonl,
                )
            progress.close()
    finally:
        download_executor.shutdown(wait=True)

    print(f"Scanned rows: {checkpoint_scanned}")
    print(f"Written docs: {written}")
    print(f"Next image id: {checkpoint_next_img_id}")
    print(f"JSONL: {output_jsonl}")
    print(f"Images: {image_dir}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
