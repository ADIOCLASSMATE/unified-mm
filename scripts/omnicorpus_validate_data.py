"""
Validate OmniCorpus JSONL/image/token snapshots.

Fast default:
    uv run python scripts/omnicorpus_validate_data.py \
        --docs_jsonl public/datasets/omnicorpus/docs/snapshots/train_20260623_064740_2946703docs.jsonl

Strict scan:
    uv run python scripts/omnicorpus_validate_data.py --full --verify-images

The default mode checks small head/tail samples and uses the checkpoint, when
present, for global document/image-id counts. Use --full when you need exact
JSONL-wide missing-file statistics. Use --count-dir-files only when you want
directory-level file totals; counting millions of files can be slow.
"""

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    return f"{value:,}"


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{value:.4f}"


def checkpoint_path_for(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(jsonl_path.suffix + ".checkpoint.json")


def load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def infer_docs_from_name(path: Path) -> Optional[int]:
    match = re.search(r"_(\d+)docs\.jsonl$", path.name)
    if match:
        return int(match.group(1))
    return None


def tail_lines(path: Path, n: int, chunk_size: int = 1024 * 1024) -> List[str]:
    if n <= 0:
        return []

    chunks: List[bytes] = []
    lines_found = 0
    with path.open("rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        while pos > 0 and lines_found <= n:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            chunks.append(chunk)
            lines_found += chunk.count(b"\n")

    data = b"".join(reversed(chunks))
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def iter_head(path: Path, n: int) -> Iterable[Tuple[int, str]]:
    with path.open() as f:
        for idx, line in enumerate(f, start=1):
            if idx > n:
                break
            yield idx, line


def iter_tail(path: Path, n: int) -> Iterable[Tuple[int, str]]:
    lines = tail_lines(path, n)
    start = -len(lines) + 1
    for offset, line in enumerate(lines):
        yield start + offset, line


def iter_all(path: Path) -> Iterable[Tuple[int, str]]:
    with path.open() as f:
        for idx, line in enumerate(f, start=1):
            yield idx, line


class ValidationStats:
    def __init__(self) -> None:
        self.docs = 0
        self.bad_json = 0
        self.schema_errors = 0
        self.docs_without_text = 0
        self.docs_without_images = 0
        self.segments = 0
        self.text_segments = 0
        self.image_segments = 0
        self.text_chars = 0
        self.image_refs = 0
        self.unique_image_ids = set()
        self.image_count_dist = Counter()
        self.min_img_id: Optional[int] = None
        self.max_img_id: Optional[int] = None
        self.missing_images = 0
        self.present_images = 0
        self.invalid_images = 0
        self.missing_tokens = 0
        self.present_tokens = 0
        self.bad_tokens = 0
        self.example_errors: List[str] = []

    def add_error(self, message: str) -> None:
        self.schema_errors += 1
        if len(self.example_errors) < 12:
            self.example_errors.append(message)

    def merge(self, other: "ValidationStats") -> None:
        self.docs += other.docs
        self.bad_json += other.bad_json
        self.schema_errors += other.schema_errors
        self.docs_without_text += other.docs_without_text
        self.docs_without_images += other.docs_without_images
        self.segments += other.segments
        self.text_segments += other.text_segments
        self.image_segments += other.image_segments
        self.text_chars += other.text_chars
        self.image_refs += other.image_refs
        self.unique_image_ids.update(other.unique_image_ids)
        self.image_count_dist.update(other.image_count_dist)
        if other.min_img_id is not None:
            self.min_img_id = other.min_img_id if self.min_img_id is None else min(self.min_img_id, other.min_img_id)
        if other.max_img_id is not None:
            self.max_img_id = other.max_img_id if self.max_img_id is None else max(self.max_img_id, other.max_img_id)
        self.missing_images += other.missing_images
        self.present_images += other.present_images
        self.invalid_images += other.invalid_images
        self.missing_tokens += other.missing_tokens
        self.present_tokens += other.present_tokens
        self.bad_tokens += other.bad_tokens
        for error in other.example_errors:
            if len(self.example_errors) < 12:
                self.example_errors.append(error)


def validate_record(
    record: Dict[str, Any],
    context: str,
    stats: ValidationStats,
    image_dir: Optional[Path],
    token_dir: Optional[Path],
    verify_images: bool,
    verify_tokens: bool,
    max_verify_images: int,
    max_verify_tokens: int,
) -> None:
    doc_id = record.get("doc_id", "")
    img_ids_raw = record.get("img_ids")
    segments = record.get("segments")

    if not isinstance(img_ids_raw, list):
        stats.add_error(f"{context}: img_ids is not a list")
        img_ids_raw = []
    if not isinstance(segments, list):
        stats.add_error(f"{context}: segments is not a list")
        segments = []

    img_ids: List[int] = []
    for i, value in enumerate(img_ids_raw):
        try:
            img_ids.append(int(value))
        except Exception:
            stats.add_error(f"{context}: img_ids[{i}] is not an int: {value!r}")

    text_seen = False
    image_seen = False
    image_indices = set()
    text_chars = 0

    for seg_idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            stats.add_error(f"{context}: segment {seg_idx} is not an object")
            continue

        seg_type = seg.get("type")
        stats.segments += 1
        if seg_type == "text":
            content = seg.get("content")
            if not isinstance(content, str) or not content.strip():
                stats.add_error(f"{context}: empty/non-string text segment {seg_idx}")
                continue
            text_seen = True
            stats.text_segments += 1
            text_chars += len(content)
        elif seg_type == "image":
            img_idx = seg.get("img_idx")
            if not isinstance(img_idx, int) or img_idx < 0 or img_idx >= len(img_ids):
                stats.add_error(f"{context}: bad image img_idx={img_idx!r} in segment {seg_idx}")
                continue
            image_seen = True
            image_indices.add(img_idx)
            stats.image_segments += 1
        else:
            stats.add_error(f"{context}: unknown segment type {seg_type!r}")

    if not text_seen:
        stats.docs_without_text += 1
        stats.add_error(f"{context}: no valid text segment, doc_id={doc_id!r}")
    if not img_ids or not image_seen:
        stats.docs_without_images += 1
        stats.add_error(f"{context}: no valid image segment, doc_id={doc_id!r}")

    for img_idx in range(len(img_ids)):
        if img_idx not in image_indices:
            stats.add_error(f"{context}: img_ids[{img_idx}] is not referenced by any image segment")

    stats.docs += 1
    stats.text_chars += text_chars
    stats.image_refs += len(img_ids)
    stats.image_count_dist[len(img_ids)] += 1

    for img_id in img_ids:
        stats.unique_image_ids.add(img_id)
        stats.min_img_id = img_id if stats.min_img_id is None else min(stats.min_img_id, img_id)
        stats.max_img_id = img_id if stats.max_img_id is None else max(stats.max_img_id, img_id)

        if image_dir is not None:
            image_path = image_dir / f"{img_id:012d}.jpg"
            if image_path.exists() and image_path.stat().st_size > 0:
                stats.present_images += 1
                if verify_images and stats.present_images <= max_verify_images:
                    try:
                        from PIL import Image

                        with Image.open(image_path) as image:
                            image.verify()
                    except Exception as exc:
                        stats.invalid_images += 1
                        stats.add_error(f"{context}: invalid image {image_path}: {exc}")
            else:
                stats.missing_images += 1
                stats.add_error(f"{context}: missing image {image_path}")

        if token_dir is not None:
            token_path = token_dir / f"{img_id:012d}.pt"
            if token_path.exists() and token_path.stat().st_size > 0:
                stats.present_tokens += 1
                if verify_tokens and stats.present_tokens <= max_verify_tokens:
                    try:
                        import torch

                        tokens = torch.load(token_path, map_location="cpu").long().view(-1)
                        if tokens.numel() != 256:
                            stats.bad_tokens += 1
                            stats.add_error(f"{context}: bad token length {token_path}: {tokens.numel()}")
                    except Exception as exc:
                        stats.bad_tokens += 1
                        stats.add_error(f"{context}: unreadable token {token_path}: {exc}")
            else:
                stats.missing_tokens += 1


def validate_lines(
    lines: Iterable[Tuple[int, str]],
    label: str,
    image_dir: Optional[Path],
    token_dir: Optional[Path],
    verify_images: bool,
    verify_tokens: bool,
    max_verify_images: int,
    max_verify_tokens: int,
    progress_every: int,
) -> ValidationStats:
    stats = ValidationStats()
    recent = deque(maxlen=1)

    for idx, line in lines:
        if not line.strip():
            continue
        recent.append(idx)
        try:
            record = json.loads(line)
        except Exception as exc:
            stats.bad_json += 1
            stats.add_error(f"{label}:{idx}: bad JSON: {exc}")
            continue

        if not isinstance(record, dict):
            stats.add_error(f"{label}:{idx}: JSON value is not an object")
            continue

        validate_record(
            record=record,
            context=f"{label}:{idx}",
            stats=stats,
            image_dir=image_dir,
            token_dir=token_dir,
            verify_images=verify_images,
            verify_tokens=verify_tokens,
            max_verify_images=max_verify_images,
            max_verify_tokens=max_verify_tokens,
        )

        if progress_every > 0 and stats.docs > 0 and stats.docs % progress_every == 0:
            print(f"[{label}] checked {format_int(stats.docs)} docs; last line={recent[-1]}", flush=True)

    return stats


def count_files(path: Path, pattern: str) -> Optional[int]:
    if not path.exists():
        return None
    return sum(1 for _ in path.glob(pattern))


def print_stats(title: str, stats: ValidationStats) -> None:
    missing_ratio = None
    image_checked = stats.present_images + stats.missing_images
    if image_checked:
        missing_ratio = stats.missing_images / image_checked

    token_missing_ratio = None
    token_checked = stats.present_tokens + stats.missing_tokens
    if token_checked:
        token_missing_ratio = stats.missing_tokens / token_checked

    avg_images = stats.image_refs / stats.docs if stats.docs else 0.0
    avg_text_chars = stats.text_chars / stats.docs if stats.docs else 0.0

    print(f"\n[{title}]")
    print(f"docs_checked: {format_int(stats.docs)}")
    print(f"bad_json: {format_int(stats.bad_json)}")
    print(f"schema_errors: {format_int(stats.schema_errors)}")
    print(f"docs_without_text: {format_int(stats.docs_without_text)}")
    print(f"docs_without_images: {format_int(stats.docs_without_images)}")
    print(f"image_refs: {format_int(stats.image_refs)}")
    print(f"unique_image_ids_in_checked_docs: {format_int(len(stats.unique_image_ids))}")
    print(f"avg_images_per_doc: {avg_images:.4f}")
    print(f"avg_text_chars_per_doc: {avg_text_chars:.1f}")
    print(f"image_count_dist: {dict(sorted(stats.image_count_dist.items()))}")
    print(f"img_id_range: {stats.min_img_id}..{stats.max_img_id}")
    print(f"present_images_checked: {format_int(stats.present_images)}")
    print(f"missing_images_checked: {format_int(stats.missing_images)}")
    print(f"missing_image_ratio_checked: {format_float(missing_ratio)}")
    print(f"invalid_images_checked: {format_int(stats.invalid_images)}")
    if token_checked:
        print(f"present_tokens_checked: {format_int(stats.present_tokens)}")
        print(f"missing_tokens_checked: {format_int(stats.missing_tokens)}")
        print(f"missing_token_ratio_checked: {format_float(token_missing_ratio)}")
        print(f"bad_tokens_checked: {format_int(stats.bad_tokens)}")
    if stats.example_errors:
        print("example_errors:")
        for error in stats.example_errors:
            print(f"  - {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_jsonl", default="public/datasets/omnicorpus/docs/train.jsonl")
    parser.add_argument("--image_dir", default="public/datasets/omnicorpus/images")
    parser.add_argument("--token_dir", default=None,
                        help="Optional image token directory, e.g. public/datasets/omnicorpus/image_tokens_magvit2")
    parser.add_argument("--checkpoint", default=None,
                        help="Defaults to <docs_jsonl>.checkpoint.json when it exists.")
    parser.add_argument("--sample_docs", type=int, default=2000,
                        help="Head and tail docs to check in fast mode.")
    parser.add_argument("--full", action="store_true",
                        help="Scan every JSONL row. This is slower but gives exact missing-image counts.")
    parser.add_argument("--no-image-files", action="store_true",
                        help="Only validate JSONL references; do not stat image files.")
    parser.add_argument("--count-dir-files", action="store_true",
                        help="Count all *.jpg/*.part/*.pt files in data directories. Slow for millions of files.")
    parser.add_argument("--verify-images", action="store_true",
                        help="Open sampled/found images with PIL Image.verify().")
    parser.add_argument("--max-verify-images", type=int, default=1000)
    parser.add_argument("--verify-tokens", action="store_true",
                        help="Load sampled/found .pt files and check that they contain 256 tokens.")
    parser.add_argument("--max-verify-tokens", type=int, default=1000)
    parser.add_argument("--min-docs", type=int, default=100000,
                        help="Minimum global docs expected for early validation.")
    parser.add_argument("--min-images", type=int, default=100000,
                        help="Minimum global referenced images expected for early validation.")
    parser.add_argument("--max-missing-image-ratio", type=float, default=0.001)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    docs_jsonl = Path(args.docs_jsonl)
    if not docs_jsonl.exists():
        raise FileNotFoundError(docs_jsonl)

    image_dir = None if args.no_image_files else Path(args.image_dir)
    if image_dir is not None and not image_dir.exists():
        raise FileNotFoundError(image_dir)

    token_dir = Path(args.token_dir) if args.token_dir else None
    if token_dir is not None and not token_dir.exists():
        raise FileNotFoundError(token_dir)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_path_for(docs_jsonl)
    checkpoint = load_checkpoint(checkpoint_path)

    global_docs = None
    global_images = None
    if checkpoint:
        global_docs = int(checkpoint.get("written_docs", 0))
        start_img_id = checkpoint.get("start_img_id")
        next_img_id = checkpoint.get("next_img_id")
        if start_img_id is not None and next_img_id is not None:
            global_images = int(next_img_id) - int(start_img_id)
    else:
        global_docs = infer_docs_from_name(docs_jsonl)

    jpg_count = count_files(image_dir, "*.jpg") if image_dir is not None and args.count_dir_files else None
    part_count = count_files(image_dir, "*.part") if image_dir is not None and args.count_dir_files else None
    token_count = count_files(token_dir, "*.pt") if token_dir is not None and args.count_dir_files else None

    print("OmniCorpus data validation")
    print(f"docs_jsonl: {docs_jsonl}")
    print(f"jsonl_size_bytes: {format_int(docs_jsonl.stat().st_size)}")
    print(f"checkpoint: {checkpoint_path if checkpoint else 'not found'}")
    print(f"global_docs_from_checkpoint_or_name: {format_int(global_docs)}")
    print(f"global_allocated_image_ids_from_checkpoint: {format_int(global_images)}")
    if image_dir is not None:
        print(f"image_dir: {image_dir}")
        print(f"jpg_files: {format_int(jpg_count)}")
        print(f"part_files: {format_int(part_count)}")
        if not args.count_dir_files:
            print("directory_file_counts: skipped (pass --count-dir-files to enable)")
    if token_dir is not None:
        print(f"token_dir: {token_dir}")
        print(f"token_files: {format_int(token_count)}")
        if not args.count_dir_files:
            print("token_file_count: skipped (pass --count-dir-files to enable)")

    if args.full:
        stats = validate_lines(
            iter_all(docs_jsonl),
            "full",
            image_dir,
            token_dir,
            args.verify_images,
            args.verify_tokens,
            args.max_verify_images,
            args.max_verify_tokens,
            args.progress_every,
        )
        print_stats("full scan", stats)
        total_stats = stats
        global_docs = stats.docs
        global_images = stats.image_refs
    else:
        head_stats = validate_lines(
            iter_head(docs_jsonl, args.sample_docs),
            "head",
            image_dir,
            token_dir,
            args.verify_images,
            args.verify_tokens,
            args.max_verify_images,
            args.max_verify_tokens,
            0,
        )
        tail_stats = validate_lines(
            iter_tail(docs_jsonl, args.sample_docs),
            "tail",
            image_dir,
            token_dir,
            args.verify_images,
            args.verify_tokens,
            args.max_verify_images,
            args.max_verify_tokens,
            0,
        )
        print_stats("head sample", head_stats)
        print_stats("tail sample", tail_stats)
        total_stats = ValidationStats()
        total_stats.merge(head_stats)
        total_stats.merge(tail_stats)

    checked_image_refs = total_stats.present_images + total_stats.missing_images
    missing_ratio = total_stats.missing_images / checked_image_refs if checked_image_refs else 0.0
    fatal_errors = (
        total_stats.bad_json
        + total_stats.schema_errors
        + total_stats.invalid_images
        + total_stats.bad_tokens
    )

    enough_docs = global_docs is not None and global_docs >= args.min_docs
    enough_images = global_images is not None and global_images >= args.min_images
    if global_images is None and jpg_count is not None:
        enough_images = jpg_count >= args.min_images

    pass_early = (
        enough_docs
        and enough_images
        and fatal_errors == 0
        and missing_ratio <= args.max_missing_image_ratio
    )

    print("\n[decision]")
    print(f"enough_docs: {enough_docs} ({format_int(global_docs)} >= {format_int(args.min_docs)})")
    print(f"enough_images: {enough_images} ({format_int(global_images or jpg_count)} >= {format_int(args.min_images)})")
    print(f"fatal_errors_in_checked_rows: {format_int(fatal_errors)}")
    print(f"missing_image_ratio_in_checked_rows: {missing_ratio:.6f}")
    print(f"early_validation_ready: {'YES' if pass_early else 'NO'}")

    if not pass_early:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
