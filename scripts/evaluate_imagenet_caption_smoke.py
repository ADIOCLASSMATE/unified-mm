#!/usr/bin/env python3
"""Evaluate a small ImageNet multi-caption distillation smoke run.

The script keeps preparation, CLIP evaluation, and aggregation as separate
observable stages:

1. ``prepare`` joins the response SQLite database to the original captions.
2. ``clip`` computes image/text and original/distilled CLIP similarities.
3. ``summarize`` writes aggregate JSON and a compact Markdown report.

Hallucination judging is intentionally excluded from the operational smoke
workflow because it adds paid model calls and does not belong in the efficient
CPU-side synthesis path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Sequence

if __package__:
    from scripts.distill_imagenet_captions import canonical_json, word_tokens
else:
    from distill_imagenet_captions import canonical_json, word_tokens


DEFAULT_ORIGINALS = Path(
    "public/datasets/imagenet_ablation_100c_balanced/captions/"
    "imagenet100_recaption_short_join.jsonl"
)
DEFAULT_SYNSET_MAPPING = Path("/inspire/dataset/imagenet/v1/LOC_synset_mapping.txt")
VOCAB_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - depends on the runtime environment
    _tqdm = None


class TextProgress:
    """Dependency-free terminal progress bar used when tqdm is unavailable."""

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        *,
        total: int | None = None,
        desc: str = "Progress",
        unit: str = "item",
        disable: bool = False,
        **_: Any,
    ) -> None:
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)  # type: ignore[arg-type]
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.completed = 0
        self.closed = False
        self.last_rendered_at = 0.0
        self.last_rendered_completed = -1
        self._render(force=True)

    def __enter__(self) -> TextProgress:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __iter__(self) -> Any:
        if self.iterable is None:
            raise TypeError("progress iterable is not set")
        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def update(self, amount: int = 1) -> None:
        self.completed += amount
        self._render(force=self.completed >= self.total)

    def _render(self, *, force: bool) -> None:
        if self.disable:
            return
        now = monotonic()
        if force and self.completed == self.last_rendered_completed:
            return
        if not force and now - self.last_rendered_at < 0.25:
            return
        self.last_rendered_at = now
        self.last_rendered_completed = self.completed
        fraction = min(1.0, self.completed / self.total) if self.total else 1.0
        width = 24
        filled = round(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r{self.desc}: {fraction:6.1%}|{bar}| "
            f"{self.completed}/{self.total} {self.unit}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        if self.closed:
            return
        self._render(force=True)
        if not self.disable:
            print(file=sys.stderr, flush=True)
        self.closed = True


class RuntimeLog:
    """Append-only JSONL log that is flushed after every progress event."""

    def __init__(self, path: Path, command: str) -> None:
        self.path = path
        self.command = command
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + f"-pid{os.getpid()}"
        )
        self.started_at = monotonic()
        self._handle: Any | None = None

    def __enter__(self) -> RuntimeLog:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.emit("run_started", pid=os.getpid())
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def emit(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            raise RuntimeError("runtime log is not open")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "command": self.command,
            "event": event,
            "elapsed_seconds": round(monotonic() - self.started_at, 3),
            **fields,
        }
        self._handle.write(canonical_json(row) + "\n")
        self._handle.flush()


def default_runtime_log_path(args: argparse.Namespace) -> Path:
    if args.log_file:
        return Path(args.log_file)
    if args.command in {"prepare", "clip"}:
        output = Path(args.output)
        return output.with_suffix(output.suffix + ".runtime.jsonl")
    return Path(args.output_dir) / "summarize.runtime.jsonl"


def log_progress(
    args: argparse.Namespace,
    stage: str,
    completed: int,
    total: int,
    **fields: Any,
) -> None:
    every = int(args.log_every)
    if completed == 1 or completed == total or completed % every == 0:
        args.runtime_log.emit(
            "progress",
            stage=stage,
            completed=completed,
            total=total,
            **fields,
        )


def progress_bar(
    args: argparse.Namespace,
    iterable: Iterable[Any] | None = None,
    **kwargs: Any,
) -> Any:
    progress_type = _tqdm or TextProgress
    return progress_type(
        iterable,
        dynamic_ncols=True,
        disable=args.no_progress,
        **kwargs,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_synset_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            synset, _, name = line.strip().partition(" ")
            if synset:
                names[synset] = name
    return names


def run_prepare(args: argparse.Namespace) -> int:
    response_db = Path(args.response_db)
    originals_path = Path(args.original_captions)
    output_path = Path(args.output)

    connection = sqlite3.connect(response_db)
    try:
        records = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT record_json FROM responses ORDER BY img_id, model"
            )
        ]
        errors = connection.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
    finally:
        connection.close()
    if errors:
        raise ValueError(f"response database still contains {errors} errors")
    if not records:
        raise ValueError("response database contains no successful responses")
    args.runtime_log.emit(
        "inputs_loaded",
        response_db=str(response_db),
        response_records=len(records),
    )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for completed, record in enumerate(
        progress_bar(
            args,
            records,
            desc="Grouping responses",
            unit="record",
        ),
        start=1,
    ):
        grouped[int(record["img_id"])].append(record)
        log_progress(
            args,
            "group_responses",
            completed,
            len(records),
            img_id=int(record["img_id"]),
        )
    selected_ids = set(grouped)

    originals: dict[int, dict[str, Any]] = {}
    with (
        originals_path.open(encoding="utf-8") as handle,
        progress_bar(
            args,
            total=len(selected_ids),
            desc="Matching originals",
            unit="image",
        ) as progress,
    ):
        for line in handle:
            row = json.loads(line)
            img_id = int(row["img_id"])
            if img_id in selected_ids and img_id not in originals:
                originals[img_id] = row
                progress.update(1)
                log_progress(
                    args,
                    "match_originals",
                    len(originals),
                    len(selected_ids),
                    img_id=img_id,
                )
                if len(originals) == len(selected_ids):
                    break
    missing = selected_ids.difference(originals)
    if missing:
        raise ValueError(f"missing original captions for img_ids={sorted(missing)}")

    synset_names = load_synset_names(Path(args.synset_mapping))
    output_rows: list[dict[str, Any]] = []
    sorted_img_ids = sorted(grouped)
    for completed, img_id in enumerate(
        progress_bar(
            args,
            sorted_img_ids,
            desc="Preparing samples",
            unit="image",
        ),
        start=1,
    ):
        response_records = grouped[img_id]
        identity = response_records[0]
        original = originals[img_id]
        for record in response_records[1:]:
            for key in ("manifest_index", "img_id", "id", "path", "synset"):
                if record[key] != identity[key]:
                    raise ValueError(
                        f"identity disagreement for img_id={img_id}: {key}"
                    )
            if record["image_sha256"] != identity["image_sha256"]:
                raise ValueError(f"image hash disagreement for img_id={img_id}")
        for key in ("img_id", "id", "path", "synset"):
            if original[key] != identity[key]:
                raise ValueError(
                    f"original identity mismatch for img_id={img_id}: {key}"
                )

        captions = [
            {
                "caption_key": "original",
                "source": "original",
                "caption_index": 0,
                "text": original["recaption_short"],
                "word_count": len(word_tokens(original["recaption_short"])),
            }
        ]
        for record in sorted(response_records, key=lambda item: item["model"]):
            for caption in record["captions"]:
                captions.append(
                    {
                        "caption_key": (
                            f"{record['model']}:{int(caption['caption_index'])}"
                        ),
                        "source": record["model"],
                        "caption_index": int(caption["caption_index"]),
                        "text": caption["text"],
                        "word_count": int(caption["word_count"]),
                    }
                )
        output_rows.append(
            {
                "schema": "imagenet_caption_smoke_sample_v1",
                "manifest_index": int(identity["manifest_index"]),
                "img_id": img_id,
                "id": identity["id"],
                "path": identity["path"],
                "source_path": identity["source_path"],
                "synset": identity["synset"],
                "class_name": synset_names.get(identity["synset"], ""),
                "image_sha256": identity["image_sha256"],
                "captions": captions,
            }
        )
        log_progress(
            args,
            "prepare_samples",
            completed,
            len(sorted_img_ids),
            img_id=img_id,
            captions=len(captions),
        )
    atomic_write_jsonl(output_path, output_rows)
    args.runtime_log.emit(
        "artifact_written",
        path=str(output_path),
        images=len(output_rows),
        captions=sum(len(row["captions"]) for row in output_rows),
    )
    print(
        canonical_json(
            {
                "output": str(output_path),
                "images": len(output_rows),
                "captions": sum(len(row["captions"]) for row in output_rows),
                "runtime_log": str(args.runtime_log.path),
            }
        )
    )
    return 0


def _feature_tensor(value: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("torch is required for CLIP evaluation") from exc
    if torch.is_tensor(value):
        return value
    if getattr(value, "pooler_output", None) is not None:
        return value.pooler_output
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    raise TypeError(f"cannot extract feature tensor from {type(value)}")


def run_clip(args: argparse.Namespace) -> int:
    try:
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("torch, Pillow, and transformers are required") from exc

    samples = read_jsonl(Path(args.samples))
    if not samples:
        raise ValueError("samples file is empty")
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    args.runtime_log.emit(
        "model_load_started",
        model_dir=str(args.model_dir),
        device=str(device),
        images=len(samples),
    )
    model = CLIPModel.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer = processor.tokenizer
    max_length = int(tokenizer.model_max_length)
    args.runtime_log.emit(
        "model_load_completed",
        model_dir=str(args.model_dir),
        device=str(device),
        max_text_tokens=max_length,
    )

    score_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for completed, sample in enumerate(
            progress_bar(
                args,
                samples,
                desc="CLIP evaluation",
                unit="image",
            ),
            start=1,
        ):
            captions = sample["captions"]
            texts = [item["text"] for item in captions]
            raw_token_lengths = [
                len(tokenizer(text, truncation=False)["input_ids"]) for text in texts
            ]
            text_inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            with Image.open(sample["source_path"]) as source_image:
                image = source_image.convert("RGB")
                image_inputs = processor(images=image, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(device=device, dtype=dtype)

            context = (
                torch.autocast(device_type="cuda", dtype=dtype)
                if device.type == "cuda"
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with context:
                image_features = _feature_tensor(
                    model.get_image_features(pixel_values=pixel_values)
                )
                text_features = _feature_tensor(model.get_text_features(**text_inputs))
            image_features = functional.normalize(image_features.float(), dim=-1)
            text_features = functional.normalize(text_features.float(), dim=-1)
            image_scores = (text_features @ image_features.T).squeeze(1).cpu().tolist()
            original_feature = text_features[0:1]
            original_text_scores = (
                (text_features @ original_feature.T).squeeze(1).cpu().tolist()
            )

            for index, caption in enumerate(captions):
                score_rows.append(
                    {
                        "schema": "imagenet_caption_clip_score_v1",
                        "img_id": sample["img_id"],
                        "id": sample["id"],
                        "synset": sample["synset"],
                        "caption_key": caption["caption_key"],
                        "source": caption["source"],
                        "caption_index": caption["caption_index"],
                        "word_count": caption["word_count"],
                        "clip_image_text_cosine": float(image_scores[index]),
                        "clip_text_cosine_to_original": float(
                            original_text_scores[index]
                        ),
                        "clip_token_count": raw_token_lengths[index],
                        "clip_max_tokens": max_length,
                        "clip_truncated": raw_token_lengths[index] > max_length,
                        "clip_model": str(args.model_dir),
                    }
                )
            log_progress(
                args,
                "clip_images",
                completed,
                len(samples),
                img_id=int(sample["img_id"]),
                scores_written=len(score_rows),
            )
    output_path = Path(args.output)
    atomic_write_jsonl(output_path, score_rows)
    args.runtime_log.emit(
        "artifact_written",
        path=str(output_path),
        images=len(samples),
        scores=len(score_rows),
    )
    print(
        canonical_json(
            {
                "output": str(output_path),
                "images": len(samples),
                "scores": len(score_rows),
                "clip_model": str(args.model_dir),
                "device": str(device),
                "runtime_log": str(args.runtime_log.path),
            }
        )
    )
    return 0


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.9),
        "max": max(values),
    }


def vocabulary(texts: Iterable[str]) -> set[str]:
    return {token for text in texts for token in VOCAB_RE.findall(text.lower())}


def run_summarize(args: argparse.Namespace) -> int:
    samples = read_jsonl(Path(args.samples))
    clip_rows = read_jsonl(Path(args.clip_scores))
    output_dir = Path(args.output_dir)

    captions = [caption for sample in samples for caption in sample["captions"]]
    sources = sorted({caption["source"] for caption in captions})
    caption_by_source: dict[str, list[dict[str, Any]]] = {
        source: [item for item in captions if item["source"] == source]
        for source in sources
    }
    original_vocab = vocabulary(item["text"] for item in caption_by_source["original"])
    args.runtime_log.emit(
        "inputs_loaded",
        images=len(samples),
        captions=len(captions),
        clip_scores=len(clip_rows),
        sources=sources,
    )

    lengths: dict[str, Any] = {}
    vocab_stats: dict[str, Any] = {}
    source_items = list(caption_by_source.items())
    for completed, (source, items) in enumerate(
        progress_bar(
            args,
            source_items,
            desc="Caption statistics",
            unit="source",
        ),
        start=1,
    ):
        values = [float(item["word_count"]) for item in items]
        lengths[source] = {
            **distribution(values),
            "below_32_rate": sum(value < 32 for value in values) / len(values),
            "within_32_60_rate": sum(32 <= value <= 60 for value in values)
            / len(values),
            "above_60_rate": sum(value > 60 for value in values) / len(values),
        }
        source_vocab = vocabulary(item["text"] for item in items)
        novel = source_vocab.difference(original_vocab)
        vocab_stats[source] = {
            "unique_types": len(source_vocab),
            "novel_types_vs_original": len(novel),
            "combined_types_with_original": len(source_vocab | original_vocab),
            "combined_increase_vs_original": (
                len(source_vocab | original_vocab) / len(original_vocab) - 1.0
            ),
            "token_count": sum(
                len(VOCAB_RE.findall(item["text"].lower())) for item in items
            ),
        }
        log_progress(
            args,
            "caption_statistics",
            completed,
            len(source_items),
            source=source,
            captions=len(items),
        )
    distilled_vocab = vocabulary(
        item["text"] for item in captions if item["source"] != "original"
    )
    vocab_stats["all_distilled"] = {
        "unique_types": len(distilled_vocab),
        "novel_types_vs_original": len(distilled_vocab - original_vocab),
        "combined_types_with_original": len(distilled_vocab | original_vocab),
        "combined_increase_vs_original": (
            len(distilled_vocab | original_vocab) / len(original_vocab) - 1.0
        ),
    }

    original_clip = {
        int(row["img_id"]): float(row["clip_image_text_cosine"])
        for row in clip_rows
        if row["source"] == "original"
    }
    clip_by_source: dict[str, Any] = {}
    for completed, source in enumerate(
        progress_bar(
            args,
            sources,
            desc="CLIP statistics",
            unit="source",
        ),
        start=1,
    ):
        rows = [row for row in clip_rows if row["source"] == source]
        scores = [float(row["clip_image_text_cosine"]) for row in rows]
        text_scores = [float(row["clip_text_cosine_to_original"]) for row in rows]
        deltas = [
            float(row["clip_image_text_cosine"]) - original_clip[int(row["img_id"])]
            for row in rows
        ]
        grouped_scores: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            grouped_scores[int(row["img_id"])].append(
                float(row["clip_image_text_cosine"])
            )
        clip_by_source[source] = {
            "image_text_cosine": distribution(scores),
            "text_cosine_to_original": distribution(text_scores),
            "mean_delta_vs_original_image_text": statistics.fmean(deltas),
            "caption_win_rate_vs_original": sum(delta > 0 for delta in deltas)
            / len(deltas),
            "best_of_3_image_text_mean": statistics.fmean(
                max(values) for values in grouped_scores.values()
            ),
            "clip_truncation_rate": sum(bool(row["clip_truncated"]) for row in rows)
            / len(rows),
        }
        log_progress(
            args,
            "clip_statistics",
            completed,
            len(sources),
            source=source,
            scores=len(rows),
        )

    summary = {
        "schema": "imagenet_caption_smoke_summary_v1",
        "images": len(samples),
        "caption_rows": len(captions),
        "distilled_caption_rows": sum(
            item["source"] != "original" for item in captions
        ),
        "sources": sources,
        "length_distribution_words": lengths,
        "vocabulary": vocab_stats,
        "clip": clip_by_source,
        "hallucination": {
            "status": "not_measured",
            "reason": (
                "VLM judging was disabled to keep synthesis efficient and low-cost; "
                "CLIP similarity is not treated as a hallucination label"
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )

    lines = [
        "# ImageNet caption distillation smoke (20 images)",
        "",
        f"- Images: {len(samples)}",
        f"- Distilled captions: {summary['distilled_caption_rows']}",
        "- Hallucination: not measured (paid VLM judge disabled)",
        "",
        "## Aggregate metrics",
        "",
        "| source | mean words | p25 / median / p75 | vocab novel vs original | CLIP image-text | delta vs original |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in sources:
        length = lengths[source]
        vocab = vocab_stats[source]
        clip = clip_by_source[source]
        lines.append(
            f"| {source} | {length['mean']:.2f} | "
            f"{length['p25']:.1f} / {length['median']:.1f} / {length['p75']:.1f} | "
            f"{vocab['novel_types_vs_original']} | "
            f"{clip['image_text_cosine']['mean']:.4f} | "
            f"{clip['mean_delta_vs_original_image_text']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Hallucination rate is intentionally not reported: no paid VLM judge or human annotation was run, and CLIP similarity is not a valid hallucination label.",
            "",
        ]
    )
    report_path = output_dir / "report.md"
    atomic_write_text(report_path, "\n".join(lines))
    args.runtime_log.emit(
        "artifacts_written",
        summary=str(summary_path),
        report=str(report_path),
        images=len(samples),
    )
    print(
        canonical_json(
            {
                "summary": str(summary_path),
                "report": str(report_path),
                "images": len(samples),
                "runtime_log": str(args.runtime_log.path),
            }
        )
    )
    return 0


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-file",
        help=(
            "Append JSONL runtime events to this path. By default a "
            "*.runtime.jsonl file is created beside the command output."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        metavar="N",
        help="Flush a progress event every N processed items (default: 1).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable terminal progress bars; runtime JSONL logging remains enabled.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--response-db", required=True)
    prepare.add_argument("--original-captions", default=str(DEFAULT_ORIGINALS))
    prepare.add_argument("--synset-mapping", default=str(DEFAULT_SYNSET_MAPPING))
    prepare.add_argument("--output", required=True)
    add_runtime_arguments(prepare)
    prepare.set_defaults(func=run_prepare)

    clip = subparsers.add_parser("clip")
    clip.add_argument("--samples", required=True)
    clip.add_argument("--model-dir", required=True)
    clip.add_argument("--output", required=True)
    clip.add_argument("--device", default="cpu")
    add_runtime_arguments(clip)
    clip.set_defaults(func=run_clip)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--samples", required=True)
    summarize.add_argument("--clip-scores", required=True)
    summarize.add_argument("--output-dir", required=True)
    add_runtime_arguments(summarize)
    summarize.set_defaults(func=run_summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    log_path = default_runtime_log_path(args)
    with RuntimeLog(log_path, args.command) as runtime_log:
        args.runtime_log = runtime_log
        try:
            result = int(args.func(args))
        except BaseException as exc:
            runtime_log.emit(
                "run_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        runtime_log.emit("run_completed", exit_code=result)
        return result


if __name__ == "__main__":
    raise SystemExit(main())
