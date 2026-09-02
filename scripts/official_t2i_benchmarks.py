#!/usr/bin/env python3
"""Frozen dataset contracts shared by the official T2I benchmarks."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENEVAL_COMMIT = "af4902f24d3ca90ebbb446dd9891a59e0f82725f"
DPGBENCH_COMMIT = "3c228f1dc6c4d3cad0a47493816151a419f14db3"
MJHQ_COMMIT = "15b0a659e066e763d0e9a6cd8f00e25f8af5e084"
CLEANFID_COMMIT = "e88c4d6269a4bbf04c04deeb578475b57719acee"
CLEANFID_VERSION = "0.1.35"

GENEVAL_PROMPTS = 553
GENEVAL_IMAGES_PER_PROMPT = 4
GENEVAL_TAG_COUNTS = {
    "single_object": 80,
    "two_object": 99,
    "counting": 80,
    "colors": 94,
    "position": 100,
    "color_attr": 100,
}
DPGBENCH_PROMPTS = 1_065
DPGBENCH_IMAGES_PER_PROMPT = 4
MJHQ_PROMPTS = 30_000
MJHQ_CATEGORY_COUNTS = {
    category: 3_000
    for category in (
        "animals",
        "art",
        "fashion",
        "food",
        "indoor",
        "landscape",
        "logo",
        "people",
        "plants",
        "vehicles",
    )
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class BenchmarkPrompt:
    index: int
    prompt_id: str
    prompt: str
    category: str | None = None
    metadata: dict[str, Any] | None = None


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part
    )


def require_git_revision(root: Path, expected: str, label: str) -> str:
    root = root.resolve()
    if not (root / ".git").exists():
        raise ValueError(
            f"{label} must be a git checkout pinned to {expected}; "
            f"missing {root / '.git'}"
        )
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != expected:
        raise ValueError(
            f"{label} revision mismatch: expected {expected}, got {actual}"
        )
    return actual


def _require_tracked_paths_clean(root: Path, paths: list[str], label: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *paths],
        check=False,
    )
    if completed.returncode == 1:
        raise ValueError(f"{label} official benchmark files have local modifications")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)


def _require_prompt_text(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"empty benchmark prompt: {label}")
    return text


def load_geneval_prompts(
    repository: Path, *, verify_revision: bool = True
) -> list[BenchmarkPrompt]:
    repository = repository.resolve()
    if verify_revision:
        require_git_revision(repository, GENEVAL_COMMIT, "GenEval")
        _require_tracked_paths_clean(
            repository, ["prompts/evaluation_metadata.jsonl"], "GenEval"
        )
    path = repository / "prompts" / "evaluation_metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"GenEval row {line_number} is not an object")
            rows.append(value)
    if len(rows) != GENEVAL_PROMPTS:
        raise ValueError(
            f"GenEval requires {GENEVAL_PROMPTS} prompts, got {len(rows)}"
        )
    tag_counts = Counter(str(row.get("tag", "")) for row in rows)
    if dict(tag_counts) != GENEVAL_TAG_COUNTS:
        raise ValueError(
            f"GenEval task distribution mismatch: {dict(tag_counts)}"
        )
    return [
        BenchmarkPrompt(
            index=index,
            prompt_id=f"{index:05d}",
            prompt=_require_prompt_text(row.get("prompt"), f"GenEval row {index}"),
            category=str(row["tag"]),
            metadata=row,
        )
        for index, row in enumerate(rows)
    ]


def load_dpgbench_prompts(
    repository: Path, *, verify_revision: bool = True
) -> list[BenchmarkPrompt]:
    repository = repository.resolve()
    if verify_revision:
        require_git_revision(repository, DPGBENCH_COMMIT, "DPG-Bench/ELLA")
        _require_tracked_paths_clean(
            repository,
            ["dpg_bench/dpg_bench.csv", "dpg_bench/prompts"],
            "DPG-Bench/ELLA",
        )
    benchmark_root = repository / "dpg_bench"
    csv_path = benchmark_root / "dpg_bench.csv"
    prompt_dir = benchmark_root / "prompts"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not prompt_dir.is_dir():
        raise FileNotFoundError(prompt_dir)

    csv_prompts: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            prompt_id = str(row.get("item_id", "")).strip()
            prompt = _require_prompt_text(
                row.get("text"), f"DPG-Bench CSV row {row_number}"
            )
            previous = csv_prompts.setdefault(prompt_id, prompt)
            if not prompt_id or previous != prompt:
                raise ValueError(
                    f"inconsistent DPG-Bench item at CSV row {row_number}"
                )

    prompt_paths = {path.stem: path for path in prompt_dir.glob("*.txt")}
    if len(csv_prompts) != DPGBENCH_PROMPTS:
        raise ValueError(
            f"DPG-Bench requires {DPGBENCH_PROMPTS} prompts, "
            f"got {len(csv_prompts)}"
        )
    if set(prompt_paths) != set(csv_prompts):
        raise ValueError("DPG-Bench prompt files and CSV item IDs disagree")

    prompt_ids = sorted(csv_prompts, key=_natural_key)
    result: list[BenchmarkPrompt] = []
    for index, prompt_id in enumerate(prompt_ids):
        file_prompt = _require_prompt_text(
            prompt_paths[prompt_id].read_text(encoding="utf-8"),
            f"DPG-Bench prompt {prompt_id}",
        )
        # A few official prompt files use typographic punctuation while the
        # evaluator CSV uses ASCII. Generation follows the prompt files; the
        # CSV remains authoritative for questions and dependencies.
        result.append(
            BenchmarkPrompt(index=index, prompt_id=prompt_id, prompt=file_prompt)
        )
    return result


def load_mjhq_prompts(
    repository: Path, *, verify_revision: bool = True
) -> list[BenchmarkPrompt]:
    repository = repository.resolve()
    if verify_revision:
        require_git_revision(repository, MJHQ_COMMIT, "MJHQ-30K")
        _require_tracked_paths_clean(repository, ["meta_data.json"], "MJHQ-30K")
    path = repository / "meta_data.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or len(metadata) != MJHQ_PROMPTS:
        observed = len(metadata) if isinstance(metadata, dict) else type(metadata)
        raise ValueError(f"MJHQ-30K requires {MJHQ_PROMPTS} rows, got {observed}")
    categories = Counter(
        str(value.get("category", ""))
        for value in metadata.values()
        if isinstance(value, dict)
    )
    if dict(categories) != MJHQ_CATEGORY_COUNTS:
        raise ValueError(f"MJHQ-30K category distribution mismatch: {dict(categories)}")

    prompts: list[BenchmarkPrompt] = []
    for index, (prompt_id, value) in enumerate(metadata.items()):
        if not isinstance(value, dict):
            raise TypeError(f"MJHQ-30K metadata is not an object: {prompt_id}")
        category = str(value.get("category", "")).strip()
        prompts.append(
            BenchmarkPrompt(
                index=index,
                prompt_id=str(prompt_id),
                prompt=_require_prompt_text(
                    value.get("prompt"), f"MJHQ-30K row {prompt_id}"
                ),
                category=category,
                metadata=value,
            )
        )
    return prompts


def load_benchmark_prompts(
    benchmark: str, source: Path, *, verify_revision: bool = True
) -> list[BenchmarkPrompt]:
    loaders = {
        "geneval": load_geneval_prompts,
        "dpgbench": load_dpgbench_prompts,
        "mjhq": load_mjhq_prompts,
    }
    try:
        loader = loaders[benchmark]
    except KeyError as error:
        raise ValueError(f"unknown T2I benchmark: {benchmark}") from error
    return loader(source, verify_revision=verify_revision)


def benchmark_revision(benchmark: str) -> str:
    return {
        "geneval": GENEVAL_COMMIT,
        "dpgbench": DPGBENCH_COMMIT,
        "mjhq": MJHQ_COMMIT,
    }[benchmark]


def expected_images_per_prompt(benchmark: str) -> int:
    return {
        "geneval": GENEVAL_IMAGES_PER_PROMPT,
        "dpgbench": DPGBENCH_IMAGES_PER_PROMPT,
        "mjhq": 1,
    }[benchmark]


def index_images(root: Path) -> dict[str, Path]:
    """Index image stems recursively and reject ambiguous benchmark IDs."""

    indexed: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in indexed:
            raise ValueError(
                f"duplicate image stem {path.stem!r}: {indexed[path.stem]} and {path}"
            )
        indexed[path.stem] = path
    return indexed


def require_exact_ids(actual: Iterable[str], expected: Iterable[str], label: str) -> None:
    actual_ids = set(actual)
    expected_ids = set(expected)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids, key=_natural_key)[:10]
        extra = sorted(actual_ids - expected_ids, key=_natural_key)[:10]
        raise ValueError(f"{label} ID coverage mismatch: missing={missing}, extra={extra}")
