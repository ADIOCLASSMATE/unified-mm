#!/usr/bin/env python3
"""Distill and merge multi-caption ImageNet datasets through a vision API.

The final JSONL keeps exactly one row per ImageNet image.  Each row contains
the original caption first and zero or more API-distilled captions in a nested
``captions`` list.  ``ImageNetFlowCacheDataset`` can deterministically cycle
through that list during training while keeping validation fixed.

Generation is resumable.  API responses are committed into sharded SQLite
staging databases, so interruption cannot leave a half-written JSONL record.
The merge step validates img_id, synset, relative path, image id, coverage, and
caption cardinality against the canonical cache manifest before publishing an
atomic final JSONL file.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
import functools
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, time
from typing import Any, Iterable, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "imagenet_multicap_en_v1"
DISTILL_RECORD_SCHEMA = "imagenet_caption_distill_response_v1"
MERGED_SCHEMA = "imagenet_multicap_v1"
DEFAULT_BASE_URL = "https://apicz.boyuerichdata.com/"

# Capability is based on image-input probes against this exact proxy on
# 2026-08-03.  Known non-vision aliases are also recorded so a large run cannot
# accidentally pay for a model that silently ignores the image.
MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "qwen3.5-flash": {
        "vision": True,
        "recommended": False,
        "api_enabled": False,
        "reason": "API use disabled; future Qwen captions are generated locally",
    },
    "qwen3.6-flash": {
        "vision": True,
        "recommended": False,
        "api_enabled": False,
        "reason": "API use disabled; future Qwen captions are generated locally",
    },
    "qwen3.7-flash": {
        "vision": True,
        "recommended": False,
        "api_enabled": False,
        "reason": "API use disabled; future Qwen captions are generated locally",
    },
    "kimi-k2.6": {
        "vision": True,
        "recommended": True,
        "reason": "proxy probe passed; fast non-thinking cross-family teacher",
    },
    "kimi-k3": {
        "vision": True,
        "recommended": False,
        "reason": "proxy probe passed, but reasoning made short caption requests slow",
    },
    "MiniMax-M3": {
        "vision": True,
        "recommended": True,
        "reason": "proxy probe passed; useful as a smaller cross-family diversity source",
    },
    "MiniMax-M2.5": {
        "vision": False,
        "reason": "text-only; MiniMax documents M3 as its image-input language model",
    },
    "MiniMax/MiniMax-M2.5": {
        "vision": False,
        "reason": "text-only alias of MiniMax-M2.5",
    },
    "MiniMax/MiniMax-M2.7": {
        "vision": False,
        "reason": "text-only alias of MiniMax-M2.7",
    },
    "doubao-seed-2-1-turbo-260628": {
        "vision": True,
        "recommended": False,
        "reason": "proxy probe passed, but a one-caption request took over 30 seconds",
    },
    "doubao-embedding-vision-251215": {
        "vision": False,
        "reason": "embedding model; it returns vectors rather than caption text",
    },
    "deepseek-v4-flash": {
        "vision": False,
        "reason": "text-only; the dated proxy alias could not see the supplied image",
    },
    "deepseek-v4-flash-0731": {
        "vision": False,
        "reason": "text-only; proxy probe returned CANNOT_SEE_IMAGE",
    },
    "deepseek-v4-pro": {
        "vision": False,
        "reason": "text-only; proxy ignored the image and said it could not see it",
    },
    "glm-5.2": {
        "vision": False,
        "reason": "text-only; use a GLM *V* model for image input",
    },
    "qwen3-vl-embedding": {
        "vision": False,
        "reason": "embedding model; it returns vectors rather than caption text",
    },
    "qwen-image-2.0": {
        "vision": False,
        "reason": "image generation/editing model; it does not return text captions",
    },
    "qwen-image-2.0-pro": {
        "vision": False,
        "reason": "image generation/editing model; it does not return text captions",
    },
    "doubao-seedance-1-0-pro-250528": {
        "vision": False,
        "reason": "video generation model; it does not return text captions",
    },
    "doubao-seedance-1-0-pro-fast-251015": {
        "vision": False,
        "reason": "video generation model; it does not return text captions",
    },
    "doubao-seedream-4-0-250828": {
        "vision": False,
        "reason": "image generation model; it does not return text captions",
    },
    "doubao-seedream-4-5-251128": {
        "vision": False,
        "reason": "image generation model; it does not return text captions",
    },
    "doubao-seedream-5-0-260128": {
        "vision": False,
        "reason": "image generation model; it does not return text captions",
    },
}


# Captioning should not spend tokens on hidden chain-of-thought.  Kimi K2.6
# additionally requires temperature=0.6 in non-thinking mode, while Kimi K3
# only accepts temperature=1.0 on this proxy.  ``thinking=None`` means omit the
# field, which leaves MiniMax-M3 thinking disabled by its API default.
MODEL_REQUEST_POLICIES: dict[str, dict[str, Any]] = {
    "qwen3.5-flash": {"temperature": 0.8, "thinking": "disabled"},
    "qwen3.6-flash": {"temperature": 0.8, "thinking": "disabled"},
    "qwen3.7-flash": {"temperature": 0.8, "thinking": "disabled"},
    "kimi-k2.6": {"temperature": 0.6, "thinking": "disabled"},
    "kimi-k3": {"temperature": 1.0, "thinking": None},
    "MiniMax-M3": {"temperature": 0.8, "thinking": None},
    "doubao-seed-2-1-turbo-260628": {
        "temperature": 0.8,
        "thinking": "disabled",
    },
}


# Sustained full-dataset runs need request pacing in addition to a concurrency
# cap.  A provider returning an empty response quickly can otherwise turn a
# large concurrency pool into an accidental request flood.
DEFAULT_MODEL_REQUESTS_PER_SECOND: dict[str, float] = {
    "kimi-k2.6": 0.8,
    "MiniMax-M3": 4.0,
}


# Content-level failures are generally tied to a specific image (for example,
# a provider safety filter returning null) or one malformed model generation,
# rather than provider availability. They remain recorded for residual retry,
# but must never stop the healthy bulk queue. Local image-read errors likewise
# do not describe API health.
CONTENT_ERROR_TYPES = frozenset({"empty_response", "malformed_response"})
LOCAL_DATA_ERROR_TYPES = frozenset({"image_read"})


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    manifest: Path
    original_captions: Path
    output_root: Path
    merged_filename: str


DATASET_PROFILES = {
    "imagenet100": DatasetProfile(
        name="imagenet100",
        manifest=REPO_ROOT
        / "public/datasets/imagenet_ablation_100c_balanced/manifest.jsonl",
        original_captions=REPO_ROOT
        / "public/datasets/imagenet_ablation_100c_balanced/captions/imagenet100_recaption_short_join.jsonl",
        output_root=REPO_ROOT
        / "public/datasets/imagenet_distilled_captions/imagenet100",
        merged_filename="imagenet100_multicap_v1.jsonl",
    ),
    "imagenet1k": DatasetProfile(
        name="imagenet1k",
        manifest=REPO_ROOT / "public/datasets/imagenet_full/manifest.jsonl",
        original_captions=REPO_ROOT
        / "public/datasets/imagenet_full/captions/imagenet1k_train_caption_join.jsonl",
        output_root=REPO_ROOT
        / "public/datasets/imagenet_distilled_captions/imagenet1k",
        merged_filename="imagenet1k_multicap_v1.jsonl",
    ),
}


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class GenerationRuntimeLog:
    """Append-only request log flushed immediately after every event."""

    def __init__(self, path: Path, *, models: Sequence[str]) -> None:
        self.path = path
        self.models = list(models)
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + f"-pid{os.getpid()}"
        )
        self._handle: Any | None = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, event: str, **fields: Any) -> None:
        if self._handle is None:
            raise RuntimeError("generation runtime log is not open")
        self._handle.write(
            canonical_json(
                {
                    "timestamp": utc_now(),
                    "run_id": self.run_id,
                    "event": event,
                    **fields,
                }
            )
            + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class GenerationProgress:
    """Small dependency-free terminal progress display."""

    def __init__(self, total: int, *, disabled: bool = False) -> None:
        self.total = total
        self.disabled = disabled
        self.completed = 0
        self.last_rendered = 0.0
        self.last_rendered_completed = -1
        self.closed = False
        self.render(force=True)

    def update(self, amount: int = 1) -> None:
        self.completed += amount
        self.render(force=self.completed >= self.total)

    def render(self, *, force: bool) -> None:
        if self.disabled:
            return
        now = monotonic()
        if force and self.completed == self.last_rendered_completed:
            return
        if not force and now - self.last_rendered < 0.25:
            return
        self.last_rendered = now
        self.last_rendered_completed = self.completed
        fraction = min(1.0, self.completed / self.total) if self.total else 1.0
        width = 24
        filled = round(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\rAPI caption generation: {fraction:6.1%}|{bar}| "
            f"{self.completed}/{self.total} requests",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.render(force=True)
        if not self.disabled:
            print(file=sys.stderr, flush=True)
        self.closed = True


class AsyncRateLimiter:
    """Start API attempts at a bounded rate, shared by one model pool."""

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.interval_seconds = 1.0 / requests_per_second
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = monotonic()
            delay = self._next_start - now
            if delay > 0:
                await asyncio.sleep(delay)
                now = monotonic()
            self._next_start = max(self._next_start, now) + self.interval_seconds


class FailureCircuitBreaker:
    """Stop scheduling a model when a rolling request window is unhealthy."""

    def __init__(
        self,
        models: Sequence[str],
        *,
        window: int,
        failure_rate: float,
    ) -> None:
        if window < 1:
            raise ValueError("circuit window must be positive")
        if not 0.0 < failure_rate <= 1.0:
            raise ValueError("circuit failure rate must be in (0, 1]")
        self.window = window
        self.failure_rate = failure_rate
        self._outcomes = {model: deque(maxlen=window) for model in models}
        self._opened: dict[str, dict[str, Any]] = {}

    def record(self, model: str, *, success: bool) -> dict[str, Any] | None:
        if model in self._opened:
            return None
        outcomes = self._outcomes[model]
        outcomes.append(success)
        if len(outcomes) < self.window:
            return None
        failures = sum(not outcome for outcome in outcomes)
        observed_rate = failures / len(outcomes)
        if observed_rate < self.failure_rate:
            return None
        details = {
            "model": model,
            "window": len(outcomes),
            "failures": failures,
            "failure_rate": observed_rate,
            "threshold": self.failure_rate,
            "opened_at": utc_now(),
        }
        self._opened[model] = details
        return details

    def is_open(self, model: str) -> bool:
        return model in self._opened

    @property
    def opened(self) -> dict[str, dict[str, Any]]:
        return dict(self._opened)


def should_record_circuit_outcome(
    *,
    success: bool,
    error_type: str | None,
    previous_error_type: str | None,
) -> bool:
    """Return whether a completed group is evidence about provider health."""
    if success:
        return True
    if error_type in CONTENT_ERROR_TYPES or error_type in LOCAL_DATA_ERROR_TYPES:
        return False
    return True


def classify_request_error(error: str) -> str:
    lowered = error.lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "ratelimit" in lowered or "rate limit" in lowered or "429" in lowered:
        return "rate_limit_429"
    if (
        "overloadederror" in lowered
        or "error code: 529" in lowered
        or "service cluster is heavily loaded" in lowered
        or "服务集群负载较高" in error
    ):
        return "overloaded_529"
    if any(
        marker in lowered
        for marker in (
            "connection reset",
            "connectionerror",
            "remoteprotocolerror",
            "server disconnected",
            "connection error",
            "broken pipe",
        )
    ):
        return "connection_reset"
    if "empty" in lowered or "no text" in lowered:
        return "empty_response"
    if any(
        marker in lowered
        for marker in (
            "valid json",
            "json field",
            "expected 3 captions",
            "expected exactly",
            "caption 0 is not",
            "caption 1 is not",
            "caption 2 is not",
        )
    ):
        return "malformed_response"
    if "image read failed" in lowered:
        return "image_read"
    if "api" in lowered or "http" in lowered:
        return "api_other"
    return "other"


def exact_percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "p50": exact_percentile(values, 0.50),
        "p90": exact_percentile(values, 0.90),
        "p95": exact_percentile(values, 0.95),
        "max": max(values),
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def word_tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def normalized_caption(text: str) -> str:
    return " ".join(word_tokens(text))


def caption_jaccard(left: str, right: str) -> float:
    left_tokens = set(word_tokens(left))
    right_tokens = set(word_tokens(right))
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def relative_image_path(path: str) -> str:
    parts = Path(path).parts
    if len(parts) < 2:
        raise ValueError(f"image path has fewer than two components: {path!r}")
    return "/".join(parts[-2:])


def manifest_identity(row: dict[str, Any], index: int) -> dict[str, Any]:
    img_id = int(row["img_id"])
    source_path = str(row["source_path"])
    synset = str(row.get("synset", ""))
    rel_path = relative_image_path(source_path)
    path_obj = Path(rel_path)
    image_id = path_obj.stem
    if not synset:
        raise ValueError(f"manifest row {index} img_id={img_id} has no synset")
    if path_obj.parent.name != synset:
        raise ValueError(
            f"manifest row {index} img_id={img_id} path/synset mismatch: "
            f"{rel_path!r} vs {synset!r}"
        )
    return {
        "manifest_index": index,
        "img_id": img_id,
        "id": image_id,
        "path": rel_path,
        "source_path": source_path,
        "synset": synset,
    }


def iter_manifest(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        index = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                yield manifest_identity(row, index)
            except Exception as exc:
                raise ValueError(f"invalid manifest row {path}:{line_number}") from exc
            index += 1


def load_synset_names(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    names: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            synset, _, raw_names = line.strip().partition(" ")
            if synset:
                names[synset] = raw_names.strip()
    return names


def image_path_for_row(row: dict[str, Any], image_root: Path | None) -> Path:
    if image_root is None:
        return Path(row["source_path"])
    return image_root / row["path"]


def image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    raise ValueError(f"unsupported API image type: {path}")


def build_caption_prompt(
    captions_per_model: int,
    min_words: int,
    max_words: int,
    class_hint: str,
    previous_error: str | None = None,
) -> str:
    hint = (
        f"ImageNet class metadata: {class_hint}. Treat it only as a weak hint "
        "and do not mention a class that conflicts with visible pixels."
        if class_hint
        else "No class-name hint is provided; rely only on visible pixels."
    )
    repair = (
        f"\nThe previous response was invalid: {previous_error}. Correct it now."
        if previous_error
        else ""
    )
    json_example = ",".join('{"text":"..."}' for _ in range(captions_per_model))
    return f"""You are producing diverse text-to-image training captions for one ImageNet image.
Inspect the image itself carefully. {hint}

Return exactly {captions_per_model} distinct English captions. Each caption must:
- contain at least {min_words} words, preferably no more than {max_words}, and use complete, natural sentences; never cut a sentence merely to meet the preferred upper length;
- describe only visible subjects, attributes, actions, spatial relations, setting, lighting, viewpoint, and composition;
- be independently useful as a text-to-image prompt, without saying "this image", "the photo shows", or mentioning ImageNet;
- avoid proper identities, hidden intent, unsupported counts, and details that cannot be seen;
- emphasize a different description axis from the other captions: subject/appearance, spatial/compositional relations, and scene/photographic context.

Return JSON only in this exact shape:
{{"captions":[{json_example}]}}
The captions array must contain exactly {captions_per_model} objects.{repair}"""


def _json_candidates(text: str) -> Iterator[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        yield value


def parse_caption_response(
    text: str,
    *,
    expected_count: int,
    min_words: int,
    max_words: int,
    max_jaccard: float,
) -> list[dict[str, Any]]:
    payload = next(_json_candidates(text), None)
    if payload is None:
        raise ValueError("response did not contain valid JSON")
    raw_captions = payload.get("captions") if isinstance(payload, dict) else payload
    if not isinstance(raw_captions, list):
        raise ValueError("JSON field 'captions' is not a list")
    if not raw_captions:
        raise ValueError("JSON field 'captions' is empty")
    if len(raw_captions) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} captions, received {len(raw_captions)}"
        )

    captions: list[dict[str, Any]] = []
    for index, item in enumerate(raw_captions):
        if isinstance(item, dict):
            text_value = item.get("text", "")
        elif isinstance(item, str):
            text_value = item
        else:
            raise ValueError(f"caption {index} is not a string/object")
        caption = " ".join(str(text_value).strip().split())
        if not caption:
            raise ValueError(f"caption {index} is empty")
        tokens = word_tokens(caption)
        captions.append(
            {
                "caption_index": index,
                "text": caption,
                "word_count": len(tokens),
                "text_sha256": sha256_text(caption),
            }
        )

    return captions


def response_has_exact_caption_count(record: dict[str, Any], expected_count: int) -> bool:
    captions = record.get("captions")
    return (
        isinstance(captions, list)
        and len(captions) == expected_count
        and all(
            isinstance(caption, dict) and bool(str(caption.get("text", "")).strip())
            for caption in captions
        )
    )


def model_request_options(model: str, args: argparse.Namespace) -> dict[str, Any]:
    """Resolve fast caption-generation parameters for a provider alias."""
    policy = MODEL_REQUEST_POLICIES.get(model, {})
    temperature = args.temperature
    if temperature is None:
        temperature = float(policy.get("temperature", 0.8))

    options: dict[str, Any] = {"temperature": temperature}
    thinking_mode = args.thinking
    if thinking_mode == "auto":
        thinking_mode = policy.get("thinking")
    if thinking_mode == "disabled":
        options["thinking"] = {"type": "disabled"}
    elif thinking_mode == "enabled":
        options["thinking"] = {
            "type": "enabled",
            "budget_tokens": args.thinking_budget_tokens,
        }
    return options


def request_fingerprint(
    args: argparse.Namespace,
    models: Sequence[str],
    *,
    manifest_sha256: str,
    synset_mapping_sha256: str | None,
) -> str:
    contract = {
        "schema": DISTILL_RECORD_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "dataset": args.dataset,
        "manifest_sha256": manifest_sha256,
        "models": list(models),
        "captions_per_model": args.captions_per_model,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_jaccard": args.max_jaccard,
        "include_class_hint": not args.no_class_hint,
        "synset_mapping_sha256": synset_mapping_sha256,
        "base_url": args.base_url,
        "model_request_options": {
            model: model_request_options(model, args) for model in models
        },
        "max_tokens": args.max_tokens,
        "image_root": str(args.image_root or ""),
    }
    return sha256_text(canonical_json(contract))[:16]


def validate_generation_models(models: Sequence[str], allow_unverified: bool) -> None:
    if not models:
        raise ValueError("at least one model is required")
    for model in models:
        capability = MODEL_CAPABILITIES.get(model)
        if model.lower().startswith("qwen"):
            raise ValueError(
                f"{model} API generation is disabled; generate Qwen captions locally"
            )
        if capability is not None and not capability["vision"]:
            raise ValueError(
                f"{model} is not a multimodal image-input model: {capability['reason']}"
            )
        if capability is None and not allow_unverified:
            raise ValueError(
                f"{model} has not been vision-verified on this proxy; pass "
                "--allow-unverified-models only after an image capability probe"
            )


def resolve_model_concurrency(
    values: Sequence[str] | None,
    models: Sequence[str],
) -> dict[str, int] | None:
    """Parse optional independent per-model concurrency limits."""
    if not values:
        return None
    selected = set(models)
    resolved: dict[str, int] = {}
    for value in values:
        model, separator, limit_text = value.rpartition("=")
        if not separator or not model or not limit_text:
            raise ValueError(
                f"--model-concurrency entries must use MODEL=N, received {value!r}"
            )
        if model not in selected:
            raise ValueError(f"concurrency specified for unselected model: {model}")
        if model in resolved:
            raise ValueError(f"duplicate model concurrency entry: {model}")
        try:
            limit = int(limit_text)
        except ValueError as exc:
            raise ValueError(f"invalid concurrency for {model}: {limit_text}") from exc
        if limit < 1:
            raise ValueError(f"model concurrency must be positive for {model}")
        resolved[model] = limit
    missing = selected.difference(resolved)
    if missing:
        raise ValueError(
            "--model-concurrency must specify every selected model; missing "
            + ", ".join(sorted(missing))
        )
    return resolved


def resolve_model_rate_limits(
    values: Sequence[str] | None,
    models: Sequence[str],
) -> dict[str, float]:
    """Resolve per-model API start-rate limits with conservative defaults."""
    if not values:
        return {
            model: DEFAULT_MODEL_REQUESTS_PER_SECOND[model]
            for model in models
            if model in DEFAULT_MODEL_REQUESTS_PER_SECOND
        }
    selected = set(models)
    resolved: dict[str, float] = {}
    for value in values:
        model, separator, limit_text = value.rpartition("=")
        if not separator or not model or not limit_text:
            raise ValueError(
                f"--model-rps entries must use MODEL=RATE, received {value!r}"
            )
        if model not in selected:
            raise ValueError(f"request rate specified for unselected model: {model}")
        if model in resolved:
            raise ValueError(f"duplicate model request rate entry: {model}")
        try:
            limit = float(limit_text)
        except ValueError as exc:
            raise ValueError(f"invalid request rate for {model}: {limit_text}") from exc
        if limit <= 0:
            raise ValueError(f"model request rate must be positive for {model}")
        resolved[model] = limit
    missing = selected.difference(resolved)
    if missing:
        raise ValueError(
            "--model-rps must specify every selected model; missing "
            + ", ".join(sorted(missing))
        )
    return resolved


def open_response_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS responses (
            img_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            record_json TEXT NOT NULL,
            PRIMARY KEY (img_id, model)
        );
        CREATE TABLE IF NOT EXISTS errors (
            img_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            error TEXT NOT NULL,
            error_type TEXT NOT NULL DEFAULT 'other',
            request_latency_seconds REAL,
            request_started_at TEXT,
            request_finished_at TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (img_id, model)
        );
        """
    )
    existing_error_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(errors)")
    }
    migrations = {
        "error_type": "TEXT NOT NULL DEFAULT 'other'",
        "request_latency_seconds": "REAL",
        "request_started_at": "TEXT",
        "request_finished_at": "TEXT",
    }
    for column, definition in migrations.items():
        if column not in existing_error_columns:
            connection.execute(f"ALTER TABLE errors ADD COLUMN {column} {definition}")
    return connection


def load_completed_response_ids(
    connection: sqlite3.Connection,
    models: Sequence[str],
    *,
    expected_count: int,
) -> dict[str, set[int]]:
    completed: dict[str, set[int]] = {model: set() for model in models}
    for img_id, model, record_json in connection.execute(
        "SELECT img_id, model, record_json FROM responses"
    ):
        if model not in completed:
            continue
        try:
            record = json.loads(record_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if response_has_exact_caption_count(record, expected_count):
            completed[str(model)].add(int(img_id))
    return completed


def load_response_error_types(
    connection: sqlite3.Connection,
    models: Sequence[str],
) -> dict[str, dict[int, str]]:
    error_types: dict[str, dict[int, str]] = {model: {} for model in models}
    for img_id, model, error_type in connection.execute(
        "SELECT img_id, model, error_type FROM errors"
    ):
        if model in error_types:
            error_types[str(model)][int(img_id)] = str(error_type or "other")
    return error_types


def reuse_successful_responses(
    connection: sqlite3.Connection,
    source_dirs: Sequence[str] | None,
    *,
    target_db: Path,
    models: Sequence[str],
    manifest_sha256: str,
    fingerprint: str,
    expected_run: dict[str, Any],
) -> int:
    """Copy compatible successful records into a new model-set response DB."""
    if not source_dirs:
        return 0
    selected_models = set(models)
    imported = 0
    for source_dir_text in source_dirs:
        source_dir = Path(source_dir_text)
        run_path = source_dir / "run.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"reuse source has no run.json: {source_dir}")
        source_run = json.loads(run_path.read_text(encoding="utf-8"))
        if source_run.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"reuse source manifest mismatch: {source_dir}")
        if source_run.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"reuse source prompt mismatch: {source_dir}")
        reusable_models = selected_models.intersection(source_run.get("models", []))
        if not reusable_models:
            raise ValueError(f"reuse source has no selected model: {source_dir}")

        for key in (
            "dataset",
            "captions_per_model",
            "min_words",
            "max_words",
            "include_class_hint",
            "synset_mapping_sha256",
            "base_url",
        ):
            if source_run.get(key) != expected_run.get(key):
                raise ValueError(f"reuse source {key} mismatch: {source_dir}")
        source_options = source_run.get("model_request_options", {})
        target_options = expected_run.get("model_request_options", {})
        for model in reusable_models:
            if source_options.get(model) != target_options.get(model):
                raise ValueError(
                    f"reuse source request options mismatch for {model}: {source_dir}"
                )
        for database in sorted(source_dir.glob("part-*.sqlite3")):
            if database.resolve() == target_db.resolve():
                continue
            source = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            try:
                for (record_json,) in source.execute(
                    "SELECT record_json FROM responses ORDER BY img_id, model"
                ):
                    record = json.loads(record_json)
                    model = str(record.get("model", ""))
                    if model not in reusable_models:
                        continue
                    if record.get("manifest_sha256") != manifest_sha256:
                        raise ValueError(
                            f"reuse record manifest mismatch in {database}"
                        )
                    captions = record.get("captions")
                    if not response_has_exact_caption_count(
                        record, int(expected_run["captions_per_model"])
                    ):
                        continue
                    source_fingerprint = record.get("request_fingerprint")
                    record["source_request_fingerprint"] = source_fingerprint
                    record["request_fingerprint"] = fingerprint
                    record["reused_at"] = utc_now()
                    before = connection.total_changes
                    connection.execute(
                        "INSERT OR IGNORE INTO responses(img_id, model, record_json) "
                        "VALUES (?, ?, ?)",
                        (
                            int(record["img_id"]),
                            model,
                            canonical_json(record),
                        ),
                    )
                    if connection.total_changes > before:
                        connection.execute(
                            "DELETE FROM errors WHERE img_id=? AND model=?",
                            (int(record["img_id"]), model),
                        )
                        imported += 1
            finally:
                source.close()
    connection.commit()
    return imported


def content_text(response: Any) -> str:
    parts = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", "") == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(parts).strip()


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
    }


def exception_summary(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"caused by {type(cause).__name__}: {cause}")
    return "; ".join(parts)


async def distill_one(
    *,
    client: Any,
    executor: ThreadPoolExecutor,
    row: dict[str, Any],
    model: str,
    image_root: Path | None,
    class_hint: str,
    args: argparse.Namespace,
    manifest_sha256: str,
    fingerprint: str,
    runtime_log: GenerationRuntimeLog,
    concurrency_config: dict[str, Any],
    request_limiter: AsyncRateLimiter | None,
) -> tuple[bool, dict[str, Any]]:
    image_path = image_path_for_row(row, image_root)
    request_started_monotonic = monotonic()
    request_started_epoch = time()
    request_started_at = utc_now()
    loop = asyncio.get_running_loop()
    runtime_log.emit(
        "request_started",
        model=model,
        img_id=int(row["img_id"]),
        manifest_index=int(row["manifest_index"]),
        concurrency=concurrency_config,
        request_started_at=request_started_at,
    )

    def finish(
        ok: bool,
        payload: dict[str, Any],
        *,
        error_type: str | None,
    ) -> tuple[bool, dict[str, Any]]:
        request_finished_at = utc_now()
        request_finished_epoch = time()
        latency = round(monotonic() - request_started_monotonic, 3)
        payload.update(
            {
                "request_started_at": request_started_at,
                "request_finished_at": request_finished_at,
                "request_started_epoch": request_started_epoch,
                "request_finished_epoch": request_finished_epoch,
                "request_latency_seconds": latency,
                "error_type": error_type,
            }
        )
        runtime_log.emit(
            "request_finished",
            model=model,
            img_id=int(row["img_id"]),
            manifest_index=int(row["manifest_index"]),
            concurrency=concurrency_config,
            request_started_at=request_started_at,
            request_finished_at=request_finished_at,
            latency_seconds=latency,
            success=ok,
            error_type=error_type,
            error=payload.get("error"),
            captions_saved=(len(payload.get("captions", [])) if ok else 0),
        )
        return ok, payload

    try:
        image_bytes = await loop.run_in_executor(executor, image_path.read_bytes)
        media_type = image_media_type(image_path)
    except Exception as exc:
        error = f"image read failed for {image_path}: {exc}"
        return finish(
            False,
            {
                "img_id": row["img_id"],
                "model": model,
                "attempts": 0,
                "error": error,
            },
            error_type="image_read",
        )

    image_sha256 = sha256_bytes(image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    previous_error: str | None = None
    previous_error_type: str | None = None
    for attempt in range(1, args.max_retries + 1):
        prompt = build_caption_prompt(
            captions_per_model=args.captions_per_model,
            min_words=args.min_words,
            max_words=args.max_words,
            class_hint=class_hint,
            previous_error=previous_error,
        )
        try:
            if request_limiter is not None:
                await request_limiter.acquire()
            request_options = model_request_options(model, args)
            request = functools.partial(
                client.messages.create,
                model=model,
                max_tokens=args.max_tokens,
                **request_options,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            response = await loop.run_in_executor(executor, request)
            response_text = content_text(response)
            if not response_text:
                raise ValueError("empty response text")
            captions = parse_caption_response(
                response_text,
                expected_count=args.captions_per_model,
                min_words=args.min_words,
                max_words=args.max_words,
                max_jaccard=args.max_jaccard,
            )
            record = {
                "schema": DISTILL_RECORD_SCHEMA,
                "request_fingerprint": fingerprint,
                "prompt_version": PROMPT_VERSION,
                "dataset": args.dataset,
                "manifest_sha256": manifest_sha256,
                **row,
                "image_sha256": image_sha256,
                "model": model,
                "captions": captions,
                "response_id": getattr(response, "id", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "usage": usage_dict(response),
                "request_options": request_options,
                "attempts": attempt,
                "created_at": utc_now(),
            }
            return finish(True, record, error_type=None)
        except Exception as exc:
            previous_error = exception_summary(exc)
            previous_error_type = classify_request_error(previous_error)
            if attempt < args.max_retries:
                delay = min(
                    args.retry_max_seconds,
                    args.retry_base_seconds * (2 ** (attempt - 1)),
                )
                delay *= random.uniform(0.75, 1.25)
                await asyncio.sleep(delay)

    return finish(
        False,
        {
            "img_id": row["img_id"],
            "model": model,
            "attempts": args.max_retries,
            "error": previous_error or "unknown generation failure",
        },
        error_type=previous_error_type or "other",
    )


def selected_manifest_rows(
    manifest_path: Path,
    *,
    shard_id: int,
    num_shards: int,
    start_index: int,
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    emitted = 0
    for row in iter_manifest(manifest_path):
        index = row["manifest_index"]
        if index < start_index or index % num_shards != shard_id:
            continue
        if limit is not None and emitted >= limit:
            return
        yield row
        emitted += 1


async def run_generation(args: argparse.Namespace) -> int:
    profile, manifest_path, _, output_root = resolve_profile_paths(args)
    models = tuple(dict.fromkeys(args.models))
    validate_generation_models(models, args.allow_unverified_models)
    model_concurrency = resolve_model_concurrency(args.model_concurrency, models)
    model_rate_limits = resolve_model_rate_limits(args.model_rps, models)
    if len(models) > 1 and model_concurrency is None:
        raise ValueError(
            "multiple API models require independent --model-concurrency MODEL=N "
            "entries"
        )
    scheduler_concurrency = (
        sum(model_concurrency.values()) if model_concurrency else args.concurrency
    )
    scheduler_config: dict[str, Any] = (
        {"mode": "per_model", "limits": model_concurrency}
        if model_concurrency
        else {"mode": "global", "concurrency": args.concurrency}
    )
    scheduler_config["requests_per_second"] = model_rate_limits
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")
    if args.captions_per_model < 2:
        raise ValueError("captions-per-model must be at least 2")
    if args.concurrency < 1 or args.max_retries < 1 or args.commit_every < 1:
        raise ValueError("concurrency, max-retries, and commit-every must be positive")
    if args.circuit_window < 1:
        raise ValueError("circuit-window must be positive")
    if not 0.0 < args.circuit_failure_rate <= 1.0:
        raise ValueError("circuit-failure-rate must be in (0, 1]")
    if args.min_words < 1 or args.max_words < args.min_words:
        raise ValueError("invalid caption word limits")
    if not 0.0 <= args.max_jaccard <= 1.0:
        raise ValueError("max-jaccard must be between zero and one")
    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        raise ValueError("temperature override must be between zero and two")
    if args.thinking == "enabled" and args.thinking_budget_tokens >= args.max_tokens:
        raise ValueError("thinking budget must be smaller than max-tokens")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest_digest = sha256_file(manifest_path)
    synset_mapping_path = Path(args.synset_mapping)
    if not args.no_class_hint and not synset_mapping_path.is_file():
        raise FileNotFoundError(synset_mapping_path)
    synset_mapping_digest = (
        None
        if args.no_class_hint or not synset_mapping_path.is_file()
        else sha256_file(synset_mapping_path)
    )
    fingerprint = request_fingerprint(
        args,
        models,
        manifest_sha256=manifest_digest,
        synset_mapping_sha256=synset_mapping_digest,
    )
    response_dir = output_root / "responses" / fingerprint
    response_db = response_dir / (
        f"part-{args.shard_id:05d}-of-{args.num_shards:05d}.sqlite3"
    )
    existing_completed: dict[str, set[int]] = {model: set() for model in models}
    if response_db.is_file():
        existing_connection = sqlite3.connect(
            f"file:{response_db.resolve()}?mode=ro", uri=True
        )
        try:
            existing_completed = load_completed_response_ids(
                existing_connection,
                models,
                expected_count=args.captions_per_model,
            )
        finally:
            existing_connection.close()
    selected_image_count = 0
    exact_completed_in_selection = 0
    for selected_row in selected_manifest_rows(
        manifest_path,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        start_index=args.start_index,
        limit=args.limit,
    ):
        selected_image_count += 1
        exact_completed_in_selection += sum(
            int(selected_row["img_id"] in existing_completed[model]) for model in models
        )
    rows = selected_manifest_rows(
        manifest_path,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        start_index=args.start_index,
        limit=args.limit,
    )

    if args.dry_run:
        print(
            canonical_json(
                {
                    "dataset": profile.name,
                    "manifest": str(manifest_path),
                    "manifest_sha256": manifest_digest,
                    "models": models,
                    "model_request_options": {
                        model: model_request_options(model, args) for model in models
                    },
                    "selected_images": selected_image_count,
                    "exact_completed_groups": exact_completed_in_selection,
                    "planned_api_requests": (
                        selected_image_count * len(models) - exact_completed_in_selection
                    ),
                    "captions_per_request": args.captions_per_model,
                    "scheduler": scheduler_config,
                    "worker_threads": scheduler_concurrency,
                    "response_db": str(response_db),
                    "request_fingerprint": fingerprint,
                    "dry_run": True,
                }
            )
        )
        return 0

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"environment variable {args.api_key_env} is not set")
    try:
        from anthropic import Anthropic
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "anthropic and httpx are required; run through the project's uv environment"
        ) from exc

    response_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "schema": DISTILL_RECORD_SCHEMA,
        "request_fingerprint": fingerprint,
        "prompt_version": PROMPT_VERSION,
        "dataset": profile.name,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "models": list(models),
        "model_request_options": {
            model: model_request_options(model, args) for model in models
        },
        "captions_per_model": args.captions_per_model,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_jaccard": args.max_jaccard,
        "include_class_hint": not args.no_class_hint,
        "synset_mapping": str(synset_mapping_path),
        "synset_mapping_sha256": synset_mapping_digest,
        "base_url": args.base_url,
        "num_shards": args.num_shards,
        "created_at": utc_now(),
    }
    run_path = response_dir / "run.json"
    if run_path.exists():
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        comparable_keys = set(run_metadata).difference({"created_at"})
        if any(existing.get(key) != run_metadata.get(key) for key in comparable_keys):
            raise RuntimeError(f"incompatible existing run metadata: {run_path}")
    else:
        run_path.write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    connection = open_response_db(response_db)
    reused_count = reuse_successful_responses(
        connection,
        args.reuse_response_dir,
        target_db=response_db,
        models=models,
        manifest_sha256=manifest_digest,
        fingerprint=fingerprint,
        expected_run=run_metadata,
    )
    completed = load_completed_response_ids(
        connection,
        models,
        expected_count=args.captions_per_model,
    )
    previous_error_types = load_response_error_types(connection, models)

    runtime_log_path = (
        Path(args.runtime_log)
        if args.runtime_log
        else response_dir / "generation.runtime.jsonl"
    )
    runtime_log = GenerationRuntimeLog(runtime_log_path, models=models)
    runtime_log.open()
    runtime_log.emit(
        "run_started",
        dataset=profile.name,
        models=list(models),
        scheduler=scheduler_config,
        selected_images=selected_image_count,
        selected_image_model_groups=selected_image_count * len(models),
        start_index=args.start_index,
        limit=args.limit,
        max_retries=args.max_retries,
        exact_completed_groups=sum(len(values) for values in completed.values()),
        response_db=str(response_db),
        reused_records=reused_count,
    )
    progress = GenerationProgress(
        selected_image_count * len(models),
        disabled=args.no_progress,
    )

    synset_names = {} if args.no_class_hint else load_synset_names(synset_mapping_path)
    image_root = Path(args.image_root) if args.image_root else None
    request_limiters = {
        model: AsyncRateLimiter(rate) for model, rate in model_rate_limits.items()
    }
    circuit_breaker = FailureCircuitBreaker(
        models,
        window=args.circuit_window,
        failure_rate=args.circuit_failure_rate,
    )
    # The Inspire HTTP proxy currently fails TLS setup through httpx's async
    # transport while the synchronous transport works. API calls therefore use
    # an explicit bounded thread pool and a matching HTTP connection pool. This
    # avoids asyncio's default ~32-thread ceiling at high concurrency.
    executor = ThreadPoolExecutor(
        max_workers=scheduler_concurrency,
        thread_name_prefix="caption-api",
    )
    http_client = httpx.Client(
        timeout=args.timeout,
        limits=httpx.Limits(
            max_connections=scheduler_concurrency,
            max_keepalive_connections=scheduler_concurrency,
        ),
    )
    client = Anthropic(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout,
        max_retries=0,
        http_client=http_client,
    )

    generation_started_at = monotonic()
    success_count = 0
    failure_count = 0
    skipped_count = 0
    completed_since_commit = 0
    pending_commit_events: list[tuple[bool, dict[str, Any]]] = []
    per_model_stats: dict[str, dict[str, Any]] = {
        model: {
            "generated": 0,
            "failed": 0,
            "resumed": 0,
            "latencies": [],
            "success_latencies": [],
            "errors": {},
            "circuit_ignored_repeated_content_failures": 0,
            "requests_per_minute": {},
            "first_started_epoch": None,
            "last_finished_epoch": None,
        }
        for model in models
    }

    async def consume_finished(
        done: Iterable[asyncio.Task[tuple[bool, dict[str, Any]]]],
    ) -> None:
        nonlocal success_count, failure_count, completed_since_commit
        for task in done:
            ok, payload = await task
            model = str(payload["model"])
            latency = float(payload.get("request_latency_seconds", 0.0))
            stats = per_model_stats[model]
            stats["latencies"].append(latency)
            started_epoch = float(payload.get("request_started_epoch", 0.0))
            finished_epoch = float(payload.get("request_finished_epoch", 0.0))
            if stats["first_started_epoch"] is None:
                stats["first_started_epoch"] = started_epoch
            else:
                stats["first_started_epoch"] = min(
                    float(stats["first_started_epoch"]), started_epoch
                )
            if stats["last_finished_epoch"] is None:
                stats["last_finished_epoch"] = finished_epoch
            else:
                stats["last_finished_epoch"] = max(
                    float(stats["last_finished_epoch"]), finished_epoch
                )
            minute = datetime.fromtimestamp(finished_epoch, timezone.utc).strftime(
                "%Y-%m-%dT%H:%MZ"
            )
            stats["requests_per_minute"][minute] = (
                int(stats["requests_per_minute"].get(minute, 0)) + 1
            )
            if ok:
                connection.execute(
                    "INSERT OR REPLACE INTO responses(img_id, model, record_json) "
                    "VALUES (?, ?, ?)",
                    (
                        int(payload["img_id"]),
                        str(payload["model"]),
                        canonical_json(payload),
                    ),
                )
                connection.execute(
                    "DELETE FROM errors WHERE img_id=? AND model=?",
                    (int(payload["img_id"]), str(payload["model"])),
                )
                success_count += 1
                stats["generated"] += 1
                stats["success_latencies"].append(latency)
            else:
                connection.execute(
                    "INSERT OR REPLACE INTO errors"
                    "(img_id, model, attempts, error, error_type, "
                    "request_latency_seconds, request_started_at, "
                    "request_finished_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        int(payload["img_id"]),
                        str(payload["model"]),
                        int(payload["attempts"]),
                        str(payload["error"]),
                        str(payload["error_type"]),
                        latency,
                        str(payload["request_started_at"]),
                        str(payload["request_finished_at"]),
                        utc_now(),
                    ),
                )
                failure_count += 1
                stats["failed"] += 1
                error_type = str(payload.get("error_type") or "other")
                stats["errors"][error_type] = (
                    int(stats["errors"].get(error_type, 0)) + 1
                )
            error_type = None if ok else str(payload.get("error_type") or "other")
            previous_error_type = previous_error_types[model].get(
                int(payload["img_id"])
            )
            if should_record_circuit_outcome(
                success=ok,
                error_type=error_type,
                previous_error_type=previous_error_type,
            ):
                opened = circuit_breaker.record(model, success=ok)
                if opened is not None:
                    runtime_log.emit("circuit_breaker_open", **opened)
            else:
                stats["circuit_ignored_repeated_content_failures"] += 1
                runtime_log.emit(
                    "circuit_outcome_ignored",
                    model=model,
                    img_id=int(payload["img_id"]),
                    error_type=error_type,
                    previous_error_type=previous_error_type,
                    reason="known_image_level_failure",
                )
            if not ok:
                previous_error_types[model][int(payload["img_id"])] = str(error_type)
            completed_since_commit += 1
            pending_commit_events.append((ok, payload))
            progress.update(1)
            if completed_since_commit >= args.commit_every:
                commit_pending_results()

    def commit_pending_results() -> None:
        nonlocal completed_since_commit
        if not pending_commit_events:
            return
        connection.commit()
        for ok, payload in pending_commit_events:
            runtime_log.emit(
                "record_committed",
                model=str(payload["model"]),
                img_id=int(payload["img_id"]),
                concurrency=scheduler_config,
                request_started_at=payload.get("request_started_at"),
                request_finished_at=payload.get("request_finished_at"),
                latency_seconds=payload.get("request_latency_seconds"),
                success=ok,
                error_type=payload.get("error_type"),
                captions_saved=(len(payload.get("captions", [])) if ok else 0),
            )
        pending_commit_events.clear()
        completed_since_commit = 0

    async def run_global_scheduler() -> None:
        nonlocal skipped_count
        pending: set[asyncio.Task[tuple[bool, dict[str, Any]]]] = set()
        for row in rows:
            for model in models:
                if circuit_breaker.is_open(model):
                    continue
                if row["img_id"] in completed[model]:
                    skipped_count += 1
                    per_model_stats[model]["resumed"] += 1
                    progress.update(1)
                    continue
                task = asyncio.create_task(
                    distill_one(
                        client=client,
                        executor=executor,
                        row=row,
                        model=model,
                        image_root=image_root,
                        class_hint=synset_names.get(row["synset"], ""),
                        args=args,
                        manifest_sha256=manifest_digest,
                        fingerprint=fingerprint,
                        runtime_log=runtime_log,
                        concurrency_config=scheduler_config,
                        request_limiter=request_limiters.get(model),
                    )
                )
                pending.add(task)
                if len(pending) >= args.concurrency:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    await consume_finished(done)
            if all(circuit_breaker.is_open(model) for model in models):
                break
        if pending:
            done, _ = await asyncio.wait(pending)
            await consume_finished(done)

    async def run_model_scheduler(model: str, concurrency: int) -> None:
        nonlocal skipped_count
        pending: set[asyncio.Task[tuple[bool, dict[str, Any]]]] = set()
        model_rows = selected_manifest_rows(
            manifest_path,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            start_index=args.start_index,
            limit=args.limit,
        )
        for row in model_rows:
            if circuit_breaker.is_open(model):
                break
            if row["img_id"] in completed[model]:
                skipped_count += 1
                per_model_stats[model]["resumed"] += 1
                progress.update(1)
                continue
            task = asyncio.create_task(
                distill_one(
                    client=client,
                    executor=executor,
                    row=row,
                    model=model,
                    image_root=image_root,
                    class_hint=synset_names.get(row["synset"], ""),
                    args=args,
                    manifest_sha256=manifest_digest,
                    fingerprint=fingerprint,
                    runtime_log=runtime_log,
                    concurrency_config=scheduler_config,
                    request_limiter=request_limiters.get(model),
                )
            )
            pending.add(task)
            if len(pending) >= concurrency:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                await consume_finished(done)
        if pending:
            done, _ = await asyncio.wait(pending)
            await consume_finished(done)
        commit_pending_results()
        runtime_log.emit(
            "model_scheduler_completed",
            model=model,
            concurrency=concurrency,
            generated=int(per_model_stats[model]["generated"]),
            failed=int(per_model_stats[model]["failed"]),
            resumed=int(per_model_stats[model]["resumed"]),
        )

    generation_error: BaseException | None = None
    try:
        if model_concurrency:
            await asyncio.gather(
                *(
                    run_model_scheduler(model, model_concurrency[model])
                    for model in models
                )
            )
        else:
            await run_global_scheduler()
        commit_pending_results()
    except BaseException as exc:
        generation_error = exc
        runtime_log.emit(
            "run_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            generated=success_count,
            failed=failure_count,
            resumed=skipped_count,
        )
        raise
    finally:
        commit_pending_results()
        connection.close()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(executor, client.close)
        finally:
            executor.shutdown(wait=True)
            progress.close()
            if generation_error is not None:
                runtime_log.close()

    elapsed_seconds = monotonic() - generation_started_at
    completed_requests = success_count + failure_count
    finalized_model_stats: dict[str, dict[str, Any]] = {}
    all_requests_per_minute: dict[str, int] = {}
    for model, stats in per_model_stats.items():
        model_completed = int(stats["generated"]) + int(stats["failed"])
        model_started = stats["first_started_epoch"]
        model_finished = stats["last_finished_epoch"]
        model_elapsed = (
            float(model_finished) - float(model_started)
            if model_started is not None and model_finished is not None
            else 0.0
        )
        for minute, count in stats["requests_per_minute"].items():
            all_requests_per_minute[minute] = all_requests_per_minute.get(
                minute, 0
            ) + int(count)
        finalized_model_stats[model] = {
            "total_requests": model_completed,
            "generated": int(stats["generated"]),
            "failed": int(stats["failed"]),
            "success_rate": (
                int(stats["generated"]) / model_completed if model_completed else 0.0
            ),
            "resumed": int(stats["resumed"]),
            "concurrency": (
                model_concurrency[model] if model_concurrency else args.concurrency
            ),
            "elapsed_seconds": model_elapsed,
            "requests_per_second": (
                model_completed / model_elapsed if model_elapsed else 0.0
            ),
            "successful_requests_per_second": (
                int(stats["generated"]) / model_elapsed if model_elapsed else 0.0
            ),
            "latency_seconds": latency_summary(stats["latencies"]),
            "successful_latency_seconds": latency_summary(stats["success_latencies"]),
            "error_types": dict(sorted(stats["errors"].items())),
            "circuit_ignored_repeated_content_failures": int(
                stats["circuit_ignored_repeated_content_failures"]
            ),
            "requests_per_minute": dict(sorted(stats["requests_per_minute"].items())),
            "circuit_breaker": circuit_breaker.opened.get(model),
        }

    summary = {
        "schema": "imagenet_caption_api_benchmark_v1",
        "run_id": runtime_log.run_id,
        "response_db": str(response_db),
        "request_fingerprint": fingerprint,
        "runtime_log": str(runtime_log_path),
        "selected_images": selected_image_count,
        "total_requests": completed_requests,
        "generated": success_count,
        "failed": failure_count,
        "success_rate": (
            success_count / completed_requests if completed_requests else 0.0
        ),
        "resumed": skipped_count,
        "reused_records": reused_count,
        "elapsed_seconds": elapsed_seconds,
        "requests_per_second": (
            completed_requests / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "successful_requests_per_second": (
            success_count / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "requests_per_minute": dict(sorted(all_requests_per_minute.items())),
        "scheduler": scheduler_config,
        "circuit_breakers": circuit_breaker.opened,
        "worker_threads": scheduler_concurrency,
        "per_model": finalized_model_stats,
    }
    benchmark_dir = response_dir / "benchmark_runs"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_dir / f"{runtime_log.run_id}.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, summary_path)
    summary["summary_path"] = str(summary_path)
    runtime_log.emit(
        "run_completed",
        summary=str(summary_path),
        generated=success_count,
        failed=failure_count,
        resumed=skipped_count,
        elapsed_seconds=elapsed_seconds,
    )
    runtime_log.close()
    print(canonical_json(summary))
    if circuit_breaker.opened:
        return 2
    return 1 if failure_count else 0


def resolve_profile_paths(
    args: argparse.Namespace,
) -> tuple[DatasetProfile, Path, Path, Path]:
    profile = DATASET_PROFILES[args.dataset]
    manifest = Path(args.manifest) if args.manifest else profile.manifest
    originals = (
        Path(args.original_captions)
        if getattr(args, "original_captions", None)
        else profile.original_captions
    )
    output_root = Path(args.output_root) if args.output_root else profile.output_root
    return profile, manifest, originals, output_root


def create_merge_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE manifest (
            img_id INTEGER PRIMARY KEY,
            manifest_index INTEGER NOT NULL UNIQUE,
            image_id TEXT NOT NULL UNIQUE,
            rel_path TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL,
            synset TEXT NOT NULL
        );
        CREATE TABLE originals (
            img_id INTEGER PRIMARY KEY,
            image_id TEXT NOT NULL UNIQUE,
            rel_path TEXT NOT NULL UNIQUE,
            synset TEXT,
            text TEXT NOT NULL
        );
        CREATE TABLE distilled (
            img_id INTEGER NOT NULL,
            manifest_index INTEGER NOT NULL,
            image_id TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            source_path TEXT NOT NULL,
            synset TEXT NOT NULL,
            model TEXT NOT NULL,
            caption_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            text_sha256 TEXT NOT NULL,
            image_sha256 TEXT NOT NULL,
            PRIMARY KEY (img_id, model, caption_index)
        );
        """
    )
    return connection


def batched_insert(
    connection: sqlite3.Connection,
    sql: str,
    rows: Iterable[tuple[Any, ...]],
    batch_size: int = 10_000,
) -> int:
    batch: list[tuple[Any, ...]] = []
    count = 0
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            connection.executemany(sql, batch)
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(sql, batch)
        count += len(batch)
    connection.commit()
    return count


def original_rows(path: Path) -> Iterator[tuple[Any, ...]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            caption = str(row.get("recaption_short", "")).strip()
            if not caption:
                raise ValueError(f"missing recaption_short at {path}:{line_number}")
            if "img_id" not in row:
                raise ValueError(f"missing img_id at {path}:{line_number}")
            rel_path = relative_image_path(str(row["path"]))
            yield (
                int(row["img_id"]),
                str(row.get("id", Path(rel_path).stem)),
                rel_path,
                str(row.get("synset", "")),
                caption,
            )


def discover_response_dir(output_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    root = output_root / "responses"
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one response run under {root}, found "
            f"{len(candidates)}; pass --response-dir"
        )
    return candidates[0]


def response_records(response_dir: Path) -> Iterator[dict[str, Any]]:
    databases = sorted(response_dir.glob("part-*.sqlite3"))
    if not databases:
        raise FileNotFoundError(f"no response databases in {response_dir}")
    for database in databases:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            for (record_json,) in connection.execute(
                "SELECT record_json FROM responses ORDER BY img_id, model"
            ):
                yield json.loads(record_json)
        finally:
            connection.close()


def validate_staged_identity(connection: sqlite3.Connection) -> None:
    mismatch = connection.execute(
        """
        SELECT o.img_id, o.image_id, o.rel_path, o.synset,
               m.image_id, m.rel_path, m.synset
        FROM originals o LEFT JOIN manifest m USING (img_id)
        WHERE m.img_id IS NULL OR o.image_id != m.image_id
           OR o.rel_path != m.rel_path
           OR (o.synset != '' AND o.synset != m.synset)
        LIMIT 1
        """
    ).fetchone()
    if mismatch:
        raise ValueError(f"original caption/manifest identity mismatch: {mismatch}")
    missing_original = connection.execute(
        """
        SELECT m.img_id FROM manifest m LEFT JOIN originals o USING (img_id)
        WHERE o.img_id IS NULL LIMIT 1
        """
    ).fetchone()
    if missing_original:
        raise ValueError(
            f"manifest img_id={missing_original[0]} has no original caption"
        )


def distilled_rows(
    records: Iterable[dict[str, Any]],
    *,
    expected_fingerprint: str,
    expected_manifest_sha256: str,
    expected_models: set[str],
    expected_count: int,
) -> Iterator[tuple[Any, ...]]:
    for record in records:
        if record.get("schema") != DISTILL_RECORD_SCHEMA:
            raise ValueError(f"unsupported response schema: {record.get('schema')}")
        if record.get("request_fingerprint") != expected_fingerprint:
            raise ValueError("mixed request fingerprints in response staging")
        if record.get("manifest_sha256") != expected_manifest_sha256:
            raise ValueError("response was generated from a different manifest")
        if record.get("model") not in expected_models:
            raise ValueError(f"unexpected response model: {record.get('model')}")
        required_identity = (
            "manifest_index",
            "img_id",
            "id",
            "path",
            "source_path",
            "synset",
            "image_sha256",
        )
        if any(key not in record for key in required_identity) or any(
            not record.get(key)
            for key in ("id", "path", "source_path", "synset", "image_sha256")
        ):
            raise ValueError(f"incomplete response identity: {record}")
        captions = record.get("captions")
        if not isinstance(captions, list) or len(captions) != expected_count:
            actual_count = len(captions) if isinstance(captions, list) else None
            raise ValueError(
                f"response must contain exactly {expected_count} captions; "
                f"received {actual_count}"
            )
        for expected_index, caption in enumerate(captions):
            if int(caption.get("caption_index", -1)) != expected_index:
                raise ValueError(
                    "response caption indexes must be contiguous from zero"
                )
            text = str(caption["text"]).strip()
            if not text:
                raise ValueError("distilled caption is empty")
            if sha256_text(text) != caption["text_sha256"]:
                raise ValueError("distilled caption text digest mismatch")
            words = len(word_tokens(text))
            if int(caption.get("word_count", -1)) != words:
                raise ValueError("distilled caption word count mismatch")
            yield (
                int(record["img_id"]),
                int(record["manifest_index"]),
                str(record["id"]),
                relative_image_path(str(record["path"])),
                str(record["source_path"]),
                str(record["synset"]),
                str(record["model"]),
                int(caption["caption_index"]),
                text,
                words,
                str(caption["text_sha256"]),
                str(record["image_sha256"]),
            )


def validate_distilled_coverage(
    connection: sqlite3.Connection,
    *,
    models: Sequence[str],
    captions_per_model: int,
    allow_incomplete: bool,
) -> dict[str, Any]:
    manifest_count = connection.execute("SELECT COUNT(*) FROM manifest").fetchone()[0]
    mismatch = connection.execute(
        """
        SELECT d.img_id, d.manifest_index, d.image_id, d.rel_path,
               d.source_path, d.synset, m.manifest_index, m.image_id,
               m.rel_path, m.source_path, m.synset
        FROM distilled d LEFT JOIN manifest m USING (img_id)
        WHERE m.img_id IS NULL OR d.manifest_index != m.manifest_index
           OR d.image_id != m.image_id OR d.rel_path != m.rel_path
           OR d.source_path != m.source_path OR d.synset != m.synset
        LIMIT 1
        """
    ).fetchone()
    if mismatch:
        raise ValueError(f"distilled response has unknown img_id: {mismatch}")
    hash_conflict = connection.execute(
        """
        SELECT img_id, COUNT(DISTINCT image_sha256) AS hashes
        FROM distilled GROUP BY img_id HAVING hashes != 1 LIMIT 1
        """
    ).fetchone()
    if hash_conflict:
        raise ValueError(f"conflicting image hashes for img_id={hash_conflict[0]}")

    model_set = set(models)
    actual_models = {
        str(row[0])
        for row in connection.execute("SELECT DISTINCT model FROM distilled")
    }
    if actual_models.difference(model_set):
        raise ValueError(
            f"unexpected response models: {sorted(actual_models.difference(model_set))}"
        )
    complete_groups = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT img_id, model, COUNT(*) AS n FROM distilled
            GROUP BY img_id, model HAVING n = ?
        )
        """,
        (captions_per_model,),
    ).fetchone()[0]
    expected_groups = manifest_count * len(models)
    missing_groups = expected_groups - complete_groups
    if missing_groups and not allow_incomplete:
        raise ValueError(
            f"distillation is incomplete: missing {missing_groups} of "
            f"{expected_groups} image/model groups"
        )
    return {
        "manifest_images": manifest_count,
        "expected_image_model_groups": expected_groups,
        "complete_image_model_groups": complete_groups,
        "missing_image_model_groups": missing_groups,
    }


def write_merged_dataset(
    connection: sqlite3.Connection,
    output_path: Path,
    *,
    profile_name: str,
    manifest_sha256: str,
    models: Sequence[str],
) -> tuple[int, str, dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    digest = hashlib.sha256()
    caption_source_counts: dict[str, int] = {"original": 0}
    for model in models:
        caption_source_counts[model] = 0

    images = connection.execute(
        """
        SELECT m.manifest_index, m.img_id, m.image_id, m.rel_path,
               m.source_path, m.synset, o.text
        FROM manifest m JOIN originals o USING (img_id)
        ORDER BY m.manifest_index
        """
    )
    distilled = iter(
        connection.execute(
            """
            SELECT m.manifest_index, d.img_id, d.model, d.caption_index,
                   d.text, d.word_count, d.text_sha256, d.image_sha256
            FROM distilled d JOIN manifest m USING (img_id)
            ORDER BY m.manifest_index, d.model, d.caption_index
            """
        )
    )
    next_distilled = next(distilled, None)
    row_count = 0
    with temporary.open("wb") as handle:
        for (
            manifest_index,
            img_id,
            image_id,
            rel_path,
            source_path,
            synset,
            original_text,
        ) in images:
            original_digest = sha256_text(original_text)
            captions = [
                {
                    "caption_id": f"original:{original_digest[:16]}",
                    "text": original_text,
                    "source": "original",
                    "model": None,
                    "caption_index": 0,
                    "word_count": len(word_tokens(original_text)),
                    "text_sha256": original_digest,
                }
            ]
            caption_source_counts["original"] += 1
            normalized_seen = {normalized_caption(original_text)}
            image_hashes: set[str] = set()
            while next_distilled is not None and next_distilled[0] == manifest_index:
                (
                    _,
                    distilled_img_id,
                    model,
                    caption_index,
                    text,
                    word_count,
                    text_digest,
                    image_digest,
                ) = next_distilled
                if distilled_img_id != img_id:
                    raise RuntimeError("internal merge ordering mismatch")
                normalized = normalized_caption(text)
                if normalized in normalized_seen:
                    raise ValueError(
                        f"duplicate merged caption for img_id={img_id}, model={model}"
                    )
                normalized_seen.add(normalized)
                image_hashes.add(image_digest)
                captions.append(
                    {
                        "caption_id": f"api:{model}:{caption_index}:{text_digest[:16]}",
                        "text": text,
                        "source": "api_distilled",
                        "model": model,
                        "caption_index": caption_index,
                        "word_count": word_count,
                        "text_sha256": text_digest,
                        "prompt_version": PROMPT_VERSION,
                    }
                )
                caption_source_counts[model] = caption_source_counts.get(model, 0) + 1
                next_distilled = next(distilled, None)
            if len(image_hashes) > 1:
                raise RuntimeError(f"image hash conflict for img_id={img_id}")
            row = {
                "schema": MERGED_SCHEMA,
                "dataset": profile_name,
                "manifest_sha256": manifest_sha256,
                "manifest_index": manifest_index,
                "img_id": img_id,
                "id": image_id,
                "path": rel_path,
                "source_path": source_path,
                "synset": synset,
                "source_image_sha256": next(iter(image_hashes), None),
                "recaption_short": original_text,
                "caption_count": len(captions),
                "captions": captions,
            }
            encoded = (canonical_json(row) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    if next_distilled is not None:
        raise RuntimeError("unconsumed distilled rows remain after merge")
    os.replace(temporary, output_path)
    return row_count, digest.hexdigest(), caption_source_counts


def run_merge(args: argparse.Namespace) -> int:
    profile, manifest_path, originals_path, output_root = resolve_profile_paths(args)
    for path in (manifest_path, originals_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    response_dir = discover_response_dir(output_root, args.response_dir)
    run_path = response_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("dataset") != profile.name:
        raise ValueError(f"response run dataset mismatch in {run_path}")
    manifest_digest = sha256_file(manifest_path)
    if run.get("manifest_sha256") != manifest_digest:
        raise ValueError("response run manifest digest does not match current manifest")
    models = tuple(run["models"])
    captions_per_model = int(run["captions_per_model"])

    output_root.mkdir(parents=True, exist_ok=True)
    work_db = output_root / f".merge-{run['request_fingerprint']}.sqlite3"
    connection = create_merge_db(work_db)
    try:
        manifest_count = batched_insert(
            connection,
            "INSERT INTO manifest VALUES (?, ?, ?, ?, ?, ?)",
            (
                (
                    row["img_id"],
                    row["manifest_index"],
                    row["id"],
                    row["path"],
                    row["source_path"],
                    row["synset"],
                )
                for row in iter_manifest(manifest_path)
            ),
        )
        original_count = batched_insert(
            connection,
            "INSERT INTO originals VALUES (?, ?, ?, ?, ?)",
            original_rows(originals_path),
        )
        validate_staged_identity(connection)
        distilled_count = batched_insert(
            connection,
            "INSERT INTO distilled VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            distilled_rows(
                response_records(response_dir),
                expected_fingerprint=run["request_fingerprint"],
                expected_manifest_sha256=manifest_digest,
                expected_models=set(models),
                expected_count=captions_per_model,
            ),
        )
        coverage = validate_distilled_coverage(
            connection,
            models=models,
            captions_per_model=captions_per_model,
            allow_incomplete=args.allow_incomplete,
        )
        output_path = (
            Path(args.merged_output)
            if args.merged_output
            else output_root / "captions" / profile.merged_filename
        )
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to replace existing merged dataset: {output_path}; "
                "pass --overwrite after verifying the selected response run"
            )
        output_count, output_digest, source_counts = write_merged_dataset(
            connection,
            output_path,
            profile_name=profile.name,
            manifest_sha256=manifest_digest,
            models=models,
        )
    finally:
        connection.close()
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(work_db) + suffix)
            if candidate.exists():
                candidate.unlink()

    metadata = {
        "schema": MERGED_SCHEMA,
        "dataset": profile.name,
        "created_at": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "original_captions": str(originals_path),
        "original_captions_sha256": sha256_file(originals_path),
        "response_dir": str(response_dir),
        "request_fingerprint": run["request_fingerprint"],
        "models": list(models),
        "captions_per_model": captions_per_model,
        "allow_incomplete": bool(args.allow_incomplete),
        "manifest_rows": manifest_count,
        "original_rows": original_count,
        "distilled_caption_rows": distilled_count,
        "output_rows": output_count,
        "output_sha256": output_digest,
        "caption_source_counts": source_counts,
        **coverage,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    print(
        canonical_json(
            {
                "merged_output": str(output_path),
                "metadata": str(metadata_path),
                "rows": output_count,
                "sha256": output_digest,
                **coverage,
            }
        )
    )
    return 0


def run_validate(args: argparse.Namespace) -> int:
    profile, manifest_path, _, output_root = resolve_profile_paths(args)
    output_path = (
        Path(args.merged_output)
        if args.merged_output
        else output_root / "captions" / profile.merged_filename
    )
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(output_path) != metadata["output_sha256"]:
        raise ValueError("merged output SHA-256 does not match metadata")

    output_count = 0
    with output_path.open(encoding="utf-8") as output_handle:
        output_rows = (json.loads(line) for line in output_handle if line.strip())
        for manifest_row, output_row in zip(
            iter_manifest(manifest_path), output_rows, strict=True
        ):
            for key in ("manifest_index", "img_id", "id", "path", "synset"):
                if output_row.get(key) != manifest_row[key]:
                    raise ValueError(
                        f"merged identity mismatch at row {output_count}: {key}"
                    )
            captions = output_row.get("captions")
            if not isinstance(captions, list) or not captions:
                raise ValueError(
                    f"missing captions for img_id={manifest_row['img_id']}"
                )
            if captions[0].get("source") != "original":
                raise ValueError("original caption must be first")
            normalized = [normalized_caption(item["text"]) for item in captions]
            if len(normalized) != len(set(normalized)):
                raise ValueError(
                    f"duplicate captions for img_id={manifest_row['img_id']}"
                )
            for item in captions:
                if sha256_text(item["text"]) != item["text_sha256"]:
                    raise ValueError(
                        f"caption digest mismatch for img_id={manifest_row['img_id']}"
                    )
            if args.verify_images and output_row.get("source_image_sha256"):
                image_path = image_path_for_row(
                    manifest_row,
                    Path(args.image_root) if args.image_root else None,
                )
                if sha256_file(image_path) != output_row["source_image_sha256"]:
                    raise ValueError(f"image digest mismatch: {image_path}")
            output_count += 1
    if output_count != int(metadata["manifest_rows"]):
        raise ValueError(
            f"row count mismatch: validated={output_count}, "
            f"metadata={metadata['manifest_rows']}"
        )
    print(
        canonical_json(
            {
                "validated": str(output_path),
                "rows": output_count,
                "verify_images": bool(args.verify_images),
            }
        )
    )
    return 0


def run_coverage(args: argparse.Namespace) -> int:
    """Check only image/model coverage after a low-cost one-shot API run."""
    profile, manifest_path, _, _ = resolve_profile_paths(args)
    response_dir = Path(args.response_dir)
    run_path = response_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(run_path)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("dataset") != profile.name:
        raise ValueError(
            f"response dataset={run.get('dataset')} does not match {profile.name}"
        )
    manifest_digest = sha256_file(manifest_path)
    if run.get("manifest_sha256") != manifest_digest:
        raise ValueError("coverage manifest SHA-256 does not match generation run")
    models = tuple(str(model) for model in run["models"])
    expected_caption_count = int(run["captions_per_model"])
    response_dbs = sorted(response_dir.glob("part-*.sqlite3"))
    if not response_dbs:
        raise FileNotFoundError(f"no response SQLite shards under {response_dir}")

    with tempfile.TemporaryDirectory(prefix=".coverage-", dir=response_dir) as temp_dir:
        audit_db = Path(temp_dir) / "coverage.sqlite3"
        connection = sqlite3.connect(audit_db)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.executescript(
            """
            CREATE TABLE expected (img_id INTEGER PRIMARY KEY);
            CREATE TABLE actual (
                img_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                PRIMARY KEY (img_id, model)
            );
            CREATE INDEX actual_model ON actual(model);
            """
        )
        expected_images = batched_insert(
            connection,
            "INSERT INTO expected VALUES (?)",
            ((int(row["img_id"]),) for row in iter_manifest(manifest_path)),
        )

        observed_valid_groups = 0
        inserted_groups = 0
        error_rows = 0
        invalid_caption_count_groups = 0
        invalid_caption_count_samples: list[dict[str, Any]] = []
        for response_db in response_dbs:
            source = sqlite3.connect(f"file:{response_db.resolve()}?mode=ro", uri=True)
            try:
                error_rows += int(
                    source.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
                )
                cursor = source.execute(
                    "SELECT img_id, model, record_json FROM responses"
                )
                while batch := cursor.fetchmany(10_000):
                    valid_batch: list[tuple[int, str]] = []
                    for img_id, model, record_json in batch:
                        try:
                            record = json.loads(record_json)
                        except (TypeError, json.JSONDecodeError):
                            record = {}
                        captions = record.get("captions")
                        actual_count = len(captions) if isinstance(captions, list) else None
                        if not response_has_exact_caption_count(
                            record, expected_caption_count
                        ):
                            invalid_caption_count_groups += 1
                            if len(invalid_caption_count_samples) < args.sample_limit:
                                invalid_caption_count_samples.append(
                                    {
                                        "img_id": int(img_id),
                                        "model": str(model),
                                        "caption_count": actual_count,
                                    }
                                )
                            continue
                        valid_batch.append((int(img_id), str(model)))
                    observed_valid_groups += len(valid_batch)
                    before = connection.total_changes
                    connection.executemany(
                        "INSERT OR IGNORE INTO actual VALUES (?, ?)", valid_batch
                    )
                    inserted_groups += connection.total_changes - before
            finally:
                source.close()
        connection.commit()

        expected_model_groups = expected_images * len(models)
        per_model: dict[str, dict[str, Any]] = {}
        sample_missing: list[dict[str, Any]] = []
        missing_groups = 0
        for model in models:
            present = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM actual a
                    INNER JOIN expected e USING (img_id)
                    WHERE a.model=?
                    """,
                    (model,),
                ).fetchone()[0]
            )
            missing = expected_images - present
            missing_groups += missing
            rows = connection.execute(
                """
                SELECT e.img_id FROM expected e
                LEFT JOIN actual a ON a.img_id=e.img_id AND a.model=?
                WHERE a.img_id IS NULL LIMIT ?
                """,
                (model, args.sample_limit),
            ).fetchall()
            per_model[model] = {
                "present": present,
                "missing": missing,
                "coverage": present / expected_images,
            }
            sample_missing.extend(
                {"img_id": int(row[0]), "model": model} for row in rows
            )

        placeholders = ",".join("?" for _ in models)
        unexpected_groups = int(
            connection.execute(
                f"""
                SELECT COUNT(*) FROM actual a
                LEFT JOIN expected e USING (img_id)
                WHERE e.img_id IS NULL OR a.model NOT IN ({placeholders})
                """,
                models,
            ).fetchone()[0]
        )
        connection.close()

    duplicate_groups = observed_valid_groups - inserted_groups
    expected_shards = int(run.get("num_shards", 1))
    complete = not any(
        (
            missing_groups,
            unexpected_groups,
            duplicate_groups,
            error_rows,
            invalid_caption_count_groups,
            len(response_dbs) != expected_shards,
        )
    )
    report = {
        "schema": "imagenet_caption_distill_coverage_v1",
        "status": "complete" if complete else "incomplete",
        "dataset": profile.name,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "response_dir": str(response_dir),
        "models": list(models),
        "expected_images": expected_images,
        "expected_image_model_groups": expected_model_groups,
        "present_unique_groups": inserted_groups - unexpected_groups,
        "missing_groups": missing_groups,
        "unexpected_groups": unexpected_groups,
        "duplicate_groups_across_shards": duplicate_groups,
        "error_rows": error_rows,
        "required_captions_per_group": expected_caption_count,
        "invalid_caption_count_groups": invalid_caption_count_groups,
        "invalid_caption_count_samples": invalid_caption_count_samples,
        "expected_shard_databases": expected_shards,
        "found_shard_databases": len(response_dbs),
        "per_model": per_model,
        "sample_missing": sample_missing[: args.sample_limit],
    }
    output_path = (
        Path(args.coverage_output)
        if args.coverage_output
        else response_dir / "coverage.json"
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(canonical_json({"coverage": str(output_path), **report}))
    return 0 if complete else 1


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=sorted(DATASET_PROFILES), required=True)
    parser.add_argument("--manifest", help="override the profile manifest")
    parser.add_argument("--output-root", help="override the public output root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capabilities = subparsers.add_parser(
        "capabilities", help="print known test_api.py model capabilities"
    )
    capabilities.set_defaults(func=None)

    generate = subparsers.add_parser(
        "generate", help="generate resumable teacher captions"
    )
    add_profile_arguments(generate)
    generate.add_argument(
        "--models",
        nargs="+",
        default=["MiniMax-M3"],
    )
    generate.add_argument("--allow-unverified-models", action="store_true")
    generate.add_argument("--base-url", default=DEFAULT_BASE_URL)
    generate.add_argument("--api-key-env", default="SII_API_KEY")
    generate.add_argument(
        "--synset-mapping",
        default="/inspire/dataset/imagenet/v1/LOC_synset_mapping.txt",
    )
    generate.add_argument("--no-class-hint", action="store_true")
    generate.add_argument("--image-root")
    generate.add_argument("--captions-per-model", type=int, default=3)
    generate.add_argument("--min-words", type=int, default=32)
    generate.add_argument(
        "--max-words",
        type=int,
        default=60,
        help="preferred prompt length, not a hard validation ceiling",
    )
    generate.add_argument("--max-jaccard", type=float, default=0.82)
    generate.add_argument(
        "--temperature",
        type=float,
        help="override provider-aware defaults (normally leave unset)",
    )
    generate.add_argument(
        "--thinking",
        choices=("auto", "disabled", "enabled"),
        default="auto",
        help="auto applies fast per-model defaults",
    )
    generate.add_argument("--thinking-budget-tokens", type=int, default=1024)
    generate.add_argument("--max-tokens", type=int, default=1400)
    generate.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="global in-flight request limit when --model-concurrency is omitted",
    )
    generate.add_argument(
        "--model-concurrency",
        action="append",
        metavar="MODEL=N",
        help=(
            "independent in-flight limit for a selected model; repeat for every "
            "model. When provided, these limits replace --concurrency."
        ),
    )
    generate.add_argument(
        "--model-rps",
        action="append",
        metavar="MODEL=RATE",
        help=(
            "maximum API attempt starts per second for each selected model; "
            "repeat for every model. Conservative provider defaults apply when omitted."
        ),
    )
    generate.add_argument("--timeout", type=float, default=45.0)
    generate.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="attempts per missing image/model group before preserving an error row",
    )
    generate.add_argument("--retry-base-seconds", type=float, default=1.0)
    generate.add_argument("--retry-max-seconds", type=float, default=30.0)
    generate.add_argument("--commit-every", type=int, default=25)
    generate.add_argument(
        "--circuit-window",
        type=int,
        default=20,
        help="completed image/model groups in the rolling failure window",
    )
    generate.add_argument(
        "--circuit-failure-rate",
        type=float,
        default=0.5,
        help="stop scheduling a model when this rolling failure rate is reached",
    )
    generate.add_argument(
        "--runtime-log",
        help="append flushed per-request JSONL events to this path",
    )
    generate.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the terminal progress bar; runtime logging remains enabled",
    )
    generate.add_argument(
        "--reuse-response-dir",
        action="append",
        help=(
            "import compatible successful records from another response directory; "
            "repeat for multiple sources"
        ),
    )
    generate.add_argument("--num-shards", type=int, default=1)
    generate.add_argument("--shard-id", type=int, default=0)
    generate.add_argument("--start-index", type=int, default=0)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--dry-run", action="store_true")
    generate.set_defaults(func=run_generation)

    coverage = subparsers.add_parser(
        "coverage", help="check only missing/error image-model groups"
    )
    add_profile_arguments(coverage)
    coverage.add_argument("--response-dir", required=True)
    coverage.add_argument("--coverage-output")
    coverage.add_argument("--sample-limit", type=int, default=20)
    coverage.set_defaults(func=run_coverage)

    merge = subparsers.add_parser(
        "merge", help="strictly join originals and completed teacher captions"
    )
    add_profile_arguments(merge)
    merge.add_argument("--original-captions")
    merge.add_argument("--response-dir")
    merge.add_argument("--merged-output")
    merge.add_argument("--allow-incomplete", action="store_true")
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(func=run_merge)

    validate = subparsers.add_parser(
        "validate", help="validate a published merged JSONL against its manifest"
    )
    add_profile_arguments(validate)
    validate.add_argument("--merged-output")
    validate.add_argument("--verify-images", action="store_true")
    validate.add_argument("--image-root")
    validate.set_defaults(func=run_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "capabilities":
        print(json.dumps(MODEL_CAPABILITIES, ensure_ascii=False, indent=2))
        return 0
    result = args.func(args)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
