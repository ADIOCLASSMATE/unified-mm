"""Adaptive SII queue; reuse has no teacher call and Codex is the last attempt."""
from __future__ import annotations

import asyncio
from collections import Counter
import json
from pathlib import Path
import random
import signal
import time

from data_synthesis.clients import CodexFallback, SIIClient
from data_synthesis.config import fingerprint, load_sii_settings
from data_synthesis.contract import CONTRACT_HASH, parse_pair
from data_synthesis.io import atomic_json, dumps, sha
from data_synthesis.reuse import choose_reuse
from data_synthesis.state import State

PIPELINE_VERSION = "b512-reuse-sii-fallback-v1"


def qualification(config, path):
    if not path or not Path(path).is_file():
        raise ValueError("bulk SII synthesis needs the one-time visual qualification report; use a bounded pilot first")
    report = json.loads(Path(path).read_text())
    if (report.get("contract_hash") != CONTRACT_HASH or report.get("runtime_sha256") != fingerprint(config)
            or report.get("status") != "passed" or report.get("reviewer") != "GPT-6"
            or report.get("image_size") != 512 or not report.get("visual_review", False)):
        raise ValueError("qualification must cover this runtime, exact 512px views and a real visual review")
    for model in config["sii"]["vision_models"]:
        if not report.get("models", {}).get(model, {}).get("qualified"):
            raise ValueError("a configured SII vision model has not passed qualification")
    return report


def accept_raw(state, item, ident, raw, tokenizer, config):
    backend = raw["backend"]
    if raw.get("image_id") != item["key"] or raw.get("view_sha256") != item["view"]["view_sha256"]:
        raise ValueError("response/attachment provenance mismatch")
    if raw.get("contract_hash") != CONTRACT_HASH or raw.get("image_attached") is not True:
        raise ValueError("missing grounded request contract")
    if backend == "codex_fallback":
        from scripts.legacy.distill_b512_codex import inspect_events
        # This verifies the CLI event stream contains the same final answer and no tool actions.
        inspect_events(raw.get("events", ""), raw.get("output_text", ""))
        if item["api_attempts"] < config["sii"]["max_attempts"]:
            raise ValueError("Codex cannot finalize an item before its SII attempts are exhausted")
    pair = parse_pair(raw, item["key"], tokenizer)
    state.check_test_prompts(pair)
    if not all(pair["usable"].values()):
        raise ValueError("model could not provide usable paired text")
    for fact in item["row"].get("verified_facts", []):
        if fact.get("verified") is not True or fact.get("view_sha256") != item["view"]["view_sha256"]:
            continue
        if fact.get("type") == "count" and fact.get("exhaustive_for_referent") is True:
            observations = {c["entity"].casefold(): c["count"] for c in pair["observations"]["counts"]}
            if fact["entity"].casefold() in observations and observations[fact["entity"].casefold()] != fact["count"]:
                raise ValueError("generated count conflicts with verified final-view annotation")
    model = raw["requested_model"]
    evidence = {"pipeline": PIPELINE_VERSION, "route": backend, "attempt_id": ident,
                "generator_models": {"i2t": model, "t2i": model}, "contract_hash": CONTRACT_HASH,
                "api_attempts": item["api_attempts"], "codex_attempts": item["codex_attempts"],
                "view_sha256": item["view"]["view_sha256"],
                "fallback_reason": item.get("error") if backend == "codex_fallback" else None,
                "semantic_accuracy_independently_verified": False}
    state.succeed(item["key"], pair, evidence, ident)
    return pair


async def run(root, config, *, tokenizer, sii=None, codex=None, max_items=None, qualification_path=None):
    with State(root, config) as state:
        frozen = state.meta("frozen")
        if not frozen:
            raise ValueError("freeze the completed image pool before synthesis")
        if not frozen["pilot"] and config["require_qualification_for_bulk"]:
            qualification(config, qualification_path)
        if max_items is not None and max_items < 1:
            raise ValueError("max_items must be positive")
        limit = min(max_items or 32, 256) if frozen["pilot"] else max_items
        state.recover()
        api, fallback = config["sii"], config["codex_fallback"]
        scheduler = state.meta("scheduler", {"effective": api["concurrency_start"], "healthy": 0,
                                             "failures": 0, "circuit_until": 0, "probe": False})
        scheduler["probe"] = False
        scheduler["effective"] = min(max(1, scheduler["effective"]), api["concurrency_max"])
        active, touched = {}, set()
        counts = Counter()
        fallback_started = 0
        stop, blocked = asyncio.Event(), None
        loop = asyncio.get_running_loop()
        installed = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
                installed.append(sig)
            except (NotImplementedError, RuntimeError):
                pass
        started = time.time()
        last_report = 0.0

        def report(status):
            nonlocal last_report
            state.set_meta("scheduler", scheduler)
            state.db.commit()
            value = {"state": status, "pipeline": PIPELINE_VERSION, "pilot": frozen["pilot"],
                     "counts": state.counts(), "run_counters": dict(counts), "scheduler": scheduler,
                     "active_sii": sum(backend == "sii" for _, backend in active.values()),
                     "active_codex": sum(backend == "codex_fallback" for _, backend in active.values()),
                     "fallback_started": fallback_started, "blocked_reason": blocked,
                     "started_at": started, "updated_at": time.time()}
            atomic_json(state.root / "status.json", value)
            last_report = time.time()
            return value

        def finish(key, ident, backend, raw):
            nonlocal blocked
            item = state.item(key)
            try:
                accept_raw(state, item, ident, raw, tokenizer, config)
            except Exception as exc:
                failure = raw.get("error_type", "content") if raw.get("status") != "completed" else "content"
                message = raw.get("error") or f"{failure}: {type(exc).__name__}: {str(exc)[:500]}"
                if backend == "codex_fallback":
                    exhausted = item["codex_attempts"] >= fallback["max_attempts"]
                    state.fail(key, "failed" if exhausted else "fallback", message, ident=ident)
                else:
                    delay = min(api["retry_max_seconds"], api["retry_base_seconds"] * 2 ** (item["api_attempts"] - 1))
                    exhausted = item["api_attempts"] >= api["max_attempts"]
                    state.fail(key, "fallback" if exhausted else "retry", message,
                               time.time() + delay + random.uniform(0, delay * 0.2), ident)
                    if failure == "configuration":
                        blocked = "SII authentication/endpoint configuration failure; fix shell SII variables before resuming"
                        stop.set()
                    if failure == "transport":
                        scheduler["failures"] += 1
                        scheduler["healthy"] = 0
                        if scheduler["failures"] >= api["circuit_failures"] or scheduler["probe"]:
                            scheduler["effective"] = max(1, scheduler["effective"] // 2)
                            scheduler["circuit_until"] = time.time() + api["circuit_seconds"]
                counts[f"{backend}_failed_attempts"] += 1
            else:
                counts[f"{backend}_accepted"] += 1
                if backend == "sii":
                    scheduler.update(failures=0, circuit_until=0)
            if backend == "sii":
                # A syntactically poor answer is healthy transport, not an outage.
                if raw.get("status") in {"completed", "incomplete"}:
                    scheduler.update(failures=0, circuit_until=0)
                    scheduler["healthy"] += 1
                    if scheduler["healthy"] >= max(8, scheduler["effective"]):
                        scheduler["effective"] = min(api["concurrency_max"], scheduler["effective"] + 8)
                        scheduler["healthy"] = 0
                scheduler["probe"] = False

        async def invoke(client, item, number, backend):
            try:
                return await client.generate(item, number)
            except Exception as exc:
                return {"backend": backend, "status": "failed", "error_type": "local",
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}", "image_id": item["key"],
                        "view_sha256": item["view"]["view_sha256"], "contract_hash": CONTRACT_HASH}

        try:
            while True:
                now = time.time()
                # Replay already received results before spending another request.
                for row in state.db.execute("SELECT key,attempt_id FROM items WHERE status='received' LIMIT 128").fetchall():
                    raw = state.raw(row["attempt_id"])
                    finish(row["key"], row["attempt_id"], raw["backend"], raw)
                if not stop.is_set():
                    rows = state.db.execute("SELECT key FROM items WHERE status IN ('pending','retry','fallback') AND retry_at<=? ORDER BY created_at LIMIT 512", (now,)).fetchall()
                    for row in rows:
                        key = row["key"]
                        if limit is not None and key not in touched and len(touched) >= limit:
                            continue
                        item = state.item(key)
                        if item["status"] == "pending":
                            pair, evidence, issues = choose_reuse(item, tokenizer, config)
                            if pair is not None:
                                try:
                                    state.check_test_prompts(pair)
                                except ValueError as exc:
                                    pair = None
                                    issues.append(str(exc))
                            if pair is not None:
                                state.succeed(key, pair, {"pipeline": PIPELINE_VERSION, **evidence,
                                                         "view_sha256": item["view"]["view_sha256"], "contract_hash": CONTRACT_HASH}, commit=False)
                                touched.add(key)
                                counts[evidence["route"]] += 1
                                continue
                            item["issues"] = issues
                        else:
                            item["issues"] = [item["error"]] if item["error"] else []
                        previous_raw = state.raw(item["attempt_id"]) if item["attempt_id"] else None
                        item["candidate"] = (previous_raw.get("output_text") if previous_raw else
                                             item["row"].get("caption_candidates", []))
                        backend = "sii" if item["api_attempts"] < api["max_attempts"] else "codex_fallback"
                        inflight = sum(b == backend for _, b in active.values())
                        if backend == "sii":
                            if scheduler["circuit_until"] > now or scheduler["probe"] or inflight >= scheduler["effective"]:
                                continue
                            if scheduler["circuit_until"]:
                                if inflight:
                                    continue
                                scheduler["probe"] = True
                            if sii is None:
                                settings = load_sii_settings()
                                provider = settings.public()
                                prior = state.meta("provider")
                                if prior and prior != provider:
                                    raise ValueError("SII endpoint changed within an existing run")
                                state.set_meta("provider", provider)
                                sii = SIIClient(settings, api)
                            client = sii
                        else:
                            if not fallback["enabled"]:
                                state.fail(key, "failed", "SII attempts exhausted; Codex fallback disabled")
                                continue
                            if item["codex_attempts"] >= fallback["max_attempts"]:
                                state.fail(key, "failed", "Codex attempts exhausted")
                                continue
                            if fallback_started >= fallback["max_items_per_run"]:
                                blocked = "Codex fallback run budget reached; remaining items preserved"
                                continue
                            if inflight >= fallback["concurrency"]:
                                continue
                            codex = codex or CodexFallback(fallback)
                            client = codex
                        ident, number = state.claim(key, backend)
                        touched.add(key)
                        if backend == "codex_fallback":
                            fallback_started += 1
                        task = asyncio.create_task(invoke(client, item, number, backend))
                        active[task] = (ident, backend)
                    # Commit a bounded local reuse batch, not two million fsyncs.
                    state.db.commit()
                if time.time() - last_report > 5:
                    report("draining" if stop.is_set() else "running")
                if active:
                    done, _ = await asyncio.wait(active, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        ident, backend = active.pop(task)
                        raw = task.result()
                        state.receive(ident, raw)
                        key = state.db.execute("SELECT item_key FROM attempts WHERE id=?", (ident,)).fetchone()[0]
                        finish(key, ident, backend, raw)
                    continue
                if stop.is_set() or blocked:
                    break
                remaining = state.db.execute("SELECT key,retry_at FROM items WHERE status IN ('pending','retry','fallback','received') ORDER BY retry_at LIMIT 512").fetchall()
                eligible = [row for row in remaining if limit is None or row["key"] in touched or len(touched) < limit]
                if not eligible:
                    break
                await asyncio.sleep(min(1, max(0.01, min(r["retry_at"] for r in eligible) - time.time())))
            pending = state.counts()
            complete = not any(n for status, n in pending.items() if status != "ready")
            return report("completed" if complete else "blocked" if blocked else "interrupted" if stop.is_set() else "incomplete")
        finally:
            for sig in installed:
                loop.remove_signal_handler(sig)
            # Ordinary SIGINT/SIGTERM stops refill and drains bounded in-flight calls.
            if active:
                results = await asyncio.gather(*active, return_exceptions=True)
                for (task, (ident, backend)), raw in zip(list(active.items()), results):
                    if isinstance(raw, dict):
                        state.receive(ident, raw)
                        key = state.db.execute("SELECT item_key FROM attempts WHERE id=?", (ident,)).fetchone()[0]
                        finish(key, ident, backend, raw)
            for client in (sii, codex):
                if client is not None:
                    await client.close()
            state.db.commit()
