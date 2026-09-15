"""Independent, durable Codex queue for explicitly authorized SII timeouts only."""
from __future__ import annotations

import asyncio
import copy
import gzip
import json
import os
from pathlib import Path
import sqlite3
import time

from data_synthesis.clients import CodexFallback
from data_synthesis.io import atomic_json, dumps
from data_synthesis.review_contract import REVIEW_SCHEMA, parse_codex_review

POLICY_VERSION = "b512-consecutive-timeouts-codex-v1"
MODEL = "gpt-5.6-sol"


def load_timeout_policy(root, contract):
    root = Path(root).resolve()
    path = root / "timeout_fallback_policy.json"
    if not path.exists():
        return None
    policy = json.loads(path.read_text())
    if (policy.get("version") != POLICY_VERSION or policy.get("authorized_by") != "user"
            or not policy.get("user_instructions") or policy.get("parent_root") != str(root)
            or policy.get("parent_contract") != contract or policy.get("model") != MODEL
            or policy.get("reasoning_effort") != "low" or policy.get("consecutive_timeouts") != 2
            or policy.get("max_attempts") != 1 or not 1 <= policy.get("concurrency", 0) <= 32
            or policy.get("timeout_seconds", 0) <= 0 or policy.get("compute_hashes") is not False):
        raise ValueError("timeout fallback must bind explicit user authorization, fixed scope and gpt-5.6-sol/low")
    frozen = root / "timeout_fallback" / "policy.json"
    if frozen.exists() and json.loads(frozen.read_text()) != policy:
        raise ValueError("frozen timeout fallback policy changed")
    atomic_json(frozen, policy)
    return policy


def repeated_timeouts(db, key, number, threshold=2):
    if number < threshold:
        return False
    for attempt in range(number - threshold + 1, number + 1):
        row = db.execute("SELECT raw_gzip FROM attempts WHERE key=? AND number=?", (key, attempt)).fetchone()
        if row is None:
            return False
        raw = json.loads(gzip.decompress(row[0]))
        error_class = str(raw.get("error", "")).split(":", 1)[0]
        if (raw.get("backend", "sii") != "sii" or raw.get("error_type") != "transport"
                or "Timeout" not in error_class):
            return False
    return True


def discover_timeouts(parent_root, state, sealed):
    """Read only deferred/terminal source rows; never steal a live SII request."""
    for path in sorted((Path(parent_root) / "repairs").glob("*/review.sqlite3")):
        if str(path) in sealed:
            continue
        # Check before scanning: a manifest written during the read is revisited.
        complete = (path.parent / "manifest.json").exists()
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as source:
            for key, payload, attempts in source.execute(
                    "SELECT key,payload,attempts FROM items WHERE status IN ('deferred_codex','failed')"):
                if state.db.execute("SELECT 1 FROM items WHERE key=?", (key,)).fetchone():
                    continue
                if not repeated_timeouts(source, key, attempts):
                    continue
                item = json.loads(payload)
                item["timeout_handoff"] = {"evidence_db": str(path), "sii_attempts": attempts,
                    "reason": "two consecutive SII transport timeouts", "policy": POLICY_VERSION}
                item["issues"] = ["SII repeatedly timed out; complete the grounded image review directly."]
                state.admit([item])
        if complete:
            sealed.add(str(path))


def export_timeouts(state, tokenizer, contract):
    output = state.root / "reviewed.jsonl.gz"
    temporary = output.with_suffix(".tmp")
    records = 0
    with gzip.open(temporary, "wt", compresslevel=1) as stream:
        for row in state.db.execute("SELECT * FROM items ORDER BY key"):
            item = json.loads(row["payload"])
            result = json.loads(row["result"]) if row["result"] else None
            if result is not None:
                blob = state.db.execute("SELECT raw_gzip FROM attempts WHERE key=? AND number=?",
                                        (row["key"], row["attempts"])).fetchone()[0]
                raw = json.loads(gzip.decompress(blob))
                if parse_codex_review(raw, item, tokenizer, expected_contract=contract) != result:
                    raise ValueError("Codex result differs from raw CLI evidence")
            stream.write(dumps({"key": row["key"], "original_key": item["original_key"],
                "source": item["row"]["source"], "group": item["group"], "source_run": item["source_run"],
                "source_path": item["view"]["source_path"], "status": row["status"], "attempts": row["attempts"],
                "error": row["error"], "result": result, "model": MODEL, "reasoning_effort": "low",
                "backend": "codex_fallback", "contract": contract, "timeout_handoff": item["timeout_handoff"],
                "evidence_db": str(state.root / "review.sqlite3")}) + "\n")
            records += 1
    temporary.replace(output)
    with gzip.open(output, "rt") as stream:
        checked = sum(1 for line in stream if json.loads(line))
    if records != checked or records != sum(state.counts().values()):
        raise ValueError("timeout fallback export count mismatch")
    return {"records": records, "counts": state.counts(), "output": str(output), "bytes": output.stat().st_size,
            "model": MODEL, "reasoning_effort": "low", "compute_hashes": False,
            "audit": "all_Codex_results_reparsed_against_raw_CLI_events;gzip_full_read;counts"}


async def run_timeout_fallback(parent_root, policy, config, tokenizer, stop, primary_done,
                               progress, prompt_factory, *, client=None):
    # Import locally: review_farm uses repeated_timeouts when committing SII rows.
    from data_synthesis.review_farm import ReviewDB
    root = Path(parent_root) / "timeout_fallback"
    scoped_config = copy.deepcopy(config)
    scoped_config["timeout_fallback_policy"] = policy
    state = ReviewDB(root, scoped_config)
    contract = policy["parent_contract"]
    active, sealed = {}, set()
    last_scan = last_report = 0
    fatal = None

    def accept(key, number, raw):
        item = json.loads(state.db.execute("SELECT payload FROM items WHERE key=?", (key,)).fetchone()[0])
        try:
            result = parse_codex_review(raw, item, tokenizer, expected_contract=contract)
        except Exception as exc:
            error = raw.get("error") or f"{type(exc).__name__}: {str(exc)[:600]}"
            state.db.execute("UPDATE items SET status='failed',error=? WHERE key=?", (error, key))
        else:
            status = "unusable" if result["review"]["decision"] == "unusable" else "ready"
            state.db.execute("UPDATE items SET status=?,result=?,error=NULL WHERE key=?", (status, dumps(result), key))
        state.db.commit()

    def report(phase):
        calls = state.db.execute("SELECT coalesce(sum(attempts),0) FROM items").fetchone()[0]
        progress.clear()
        progress.update(state=phase, pid=os.getpid(), updated_at=time.time(), counts=state.counts(),
                        active_codex=len(active), codex_calls=calls, model=MODEL, reasoning_effort="low",
                        concurrency=policy["concurrency"], compute_hashes=False, fatal=fatal)
        atomic_json(root / "status.json", progress)
        return dict(progress)

    async def invoke(item, number):
        try:
            return await client.generate(item, number)
        except Exception as exc:
            return {"status": "failed", "backend": "codex_fallback", "error_type": "local",
                    "error": f"{type(exc).__name__}: {str(exc)[:600]}"}

    try:
        # Recover raw replies before considering any new CLI invocation.
        for row in state.db.execute("SELECT key,attempts FROM items WHERE status='running'").fetchall():
            blob = state.db.execute("SELECT raw_gzip FROM attempts WHERE key=? AND number=?", tuple(row)).fetchone()
            if blob:
                accept(row["key"], row["attempts"], json.loads(gzip.decompress(blob[0])))
            else:
                state.db.execute("UPDATE items SET status='failed',error='interrupted CLI attempt; retry ceiling preserved' WHERE key=?",
                                 (row["key"],))
        state.db.commit()
        while True:
            now = time.time()
            if not stop.is_set() and (now - last_scan >= 5 or primary_done.is_set()):
                discover_timeouts(parent_root, state, sealed)
                last_scan = now
            if not stop.is_set():
                rows = state.db.execute("SELECT key,payload,attempts FROM items WHERE status='pending' LIMIT ?",
                                        (max(0, policy["concurrency"] - len(active)),)).fetchall()
                if rows and client is None:
                    client = CodexFallback(policy, prompt_factory=prompt_factory, contract_id=contract, schema=REVIEW_SCHEMA)
                for row in rows:
                    number = row["attempts"] + 1
                    state.db.execute("UPDATE items SET status='running',attempts=? WHERE key=?", (number, row["key"]))
                    state.db.commit()
                    task = asyncio.create_task(invoke(json.loads(row["payload"]), number))
                    active[task] = (row["key"], number)
            if now - last_report >= 5:
                report("draining" if stop.is_set() else "running")
                last_report = now
            if active:
                done, _ = await asyncio.wait(active, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    key, number = active.pop(task)
                    raw = task.result()
                    state.db.execute("INSERT OR REPLACE INTO attempts VALUES (?,?,?)",
                                     (key, number, gzip.compress(dumps(raw).encode(), compresslevel=1)))
                    state.db.commit()
                    accept(key, number, raw)
            elif stop.is_set():
                return report("interrupted")
            elif primary_done.is_set() and not state.counts().get("pending"):
                manifest = export_timeouts(state, tokenizer, contract)
                value = report("needs_attention" if state.counts().get("failed") else "completed")
                atomic_json(root / "manifest.json", {**manifest, **value})
                return value
            else:
                await asyncio.sleep(0.5)
    except BaseException as exc:
        fatal = type(exc).__name__
        stop.set()
        # Never abandon completed CLI work because the producer or a scan failed.
        for task, (key, number) in list(active.items()):
            raw = await task
            state.db.execute("INSERT OR REPLACE INTO attempts VALUES (?,?,?)",
                             (key, number, gzip.compress(dumps(raw).encode(), compresslevel=1)))
            state.db.commit()
            accept(key, number, raw)
            active.pop(task)
        report("interrupted")
        raise
    finally:
        if client:
            await client.close()
        state.close()
