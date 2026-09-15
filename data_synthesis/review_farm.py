"""SII-only full review of a fixed prepared selection; bounded sharded state."""
from __future__ import annotations

import asyncio
import bisect
from collections import Counter
import fcntl
import gzip
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import time

from data_synthesis.clients import SIIClient
from data_synthesis.config import load_sii_settings
from data_synthesis.io import atomic_json, dumps
from data_synthesis.review_contract import VERSION, parse_review, prompt_for_review

GROUPS = ("long_cc12m", "long_sa1b", "long_journeydb", "pixmo_cap",
          "pixmo_points", "openimages", "pd3m", "textcaps")
SCHEDULER_VERSION = "sii-circuit-generation-v2"


def read_batch(batch, *, keep_source_fields=False):
    path = Path(batch["source_run"]) / "state.sqlite3"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT key,row,view FROM tasks WHERE status='prepared' ORDER BY key").fetchall()
    if len(rows) != batch["records"]:
        raise ValueError(f"frozen batch count changed: {batch['batch_id']}")
    result = []
    for key, row_json, view_json in rows:
        row, view = json.loads(row_json), json.loads(view_json)
        view.update(hashes_computed=False, view_sha256=None, source_sha256=None)
        group = (row.get("selection_bucket", "").replace("blip3o_long_", "long_")
                 if batch["cohort"] == "blip3o_long" else
                 "pixmo_cap" if batch["cohort"].startswith("pixmo_cap_") else batch["cohort"])
        # Retain the complete original annotation in source_run, not thousands
        # of redundant point/relationship annotations in every review request.
        kept = {k: row[k] for k in ("source", "source_id", "split", "caption_candidates",
                                   "selection_bucket", "capabilities") if k in row}
        if keep_source_fields:
            kept = row
        result.append({"key": batch["batch_id"] + ":" + key, "original_key": key,
                       "row": kept, "view": view, "group": group,
                       "cohort": batch["cohort"], "source_run": batch["source_run"],
                       "batch_id": batch["batch_id"]})
    return result


def load_selection(path):
    selection = json.loads(Path(path).read_text())
    if selection["downloads_enabled"] or selection["compute_hashes"]:
        raise ValueError("review requires the user's fixed no-hash image selection")
    index = Path(selection["batch_index"])
    if index.stat().st_size != selection["batch_index_bytes"]:
        raise ValueError("fixed batch index size changed")
    batches = [json.loads(line) for line in index.open()]
    if sum(b["records"] for b in batches) != selection["images"]:
        raise ValueError("fixed source count changed")
    if len({b["batch_id"] for b in batches}) != len(batches):
        raise ValueError("duplicate prepared batch identity")
    return selection, batches


def prepare_pilot(selection_path, root, per_group=40):
    root = Path(root)
    target = root / "pilot_inputs.jsonl"
    if target.exists():
        return target
    selection, batches = load_selection(selection_path)
    rng = random.Random(20260914)
    pools = {}
    for b in batches:
        if b["records"]:
            family = "pixmo_cap" if b["cohort"].startswith("pixmo_cap_") else b["cohort"]
            pools.setdefault(family, []).append(b)
    sampled, seen = {g: [] for g in GROUPS}, set()
    for family, pool in pools.items():
        wanted = GROUPS[:3] if family == "blip3o_long" else (family,)
        cumul = []
        total = 0
        for b in pool:
            total += b["records"]
            cumul.append(total)
        cache = {}
        while any(len(sampled[g]) < per_group for g in wanted):
            offset = rng.randrange(total)
            index = bisect.bisect_right(cumul, offset)
            if index not in cache:
                cache[index] = read_batch(pool[index])
            item = cache[index][offset - (cumul[index - 1] if index else 0)]
            if item["key"] in seen or len(sampled[item["group"]]) >= per_group:
                continue
            seen.add(item["key"])
            sampled[item["group"]].append(item)
    items = []
    for group in GROUPS:
        for i, item in enumerate(sampled[group]):
            item = {**item, "sample_id": f"{group}-{i:02d}", "test_kind": "natural"}
            items.append(item)
    # Explicitly synthetic negative candidates, never included in production.
    challenges = []
    for group in GROUPS:
        for item in [v for v in items if v["group"] == group][:10]:
            challenges.append({**item, "key": item["key"] + ":negative",
                "sample_id": item["sample_id"] + "-negative", "test_kind": "negative_control",
                "candidate": [{"text": "A completely blank solid white square, with no objects, writing, colors, texture, or visible scene."}]})
    root.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with tmp.open("w") as f:
        for item in items + challenges:
            f.write(dumps(item) + "\n")
    tmp.replace(target)
    atomic_json(root / "pilot_sampling.json", {"seed": 20260914,
        "selection": str(Path(selection_path).resolve()), "population": selection["images"],
        "natural_images": len(items), "negative_control_requests": len(challenges),
        "natural_groups": {g: len(sampled[g]) for g in GROUPS},
        "sampling": "uniform accepted-row draws within each of eight source strata, without replacement",
        "content_hashes": False, "negative_controls_are_not_training_data": True})
    return target


class ReviewDB:
    def __init__(self, root, config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "controller.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.db = sqlite3.connect(self.root / "review.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT);
          CREATE TABLE IF NOT EXISTS items(key TEXT PRIMARY KEY,payload TEXT,status TEXT,
            attempts INTEGER DEFAULT 0,retry_at REAL DEFAULT 0,error TEXT,result TEXT);
          CREATE INDEX IF NOT EXISTS ready_queue ON items(status,retry_at);
          CREATE TABLE IF NOT EXISTS attempts(key TEXT,number INTEGER,raw_gzip BLOB,
            PRIMARY KEY(key,number));
        """)
        contract = {"version": config.get("review_contract", VERSION), "config": config}
        previous = self.meta("contract")
        if previous is not None and previous != contract:
            self.close()
            raise ValueError("review contract changed; start a new run version")
        self.set_meta("contract", contract)
        self.db.commit()

    def meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, dumps(value)))

    def admit(self, items):
        self.db.executemany("INSERT OR IGNORE INTO items(key,payload,status) VALUES (?,?,'pending')",
                            [(i["key"], dumps(i)) for i in items])
        self.db.commit()

    def counts(self):
        return dict(self.db.execute("SELECT status,count(*) FROM items GROUP BY status"))

    def close(self):
        self.db.close()
        self.lock.close()


def update_transport_health(scheduler, api, raw, *, generation, probe=False, now=None):
    """A drained burst is one outage; only a failed recovery probe extends it.

    Tag requests with the circuit generation at admission. Once a circuit opens,
    late successes/failures from that generation still commit their item results,
    but cannot repeatedly halve concurrency, extend the delay, or close a newer
    circuit. This state is persisted alongside raw responses for safe replay.
    """
    if generation != scheduler["generation"]:
        return
    if raw.get("error_type") == "transport":
        scheduler["failures"] += 1
        scheduler["healthy"] = 0
        if probe or (not scheduler["circuit_until"] and scheduler["failures"] >= api["circuit_failures"]):
            scheduler["outages"] += 1
            scheduler["effective"] = max(1, scheduler["effective"] // 2)
            delay = min(900, api["circuit_seconds"] * 2 ** min(5, scheduler["outages"] - 1))
            scheduler["circuit_until"] = (time.time() if now is None else now) + delay
            scheduler["generation"] += 1
            scheduler["failures"] = 0
    elif (raw.get("status") in {"completed", "incomplete"}
          or (200 <= (raw.get("http_status") or 0) < 500
              and raw.get("error_type") != "configuration")):
        # Bad content is healthy transport, even when it fails output validation.
        scheduler.update(failures=0, circuit_until=0, outages=0)
        scheduler["healthy"] += 1
        if scheduler["healthy"] >= max(8, scheduler["effective"]):
            scheduler["effective"] = min(api["concurrency_max"], scheduler["effective"] + 8)
            scheduler["healthy"] = 0


async def review_items(state, client, config, tokenizer, stop, *, timeout_policy=None):
    api = config["sii"]
    model = api["vision_models"][0]
    db = state.db
    scheduler = state.meta("scheduler", {"effective": api["concurrency_start"], "healthy": 0,
                                         "failures": 0, "circuit_until": 0, "outages": 0})
    scheduler.setdefault("generation", 0)
    scheduler["policy"] = SCHEDULER_VERSION
    active, last_report = {}, 0
    probe = False
    fatal = None
    if timeout_policy:
        state.set_meta("timeout_fallback_policy", timeout_policy)

    def handoff_timeout(key, number):
        if not timeout_policy:
            return False
        from data_synthesis.timeout_fallback import repeated_timeouts
        if not repeated_timeouts(db, key, number, timeout_policy["consecutive_timeouts"]):
            return False
        db.execute("UPDATE items SET status='deferred_codex',retry_at=0,error=? WHERE key=?",
                   ("two consecutive SII timeouts; handed to independent gpt-5.6-sol/low queue", key))
        return True

    def accept(key, number, raw):
        nonlocal fatal
        row = db.execute("SELECT payload FROM items WHERE key=?", (key,)).fetchone()
        item = json.loads(row[0])
        try:
            result = parse_review(raw, item, tokenizer, expected_model=model,
                                  expected_contract=config.get("review_contract", VERSION))
        except Exception as exc:
            error = raw.get("error") or f"{type(exc).__name__}: {str(exc)[:600]}"
            kind = raw.get("error_type", "content")
            status = "failed" if number >= api["max_attempts"] else "retry"
            retry = time.time() + min(60, 2 ** number)
            db.execute("UPDATE items SET status=?,retry_at=?,error=? WHERE key=?", (status, retry, error, key))
            handoff_timeout(key, number)
            if kind == "configuration":
                fatal = "SII configuration/authentication error; no other backend is permitted"
                stop.set()
        else:
            status = "unusable" if result["review"]["decision"] == "unusable" else "ready"
            db.execute("UPDATE items SET status=?,result=?,error=NULL,retry_at=0 WHERE key=?", (status, dumps(result), key))
        update_transport_health(scheduler, api, raw,
            generation=raw.get("scheduler_generation", scheduler["generation"]),
            probe=raw.get("scheduler_probe", False))
        state.set_meta("scheduler", scheduler)
        db.commit()

    # Replay durable responses after interruption without paying for new calls.
    for row in db.execute("SELECT key,attempts FROM items WHERE status='running'").fetchall():
        raw = db.execute("SELECT raw_gzip FROM attempts WHERE key=? AND number=?", tuple(row)).fetchone()
        if raw:
            accept(row["key"], row["attempts"], json.loads(gzip.decompress(raw[0])))
        else:
            db.execute("UPDATE items SET status='retry' WHERE key=?", (row["key"],))
    db.commit()
    if timeout_policy:
        for row in db.execute("SELECT key,attempts FROM items WHERE status IN ('retry','failed')").fetchall():
            handoff_timeout(row["key"], row["attempts"])
        db.commit()

    async def invoke(item, number):
        try:
            return await client.generate(item, number)
        except Exception as exc:
            return {"status": "failed", "error_type": "local", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    def report(phase):
        value = {"state": phase, "pid": os.getpid(), "updated_at": time.time(),
                 "counts": state.counts(), "scheduler": scheduler, "active_sii": len(active),
                 "model": model, "active_codex": 0, "codex_fallback_enabled": False,
                 "timeout_handoff_enabled": bool(timeout_policy), "compute_hashes": False, "fatal": fatal}
        atomic_json(state.root / "status.json", value)
        return value

    while True:
        now = time.time()
        if not stop.is_set() and now >= scheduler["circuit_until"] and not probe:
            capacity = scheduler["effective"] - len(active)
            if scheduler["circuit_until"]:
                capacity = 1 if not active else 0
            rows = db.execute("SELECT * FROM items WHERE status IN ('pending','retry') AND retry_at<=? LIMIT ?",
                              (now, max(0, capacity))).fetchall()
            for row in rows:
                if row["attempts"] >= api["max_attempts"]:
                    db.execute("UPDATE items SET status='failed',error='retry ceiling reached after interruption' WHERE key=?", (row["key"],))
                    continue
                item = json.loads(row["payload"])
                item["issues"] = [row["error"]] if row["error"] else []
                number = row["attempts"] + 1
                db.execute("UPDATE items SET status='running',attempts=? WHERE key=?", (number, row["key"]))
                db.commit()  # Item identity and attempt are durable before dispatch.
                is_probe = bool(scheduler["circuit_until"])
                active[asyncio.create_task(invoke(item, number))] = (row["key"], number, scheduler["generation"], is_probe)
                if is_probe:
                    probe = True
        if now - last_report >= 5:
            report("draining" if stop.is_set() else "running")
            last_report = now
        if active:
            done, _ = await asyncio.wait(active, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                key, number, generation, is_probe = active.pop(task)
                raw = {**task.result(), "scheduler_generation": generation,
                       "scheduler_probe": is_probe, "scheduler_policy": SCHEDULER_VERSION}
                db.execute("INSERT OR REPLACE INTO attempts VALUES (?,?,?)", (key, number, gzip.compress(dumps(raw).encode(), compresslevel=1)))
                db.commit()  # Preserve raw evidence BEFORE content validation.
                accept(key, number, raw)
                if is_probe:
                    probe = False
        elif stop.is_set():
            return report("blocked" if fatal else "interrupted")
        elif not db.execute("SELECT 1 FROM items WHERE status IN ('pending','retry','running') LIMIT 1").fetchone():
            return report("needs_attention" if state.counts().get("failed") else "completed")
        else:
            await asyncio.sleep(0.5)


def export_review(state, tokenizer):
    """Revalidate every final result against its raw response, atomically save it."""
    model = state.meta("contract")["config"]["sii"]["vision_models"][0]
    contract = state.meta("contract")["version"]
    counts = state.counts()
    if any(counts.get(k, 0) for k in ("pending", "retry", "running")):
        raise ValueError("cannot finalize an unfinished review shard")
    output = state.root / "reviewed.jsonl.gz"
    temp = output.with_suffix(".tmp")
    outcomes = Counter()
    with gzip.open(temp, "wt", compresslevel=1) as f:
        for row in state.db.execute("SELECT * FROM items ORDER BY key"):
            item = json.loads(row["payload"])
            result = json.loads(row["result"]) if row["result"] else None
            if result is not None:
                raw_row = state.db.execute("SELECT raw_gzip FROM attempts WHERE key=? AND number=?", (row["key"], row["attempts"])).fetchone()
                raw = json.loads(gzip.decompress(raw_row[0]))
                if parse_review(raw, item, tokenizer, expected_model=model, expected_contract=contract) != result:
                    raise ValueError("saved review differs from raw provider response")
                outcomes[result["review"]["decision"]] += 1
            else:
                outcomes[row["status"]] += 1
            f.write(dumps({"key": row["key"], "original_key": item["original_key"],
                          "source": item["row"]["source"], "group": item["group"],
                          "source_run": item["source_run"], "source_path": item["view"]["source_path"],
                          "test_kind": item.get("test_kind", "production"),
                          "status": row["status"], "attempts": row["attempts"], "error": row["error"],
                          "result": result, "model": model, "contract": contract,
                          "evidence_db": str(state.root / "review.sqlite3")}) + "\n")
    temp.replace(output)
    records = 0
    with gzip.open(output, "rt") as f:
        for line in f:
            json.loads(line)
            records += 1
    assert records == sum(counts.values())
    value = {"state": ("needs_attention" if counts.get("failed") else
                       "awaiting_codex" if counts.get("deferred_codex") else "completed"),
             "records": records, "outcomes": dict(outcomes), "counts": counts,
             "output": str(output), "bytes": output.stat().st_size, "compute_hashes": False,
             "audit": "all_results_reparsed_against_raw_SII_response;gzip_full_read;counts",
             "finished_at": time.time(), "codex_calls": 0, "model": model}
    atomic_json(state.root / "manifest.json", value)
    return value


def validate_qualification(args, config):
    if not args.qualification:
        raise ValueError("bulk requires the completed pilot visual qualification")
    qualification = json.loads(Path(args.qualification).read_text())
    if qualification.get("status") == "user_accepted_for_generation":
        if (qualification.get("authorized_by") != "user"
                or qualification.get("contract") != VERSION
                or qualification.get("config") != config
                or qualification.get("selection") != str(Path(args.selection).resolve())
                or qualification.get("semantic_quality_passed") is not False
                or qualification.get("defer_semantic_repair") is not True
                or qualification.get("codex_enabled_this_stage") is not False):
            raise ValueError("user acceptance must bind this exact SII-only generation scope")
        return qualification
    if (qualification.get("status") != "passed" or not qualification.get("visual_review")
            or qualification.get("reviewer") != "GPT-6"
            or qualification.get("contract") != VERSION or qualification.get("config") != config
            or qualification.get("selection") != str(Path(args.selection).resolve())):
        raise ValueError("bulk requires real pilot visual qualification for this exact review contract")
    pilot_manifest = json.loads(Path(qualification["pilot_manifest"]).read_text())
    if pilot_manifest["state"] != "completed" or pilot_manifest["records"] < 320:
        raise ValueError("pilot has unresolved failures or insufficient completed requests")
    return qualification


async def run_farm(args, config, tokenizer):
    models = config["sii"]["vision_models"]
    if (config["codex_fallback"]["enabled"] or len(models) != 1
            or models[0] not in {"deepseek-v4.1-flash", "qwen3.8-max"} or config["compute_hashes"]):
        raise ValueError("this farm requires one explicitly selected SII review model, no Codex fallback, no hashing")
    model = models[0]
    acceptance = None
    if args.mode == "bulk":
        acceptance = validate_qualification(args, config)  # Pass or explicit user acceptance, before credentials.
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "farm.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_contract = {"config": config, "contract": VERSION,
                    "selection": str(Path(args.selection).resolve()), "mode": args.mode}
    contract_path = root / (args.mode + "_run.json")
    if contract_path.exists() and json.loads(contract_path.read_text()) != run_contract:
        lock.close()
        raise ValueError("farm runtime or fixed selection changed; use a new root")
    atomic_json(contract_path, run_contract)
    if acceptance is not None:
        atomic_json(root / "generation_acceptance.json", acceptance)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    client = SIIClient(load_sii_settings(), config["sii"], prompt_factory=prompt_for_review, contract_id=VERSION)
    try:
        if args.mode == "pilot":
            inputs = prepare_pilot(args.selection, root)
            state = ReviewDB(root / "pilot", config)
            try:
                state.admit([json.loads(line) for line in inputs.open()])
                result = await review_items(state, client, config, tokenizer, stop)
                if result["state"] in {"completed", "needs_attention"}:
                    result = export_review(state, tokenizer)
                return result
            finally:
                state.close()
        selection, batches = load_selection(args.selection)
        # Input batch list is frozen before any API request. Each bounded shard
        # admits all of its identities before inference; finished shards resume
        # from their audited manifests without reopening original image pixels.
        summary = Counter()
        completed, shards = 0, []
        previous_scheduler = None
        for start in range(0, len(batches), 64):
            if stop.is_set():
                break
            group = batches[start:start + 64]
            destination = root / "shards" / f"{start // 64:05d}"
            expected = sum(b["records"] for b in group)
            manifest = destination / "manifest.json"
            if manifest.exists():
                result = json.loads(manifest.read_text())
                if result["records"] != expected or Path(result["output"]).stat().st_size != result["bytes"]:
                    raise ValueError("completed review shard changed")
            else:
                state = ReviewDB(destination, config)
                try:
                    if previous_scheduler and state.meta("scheduler") is None:
                        state.set_meta("scheduler", previous_scheduler)
                    if not state.meta("admission_complete"):
                        for b in group:
                            state.admit(await asyncio.to_thread(read_batch, b))
                        assert sum(state.counts().values()) == expected
                        state.set_meta("admission_complete", True)
                        state.db.commit()
                    atomic_json(root / "status.json", {"state": "running", "pid": os.getpid(),
                        "updated_at": time.time(), "expected_images": selection["images"],
                        "completed_images": completed, "outcomes": dict(summary),
                        "active_shard": str(destination), "shards_completed": len(shards),
                        "model": model, "codex_calls": 0,
                        "acceptance": acceptance["status"],
                        "semantic_quality_passed": acceptance.get("semantic_quality_passed", True)})
                    result = await review_items(state, client, config, tokenizer, stop)
                    previous_scheduler = state.meta("scheduler")
                    if result["state"] not in {"completed", "needs_attention"}:
                        break
                    result = export_review(state, tokenizer)
                finally:
                    state.close()
            completed += result["records"]
            summary.update(result["outcomes"])
            shards.append(result)
            atomic_json(root / "progress.json", {"updated_at": time.time(), "completed_images": completed,
                        "expected_images": selection["images"], "outcomes": dict(summary), "shards_completed": len(shards),
                        "model": model})
        finished = completed == selection["images"]
        result = {"state": "completed" if finished and not summary["failed"] else "needs_attention" if finished else "interrupted",
                  "expected_images": selection["images"], "reviewed_images": completed, "outcomes": dict(summary),
                  "shards": shards, "codex_calls": 0, "compute_hashes": False, "model": model,
                  "acceptance": acceptance["status"],
                  "semantic_quality_passed": acceptance.get("semantic_quality_passed", True),
                  "later_semantic_repair_required": acceptance.get("defer_semantic_repair", False),
                  "next_stage_started": False, "finished_at": time.time()}
        atomic_json(root / ("manifest.json" if finished else "interrupted.json"), result)
        atomic_json(root / "status.json", {k: v for k, v in result.items() if k != "shards"})
        return {k: v for k, v in result.items() if k != "shards"}
    finally:
        await client.close()
        lock.close()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
