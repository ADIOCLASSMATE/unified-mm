"""One synthesis writer, WAL state, durable attempt counts and raw responses."""
from __future__ import annotations

import fcntl
import gzip
import json
from pathlib import Path
import sqlite3
import time
import uuid

from data_synthesis.config import fingerprint
from data_synthesis.contract import CONTRACT_HASH
from data_synthesis.io import dumps, sha


class State:
    def __init__(self, root, config=None, *, readonly=False):
        self.root = Path(root).resolve()
        self.lock = None
        if readonly:
            self.db = sqlite3.connect(f"file:{self.root / 'state.sqlite3'}?mode=ro", uri=True)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.lock = (self.root / "controller.lock").open("a")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                self.lock.close()
                raise
            self.db = sqlite3.connect(self.root / "state.sqlite3")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
              CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS items(
                key TEXT PRIMARY KEY, identity TEXT NOT NULL UNIQUE,
                source_sha256 TEXT NOT NULL UNIQUE, view_sha256 TEXT NOT NULL UNIQUE,
                row_json TEXT NOT NULL, view_json TEXT NOT NULL, status TEXT NOT NULL,
                api_attempts INTEGER NOT NULL DEFAULT 0, codex_attempts INTEGER NOT NULL DEFAULT 0,
                retry_at REAL NOT NULL DEFAULT 0, attempt_id TEXT, result_json TEXT,
                evidence_json TEXT, error TEXT, created_at REAL NOT NULL);
              CREATE INDEX IF NOT EXISTS item_queue ON items(status,retry_at,created_at);
              CREATE TABLE IF NOT EXISTS attempts(
                id TEXT PRIMARY KEY, item_key TEXT NOT NULL, backend TEXT NOT NULL, number INTEGER NOT NULL,
                status TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL,
                raw_gzip BLOB, raw_sha256 TEXT, result_sha256 TEXT, error TEXT);
              CREATE INDEX IF NOT EXISTS attempt_item ON attempts(item_key,backend,number);
              CREATE TABLE IF NOT EXISTS admissions(
                id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL, rows INTEGER NOT NULL);
              CREATE TABLE IF NOT EXISTS exclusions(
                identity TEXT PRIMARY KEY, reason TEXT NOT NULL, row_json TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS phashes(part INTEGER,bucket INTEGER,value TEXT,item_key TEXT,
                PRIMARY KEY(part,bucket,value,item_key));
              CREATE INDEX IF NOT EXISTS phash_lookup ON phashes(part,bucket);
            """)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=30000")
        if config is not None:
            expected = {"runtime": fingerprint(config), "pair_contract": CONTRACT_HASH}
            previous = self.meta("contract")
            if previous is not None and previous != expected:
                self.close()
                raise ValueError("runtime or prompt contract changed; start a new state root")
            if previous is None:
                self.set_meta("contract", expected)
                self.set_meta("config", config)
                self.db.commit()

    def meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, dumps(value)))

    def check_test_prompts(self, pair):
        from data_synthesis.sources import prompt_hash
        if not hasattr(self, "_prompt_exclusions"):
            self._prompt_exclusions = set(self.meta("excluded_prompt_hashes", []))
        if any(prompt_hash(pair[task]) in self._prompt_exclusions for task in ("i2t", "t2i")):
            raise ValueError("text overlaps an excluded evaluation test prompt")

    def counts(self):
        return dict(self.db.execute("SELECT status,count(*) FROM items GROUP BY status"))

    def item(self, key):
        row = dict(self.db.execute("SELECT * FROM items WHERE key=?", (key,)).fetchone())
        row["row"], row["view"] = json.loads(row.pop("row_json")), json.loads(row.pop("view_json"))
        return row

    def claim(self, key, backend):
        column = "api_attempts" if backend == "sii" else "codex_attempts"
        ident = uuid.uuid4().hex
        number = self.db.execute(f"SELECT {column} FROM items WHERE key=?", (key,)).fetchone()[0] + 1
        self.db.execute(f"UPDATE items SET status='running',{column}=?,attempt_id=? WHERE key=?", (number, ident, key))
        self.db.execute("INSERT INTO attempts(id,item_key,backend,number,status,started_at) VALUES (?,?,?,?,'running',?)",
                        (ident, key, backend, number, time.time()))
        self.db.commit()
        return ident, number

    def receive(self, ident, raw):
        encoded = dumps(raw).encode()
        self.db.execute("UPDATE attempts SET status='received',raw_gzip=?,raw_sha256=?,finished_at=? WHERE id=?",
                        (gzip.compress(encoded, compresslevel=1), sha(encoded), time.time(), ident))
        self.db.commit()  # Raw provider response is durable BEFORE validation.

    def raw(self, ident):
        row = self.db.execute("SELECT raw_gzip,raw_sha256 FROM attempts WHERE id=?", (ident,)).fetchone()
        if not row or row[0] is None:
            return None
        data = gzip.decompress(row[0])
        if sha(data) != row[1]:
            raise ValueError("raw attempt checksum mismatch")
        return json.loads(data)

    def succeed(self, key, pair, evidence, ident=None, *, commit=True):
        self.db.execute("UPDATE items SET status='ready',result_json=?,evidence_json=?,error=NULL,retry_at=0 WHERE key=?",
                        (dumps(pair), dumps(evidence), key))
        if ident:
            self.db.execute("UPDATE attempts SET status='succeeded',result_sha256=?,error=NULL WHERE id=?",
                            (sha(dumps(pair).encode()), ident))
        if commit:
            self.db.commit()

    def fail(self, key, status, error, retry_at=0, ident=None):
        self.db.execute("UPDATE items SET status=?,error=?,retry_at=? WHERE key=?", (status, str(error), retry_at, key))
        if ident:
            self.db.execute("UPDATE attempts SET status='failed',error=? WHERE id=?", (str(error), ident))
        self.db.commit()

    def recover(self):
        # Retain received-but-unvalidated attempts for local replay, without a new API call.
        self.db.execute("UPDATE items SET status='received' WHERE status='running' AND attempt_id IN (SELECT id FROM attempts WHERE raw_gzip IS NOT NULL)")
        self.db.execute("UPDATE attempts SET status='interrupted',error='controller interrupted before a durable response' WHERE status='running'")
        self.db.execute("UPDATE items SET status='retry',error='interrupted attempt retained' WHERE status='running'")
        self.db.commit()

    def quarantine_failed(self, config):
        keys = []
        for item in self.db.execute("SELECT key,identity,error,api_attempts,codex_attempts,row_json FROM items WHERE status='failed'"):
            if item["api_attempts"] < config["sii"]["max_attempts"]:
                raise ValueError("cannot quarantine an item before SII retries are exhausted")
            if config["codex_fallback"]["enabled"] and item["codex_attempts"] < config["codex_fallback"]["max_attempts"]:
                raise ValueError("cannot quarantine before the configured last fallback")
            self.db.execute("INSERT OR REPLACE INTO exclusions VALUES (?,?,?)",
                            (item["identity"], "terminal_synthesis_failure:" + str(item["error"]), item["row_json"]))
            keys.append(item["key"])
        self.db.executemany("UPDATE items SET status='quarantined' WHERE key=?", [(key,) for key in keys])
        self.db.commit()
        return {"quarantined": len(keys), "counts": self.counts(), "attempt_history_preserved": True}

    def close(self):
        self.db.close()
        if self.lock:
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
