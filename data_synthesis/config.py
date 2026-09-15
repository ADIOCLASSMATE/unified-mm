"""Versioned policy; SII credentials come from environment or static shell rc exports."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex

import httpx

from data_synthesis.io import dumps, sha

SCHEMA = "b512_reuse_sii_fallback_runtime_v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/data_synthesis/b512_sii_v1.json"


@dataclass(frozen=True)
class SIISettings:
    base_url: str
    api_key: str = field(repr=False)

    def endpoint(self, protocol):
        suffix = "chat/completions" if protocol == "openai_chat" else "messages"
        base = self.base_url.rstrip("/")
        return base + ("/" if base.endswith("/v1") else "/v1/") + suffix

    def public(self):
        return {"base_url": self.base_url, "credential_source": "SII_API_KEY/SII_BASE_URL environment or literal bashrc/zshrc exports", "proxy": False}

    def redact(self, text):
        return str(text).replace(self.api_key, "[REDACTED]") if self.api_key else str(text)


def load_sii_settings(*, require_key=True, environ=None, rc_paths=None):
    """Read exported SII variables, or their literal rc declarations; execute no shell."""
    env = os.environ if environ is None else environ
    values = {name: env.get(name, "") for name in ("SII_BASE_URL", "SII_API_KEY")}
    missing = [name for name, value in values.items() if not value and (require_key or name != "SII_API_KEY")]
    if missing:
        found = {name: set() for name in missing}
        paths = (Path.home() / ".bashrc", Path.home() / ".zshrc") if rc_paths is None else rc_paths
        for path in paths:
            path = Path(path)
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                for name in missing:
                    if not re.match(r"\s*export\s+" + name + r"=", line):
                        continue
                    if "$" in line or "`" in line:
                        raise ValueError(f"{name} rc declaration must be literal, or exported in the environment")
                    parts = shlex.split(line, comments=True)
                    if len(parts) != 2 or not parts[1].startswith(name + "="):
                        raise ValueError(f"{name} rc declaration must be one literal export assignment")
                    found[name].add(parts[1].split("=", 1)[1])
        for name, candidates in found.items():
            if len(candidates) > 1:
                raise ValueError(f"conflicting {name} declarations; export the intended value in the invoking shell")
            if candidates:
                values[name] = candidates.pop()
    base, key = values["SII_BASE_URL"], values["SII_API_KEY"] if require_key else ""
    if not base:
        raise ValueError("SII_BASE_URL is missing; export it from your bashrc/zshrc")
    url = httpx.URL(base)
    if url.scheme != "https" or not url.host or url.username or url.password or url.query or url.fragment:
        raise ValueError("SII_BASE_URL must be HTTPS without embedded credentials/query/fragment")
    if require_key and not key:
        raise ValueError("SII_API_KEY is missing; export it from your bashrc/zshrc")
    return SIISettings(str(base), key or "")


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(c):
    if c.get("schema") != SCHEMA:
        raise ValueError("unknown synthesis runtime schema")
    if not isinstance(c.get("compute_hashes", True), bool):
        raise ValueError("compute_hashes must be boolean")
    if (c.get("image_size"), c.get("image_tokens"), c.get("text_tokens_max")) != (512, 1024, 960):
        raise ValueError("B512 requires 512px, 1024 image tokens and a 960 text-token budget")
    if c.get("minimum_images", 0) < 1 or c.get("source_targets_are_caps") is not False:
        raise ValueError("image count is a positive minimum, source targets are not caps")
    api, fallback = c["sii"], c["codex_fallback"]
    if api["protocol"] not in {"openai_chat", "anthropic_messages"}:
        raise ValueError("unsupported SII protocol")
    if api["tls_maximum_version"] not in {"TLSv1_2", "TLSv1_3"}:
        raise ValueError("SII TLS maximum must be TLSv1_2 or TLSv1_3, with certificate verification")
    if not api["vision_models"] or any(not isinstance(m, str) or not m for m in api["vision_models"]):
        raise ValueError("configure at least one SII vision model")
    if set(api["vision_models"]) & set(api.get("text_only_models", [])):
        raise ValueError("text-only models cannot be used for image correction")
    for k in ("max_attempts", "concurrency_start", "concurrency_max", "max_tokens", "timeout_seconds", "rpm", "tpm"):
        if api[k] <= 0:
            raise ValueError(f"SII {k} must be positive")
    if api["concurrency_start"] > api["concurrency_max"]:
        raise ValueError("SII start concurrency exceeds ceiling")
    if c.get("review_mode") == "reuse_first_targeted":
        if not 1 <= c.get("routing_workers", 0) <= 64 or not c.get("prepared_selection"):
            raise ValueError("reuse-first requires 1..64 local workers and a fixed prepared selection")
    if fallback["model"] != "gpt-5.6-sol" or fallback["reasoning_effort"] != "low":
        raise ValueError("Codex fallback explicitly uses gpt-5.6-sol / low")
    if fallback["max_attempts"] < 1 or fallback["concurrency"] < 1 or fallback["max_items_per_run"] < 1:
        raise ValueError("Codex fallback must have bounded positive limits")
    for k in ("retry_base_seconds", "retry_max_seconds", "circuit_seconds", "circuit_failures"):
        if api[k] <= 0:
            raise ValueError(f"SII {k} must be positive")
    if api.get("request_options", {}).keys() & {"model", "messages", "system", "stream", "max_tokens"}:
        raise ValueError("request options cannot override grounded inputs or routing")


def fingerprint(config):
    return sha(dumps(config).encode())
