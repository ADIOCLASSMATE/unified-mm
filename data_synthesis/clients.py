"""Pooled, direct SII requests and a separate, bounded Codex last resort."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pathlib import Path
import shutil
import signal
import ssl
import tempfile
import time

import httpx
from PIL import Image

from data_synthesis.contract import CONTRACT_HASH, PAIR_SCHEMA, prompt_for
from data_synthesis.io import dumps, sha
from data_synthesis.integrity import view_binding
from utils.direct_network import check_direct_routes, direct_ssl_context
from utils.image_shard_io import read_image_bytes


def frozen_pixels(item, *, compute_hashes=None):
    view = item["view"]
    if compute_hashes is None:
        compute_hashes = view.get("hashes_computed", True)
    data = read_image_bytes(view["source_path"])
    if compute_hashes and sha(data) != view["view_sha256"]:
        raise ValueError("frozen image bytes changed")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.size != (512, 512) or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("expected a decoded, single RGB 512px image")
        mime = {"JPEG": "image/jpeg", "PNG": "image/png"}.get(image.format)
        if not mime:
            raise ValueError("frozen image format must be JPEG or PNG")
    return data, mime


class RequestPacer:
    def __init__(self, rpm, tpm, reserved_tokens):
        self.interval = max(60 / rpm, 60 * reserved_tokens / tpm)
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            delay = max(0, self.next_start - time.monotonic())
            if delay:
                await asyncio.sleep(delay)
            self.next_start = time.monotonic() + self.interval


class SIIClient:
    def __init__(self, settings, config, *, transport=None, prompt_factory=None, contract_id=None):
        if transport is None:
            check_direct_routes()
        self.settings, self.config = settings, config
        self.prompt_factory = prompt_factory or prompt_for
        self.contract_id = contract_id or CONTRACT_HASH
        context = direct_ssl_context()
        context.maximum_version = getattr(ssl.TLSVersion, config["tls_maximum_version"])
        if transport is None and config.get("tcp_mss_before_connect"):
            from data_synthesis.direct_backend import DirectMSSBackend
            transport = httpx.AsyncHTTPTransport(
                verify=context, trust_env=False, http2=config["http2"],
                limits=httpx.Limits(max_connections=config["concurrency_max"],
                                   max_keepalive_connections=config["concurrency_max"]),
            )
            # httpx's socket_options are applied AFTER connect and cannot
            # negotiate MSS. Use a scoped backend that sets it before SYN.
            transport._pool._network_backend = DirectMSSBackend(config["tcp_mss_before_connect"])
        self.client = httpx.AsyncClient(
            trust_env=False, proxy=None, verify=context,
            http2=config["http2"], follow_redirects=False, transport=transport,
            timeout=httpx.Timeout(config["timeout_seconds"], connect=15),
            limits=httpx.Limits(max_connections=config["concurrency_max"],
                               max_keepalive_connections=config["concurrency_max"]),
        )
        self.pacer = RequestPacer(config["rpm"], config["tpm"], config["max_tokens"] + 4096)

    async def generate(self, item, attempt):
        started = time.time()
        data, mime = await asyncio.to_thread(frozen_pixels, item)
        prompt = self.prompt_factory(item, item.get("candidate"), item.get("issues", []))
        model = self.config["vision_models"][(attempt - 1) % len(self.config["vision_models"])]
        protocol = self.config["protocol"]
        encoded = base64.b64encode(data).decode("ascii")
        if protocol == "openai_chat":
            body = {"model": model, "max_tokens": self.config["max_tokens"], "stream": False,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ]}], **self.config.get("request_options", {})}
            headers = {"Authorization": "Bearer " + self.settings.api_key}
        else:
            body = {"model": model, "max_tokens": self.config["max_tokens"], "stream": False,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}},
                    ]}], **self.config.get("request_options", {})}
            headers = {"x-api-key": self.settings.api_key, "anthropic-version": "2023-06-01"}
        evidence = {"backend": "sii", "requested_model": model, "protocol": protocol,
                    "endpoint": self.settings.endpoint(protocol), "proxy": False,
                    "image_id": item["key"], **view_binding(item["view"], item["view"].get("hashes_computed", True)),
                    "decoded_size": [512, 512], "mime_type": mime, "image_attached": True,
                    "prompt": prompt, "prompt_sha256": sha(prompt.encode()) if item["view"].get("hashes_computed", True) else None,
                    "request_sha256": sha(dumps(body).encode()) if item["view"].get("hashes_computed", True) else None,
                    "contract_hash": self.contract_id,
                    "tls_policy": {"curve": "prime256v1", "maximum_version": self.config["tls_maximum_version"],
                                   "certificate_verification": True, "http2": self.config["http2"]},
                    "tcp_mss_before_connect": self.config.get("tcp_mss_before_connect"),
                    "started_at": started, "attempt": attempt, "request_options": self.config.get("request_options", {})}
        await self.pacer.wait()
        evidence["network_events"] = []
        async def trace(name, info):
            if len(evidence["network_events"]) < 32:
                evidence["network_events"].append({"phase": name, "seconds": round(time.time() - started, 3)})
            if name == "connection.start_tls.complete":
                connection = info.get("return_value")
                tls = connection.get_extra_info("ssl_object") if connection else None
                if tls:
                    evidence["negotiated_tls_version"] = tls.version()
        try:
            async with asyncio.timeout(self.config["timeout_seconds"]):
                response = await self.client.post(self.settings.endpoint(protocol), json=body, headers=headers,
                                                  extensions={"trace": trace})
            evidence["http_status"] = response.status_code
            stream = response.extensions.get("network_stream")
            tls = stream.get_extra_info("ssl_object") if stream else None
            evidence["negotiated_tls_version"] = tls.version() if tls else None
            evidence["http_version"] = response.http_version
            # Persist provider response before parsing. No request headers or key.
            evidence["raw_response"] = self.settings.redact(response.text)
            response.raise_for_status()
            payload = response.json()
            if protocol == "openai_chat":
                choice = payload["choices"][0]
                content = choice["message"].get("content", "")
                if isinstance(content, list):
                    content = "\n".join(x.get("text", "") for x in content if x.get("type") == "text")
                completed = choice.get("finish_reason") == "stop"
            else:
                content = "\n".join(x["text"] for x in payload["content"] if x.get("type") == "text")
                completed = payload.get("stop_reason") in {"end_turn", "stop_sequence"}
            evidence.update(status="completed" if completed else "incomplete", output_text=self.settings.redact(content or ""),
                            usage=payload.get("usage", {}), returned_model=payload.get("model"))
        except Exception as exc:
            code = evidence.get("http_status")
            error_type = ("configuration" if code in {401, 403, 404} else
                          "transport" if isinstance(exc, (httpx.TransportError, TimeoutError)) or code == 429 or (code and code >= 500)
                          else "response")
            evidence.update(status="failed", output_text="", error_type=error_type,
                            error=self.settings.redact(f"{type(exc).__name__}: {exc}")[:1200])
        evidence["elapsed_seconds"] = time.time() - started
        return evidence

    async def close(self):
        await self.client.aclose()


class CodexFallback:
    def __init__(self, config, *, prompt_factory=None, contract_id=None, schema=None):
        self.config = config
        self.version = None
        self.prompt_factory = prompt_factory or prompt_for
        self.contract_id = contract_id or CONTRACT_HASH
        self.schema = schema or PAIR_SCHEMA

    async def generate(self, item, attempt):
        """Called only after persisted item API attempts have reached their limit."""
        executable = self.config["executable"]
        if shutil.which(executable) is None:
            raise ValueError("Codex fallback executable is unavailable")
        data, mime = await asyncio.to_thread(frozen_pixels, item)
        prompt = self.prompt_factory(item, item.get("candidate"), item.get("issues", []))
        started = time.time()
        if self.version is None:
            check = await asyncio.create_subprocess_exec(executable, "--version", stdout=asyncio.subprocess.PIPE)
            output, _ = await check.communicate()
            self.version = output.decode().strip()
        with tempfile.TemporaryDirectory(prefix="b512-final-fallback-") as directory:
            root = Path(directory)
            attachment = root / ("image.png" if mime == "image/png" else "image.jpg")
            attachment.write_bytes(data)
            schema, result = root / "schema.json", root / "result.json"
            schema.write_text(dumps(self.schema))
            command = [executable, "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                       "--sandbox", "read-only", "--model", "gpt-5.6-sol",
                       "-c", 'model_reasoning_effort="low"', "-c", 'approval_policy="never"',
                       "-c", 'web_search="disabled"', "--cd", str(root), "--image", str(attachment),
                       "--output-schema", str(schema), "--output-last-message", str(result), "--json", "-"]
            # Keep the parent's proxy for Codex; SII secrets are never inherited.
            env = {k: v for k, v in os.environ.items()
                   if not (k.upper().startswith("SII") and any(part in k.upper() for part in ("KEY", "TOKEN", "SECRET")))}
            process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, env=env, start_new_session=True)
            timed_out = False
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(prompt.encode()), self.config["timeout_seconds"])
            except (TimeoutError, asyncio.CancelledError) as exc:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(process.wait(), 3)
                    except TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                stdout, stderr = b"", b"Codex fallback deadline exceeded"
                timed_out = True
            return {"backend": "codex_fallback", "requested_model": "gpt-5.6-sol", "reasoning_effort": "low",
                    "cli_version": self.version, "command": command, "contract_hash": self.contract_id,
                    "prompt": prompt, "prompt_sha256": sha(prompt.encode()) if item["view"].get("hashes_computed", True) else None,
                    "image_id": item["key"], **view_binding(item["view"], item["view"].get("hashes_computed", True)),
                    "decoded_size": [512, 512], "image_attached": True, "started_at": started,
                    "elapsed_seconds": time.time() - started, "attempt": attempt,
                    "status": "completed" if process.returncode == 0 and result.is_file() and not timed_out else "failed",
                    "output_text": result.read_text() if result.is_file() else "",
                    "events": stdout.decode(errors="replace"), "stderr": stderr.decode(errors="replace"),
                    "exit_code": process.returncode, "error_type": "timeout" if timed_out else "codex_exit"}

    async def close(self):
        pass
