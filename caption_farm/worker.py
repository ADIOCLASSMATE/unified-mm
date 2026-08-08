from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

from .io import append_jsonl, canonical_json, load_json, sha256_text, utc_now
from .queue import Lease, TaskStore, default_owner


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
WORKER_TUNING_SCHEMA = "imagenet_caption_farm_worker_tuning_v1"


def load_worker_tuning(run_dir: Path, run: dict[str, Any]) -> dict[str, Any] | None:
    path = run_dir / "worker_tuning.json"
    if not path.is_file():
        return None
    tuning = load_json(path)
    if tuning.get("schema") != WORKER_TUNING_SCHEMA:
        raise ValueError(f"unexpected worker tuning schema in {path}")
    if tuning.get("run_fingerprint") != run["run_fingerprint"]:
        raise ValueError("worker tuning belongs to another run")
    if tuning.get("model_fingerprint") != run["model"]["fingerprint"]:
        raise ValueError("worker tuning belongs to another model snapshot")
    for key in ("request_concurrency", "max_num_seqs", "claim_batch_size"):
        if int(tuning.get(key) or 0) < 1:
            raise ValueError(f"worker tuning {key} must be positive")
    if int(tuning["request_concurrency"]) > int(tuning["max_num_seqs"]):
        raise ValueError("worker tuning request_concurrency cannot exceed max_num_seqs")
    for key in ("max_image_side", "max_image_pixels", "max_tasks_per_worker"):
        if key in tuning and int(tuning[key]) < 1:
            raise ValueError(f"worker tuning {key} must be positive")
    if "request_timeout_seconds" in tuning and float(tuning["request_timeout_seconds"]) <= 0:
        raise ValueError("worker tuning request_timeout_seconds must be positive")
    return tuning


def clean_caption(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:text|markdown)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    value = " ".join(value.split())
    for prefix in ("Caption:", "Image caption:", "Description:"):
        if value.lower().startswith(prefix.lower()):
            value = value[len(prefix) :].strip()
            break
    return value


def stable_seed(task: dict[str, Any], run_fingerprint: str) -> int:
    digest = sha256_text(f"{run_fingerprint}:{task['image_id']}:{task['caption_slot']}")
    return int(digest[:8], 16) & 0x7FFFFFFF


def render_prompt(run: dict[str, Any], task: dict[str, Any]) -> str:
    caption = run["caption"]
    variants = caption["prompt_variants"]
    variant = variants[int(task["caption_slot"]) % len(variants)]
    return str(variant).format(
        target_min_words=int(caption["target_min_words"]),
        target_max_words=int(caption["target_max_words"]),
        caption_slot=int(task["caption_slot"]),
        synset=task["synset"],
    )


class CaptionEngine(Protocol):
    load_seconds: float

    def start(self, health_task: dict[str, Any]) -> None: ...

    def caption(self, task: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


class FakeCaptionEngine:
    def __init__(self, run: dict[str, Any], *, delay_seconds: float = 0.0) -> None:
        self.run = run
        self.delay_seconds = delay_seconds
        self.load_seconds = 0.0

    def start(self, health_task: dict[str, Any]) -> None:
        started = time.monotonic()
        if self.delay_seconds:
            time.sleep(min(self.delay_seconds, 0.05))
        self.load_seconds = time.monotonic() - started

    def caption(self, task: dict[str, Any]) -> dict[str, Any]:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        slot = int(task["caption_slot"])
        text = (
            f"A detailed subject from class {task['synset']} occupies the frame with "
            f"clearly visible contours, natural texture, balanced lighting, and a distinct "
            f"background composition described by deterministic local caption variant {slot}."
        )
        return {"caption": text, "latency_seconds": self.delay_seconds, "seed": stable_seed(task, self.run["run_fingerprint"])}

    def close(self) -> None:
        return None


class VllmOpenAIEngine:
    """One local vLLM server plus concurrent requests for continuous batching."""

    def __init__(self, run: dict[str, Any], worker_id: str, log_path: Path) -> None:
        self.run = run
        self.worker_id = worker_id
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any | None = None
        self.load_seconds = 0.0
        inference = run["model"]["inference"]
        self.host = str(inference.get("host", "127.0.0.1"))
        self.port = int(inference.get("port", 18080))
        self.base_url = f"http://{self.host}:{self.port}/v1"

    def _command(self) -> list[str]:
        inference = self.run["model"]["inference"]
        command = inference.get("server_command")
        if not isinstance(command, list) or not command:
            raise ValueError("model.inference.server_command must be a non-empty argv list")
        replacements = {
            "model_path": self.run["model"]["path"],
            "served_model_name": inference["served_model_name"],
            "host": self.host,
            "port": str(self.port),
            "max_model_len": str(inference["max_model_len"]),
            "gpu_memory_utilization": str(inference["gpu_memory_utilization"]),
            "max_num_seqs": str(inference["max_num_seqs"]),
        }
        rendered: list[str] = []
        for part in command:
            value = str(part)
            for key, replacement in replacements.items():
                value = value.replace("{" + key + "}", str(replacement))
            rendered.append(value)
        return rendered

    def _post(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=canonical_json(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"local inference HTTP {exc.code}: {body[-2000:]}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("local inference returned a non-object JSON response")
        return value

    def _image_data_url(self, path: Path) -> str:
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        inference = self.run["model"]["inference"]
        max_side = int(inference.get("max_image_side", 1024))
        max_pixels = int(inference.get("max_image_pixels", 1024 * 1024))
        if max_side < 1 or max_pixels < 1:
            raise ValueError("max_image_side and max_image_pixels must be positive")
        with path.open("rb") as handle:
            payload = handle.read()
        # A small number of ImageNet JPEGs are several thousand pixels on a
        # side. Qwen's native-resolution vision tokenizer can turn those into
        # >10k tokens, exceeding the deliberately short caption context. Only
        # resize inputs above the audited limits; the manifest and source image
        # remain untouched.
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(payload)) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            scale = min(
                1.0,
                max_side / max(width, height),
                (max_pixels / (width * height)) ** 0.5,
            )
            if scale < 1.0:
                resized = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.LANCZOS,
                )
                if resized.mode != "RGB":
                    resized = resized.convert("RGB")
                output = io.BytesIO()
                resized.save(output, format="JPEG", quality=90, optimize=True)
                payload = output.getvalue()
                media_type = "image/jpeg"
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _payload(self, task: dict[str, Any], *, health: bool = False) -> dict[str, Any]:
        inference = self.run["model"]["inference"]
        if health:
            prompt = "Inspect this image and return one short, non-empty English description."
            max_tokens = 32
        else:
            prompt = render_prompt(self.run, task)
            max_tokens = int(self.run["caption"]["max_output_tokens"])
        return {
            "model": inference["served_model_name"],
            "messages": [
                {"role": "system", "content": self.run["caption"]["system_prompt"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(Path(task["source_path"]))},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "temperature": float(self.run["caption"]["temperature"]),
            "top_p": float(self.run["caption"]["top_p"]),
            "presence_penalty": float(self.run["caption"]["presence_penalty"]),
            "max_tokens": max_tokens,
            "seed": stable_seed(task, self.run["run_fingerprint"]),
            # These are top-level OpenAI-compatible extension fields.  The
            # Python client calls its merge argument ``extra_body``, but raw
            # HTTP requests must send the merged fields themselves.
            "top_k": int(self.run["caption"]["top_k"]),
            "chat_template_kwargs": {"enable_thinking": False},
        }

    @staticmethod
    def _response_text(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("local inference response has no message content") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(item.get("text") or "") for item in content if isinstance(item, dict)
            )
        return str(content or "")

    def _wait_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"local inference server exited with code {self.process.returncode}")
            try:
                with urllib.request.urlopen(f"http://{self.host}:{self.port}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(1)
        raise TimeoutError(f"local inference server was not ready: {last_error}")

    def start(self, health_task: dict[str, Any]) -> None:
        started = time.monotonic()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("ab", buffering=0)
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        self.process = subprocess.Popen(
            self._command(),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        self._wait_ready(float(self.run["model"]["inference"]["load_timeout_seconds"]))
        response = self._post(
            self._payload(health_task, health=True),
            timeout=float(self.run["model"]["inference"]["request_timeout_seconds"]),
        )
        if not clean_caption(self._response_text(response)):
            raise RuntimeError("vision health check returned an empty description")
        self.load_seconds = time.monotonic() - started

    def caption(self, task: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        response = self._post(
            self._payload(task),
            timeout=float(self.run["model"]["inference"]["request_timeout_seconds"]),
        )
        text = clean_caption(self._response_text(response))
        if not text:
            raise RuntimeError("local model returned an empty caption")
        usage = response.get("usage") or {}
        return {
            "caption": text,
            "latency_seconds": time.monotonic() - started,
            "seed": stable_seed(task, self.run["run_fingerprint"]),
            "usage": usage,
        }

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
        self.process = None
        self.log_handle = None


def _gpu_memory_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return max(values) if values else None
    except Exception:
        return None


class Worker:
    def __init__(
        self,
        run_dir: Path,
        *,
        worker_id: str | None = None,
        engine: CaptionEngine | None = None,
        max_tasks: int | None = None,
        request_concurrency: int | None = None,
        claim_batch_size: int | None = None,
        max_num_seqs: int | None = None,
        server_port: int | None = None,
        lease_seconds: float | None = None,
        post_claim_delay_seconds: float = 0.0,
    ) -> None:
        self.store = TaskStore(run_dir)
        self.run = deepcopy(self.store.run)
        self.tuning = load_worker_tuning(run_dir, self.run)
        self.worker_id = worker_id or default_owner()
        self.worker_log = run_dir / "workers" / f"{self.worker_id}.jsonl"
        tuned_max_tasks = (
            int(self.tuning["max_tasks_per_worker"])
            if self.tuning is not None and "max_tasks_per_worker" in self.tuning
            else None
        )
        self.max_tasks = max_tasks if max_tasks is not None else tuned_max_tasks
        tuned_claim_batch_size = (
            int(self.tuning["claim_batch_size"]) if self.tuning is not None else None
        )
        self.claim_batch_size = (
            claim_batch_size if claim_batch_size is not None else tuned_claim_batch_size
        )
        self.post_claim_delay_seconds = post_claim_delay_seconds
        if lease_seconds is not None:
            if lease_seconds <= 0:
                raise ValueError("lease_seconds must be positive")
            self.store.lease_seconds = lease_seconds
        inference = self.run["model"].setdefault("inference", {})
        if self.tuning is not None:
            inference["request_concurrency"] = int(self.tuning["request_concurrency"])
            inference["max_num_seqs"] = int(self.tuning["max_num_seqs"])
            if "max_image_side" in self.tuning:
                inference["max_image_side"] = int(self.tuning["max_image_side"])
            if "max_image_pixels" in self.tuning:
                inference["max_image_pixels"] = int(self.tuning["max_image_pixels"])
            if "request_timeout_seconds" in self.tuning:
                inference["request_timeout_seconds"] = float(
                    self.tuning["request_timeout_seconds"]
                )
        if request_concurrency is not None:
            if request_concurrency < 1:
                raise ValueError("request_concurrency must be positive")
            inference["request_concurrency"] = request_concurrency
        if max_num_seqs is not None:
            if max_num_seqs < 1:
                raise ValueError("max_num_seqs must be positive")
            inference["max_num_seqs"] = max_num_seqs
        if server_port is not None:
            if not (1 <= server_port <= 65535):
                raise ValueError("server_port must be between 1 and 65535")
            inference["port"] = server_port
        if claim_batch_size is not None and claim_batch_size < 1:
            raise ValueError("claim_batch_size must be positive")
        self.completed = 0
        self.stopping = threading.Event()
        self.peak_memory_mib = 0
        if engine is not None:
            self.engine = engine
        elif self.run["model"]["engine"] == "fake":
            self.engine = FakeCaptionEngine(self.run)
        elif self.run["model"]["engine"] == "vllm-openai":
            self.engine = VllmOpenAIEngine(
                self.run, self.worker_id, run_dir / "workers" / f"{self.worker_id}.server.log"
            )
        else:
            raise ValueError(f"unsupported inference engine {self.run['model']['engine']!r}")

    def _emit(self, event: str, **fields: Any) -> None:
        append_jsonl(
            self.worker_log,
            {
                "timestamp": utc_now(),
                "event": event,
                "worker_id": self.worker_id,
                "run_fingerprint": self.run["run_fingerprint"],
                **fields,
            },
        )

    def request_stop(self, signum: int | None = None, frame: Any = None) -> None:
        self.stopping.set()
        self._emit("signal_received", signal=signum)

    def _heartbeat_loop(self, lease: Lease, finished: threading.Event) -> None:
        interval = float(self.run["queue"]["heartbeat_seconds"])
        while not finished.wait(interval):
            try:
                expires = self.store.heartbeat(lease.lease_id, lease.owner)
                self._emit("lease_heartbeat", lease_id=lease.lease_id, expires_at=expires)
            except FileNotFoundError:
                return
            except Exception as exc:
                self._emit("lease_heartbeat_failed", lease_id=lease.lease_id, error=str(exc))
                self.stopping.set()
                return

    @staticmethod
    def _is_transport_failure(exc: BaseException) -> bool:
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, (TimeoutError, ConnectionError, urllib.error.URLError)):
                return True
            current = current.__cause__ or current.__context__
        return False

    def _infer_batch(
        self, tasks: list[dict[str, Any]]
    ) -> tuple[
        dict[str, tuple[dict[str, Any] | None, str | None]],
        str | None,
    ]:
        concurrency = int(self.run["model"].get("inference", {}).get("request_concurrency", 1))
        results: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
        unhealthy_error: str | None = None
        # Submit only one concurrency-sized wave at a time. If the local
        # server stops answering, this prevents the executor from feeding a
        # second wave into the same unhealthy process.
        for start in range(0, len(tasks), concurrency):
            wave = tasks[start : start + concurrency]
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(self.engine.caption, task): task for task in wave}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        results[task["task_key"]] = (future.result(), None)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        results[task["task_key"]] = (None, error)
                        if unhealthy_error is None and self._is_transport_failure(exc):
                            unhealthy_error = error
            if unhealthy_error is not None:
                break
        return results, unhealthy_error

    def run_forever(self) -> int:
        previous_handlers: dict[int, Any] = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, self.request_stop)
        try:
            health_task = self.store._task_from_ordinal(0)
            self._emit("model_loading_started", model_fingerprint=self.run["model"]["fingerprint"])
            self.engine.start(health_task)
            memory = _gpu_memory_used_mib()
            if memory is not None:
                self.peak_memory_mib = max(self.peak_memory_mib, memory)
            self._emit(
                "model_ready",
                load_seconds=self.engine.load_seconds,
                gpu_memory_used_mib=memory,
                request_concurrency=int(
                    self.run["model"].get("inference", {}).get("request_concurrency", 1)
                ),
                max_num_seqs=int(
                    self.run["model"].get("inference", {}).get("max_num_seqs", 1)
                ),
                inference_server_pid=getattr(getattr(self.engine, "process", None), "pid", None),
                worker_tuning=self.tuning,
            )

            claim_limit = int(
                self.claim_batch_size
                if self.claim_batch_size is not None
                else self.run["queue"]["claim_batch_size"]
            )
            idle_seconds = float(self.run["queue"].get("idle_exit_seconds", 30))
            idle_started: float | None = None
            while not self.stopping.is_set():
                if self.max_tasks is not None:
                    remaining = self.max_tasks - self.completed
                    if remaining <= 0:
                        break
                    limit = min(claim_limit, remaining)
                else:
                    limit = claim_limit
                lease = self.store.claim(self.worker_id, limit)
                if lease is None:
                    snapshot = self.store.snapshot()
                    if snapshot["PENDING"] == 0 and snapshot["LEASED"] == 0:
                        break
                    if (self.store.run_dir / "DRAIN").exists():
                        break
                    if idle_started is None:
                        idle_started = time.monotonic()
                    if time.monotonic() - idle_started >= idle_seconds:
                        self._emit("idle_exit", snapshot=snapshot)
                        break
                    self.stopping.wait(min(2.0, idle_seconds))
                    continue
                idle_started = None
                tasks = list(lease.tasks)
                self._emit("batch_started", lease_id=lease.lease_id, task_count=len(tasks))
                heartbeat_finished = threading.Event()
                heartbeat = threading.Thread(
                    target=self._heartbeat_loop,
                    args=(lease, heartbeat_finished),
                    name=f"lease-heartbeat-{lease.lease_id[:8]}",
                    daemon=True,
                )
                heartbeat.start()
                try:
                    if self.post_claim_delay_seconds and self.stopping.wait(
                        self.post_claim_delay_seconds
                    ):
                        self.store.release(lease, tasks, "worker stopping before inference")
                        break
                    inferred, unhealthy_error = self._infer_batch(tasks)
                    deferred: list[dict[str, Any]] = []
                    commits: list[
                        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
                    ] = []
                    for index, task in enumerate(tasks):
                        if self.stopping.is_set() and task["task_key"] not in inferred:
                            self.store.release(lease, tasks[index:], "worker stopping")
                            break
                        if task["task_key"] not in inferred:
                            if unhealthy_error is None:
                                raise RuntimeError(
                                    f"inference result missing for {task['task_key']}"
                                )
                            deferred.append(task)
                            self._emit(
                                "caption_deferred",
                                task_key=task["task_key"],
                                error=unhealthy_error,
                            )
                            continue
                        result, error = inferred[task["task_key"]]
                        if error is not None or result is None:
                            if unhealthy_error is not None:
                                deferred.append(task)
                                self._emit(
                                    "caption_deferred",
                                    task_key=task["task_key"],
                                    error=error or unhealthy_error,
                                )
                                continue
                            self.store.nack(lease, task, error or "unknown inference failure")
                            self._emit("caption_failed", task_key=task["task_key"], error=error)
                            continue
                        caption = clean_caption(str(result["caption"]))
                        if not caption:
                            self.store.nack(lease, task, "empty caption after format cleaning")
                            continue
                        record = {
                            **result,
                            "caption": caption,
                            "caption_sha256": sha256_text(caption),
                            "word_count": len(WORD_RE.findall(caption)),
                            "prompt_version": self.run["caption"]["prompt_version"],
                            "model_repository": self.run["model"]["repository"],
                            "generated_at": utc_now(),
                        }
                        commits.append((task, record, result))
                    commit_outcomes = self.store.commit_many(
                        lease,
                        [(task, record) for task, record, _ in commits],
                    )
                    for task, record, result in commits:
                        created = commit_outcomes[str(task["task_key"])]
                        self.completed += 1
                        self._emit(
                            "caption_committed",
                            task_key=task["task_key"],
                            created=created,
                            latency_seconds=result.get("latency_seconds"),
                            word_count=record["word_count"],
                        )
                    if deferred:
                        reason = f"unhealthy inference transport: {unhealthy_error}"
                        released = self.store.release(lease, deferred, reason)
                        self._emit(
                            "worker_retiring_unhealthy",
                            error=unhealthy_error,
                            deferred_tasks=released,
                        )
                        self.stopping.set()
                    memory = _gpu_memory_used_mib()
                    if memory is not None:
                        self.peak_memory_mib = max(self.peak_memory_mib, memory)
                    self._emit(
                        "batch_completed",
                        lease_id=lease.lease_id,
                        completed_total=self.completed,
                        gpu_memory_used_mib=memory,
                        peak_gpu_memory_used_mib=self.peak_memory_mib,
                    )
                finally:
                    heartbeat_finished.set()
                    heartbeat.join(timeout=5)
            self._emit(
                "worker_completed",
                completed=self.completed,
                stopping=self.stopping.is_set(),
                peak_gpu_memory_used_mib=self.peak_memory_mib,
            )
            return 0
        except Exception as exc:
            self._emit("worker_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.engine.close()
            self.store.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
