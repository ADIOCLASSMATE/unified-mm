from __future__ import annotations

import json
import os
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .io import utc_now


class InspireError(RuntimeError):
    pass


class DatasetMountError(InspireError):
    pass


def is_transient_platform_error(value: Any) -> bool:
    text = str(value or "").lower()
    return any(
        marker in text
        for marker in (
            "econnrefused",
            "econnreset",
            "etimedout",
            "connection refused",
            "connection reset",
            "temporarily unavailable",
            "too many requests",
            "timed out",
            "timeout",
            "http 429",
            "http 502",
            "http 503",
            "http 504",
        )
    )


@dataclass(frozen=True)
class LiveTarget:
    project: str
    project_weight: int
    project_max_active_jobs: int
    group: str
    quota: str
    target_weight: int
    target_max_active_jobs: int
    available_gpus: int
    low_priority_gpus: int

    @property
    def key(self) -> str:
        return f"{self.project}|{self.group}|{self.quota}"

    @property
    def weight(self) -> int:
        return self.project_weight * self.target_weight


class InspireClient:
    def __init__(self, run: dict[str, Any], *, executable: str = "inspire") -> None:
        self.run = run
        self.platform = run["platform"]
        self.executable = executable
        self._image_readiness_cache: tuple[float, dict[str, Any]] | None = None
        self._target_discovery_cache: (
            tuple[float, list[LiveTarget], dict[str, Any]] | None
        ) = None

    @staticmethod
    def _matches_fixed_image_address(candidate: Any, expected: str) -> bool:
        value = str(candidate or "").strip()
        if value == expected:
            return True
        # Notebook details expose private images without the registry host,
        # while Job plans use the fully-qualified registry URL. Only accept
        # that one exact host-stripped identity; do not allow tag or basename
        # matching.
        _, separator, repository = expected.partition("/")
        return bool(separator and value == repository)

    @classmethod
    def _status_has_fixed_image(cls, status: dict[str, Any], expected: str) -> bool:
        framework = status.get("framework_config")
        if not isinstance(framework, list) or len(framework) != 1:
            return False
        worker = framework[0]
        return bool(
            isinstance(worker, dict)
            and worker.get("image") == expected
            and int(worker.get("gpu_count") or 0) == 1
            and int(worker.get("instance_count") or 0) == 1
        )

    def _run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        timeout: float = 180,
    ) -> dict[str, Any]:
        command = [self.executable, "--json", *args]
        retries = int(self.platform.get("api_rate_limit_retries", 4))
        for attempt in range(retries + 1):
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
            output = completed.stdout.strip()
            try:
                value = json.loads(output) if output else {}
            except json.JSONDecodeError as exc:
                raise InspireError(
                    f"inspire returned non-JSON output for {' '.join(args)}: "
                    f"{(completed.stderr or output)[-2000:]}"
                ) from exc
            error = value.get("error") if isinstance(value, dict) else None
            message = (error or {}).get("message") if isinstance(error, dict) else None
            combined_error = str(message or completed.stderr.strip())
            retryable = is_transient_platform_error(combined_error)
            if not (check and (completed.returncode != 0 or value.get("success") is False)):
                break
            if retryable and attempt < retries:
                delay = min(15.0, 2.0 ** attempt) * random.uniform(0.8, 1.2)
                time.sleep(delay)
                continue
            raise InspireError(combined_error or f"inspire exited {completed.returncode}")
        value["_returncode"] = completed.returncode
        value["_stderr"] = completed.stderr[-4000:]
        return value

    def verify_image_ready(self) -> dict[str, Any]:
        now = time.monotonic()
        cache_seconds = float(self.platform.get("image_readiness_cache_seconds", 3600))
        if (
            self._image_readiness_cache is not None
            and now - self._image_readiness_cache[0] < cache_seconds
        ):
            return deepcopy(self._image_readiness_cache[1])
        image = self.platform["image"]
        listing = self._run(["image", "list", "--source", "all"])
        for item in (listing.get("data") or {}).get("images") or []:
            if item.get("url") == image or item.get("name") == image:
                status = str(item.get("status") or "").upper()
                if status not in {"READY", "SUCCESS"}:
                    raise InspireError(f"fixed image is not READY: {status or '<missing>'}")
                evidence = {
                    "source": "image-list",
                    "status": status,
                    "image": image,
                    "item": item,
                }
                self._image_readiness_cache = (now, evidence)
                return deepcopy(evidence)

        readiness_notebook = self.platform.get("image_readiness_notebook")
        if readiness_notebook:
            detail = self._run(
                [
                    "notebook",
                    "status",
                    str(readiness_notebook),
                    "--workspace",
                    self.platform["workspace"],
                ]
            )
            notebook = detail.get("data") or {}
            configured_image = (notebook.get("start_config") or {}).get("mirror_url")
            notebook_image = notebook.get("image") or {}
            notebook_status = str(notebook.get("status") or "").upper()
            image_address = notebook_image.get("address")
            if (
                notebook_status in {"RUNNING", "STOPPED"}
                and configured_image == image
                and self._matches_fixed_image_address(image_address, image)
            ):
                evidence = {
                    "source": "notebook-image-identity",
                    "status": "READY",
                    "image": image,
                    "notebook": readiness_notebook,
                    "evidence": {
                        "notebook_status": notebook_status,
                        "configured_image": configured_image,
                        "image_address": image_address,
                        "image_source": notebook_image.get("source"),
                    },
                }
                self._image_readiness_cache = (now, evidence)
                return deepcopy(evidence)
        raise InspireError(
            "fixed image is not visible as READY/SUCCESS and no exact live notebook image identity matched"
        )

    def discover_targets(self) -> tuple[list[LiveTarget], dict[str, Any]]:
        now = time.monotonic()
        cache_seconds = float(self.platform.get("target_discovery_cache_seconds", 300))
        if (
            self._target_discovery_cache is not None
            and now - self._target_discovery_cache[0] < cache_seconds
        ):
            _, targets, evidence = self._target_discovery_cache
            return list(targets), deepcopy(evidence)
        workspace = self.platform["workspace"]
        project_response = self._run(["project", "list", "--workspace", workspace])
        quota_response = self._run(["job", "quota", "--workspace", workspace])
        availability_response = self._run(
            ["resources", "availability", "--workspace", workspace, "--no-cache"]
        )
        live_projects = {
            item["name"]: item
            for item in (project_response.get("data") or {}).get("projects") or []
        }
        availability = {
            item["group_name"]: item
            for item in (availability_response.get("data") or {}).get("availability") or []
        }
        quota_rows = (quota_response.get("data") or {}).get("quotas") or []
        projects = {item["name"]: item for item in self.platform["projects"]}
        targets = {item["group"]: item for item in self.platform["targets"]}
        missing_projects = sorted(set(projects) - set(live_projects))
        if missing_projects:
            raise InspireError(f"whitelisted projects are not live: {missing_projects}")
        discovered: list[LiveTarget] = []
        for row in quota_rows:
            group = str(row.get("compute_group_name") or "")
            target = targets.get(group)
            if target is None:
                continue
            if str(row.get("gpu_type") or "").upper().find("H100") < 0:
                continue
            if int(row.get("gpu_count") or 0) != 1 or str(row.get("quota")) != str(target["quota"]):
                continue
            available = availability.get(group) or {}
            for project_name, project in projects.items():
                discovered.append(
                    LiveTarget(
                        project=project_name,
                        project_weight=int(project.get("weight", 1)),
                        project_max_active_jobs=int(project["max_active_jobs"]),
                        group=group,
                        quota=str(row["quota"]),
                        target_weight=int(target.get("weight", 1)),
                        target_max_active_jobs=int(target["max_active_jobs"]),
                        available_gpus=int(available.get("available_gpus") or 0),
                        low_priority_gpus=int(available.get("low_priority_gpus") or 0),
                    )
                )
        if not discovered:
            raise InspireError("no live one-card H100 target matched the farm whitelist")
        evidence = {
            "timestamp": utc_now(),
            "workspace": workspace,
            "projects": {name: live_projects[name] for name in projects},
            "quotas": quota_rows,
            "availability": availability,
            "targets": [target.__dict__ | {"key": target.key} for target in discovered],
        }
        self._target_discovery_cache = (now, list(discovered), deepcopy(evidence))
        return discovered, evidence

    def list_jobs(self, *, active: bool = True, keyword: str | None = None) -> list[dict[str, Any]]:
        args = ["job", "list", "--workspace", self.platform["workspace"]]
        if active:
            args.append("--active")
        if keyword:
            args.extend(["--keyword", keyword])
        response = self._run(args)
        return list((response.get("data") or {}).get("jobs") or [])

    def job_status(self, name: str) -> dict[str, Any]:
        response = self._run(
            ["job", "status", name, "--workspace", self.platform["workspace"]]
        )
        return (response.get("data") or {}).get("job") or {}

    def _create_args(self, name: str, command: str, target: LiveTarget) -> list[str]:
        return [
            "job",
            "create",
            "--name",
            name,
            "--workspace",
            self.platform["workspace"],
            "--project",
            target.project,
            "--group",
            target.group,
            "--quota",
            target.quota,
            "--image",
            self.platform["image"],
            "--nodes",
            "1",
            "--priority",
            "1",
            "--shm-size",
            str(self.platform.get("shm_size_gib", 64)),
            "--max-time",
            str(self.platform.get("max_time_hours", 24)),
            "--no-auto-fault-tolerance",
            "--enable-notification",
            "--command",
            command,
        ]

    @staticmethod
    def _find_values(value: Any, key: str) -> list[Any]:
        found: list[Any] = []
        if isinstance(value, dict):
            for item_key, item in value.items():
                if item_key == key:
                    found.append(item)
                found.extend(InspireClient._find_values(item, key))
        elif isinstance(value, list):
            for item in value:
                found.extend(InspireClient._find_values(item, key))
        return found

    def dry_run_job(self, name: str, command: str, target: LiveTarget) -> dict[str, Any]:
        response = self._run([*self._create_args(name, command, target), "--dry-run"])
        priority_values = [int(value) for value in self._find_values(response, "task_priority")]
        gpu_values = [int(value) for value in self._find_values(response, "gpu_count")]
        if 1 not in priority_values:
            raise InspireError(f"dry-run did not resolve task_priority=1: {priority_values}")
        if not gpu_values or any(value != 1 for value in gpu_values):
            raise InspireError(f"dry-run did not resolve a single GPU consistently: {gpu_values}")
        return response

    @staticmethod
    def _create_payload_from_dry_run(dry_run: dict[str, Any]) -> dict[str, Any]:
        candidates = [dry_run, dry_run.get("data")]
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("create_kwargs"), dict):
                return deepcopy(candidate["create_kwargs"])
        raise InspireError("inspire dry-run did not expose create_kwargs for an audited submit")

    def _submit_with_dataset(
        self,
        dry_run: dict[str, Any],
        *,
        name: str,
        command: str,
        target: LiveTarget,
    ) -> dict[str, Any]:
        """Submit the audited CLI plan through the Web API with dataset_info added.

        Inspire CLI 6.2.0 resolves projects, quotas, images, and priorities but
        does not expose the Web UI's official-dataset field.  The small helper
        runs under Inspire's own Python environment and only adds that field to
        the exact dry-run payload.
        """

        helper_python = str(
            self.platform.get("inspire_python")
            or "/root/.local/share/uv/tools/inspire-skill/bin/python"
        )
        repository = str(self.platform["repository_path"])
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            repository
            if not existing_pythonpath
            else os.pathsep.join((repository, existing_pythonpath))
        )
        request = {
            "action": "resolve-and-create-training-job",
            "plan": {
                "name": name,
                "command": command,
                "workspace": self.platform["workspace"],
                "project": target.project,
                "group": target.group,
                "quota": target.quota,
                "image": self.platform["image"],
                "priority": 1,
                "nodes": 1,
                "shm_size_gib": int(self.platform.get("shm_size_gib", 64)),
                "max_time_hours": float(self.platform.get("max_time_hours", 24)),
                "enable_notification": True,
            },
            "dry_run_create_kwargs": self._create_payload_from_dry_run(dry_run),
            "expected": {
                "image": self.platform["image"],
                "dataset_id": self.run["dataset"]["dataset_id"],
                "version_id": self.run["dataset"]["version_id"],
                "dataset_path": self.run["dataset"]["platform_path"],
            },
        }
        retries = int(self.platform.get("submission_rate_limit_retries", 3))
        base_delay = float(self.platform.get("submission_rate_limit_base_seconds", 30))
        max_delay = float(self.platform.get("submission_rate_limit_max_seconds", 120))
        for attempt in range(retries + 1):
            completed = subprocess.run(
                [helper_python, "-m", "caption_farm.inspire_submit"],
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=repository,
                env=environment,
                timeout=180,
            )
            try:
                response = json.loads(completed.stdout) if completed.stdout.strip() else {}
            except json.JSONDecodeError as exc:
                raise InspireError(
                    "official-dataset submit helper returned non-JSON output: "
                    f"{(completed.stderr or completed.stdout)[-2000:]}"
                ) from exc
            error = str(
                response.get("error")
                or completed.stderr.strip()
                or "dataset submit failed"
            )
            if completed.returncode == 0 and response.get("success") is True:
                data = response.get("data")
                if not isinstance(data, dict):
                    raise InspireError("official-dataset submit helper returned no job data")
                return data
            if not is_transient_platform_error(error) or attempt >= retries:
                raise InspireError(error)

            # A gateway may return 429 after accepting the create request. Check
            # the unique Job name before retrying so recovery is idempotent.
            try:
                existing = self.job_status(name)
            except InspireError:
                existing = {}
            if existing:
                return {
                    "recovered_after_transient_submit_error": True,
                    "job": existing,
                    "original_error": error[-2000:],
                }
            delay = min(max_delay, base_delay * 2**attempt)
            time.sleep(delay * random.uniform(0.9, 1.1))
        raise AssertionError("unreachable")

    def submit_job(self, name: str, command: str, target: LiveTarget) -> dict[str, Any]:
        image_evidence = self.verify_image_ready()
        dry_run = self.dry_run_job(name, command, target)
        audited_payload = self._create_payload_from_dry_run(dry_run)
        audited_payload["dataset_info"] = [
            {
                "dataset_id": self.run["dataset"]["dataset_id"],
                "version_id": self.run["dataset"]["version_id"],
                "path": self.run["dataset"]["platform_path"],
            }
        ]
        response = self._submit_with_dataset(
            dry_run,
            name=name,
            command=command,
            target=target,
        )
        status: dict[str, Any] = {}
        attempts = int(self.platform.get("submission_verification_attempts", 5))
        interval = float(self.platform.get("submission_verification_interval_seconds", 2))
        for attempt in range(attempts):
            status = self.job_status(name)
            priority_verified = (
                str(status.get("priority_name") or "") == "1"
                or int(status.get("task_priority") or 0) == 1
            )
            dataset_info = status.get("dataset_info") or []
            expected_path = self.run["dataset"]["platform_path"]
            dataset_verified = any(
                item.get("path") == expected_path
                for item in dataset_info
                if isinstance(item, dict)
            )
            image_verified = self._status_has_fixed_image(status, self.platform["image"])
            if priority_verified and dataset_verified and image_verified:
                break
            if attempt + 1 < attempts:
                time.sleep(interval)
        if str(status.get("priority_name") or "") != "1" and int(status.get("task_priority") or 0) != 1:
            self.stop_job(name, check=False)
            raise InspireError(
                f"submitted job did not resolve to LOW priority=1: "
                f"priority_name={status.get('priority_name')} task_priority={status.get('task_priority')}"
            )
        if not self._status_has_fixed_image(status, self.platform["image"]):
            self.stop_job(name, check=False)
            raise InspireError(
                f"submitted job did not retain the fixed one-GPU image "
                f"{self.platform['image']}"
            )
        dataset_info = status.get("dataset_info") or []
        expected_path = self.run["dataset"]["platform_path"]
        if not any(item.get("path") == expected_path for item in dataset_info if isinstance(item, dict)):
            self.stop_job(name, check=False)
            raise DatasetMountError(
                f"job {name} has no verified ImageNet dataset_info entry for {expected_path}; job stopped"
            )
        return {
            "submitted_at": utc_now(),
            "response": response,
            "dry_run": dry_run,
            # The CLI deliberately scrubs platform IDs from dry-run JSON.  The
            # helper resolves those IDs internally; this is the complete
            # non-secret portion that was audited before submission.
            "submitted_payload": audited_payload,
            "image_evidence": image_evidence,
            "status": status,
        }

    def stop_job(self, name: str, *, check: bool = True) -> dict[str, Any]:
        return self._run(
            ["job", "stop", name, "--workspace", self.platform["workspace"]],
            check=check,
        )

    def diagnose_job(self, name: str) -> dict[str, Any]:
        result: dict[str, Any] = {"name": name, "timestamp": utc_now()}
        for kind, args in (
            ("events", ["job", "events", name, "--workspace", self.platform["workspace"], "--tail", "50"]),
            ("logs", ["job", "logs", name, "--workspace", self.platform["workspace"], "--tail", "200"]),
        ):
            try:
                result[kind] = self._run(args)
            except Exception as exc:
                result[kind] = {"error": str(exc)}
            time.sleep(random.uniform(0.35, 0.65))
        return result
