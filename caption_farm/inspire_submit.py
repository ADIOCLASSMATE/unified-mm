"""Narrow Inspire Web-API adapter for attaching an official dataset.

This module is intentionally executed with Inspire CLI's own Python runtime.
It accepts one JSON request on stdin and emits one JSON response on stdout so
the caption-farm environment does not need to duplicate browser credentials.
"""

from __future__ import annotations

import json
import sys
from typing import Any


FIXED_IMAGE = "docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1"


def _validate_payload(payload: Any, expected: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(expected, dict):
        raise ValueError("payload and expected must be JSON objects")
    if int(payload.get("task_priority") or 0) != 1:
        raise ValueError("refusing to submit anything except task_priority=1")
    framework = payload.get("framework_config")
    if not isinstance(framework, list) or len(framework) != 1:
        raise ValueError("exactly one framework_config entry is required")
    worker = framework[0]
    if not isinstance(worker, dict):
        raise ValueError("framework_config entry must be an object")
    if int(worker.get("gpu_count") or 0) != 1 or int(worker.get("instance_count") or 0) != 1:
        raise ValueError("refusing to submit anything except one 1-GPU worker")
    if expected.get("image") != FIXED_IMAGE or worker.get("image") != FIXED_IMAGE:
        raise ValueError("refusing to submit a non-fixed image")
    dataset_info = payload.get("dataset_info")
    expected_dataset = {
        "dataset_id": expected.get("dataset_id"),
        "version_id": expected.get("version_id"),
        "path": expected.get("dataset_path"),
    }
    if dataset_info != [expected_dataset]:
        raise ValueError("official dataset_info is missing or differs from the audited dataset")
    return payload


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    """Pure validation entry point used by tests and defensive callers."""
    if request.get("action") != "create-training-job":
        raise ValueError("unsupported adapter action")
    return _validate_payload(request.get("payload"), request.get("expected"))


def _resolve_payload(request: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    plan = request.get("plan")
    dry_run = request.get("dry_run_create_kwargs")
    expected = request.get("expected")
    if not isinstance(plan, dict) or not isinstance(dry_run, dict):
        raise ValueError("plan and dry_run_create_kwargs must be JSON objects")

    from inspire.cli.utils import job_submit
    from inspire.cli.utils.quota_resolver import (
        SCHEDULE_TYPE_TRAIN,
        parse_quota,
        resolve_quota,
    )
    from inspire.config import Config
    from inspire.config.workspaces import select_workspace_id
    from inspire.platform.web.session import get_web_session

    config, _ = Config.from_files_and_env()
    session = get_web_session()
    workspace_id = select_workspace_id(
        config,
        explicit_workspace_name=str(plan["workspace"]),
        session=session,
    )
    if not workspace_id:
        raise ValueError("could not resolve the requested workspace")
    quota = resolve_quota(
        spec=parse_quota(str(plan["quota"])),
        workspace_id=workspace_id,
        session=session,
        schedule_config_type=SCHEDULE_TYPE_TRAIN,
        group_override=str(plan["group"]),
    )
    project, _ = job_submit.select_project_for_workspace(
        config,
        workspace_id=workspace_id,
        requested=str(plan["project"]),
    )
    resolved = job_submit.build_training_job_plan(
        config=config,
        name=str(plan["name"]),
        command=str(plan["command"]),
        quota=quota,
        framework="pytorch",
        project_id=project.project_id,
        workspace_id=workspace_id,
        image=str(plan["image"]),
        priority=int(plan["priority"]),
        nodes=int(plan["nodes"]),
        max_time_hours=float(plan["max_time_hours"]),
        project_name=project.name,
        auto_fault_tolerance=False,
        enable_notification=bool(plan["enable_notification"]),
        shm_size=int(plan["shm_size_gib"]),
    )
    payload = resolved.create_kwargs
    for key in (
        "name",
        "framework",
        "task_priority",
        "enable_notification",
        "max_running_time_ms",
    ):
        if dry_run.get(key) != payload.get(key):
            raise ValueError(f"resolved create payload drifted from CLI dry-run at {key}")
    dry_framework = (dry_run.get("framework_config") or [{}])[0]
    resolved_framework = (payload.get("framework_config") or [{}])[0]
    for key in ("image_type", "image", "instance_count", "cpu", "gpu_count", "mem_gi", "shm_gi"):
        if dry_framework.get(key) != resolved_framework.get(key):
            raise ValueError(
                f"resolved create payload drifted from CLI dry-run at framework_config.{key}"
            )
    payload["dataset_info"] = [
        {
            "dataset_id": expected.get("dataset_id"),
            "version_id": expected.get("version_id"),
            "path": expected.get("dataset_path"),
        }
    ]
    return _validate_payload(payload, expected), session


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("adapter request must be a JSON object")
        if request.get("action") == "resolve-and-create-training-job":
            payload, session = _resolve_payload(request)
        else:
            payload = _validate_request(request)
            from inspire.platform.web.session import get_web_session

            session = get_web_session()
        from inspire.platform.web.browser_api import create_training_job

        data = create_training_job(payload=payload, session=session)
        print(json.dumps({"success": True, "data": data}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"success": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
