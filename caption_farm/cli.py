from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import load_config, prepare_run, validate_config
from .controller import (
    Controller,
    acknowledge_attention,
    start_controller_daemon,
    supervise_controller,
    wait_for_controller,
)
from .io import load_json
from .model import validate_model_snapshot
from .probe import submit_probe, verify_probe
from .publish import audit_run, publish_run
from .queue import TaskStore
from .worker import Worker


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recoverable local-Qwen ImageNet caption farm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=_path, required=True)

    model = subparsers.add_parser("validate-model")
    model.add_argument("--snapshot", type=_path, required=True)
    model.add_argument("--expected-quant-method", default="fp8")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=_path, required=True)
    prepare.add_argument("--run-dir", type=_path, required=True)
    prepare.add_argument("--rebuild-manifest", action="store_true")

    queue = subparsers.add_parser("queue")
    queue_sub = queue.add_subparsers(dest="queue_command", required=True)
    queue_status = queue_sub.add_parser("status")
    queue_status.add_argument("--run-dir", type=_path, required=True)
    queue_repair = queue_sub.add_parser("repair-failed")
    queue_repair.add_argument("--run-dir", type=_path, required=True)
    queue_repair.add_argument("--reason", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--run-dir", type=_path, required=True)
    worker.add_argument("--worker-id")
    worker.add_argument("--max-tasks", type=int)
    worker.add_argument("--request-concurrency", type=int)
    worker.add_argument("--claim-batch-size", type=int)
    worker.add_argument("--max-num-seqs", type=int)
    worker.add_argument("--server-port", type=int)
    worker.add_argument("--lease-seconds", type=float)
    worker.add_argument("--post-claim-delay-seconds", type=float, default=0.0)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--run-dir", type=_path, required=True)
    audit.add_argument("--verify-images", action="store_true")

    publish = subparsers.add_parser("publish")
    publish.add_argument("--run-dir", type=_path, required=True)
    publish.add_argument("--verify-images", action="store_true")

    probe = subparsers.add_parser("probe")
    probe_sub = probe.add_subparsers(dest="probe_command", required=True)
    probe_submit = probe_sub.add_parser("submit")
    probe_submit.add_argument("--run-dir", type=_path, required=True)
    probe_submit.add_argument("--max-tasks", type=int, default=16)
    probe_verify = probe_sub.add_parser("verify")
    probe_verify.add_argument("--run-dir", type=_path, required=True)

    controller = subparsers.add_parser("controller")
    controller_sub = controller.add_subparsers(dest="controller_command", required=True)
    for name in ("run", "dry-run"):
        command = controller_sub.add_parser(name)
        command.add_argument("--run-dir", type=_path, required=True)
        command.add_argument("--once", action="store_true")
    supervise = controller_sub.add_parser("supervise")
    supervise.add_argument("--run-dir", type=_path, required=True)
    start = controller_sub.add_parser("start")
    start.add_argument("--run-dir", type=_path, required=True)
    start.add_argument("--script-path", type=_path, required=True)
    status = controller_sub.add_parser("status")
    status.add_argument("--run-dir", type=_path, required=True)
    wait = controller_sub.add_parser("wait")
    wait.add_argument("--run-dir", type=_path, required=True)
    wait.add_argument("--interval", type=float, default=10.0)
    for name in ("pause", "resume", "stop"):
        command = controller_sub.add_parser(name)
        command.add_argument("--run-dir", type=_path, required=True)
    acknowledge = controller_sub.add_parser("acknowledge")
    acknowledge.add_argument("--run-dir", type=_path, required=True)
    acknowledge.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        validate_config(config)
        _print({"status": "valid", "config": str(args.config)})
        return 0
    if args.command == "validate-model":
        _print(
            validate_model_snapshot(
                args.snapshot, expected_quant_method=args.expected_quant_method
            )
        )
        return 0
    if args.command == "prepare":
        _print(
            prepare_run(
                args.config,
                args.run_dir,
                rebuild_manifest=args.rebuild_manifest,
            )
        )
        return 0
    if args.command == "queue":
        store = TaskStore(args.run_dir)
        try:
            if args.queue_command == "status":
                _print(store.snapshot())
            elif args.queue_command == "repair-failed":
                _print(store.repair_failed(args.reason))
            else:
                raise AssertionError("unreachable")
        finally:
            store.close()
        return 0
    if args.command == "worker":
        return Worker(
            args.run_dir,
            worker_id=args.worker_id,
            max_tasks=args.max_tasks,
            request_concurrency=args.request_concurrency,
            claim_batch_size=args.claim_batch_size,
            max_num_seqs=args.max_num_seqs,
            server_port=args.server_port,
            lease_seconds=args.lease_seconds,
            post_claim_delay_seconds=args.post_claim_delay_seconds,
        ).run_forever()
    if args.command == "audit":
        _print(audit_run(args.run_dir, verify_images=args.verify_images))
        return 0
    if args.command == "publish":
        _print(publish_run(args.run_dir, verify_images=args.verify_images))
        return 0
    if args.command == "probe":
        if args.probe_command == "submit":
            _print(submit_probe(args.run_dir, max_tasks=args.max_tasks))
            return 0
        if args.probe_command == "verify":
            _print(verify_probe(args.run_dir))
            return 0
    if args.command == "controller":
        if args.controller_command == "supervise":
            return supervise_controller(args.run_dir)
        if args.controller_command in {"run", "dry-run"}:
            return Controller(
                args.run_dir,
                once=args.once,
                dry_run=args.controller_command == "dry-run",
            ).run_forever()
        if args.controller_command == "start":
            _print(start_controller_daemon(args.run_dir, args.script_path))
            return 0
        if args.controller_command == "status":
            status_path = args.run_dir / "status.json"
            if status_path.is_file():
                _print(load_json(status_path))
            else:
                store = TaskStore(args.run_dir)
                try:
                    _print({"phase": "NOT_STARTED", "queue": store.snapshot()})
                finally:
                    store.close()
            return 0
        if args.controller_command == "wait":
            return wait_for_controller(args.run_dir, interval_seconds=args.interval)
        if args.controller_command == "pause":
            (args.run_dir / "PAUSE").touch()
            _print({"status": "pause_requested", "path": str(args.run_dir / "PAUSE")})
            return 0
        if args.controller_command == "resume":
            (args.run_dir / "PAUSE").unlink(missing_ok=True)
            _print({"status": "resumed"})
            return 0
        if args.controller_command == "stop":
            (args.run_dir / "STOP").touch()
            pid_path = args.run_dir / "controller.pid"
            if pid_path.is_file():
                try:
                    os.kill(int(pid_path.read_text(encoding="utf-8").strip()), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
            _print({"status": "safe_stop_requested", "path": str(args.run_dir / "STOP")})
            return 0
        if args.controller_command == "acknowledge":
            _print(acknowledge_attention(args.run_dir, args.reason))
            return 0
    raise AssertionError("unreachable")
