#!/usr/bin/env python3
"""Submit independent 16-NPU Inspire Jobs for each phase of a prepared sweep."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_unified_t2i_sampling import locked, now, read, require, save, task, write


def missing_npu_node(environment, events):
    """Only exclude a node with both missing devices and missing driver mounts."""
    if environment.get("devices") != [] or environment.get("driver_libraries") != []:
        return None
    scheduled = [event["message"] for event in events if event.get("reason") == "Scheduled"]
    if not scheduled:
        return None
    match = re.search(r" to (infra-gpu-npu-[\w.-]+)$", scheduled[-1])
    return match.group(1) if match else None


def expand_pending_boundary(state, protocol):
    """Keep 0.5 spacing while evaluating several new boundary points in parallel."""
    extensions = state.get("boundary_extensions", [])
    if state["phase"] != "cfg" or not extensions or extensions[-1].get("parallel_expanded"):
        return False
    extension = extensions[-1]
    completed = [a["cfg"] for a in state["tasks"] if a["status"] == "done"]
    if not completed:
        return False
    existing = {a["cfg"] for a in state["tasks"]}
    additions = []
    for first in extension["cfg"]:
        direction = -1 if first < min(completed) else 1
        for offset in range(1, protocol.get("cfg_extension_points", 1)):
            cfg = first + direction * 0.5 * offset
            if 0 <= cfg <= protocol["cfg_limit"] and cfg not in existing:
                additions.append(cfg)
                existing.add(cfg)
    state["tasks"].extend(task(cfg, protocol["cfg_steps"], "cfg") for cfg in additions)
    extension["cfg"].extend(additions)
    extension["parallel_expanded"] = True
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-prefix", default="umm-bx0-sweep-0908-r1")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    protocol = read(root / "protocol.json")
    if protocol.get("require_smoke"):
        smoke = read(root / "smoke/audit.json")
        if protocol["schema"] == "unified_t2i_ablation_matrix_v1":
            require(bool(smoke.get("validated_models")), "validated per-model development smoke required")
        else:
            require(smoke["passed"] is True, "validated development Notebook smoke required")
    platform = protocol["platform"]
    workspace = platform["workspace"]
    context = root / "launch/cli-context"
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    manifest_path = root / "launch/jobs.json"
    manifest = read(manifest_path) if manifest_path.exists() else {"created_at": now(), "jobs": []}
    waiters = []
    request_lock = threading.Lock()
    last_request = [0.0]
    last_submission = 0.0

    def cli(argv, destination=None):
        command = ["inspire", "--json", *argv]
        for retry in range(7):
            with request_lock:
                time.sleep(max(0, 1.0 - (time.monotonic() - last_request[0])))
                last_request[0] = time.monotonic()
            completed = subprocess.run(command, cwd=context, env=env, text=True, capture_output=True)
            if "429" not in completed.stdout + completed.stderr and "Too Many Requests" not in completed.stdout + completed.stderr:
                break
            if retry == 6:
                break
            delay = min(60, 15 * (retry + 1))
            print(f"RATE_LIMIT: {argv[:2]}, retrying rejected request after {delay}s", flush=True)
            time.sleep(delay)
        try:
            result = json.loads(completed.stdout)
        except ValueError:
            raise RuntimeError(f"inspire {' '.join(argv[:2])} failed: {completed.stderr[-2000:]}") from None
        if destination:
            write(destination, result)
        require(completed.returncode == 0 and result.get("success"), f"inspire {' '.join(argv[:2])}: {result}")
        return result["data"]

    def preflight_once(folder):
        commands = {
            "quota": ["job", "quota", "--workspace", workspace],
            "availability": ["resources", "availability", "--workspace", workspace, "--group", platform["compute_group"]],
            "nodes": ["resources", "nodes", "--workspace", workspace, "--group", platform["compute_group"]],
            "image": ["image", "detail", "dev-wjx-ascend:v-1.3"],
            "active": ["job", "list", "--workspace", workspace, "--active", "--all"],
        }
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {name: pool.submit(cli, argv, folder / f"live-{name}.json") for name, argv in commands.items()}
            results = {name: future.result() for name, future in futures.items()}
        if not results["quota"]["items"] and not (folder / "quota-cache-refresh.json").exists():
            cli(["cache", "refresh", "--resource", "quota-job", "--workspace", workspace, "--full"],
                folder / "quota-cache-refresh.json")
            results["quota"] = cli(commands["quota"], folder / "live-quota-after-refresh.json")
        require(any(row["quota"] == platform["quota"] and row["compute_group"] == platform["compute_group"]
                    for row in results["quota"]["items"]), "live quota mismatch")
        require(results["image"]["status"] == "SUCCESS", "image is not ready")
        availability = results["availability"]["items"]
        # The platform occasionally returns GPU totals with an empty node
        # inventory. Retry that incomplete read before judging quota agreement.
        for attempt in range(3):
            if len(availability) == 1 and availability[0]["gpus_per_node"]:
                break
            time.sleep(2)
            availability = cli(commands["availability"], folder / f"live-availability-retry-{attempt + 1}.json")["items"]
        require(len(availability) == 1 and availability[0]["gpus_per_node"] == 16, "quota/availability mismatch")
        # The nodes endpoint currently hard-codes 8 GPUs for this 16-NPU group.
        # Retain its whole-node evidence, using the agreeing live quota and
        # availability row as the instance contract. HIGH Jobs may queue for preemption.

    def preflight(folder):
        for attempt in range(8):
            try:
                preflight_once(folder)
                return
            except (ValueError, RuntimeError) as error:
                write(folder / f"preflight-incomplete-{attempt + 1}.json", {"at": now(), "error": str(error)})
                if attempt == 7:
                    raise
                delay = min(60, 15 * (attempt + 1))
                print(f"PREFLIGHT_RETRY in {delay}s: {error}", flush=True)
                time.sleep(delay)

    def record_infrastructure_failure(job):
        if job.get("infrastructure_failure"):
            return
        current = next(a for a in read(root / "state.json")["tasks"] if a["id"] == job["task_id"])
        evidence_path = root / "arms" / job["task_id"] / "environment.json"
        if current["status"] != "failed" or not evidence_path.exists():
            return
        evidence = read(evidence_path)
        if evidence.get("worker") != current.get("worker") or evidence.get("devices") != [] or evidence.get("driver_libraries") != []:
            return
        folder = Path(job["folder"])
        events_path = folder / "terminal-events.json"
        events = (read(events_path)["data"] if events_path.exists()
                  else cli(["job", "events", job["name"], "--workspace", workspace, "--all-instances"], events_path))
        node = missing_npu_node(evidence, events["items"])
        if node is None:
            return
        write(folder / "environment.json", evidence)
        job["infrastructure_failure"] = {"node": node, "reason": "no NPU devices or driver mounted", "at": now()}
        excluded = sorted({*platform.get("exclude_nodes", []), node})
        require(len(excluded) <= 8, "more than eight nodes lack NPU mounts; platform diagnosis required")
        platform["exclude_nodes"] = excluded
        write(root / "protocol.json", protocol)
        write(manifest_path, manifest)
        print(f"EXCLUDED {node}: {job['name']} had no NPU mounts", flush=True)

    def fixed_matrix():
        """Fill bounded slots as each independent arm ends; keep one waiter per Job."""
        nonlocal waiters, last_submission
        limit = int(platform["max_concurrent_jobs"])
        require(1 <= limit <= 12, "matrix concurrency exceeds reviewed 192-NPU request")
        terminal = {"job_succeeded", "job_failed", "job_stopped", "job_cancelled", "job_deleted"}

        def begin_wait(job):
            log = (Path(job["folder"]) / "wait.json").open("w")
            process = subprocess.Popen(["inspire", "--json", "job", "wait", job["name"],
                "--workspace", workspace, "--interval", "60", "--timeout", "2592000"],
                cwd=context, env=env, stdout=log, stderr=subprocess.STDOUT)
            job["waiter_pid"] = process.pid
            waiters.append((job, process, log))
            write(manifest_path, manifest)

        for job in manifest["jobs"]:
            if not job.get("wait_completed_at"):
                old_pid = job.get("waiter_pid")
                if old_pid and Path(f"/proc/{old_pid}/cmdline").exists():
                    old_command = Path(f"/proc/{old_pid}/cmdline").read_bytes().replace(b"\0", b" ")
                    require(job["name"].encode() not in old_command, "existing live Job waiter must be reused before controller recovery")
                begin_wait(job)

        while True:
            for job, process, log in list(waiters):
                if process.poll() is None:
                    continue
                log.close()
                try:
                    outcome = read(Path(job["folder"]) / "wait.json")
                except (ValueError, OSError):
                    outcome = {}
                status = outcome.get("data", {}).get("status")
                if not outcome.get("success") or status not in terminal:
                    status = cli(["job", "status", job["name"], "--workspace", workspace],
                        Path(job["folder"]) / "monitor-recovery-status.json")["status"]
                waiters.remove((job, process, log))
                if status not in terminal:
                    # Observation ended, while the original GPU Job remains live.
                    begin_wait(job)
                    continue
                job.update(wait_completed_at=now(), wait_exit_code=process.returncode, terminal_status=status)
                record_infrastructure_failure(job)
                write(manifest_path, manifest)
                print(f"TERMINAL {job['name']} status={status}", flush=True)
            state = read(root / "state.json")
            if state["status"] == "complete" and not waiters:
                require(read(root / "smoke/audit.json")["passed"], "matrix smoke did not complete")
                require(all(j.get("wait_completed_at") for j in manifest["jobs"]), "unobserved Job remains")
                manifest.update(status="complete", completed_at=now())
                write(manifest_path, manifest)
                print("ALL_MATRIX_JOBS_COMPLETE", flush=True)
                return
            require(state["status"] != "failed", state.get("error", "matrix failed"))
            validated_models = read(root / "smoke/audit.json")["validated_models"]
            # Reserve both long D arms while its larger smoke finishes.
            effective_limit = limit - (2 if "d_on_b" not in validated_models else 0)
            live_task_ids = {j["task_id"] for j, _, _ in waiters}
            candidates = [a for a in state["tasks"] if a["status"] != "done" and a["id"] not in live_task_ids
                          and a["model"] in validated_models]
            if len(waiters) >= effective_limit or not candidates:
                time.sleep(10)
                continue
            arm = candidates[0]
            attempts = [j for j in manifest["jobs"] if j["task_id"] == arm["id"]]
            require(all(j.get("wait_completed_at") for j in attempts), "cannot retry a nonterminal Job")
            require(sum(not j.get("infrastructure_failure") for j in attempts) < 3,
                f"three evaluation attempts failed for {arm['id']}")
            with locked(root):
                state = read(root / "state.json")
                current = next(a for a in state["tasks"] if a["id"] == arm["id"])
                current.update(status="pending")
                current.pop("error", None)
                save(root, state, protocol)
            name = f"{args.name_prefix}-{arm['job_label']}" + (f"-a{len(attempts) + 1}" if attempts else "")
            folder = root / "launch/jobs" / name
            folder.mkdir(parents=True, exist_ok=True)
            time.sleep(max(0, 30 - (time.monotonic() - last_submission)))
            preflight(folder)
            known = {j["name"] for j in manifest["jobs"]}
            active = read(folder / "live-active.json")["data"]["items"]
            observed_gpus = 0
            for active_job in active:
                if active_job.get("compute_group") != platform["compute_group"]:
                    continue
                if active_job["name"] in known:
                    observed_gpus += 16
                else:
                    resource = cli(["job", "status", active_job["name"], "--workspace", workspace])["resource"]
                    observed_gpus += int(resource["gpu"]) * int(resource["nodes"])
            write(folder / "observed-allocation.json", {"at": now(), "observed_requested_gpus": observed_gpus,
                "new_request_gpus": 16, "limit": platform["max_observable_gpus"], "project_wide_visibility_claimed": False})
            if observed_gpus + 16 > int(platform["max_observable_gpus"]):
                print(f"WAIT_CAPACITY observed={observed_gpus}, new=16", flush=True)
                time.sleep(45)
                continue
            command = shlex.join(["bash", str(root / "launch/run.sh"), "--task-id", arm["id"]])
            argv = ["job", "create", "--name", name, "--workspace", workspace,
                "--project", platform["project"], "--group", platform["compute_group"],
                "--quota", platform["quota"], "--image", platform["image"], "--nodes", "1", "--priority", "6",
                "--shm-size", "512", "--max-time", "24", "--no-enable-notification", "--no-auto-fault-tolerance",
                "--command", command]
            for node in platform.get("exclude_nodes", []):
                argv.extend(["--exclude-node", node])
            plan = cli([*argv, "--dry-run"], folder / "dry-run.json")
            require(plan["project"] == platform["project"] and plan["workspace"] == workspace
                and plan["compute_group"] == platform["compute_group"] and plan["nodes"] == 1
                and all(plan["resource"].get(k) == v for k, v in {"gpu": 16, "cpu": 128, "memory_gib": 1024}.items())
                and plan["priority"] == 6 and plan["image"] == platform["image"]
                and plan["shared_memory_gib"] == 512 and plan["enable_notification"] is False
                and not plan.get("auto_fault_tolerance", False) and command in plan["command"]
                and plan.get("exclude_nodes", []) == platform.get("exclude_nodes", []), "matrix dry-run mismatch")
            cli(argv, folder / "create.json")
            last_submission = time.monotonic()
            job = {"name": name, "task_id": arm["id"], "model": arm["model"], "strategy": arm["strategy"],
                "cfg": 2., "steps": 10, "phase": "matrix", "attempt": len(attempts) + 1,
                "submitted_at": now(), "folder": str(folder)}
            manifest["jobs"].append(job)
            write(manifest_path, manifest)
            cli(["job", "events", name, "--workspace", workspace, "--tail", "4"], folder / "initial-events.json")
            status = cli(["job", "status", name, "--workspace", workspace], folder / "initial-status.json")
            cli(["job", "instances", name, "--workspace", workspace], folder / "initial-instances.json")
            require(status["project"] == platform["project"] and status["resource"]["gpu"] == 16
                and status["resource"]["nodes"] == 1 and status["priority_level"] == "HIGH", "matrix submitted resource/priority mismatch")
            begin_wait(job)
            print(f"SUBMITTED {name}: 16 NPUs; active owned Jobs={len(waiters)}/{limit}", flush=True)

    try:
        if read(root / "state.json")["phase"] == "matrix":
            fixed_matrix()
            return
        while True:
            with locked(root):
                state = read(root / "state.json")
                if expand_pending_boundary(state, protocol):
                    save(root, state, protocol)
            require(state["status"] != "failed", state.get("error", "sweep failed"))
            if state["status"] == "complete":
                manifest.update(status="complete", completed_at=now())
                write(manifest_path, manifest)
                print("ALL_SWEEP_JOBS_COMPLETE", flush=True)
                return
            arms = [arm for arm in state["tasks"] if arm["status"] != "done"]
            require(bool(arms), "no work and no completed phase transition")
            require(len(arms) <= platform["initial_jobs"], "phase exceeds configured concurrent 16-NPU Job count")
            batch = []
            for arm in arms:
                attempts = [job for job in manifest["jobs"] if job["task_id"] == arm["id"]]
                previous = attempts[-1] if attempts else None
                if previous:
                    if not previous.get("wait_completed_at"):
                        batch.append(previous)
                        continue
                    record_infrastructure_failure(previous)
                    require(sum(not job.get("previous_controller_stopped") and not job.get("infrastructure_failure")
                                for job in attempts) < 3,
                            f"three evaluation Job attempts failed for {arm['id']}")
                    with locked(root):
                        latest = read(root / "state.json")
                        retried = next(a for a in latest["tasks"] if a["id"] == arm["id"])
                        retried.update(status="pending")
                        retried.pop("error", None)
                        save(root, latest, protocol)
                cfg_label = f"{arm['cfg']:.1f}".replace(".", "p")
                label = arm.get("job_label", f"c{cfg_label}-h{arm['steps']}")
                name = f"{args.name_prefix}-{label}" + (f"-a{len(attempts) + 1}" if attempts else "")
                folder = root / "launch/jobs" / name
                folder.mkdir(parents=True, exist_ok=True)
                time.sleep(max(0, 30 - (time.monotonic() - last_submission)))
                preflight(folder)
                command = shlex.join(["bash", str(root / "launch/run.sh"), "--task-id", arm["id"]])
                argv = ["job", "create", "--name", name, "--workspace", workspace,
                        "--project", platform["project"], "--group", platform["compute_group"],
                        "--quota", platform["quota"], "--image", platform["image"],
                        "--nodes", "1", "--priority", str(platform["requested_priority"]),
                        "--shm-size", "512", "--max-time", "24",
                        "--no-enable-notification", "--no-auto-fault-tolerance", "--command", command]
                for node in platform.get("exclude_nodes", []):
                    argv.extend(["--exclude-node", node])
                plan = cli([*argv, "--dry-run"], folder / "dry-run.json")
                require(plan["project"] == platform["project"] and plan["workspace"] == workspace
                        and plan["compute_group"] == platform["compute_group"] and plan["nodes"] == 1
                        and plan["resource"]["gpu"] == 16 and plan["priority"] == 6
                        and plan["resource"]["cpu"] == int(platform["quota"].split(",")[1])
                        and plan["resource"]["memory_gib"] == int(platform["quota"].split(",")[2])
                        and plan["image"] == platform["image"] and plan["shared_memory_gib"] == 512
                        and plan["enable_notification"] is False and command in plan["command"], "dry-run mismatch")
                require(plan.get("exclude_nodes", []) == platform.get("exclude_nodes", []),
                        "dry-run node exclusions mismatch")
                cli(argv, folder / "create.json")
                last_submission = time.monotonic()
                job = {"name": name, "task_id": arm["id"], "cfg": arm["cfg"], "steps": arm["steps"],
                       "phase": arm["phase"], "attempt": len(attempts) + 1, "submitted_at": now(), "folder": str(folder)}
                manifest["jobs"].append(job)
                write(manifest_path, manifest)
                cli(["job", "events", name, "--workspace", workspace, "--tail", "4"], folder / "initial-events.json")
                status = cli(["job", "status", name, "--workspace", workspace], folder / "initial-status.json")
                cli(["job", "instances", name, "--workspace", workspace], folder / "initial-instances.json")
                require(status["project"] == platform["project"] and status["resource"]["gpu"] == 16
                        and status["resource"]["nodes"] == 1 and status["priority_level"] == "HIGH", "submitted resource/priority mismatch")
                batch.append(job)
                print(f"SUBMITTED {name}: strategy={arm.get('strategy', 'spatial_halton')}, CFG={arm['cfg']}, Heun={arm['steps']}, 16 NPUs, priority={status['priority_level']}", flush=True)
            # One blocking platform wait per Job, started only after the initial
            # configuration checks. No events/status/log/metrics polling while waiting.
            waiters = []
            for job in batch:
                log = (Path(job["folder"]) / "wait.json").open("w")
                process = subprocess.Popen(["inspire", "--json", "job", "wait", job["name"],
                                            "--workspace", workspace, "--interval", "60", "--timeout", "2592000"],
                                           cwd=context, env=env, stdout=log, stderr=subprocess.STDOUT)
                waiters.append((job, process, log))
                time.sleep(2)
            while any(process.poll() is None for _, process, _ in waiters):
                time.sleep(10)
            for job, process, log in waiters:
                log.close()
                outcome = read(Path(job["folder"]) / "wait.json")
                terminal = {"job_succeeded", "job_failed", "job_stopped", "job_cancelled", "job_deleted"}
                if not outcome.get("success") or outcome.get("data", {}).get("status") not in terminal:
                    # A failed monitor is not evidence that the GPU Job ended.
                    # Recheck once after the old wait exits, then reconnect its
                    # blocking wait without resubmitting the evaluation.
                    status = cli(["job", "status", job["name"], "--workspace", workspace],
                                 Path(job["folder"]) / "monitor-recovery-status.json")
                    if status["status"] not in terminal:
                        recovered = cli(["job", "wait", job["name"], "--workspace", workspace,
                                         "--interval", "60", "--timeout", "2592000"],
                                        Path(job["folder"]) / "monitor-recovery-wait.json")
                        require(recovered["status"] in terminal, "monitor did not establish terminal Job state")
                job.update(wait_completed_at=now(), wait_exit_code=process.returncode)
                record_infrastructure_failure(job)
                print(f"TERMINAL {job['name']} wait_exit={process.returncode}", flush=True)
            write(manifest_path, manifest)
            state = read(root / "state.json")
            for job in batch:
                arm = next(a for a in state["tasks"] if a["id"] == job["task_id"])
                if arm["status"] != "done":
                    print(f"RETRY_REQUIRED {job['name']}: {arm.get('error', 'platform job terminated before metrics completed')}", flush=True)
            print(f"PHASE_RESULTS phase={state['phase']} complete={sum(a['status'] == 'done' for a in state['tasks'])}/{len(state['tasks'])}", flush=True)
    except BaseException as error:
        for _, process, log in waiters:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            log.close()
        write(root / "launch/NEEDS_ATTENTION.json", {"error": str(error), "at": now()})
        # A controller/network error must not cancel healthy independent Jobs.
        # Their results and resumable progress remain valid for controller recovery.
        raise


if __name__ == "__main__":
    main()
