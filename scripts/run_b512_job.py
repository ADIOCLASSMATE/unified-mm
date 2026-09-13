"""Run ordered B512 stages after their local dependency jobs complete."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from data_synthesis.io import atomic_json


def run_job(path):
    path = Path(path).resolve()
    job, root = json.loads(path.read_text()), path.parent
    lock = (root / "job.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"controller_pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}
    proc = None
    try:
        for stage in job["stages"]:
            dependencies = stage.get("wait_for", [])
            if isinstance(dependencies, str):
                dependencies = [dependencies]
            status.update(state="queued", stage=stage["name"], waiting_for=dependencies)
            status.pop("child_pid", None)
            atomic_json(root / "job_status.json", status)
            while dependencies:
                pending = []
                for dependency in dependencies:
                    value = json.loads(Path(dependency).read_text())
                    if value["state"] == "completed":
                        continue
                    if value["state"] not in {"running", "queued"}:
                        raise RuntimeError(f"dependency is {value['state']}: {dependency}")
                    state_path = Path("/proc") / str(value["controller_pid"]) / "stat"
                    if not state_path.exists() or state_path.read_text().split()[2] == "Z":
                        raise RuntimeError(f"local dependency controller is not alive: {dependency}")
                    pending.append(dependency)
                dependencies = pending
                if dependencies:
                    time.sleep(10)
            status.update(state="running", stage=stage["name"])
            status.pop("waiting_for", None)
            with (root / (stage["name"] + ".log")).open("a") as log:
                proc = subprocess.Popen(stage["argv"], cwd=job["cwd"], stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status["child_pid"] = proc.pid
            atomic_json(root / "job_status.json", status)
            code = proc.wait()
            proc = None
            if code:
                raise RuntimeError(f"stage {stage['name']} exited with code {code}")
        status.update(state="completed", exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(root / "job_status.json", status)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        atomic_json(root / "job_status.json", status)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    args = parser.parse_args()
    def stop(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    run_job(args.job)


if __name__ == "__main__":
    main()
