"""Record verified fixed-Notebook cleanup from captured live CLI responses."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from utils.research.geometry_v5_assets import RUN, emit, write_json


def main(root):
    responses = {
        name: json.loads((root / "logs" / f"notebook-{name}.json").read_text())
        for name in ("before-stop", "stop-response", "after-stop")
    }
    assert all(r["success"] for r in responses.values())
    before, after = responses["before-stop"]["data"], responses["after-stop"]["data"]
    for value in (before, after):
        assert (
            value["name"] == "dev-wjx-ascend" and value["workspace"] == "昇腾卡公共空间"
        )
        assert value["compute_group"] == "910B资源"
    assert before["status"] == "RUNNING" and after["status"] == "STOPPED"
    assert before["created_at"] == after["created_at"]
    stages = ["f-cal-smoke", "f-full", "f-robust"]
    stages += [
        f"{name}-{stage}"
        for name in ("qwen_text", "dinov2", "mae", "siglip")
        for stage in ("cal-smoke", "full")
    ]
    stages += [
        f"{name}-{stage}"
        for name in ("janusflow", "showo2")
        for stage in ("cal-smoke", "full", "robust")
    ]
    for stage in stages:
        value = json.loads((root / "supervisor-markers" / f"{stage}.json").read_text())
        assert value["returncode"] == 0
    for name in (
        "b",
        "f",
        "qwen_text",
        "dinov2",
        "mae",
        "siglip",
        "janusflow",
        "showo2",
    ):
        value = json.loads((root / "audits" / f"features-{name}.json").read_text())
        assert value["status"] == "passed"
    result = {
        "schema": "geometry_v5_compute_cleanup_1",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "status": "verified_stopped",
        "notebook": after["name"],
        "workspace": after["workspace"],
        "notebook_object_preserved": True,
        "artifacts_deleted": False,
        "completed_extraction_stages": stages,
        "feature_audits_passed_before_stop": True,
        "no_remaining_geometry_process_evidence": str(
            root / "logs/notebook-final-process-check.log"
        ),
        "live_responses": responses,
        "remaining_cpu_analysis_is_not_completion": True,
    }
    write_json(root / "resource-cleanup.json", result)
    emit(
        "v5_notebook_cleanup_verified", status=after["status"], notebook_preserved=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    main(parser.parse_args().output_dir)
