#!/usr/bin/env python3
"""Refresh only flow-head scale data and figures in an existing evaluation report."""

import fcntl
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.build_evaluation_report import flow_head_scale_data, write
from scripts.evaluation_report_flow_scale import export_flow_head_sweep


def refresh():
    root = REPO / "output/evaluation"
    directory = root / "comparisons/flow-head-scale"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".refresh.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        selection = json.loads((REPO / "configs/protocols/evaluation_report.json").read_text())
        study = flow_head_scale_data(root, selection)
        study["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        study["cfg_sweep"] = export_flow_head_sweep(root, study, write, plots=True)
        index = root / "index.html"
        original_stat = index.stat()
        original = index.read_text()
        match = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', original, re.S)
        if match is None:
            raise ValueError("Existing report data block not found")
        data = json.loads(match.group(1))
        data["flow_head_scale"] = study
        data["selection"]["flow_head_scale"] = selection["flow_head_scale"]
        template = (REPO / "scripts/assets/evaluation_report.html").read_text()
        extensions = "\n".join((REPO / "scripts/assets" / name).read_text() for name in (
            "evaluation_dashboard.js", "evaluation_flow_head_scale.js", "evaluation_unified_training.js",
            "evaluation_training.js", "evaluation_provenance.js"))
        embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        rendered = template.replace("__REPORT_EXTENSIONS__", extensions).replace("__REPORT_DATA__", embedded)
        if index.stat().st_mtime_ns != original_stat.st_mtime_ns:
            raise RuntimeError("Report changed during refresh; rerun to preserve the latest data")
        write(index, rendered)
        for name in ("summary.json", "selection.json"):
            path = root / name
            current = json.loads(path.read_text())
            current["flow_head_scale"] = study if name == "summary.json" else selection["flow_head_scale"]
            if name == "summary.json":
                current["selection"]["flow_head_scale"] = selection["flow_head_scale"]
            write(path, json.dumps(current, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps({"updated_at": study["updated_at"], "completed": study["completed"],
                          "total": study["total"], "rows": {r["id"]: r["completed"] for r in study["rows"]}}))


if __name__ == "__main__":
    refresh()
