"""Shared run identity for launchers, histories and qualitative evaluation.

Checkpoint model contracts remain authoritative for architecture and weights.
The registry names retained experiments; new runs can declare experiment.identity
in their config. Name-based fallbacks only support historical output layouts.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from utils.atomic_io import atomic_write_text

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs/protocols/experiment_registry.json"
GROUPS = {"main", "single", "scaling", "legacy", "lr"}
SOURCES = {"climbmix", "t2i", "i2t"}


@lru_cache(maxsize=1)
def registered_experiments():
    data = json.loads(REGISTRY_PATH.read_text())
    if data.get("schema") != "experiment_registry_v1":
        raise ValueError("unsupported experiment registry")
    rows = data["experiments"]
    for field in ("run", "id"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"duplicate experiment {field}")
    return {row["run"]: row for row in rows}


def model_labels():
    return {run: (row["id"], row["label"]) for run, row in registered_experiments().items()}


def is_temporary_training_run(name: str) -> bool:
    return any(marker in name for marker in ("smoke", "debug", "replay"))


def experiment_identity(run: str, config=None, *, presentation=None):
    config = config or {}
    registered = registered_experiments().get(run, {})
    declared = dict(config.get("experiment", {}).get("identity", {}))
    display = presentation or registered
    schedule = config.get("dataset", {}).get("params", {}).get("schedule")
    sources = sorted(set(schedule)) if schedule else registered.get("active_sources")
    if sources is not None:
        sources = sorted(set(sources))
    groups_by_id = {row["id"]: row["group"] for row in registered_experiments().values()}
    group = registered.get("group", groups_by_id.get(display.get("id"), "legacy"))
    label = display.get("label", run.removeprefix("unified-").removesuffix("-100b-imagenet-split-s42-r1"))
    # Compatibility for old configs that did not declare presentation fields.
    if "-only-" in run:
        group = "single"
        if not display:
            prefix = "B" if Path(run).name.startswith("unified-b-") else "A"
            task = "T2I-only" if "t2i-only" in run else "caption-only" if "caption-only" in run else "text-only"
            label = f"{prefix} · {task}"
    elif "flowdepth" in run:
        group = "scaling"
        label = f"B_x0 · flow depth {config.get('model', {}).get('image_flow_depth', '?')}"
    elif "-lr-sweep-" in run:
        group, label = "lr", f"A 1.7B · {Path(run).name}"
    elif run.startswith("unified-e-on-b-0p6b"):
        label = "E on B · 旧 shared-condition"
    identity = {"schema": "experiment_identity_v1", "run": run,
                "id": display.get("id", run), "label": label, "group": group,
                "purpose": registered.get("purpose", "unclassified"),
                "active_sources": sources,
                "identity_source": "registry" if registered else "legacy_layout"}
    allowed = {"id", "label", "group", "purpose"}
    if set(declared) - allowed:
        raise ValueError(f"unknown experiment identity fields: {sorted(set(declared) - allowed)}")
    identity.update(declared)
    if declared:
        identity["identity_source"] = "config"
    if is_temporary_training_run(run):
        identity["purpose"] = "temporary"
    if identity["group"] not in GROUPS or identity["purpose"] not in {"formal", "temporary", "unclassified"}:
        raise ValueError("invalid experiment group/purpose")
    if sources is not None and (not sources or set(sources) - SOURCES):
        raise ValueError("invalid active sources in experiment identity")
    return identity


def task_training_labels(identity):
    sources = identity["active_sources"]
    if sources is None:
        return {task: "训练任务未记录；仅诊断" for task in ("t2i", "i2t", "text")}
    return {
        "t2i": "已训练" if "t2i" in sources else "未做 T2I 训练；仅诊断",
        "i2t": "已训练" if "i2t" in sources else "未做 I2T 训练；仅诊断",
        "text": "已训练" if "climbmix" in sources else "纯文本为迁移诊断",
    }


def write_run_identity(output_dir: Path, config) -> None:
    identity = experiment_identity(output_dir.name, config)
    path = output_dir / "experiment_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f"experiment identity changed within one output directory: {path}")
    atomic_write_text(path, json.dumps(identity, ensure_ascii=False, indent=2) + "\n")


def read_run_identity(directory: Path, config=None, *, presentation=None):
    path = directory / "experiment_identity.json"
    expected = experiment_identity(directory.name, config, presentation=presentation)
    if not path.exists():
        return expected
    identity = json.loads(path.read_text())
    if identity.get("schema") != "experiment_identity_v1" or identity.get("run") != directory.name:
        raise ValueError(f"invalid experiment identity: {path}")
    if identity.get("group") not in GROUPS or identity.get("purpose") not in {"formal", "temporary", "unclassified"}:
        raise ValueError(f"invalid experiment group/purpose: {path}")
    if not isinstance(identity.get("id"), str) or not identity["id"] or not isinstance(identity.get("label"), str):
        raise ValueError(f"invalid experiment id/label: {path}")
    sources = identity.get("active_sources")
    if sources is not None and (not isinstance(sources, list) or not sources or set(sources) - SOURCES):
        raise ValueError(f"invalid experiment active sources: {path}")
    if config and identity.get("active_sources") != expected["active_sources"]:
        raise ValueError(f"experiment task identity disagrees with configuration: {path}")
    return identity
