"""Recoverably archive unscored Show-o2 and version its cal-only precision repair.

Run only after independently verifying extraction/view/audit writers have exited.
The all-layer cal diagnostic may remain running: it writes a distinct JSON only.
No weights, samples, scored comparison setting or semantic configuration changes.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.freeze_geometry_v5_analysis import comparison_value
from utils.research.geometry_v5_assets import RUN, emit, write_json

REVISION = "showo2-fp32-cal-revision-2"
SETTINGS = {"showo2_understanding", "showo2_generation"}
TARGETS = [
    "features/showo2",
    "flow-smoke/showo2",
    "component-views/showo2_understanding",
    "component-views/showo2_generation",
    "adapter-contracts/showo2.json",
    "model-verification/showo2.json",
    "model-verification/showo2-vae.json",
    "audits/features-showo2.json",
    "audits/cal-batch-layers-showo2.json",
    "supervisor-markers/showo2-cal-smoke.json",
    "supervisor-markers/showo2-full.json",
    "supervisor-markers/showo2-robust.json",
    "logs/showo2-cal-smoke.log",
    "logs/showo2-full.log",
    "logs/showo2-robust.log",
    "logs/views-showo2.log",
    "logs/audit-showo2.log",
    "logs/cal-batch-layers-showo2.log",
]


def unscored(root):
    for directory in (
        "analysis",
        "endpoints",
        "endpoint-statistics",
        "fitted-maps",
        "sample-statistics",
        "selected-maps",
        "robustness",
    ):
        for setting in SETTINGS:
            assert not list((root / directory / setting).rglob("*")), (
                "Refuse revision of any scored Show-o2 setting",
                directory,
                setting,
            )


def archive(root):
    unscored(root)
    destination = root / "calibration-archive" / REVISION
    ledger_path = destination / "revision.json"
    if not ledger_path.exists():
        assert all((root / path).exists() for path in TARGETS)
        diagnostic = root / "audits/cal-batch-layers-showo2-float32-diagnostic.json"
        evidence = json.loads(diagnostic.read_text())
        assert (
            evidence["complete"]
            and evidence["cal_only_precision_diagnostic"] == "float32"
        )
        original = json.loads((root / "comparison-contract.json").read_text())
        assert original["settings"]["showo2_generation"]["source_contracts"]["image"][
            "precision"
        ].startswith("bfloat16;")
        write_json(destination / "comparison-contract-revision-1.json", original)
        write_json(
            ledger_path,
            {
                "schema": "geometry_v5_unscored_precision_revision_1",
                "revision": 2,
                "created_utc": datetime.now(UTC).isoformat(),
                "changed_settings": sorted(SETTINGS),
                "reason": "cal-only batch dependence and weak generation-slot signal; FP32 arithmetic/storage correction, not score optimization",
                "diagnostic": str(diagnostic),
                "prior_semantic_results": 0,
                "training_updates": 0,
                "no_files_deleted": True,
                "targets": TARGETS,
                "archive_complete": False,
                "contract_refrozen": False,
            },
        )
    for relative in TARGETS:
        old, new = root / relative, destination / relative
        assert old.exists() != new.exists(), (old, new)
        if old.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
    ledger = json.loads(ledger_path.read_text())
    ledger["archive_complete"] = True
    write_json(ledger_path, ledger)
    emit("v5_showo_unscored_bf16_archived", archive=str(destination))


def finalize(root):
    unscored(root)
    archive = root / "calibration-archive" / REVISION
    old = json.loads((archive / "comparison-contract-revision-1.json").read_text())
    new = comparison_value(root)
    assert old.keys() == new.keys()
    for key in old:
        if key != "settings":
            assert old[key] == new[key], key
    assert old["settings"].keys() == new["settings"].keys()
    for setting, previous in old["settings"].items():
        current = new["settings"][setting]
        if setting not in SETTINGS:
            assert current == previous, setting
            continue
        for key in previous:
            if key != "source_contracts":
                assert current[key] == previous[key], (setting, key)
        for modality in ("image", "text"):
            a, b = (
                previous["source_contracts"][modality],
                current["source_contracts"][modality],
            )
            changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
            assert changed <= {
                "schema",
                "precision",
                "storage_dtype",
                "generation_noise",
                "input_pooling",
            }
            assert (
                b["precision"].startswith("float32;")
                and b["storage_dtype"] == "float32"
            )
    ledger = json.loads((archive / "revision.json").read_text())
    assert ledger["archive_complete"]
    diagnostic_path = root / "audits/cal-batch-layers-showo2-storage-float32.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["complete"]
    assert (
        diagnostic["adapter_contract"]
        == new["settings"]["showo2_generation"]["source_contracts"]["image"]
    )
    diagnostic["formal_equivalence_evidence"] = {
        "source": str(diagnostic_path),
        "contract_exactly_matches_formal_extractor": True,
        "note": "explicit FP32 storage override equals the frozen formal default; original diagnostic flags retained",
    }
    write_json(root / "audits/cal-batch-layers-showo2.json", diagnostic)
    ledger["final_cal_diagnostic"] = str(diagnostic_path)
    ledger["input_pooling_repair"] = (
        "selected-token input means avoid padding-dependent rounding of identical fixed-noise slots; no semantic input/mask/weight change"
    )
    write_json(root / "comparison-contract.json", new)
    ledger.update(contract_refrozen=True, refrozen_utc=datetime.now(UTC).isoformat())
    write_json(archive / "revision.json", ledger)
    write_json(root / "audits" / "showo2-precision-revision.json", ledger)
    emit("v5_showo_precision_contract_refrozen", unaffected_settings=12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("stage", choices=("archive", "finalize"))
    args = parser.parse_args()
    (archive if args.stage == "archive" else finalize)(args.output_dir)
