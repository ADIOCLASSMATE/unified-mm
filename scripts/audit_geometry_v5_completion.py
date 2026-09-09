"""Full-scope V5 artifact gate; scientific/visual review remains a human responsibility.

This gate cannot turn missing/partial results into completion. Run the underlying
result, paired and summary auditors first, then inspect the final Chinese report
and rendered figures before marking the persistent goal complete.
"""

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from scripts.geometry_v5_protocol import SETTINGS
from utils.research.geometry_v5_assets import RUN, emit, write_json
from scripts.report_geometry_v5 import endpoint

MODELS = ("b", "f", "qwen_text", "dinov2", "mae", "siglip", "janusflow", "showo2")
COUNTS = {
    "imagenet_images": 32000,
    "imagenet_texts": 12000,
    "coco_images": 11776,
    "coco_texts": 58909,
    "aro_images": 868,
    "aro_texts": 1736,
}


def read(path):
    return json.loads(Path(path).read_text())


def passed(root, name):
    result = read(root / "audits" / name)
    assert result["status"] == "passed", name
    return result


def local_links(path):
    checked = []
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        full = path.parent / target
        assert full.exists(), (path, target)
        checked.append(str(full))
    return checked


def narrative_primary_table(root, report):
    """Check every rounded cell in the hand-written primary table against endpoints."""
    names = {
        "b_native": "B",
        "f_native": "F",
        "dinov2_qwen": "DINOv2＋Qwen",
        "mae_qwen": "MAE＋Qwen",
        "siglip": "SigLIP 内容均值",
        "janusflow_understanding": "JanusFlow 理解",
        "janusflow_generation": "JanusFlow 生成",
        "showo2_understanding": "Show-o2 理解",
        "showo2_generation": "Show-o2 生成",
        "siglip_native": "SigLIP 原生 pooler（另列参考）",
    }
    rows, mismatches = [], []
    content = report.read_text()
    for setting, name in names.items():
        data = {
            setting: [read(p) for p in (root / "endpoints" / setting).glob("*.json")]
        }
        values = []
        for family in ("imagenet", "coco"):
            result = endpoint(data, setting, family)
            assert result["valid"]
            values.extend(
                f"{result[split]['paired']['r2']:.3f}"
                for split in ("test", "transfer_test")
            )
        line = "| " + " | ".join([name, *values]) + " |"
        if line not in content:
            mismatches.append(line)
        rows.append(line)
    assert not mismatches, (
        "Narrative primary table differs from endpoints",
        mismatches,
    )
    return rows


def main(root, report):
    contract = read(root / "comparison-contract.json")
    assert set(contract["settings"]) == set(SETTINGS) and len(SETTINGS) == 14
    protocol = read(root / "protocol.json")
    assert protocol["training_updates"] == 0 and not protocol["runtime_hashing_enabled"]
    assets = read(root / "asset-manifest.json")
    assert "ProxyHandler({})" in assets["proxy_policy"]
    asset_root = (root.parent.parent / "public/models").resolve()
    for record in assets["files"]:
        path = Path(record["path"])
        assert path.resolve().is_relative_to(asset_root) and path.is_file()
        if record["bytes"] is not None:
            assert path.stat().st_size == record["bytes"]
        if "extract_to" in record:
            assert Path(record["extract_to"]).is_dir()
    source = passed(root, "model-sources-and-training.json")
    assert source["new_training_updates"] == 0 and set(source["models"]) == set(MODELS)
    assert (
        source["models"]["f"]["configuration"]["architecture_variant"]
        == "positionwise_flow_head_on_b"
    )
    assert all(
        not source["b_f_training_differences"][key]
        for key in ("dataset", "optimizer", "lr_scheduler", "training")
    )
    features = {name: passed(root, f"features-{name}.json") for name in MODELS}
    for name, result in features.items():
        assert result["model"] == name and result["shards"] == len(result["records"])
        for record in result["records"]:
            assert Path(record["path"]).stat().st_size == record["bytes"]
            if name in {"janusflow", "showo2"}:
                upstream = record["upstream_shapes"]
                text_understanding = (
                    "/understanding/" in record["path"]
                    and "_texts-rank-" in record["path"]
                )
                assert bool(upstream) != text_understanding
                assert all(
                    shape[0] == record["shape"][0] for shape in upstream.values()
                )
    parity = passed(root, "b-v4-parity.json")
    assert (
        parity["layer_readout_rows"] == 270
        and parity["maximum_absolute_r2_difference"] == 0
    )
    samples = passed(root, "samples-and-preprocessing.json")
    assert samples["counts"] == COUNTS
    assert (
        samples["f_original_v4_sample_manifest_exact"]
        and samples["non_aro_v4_rows_and_order_exact"]
    )
    assert samples["aro_original_crop_registry_exact"]
    assert all(
        row["truncated_texts"] == 0 for row in samples["runtime_text_truncations"]
    )
    for layout in contract["settings"].values():
        for axis in ("x", "y"):
            assert {p[f"index_{axis}"] for p in layout["layer_pairs"]} == set(
                range(len(layout[f"layers_{axis}"]))
            )
        assert layout["readouts"]
    for model in ("janusflow", "showo2"):
        adapter = read(root / "adapter-contracts" / f"{model}.json")
        assert adapter["training_updates"] == 0 and adapter["clean_image_time"] == 1
        assert (
            adapter["text_generation_time"] == 0
            and len(adapter["generation_noise_seeds"]) == 3
        )
        assert set(adapter["pools"]) == {"understanding", "generation"}
        assert (
            adapter["target_leakage_contract"]
            == "image forward never reads text; text forward never reads image; no paired teacher forcing"
        )
        assert Path(adapter["vae_path"]).exists()
    results = {setting: passed(root, f"results-{setting}.json") for setting in SETTINGS}
    assert sum(v["layer_readout_rows"] for v in results.values()) == 1171
    assert sum(v["endpoints"] for v in results.values()) == 1640
    paired = passed(root, "paired-model-differences.json")
    assert paired["bootstrap"] == 2000 and paired["primary_contrasts"] == 16
    assert (
        paired["contrasts_recomputed"] == 1260
        and paired["identities_and_intervals_recomputed"]
    )
    parallel = passed(root, "paired-parallel-preflight.json")
    assert parallel["real_endpoint_pairs"] == 16 and parallel["workers_compared"] == [
        1,
        4,
    ]
    assert all(
        parallel[key]
        for key in (
            "serial_parallel_rows_exact",
            "serial_parallel_audits_exact",
            "all_bootstrap_intervals_exact",
        )
    )
    summary = passed(root, "summary-tables.json")
    assert summary["csv_rows"] == 37472 and summary["layer_readout_rows"] == 1171
    assert (
        summary["dev_selections"] == 328 and summary["layer_search_null_tests"] == 984
    )
    robust = {}
    for setting in (
        "b_native",
        "f_native",
        "janusflow_generation",
        "showo2_generation",
    ):
        result = read(root / "audits" / f"robustness-{setting}.json")
        assert len(result["paths"]) == result["expected_endpoints"]
        assert results[setting]["robustness_rows"] == len(result["paths"]) * 6
        robust[setting] = results[setting]["robustness_rows"]
    decomposition = passed(root, "perturbation-error-decomposition.json")
    assert decomposition["posthoc_explanatory"]
    assert decomposition["all_original_error_changes_independently_recomputed"]
    assert not any(
        decomposition[key]
        for key in ("target_refit", "predictions_recentered", "formal_results_modified")
    )
    expected_decompositions = {
        (setting, endpoint_kind, readout, profile)
        for setting in robust
        for endpoint_kind in ("fixed", "dev_selected")
        for readout in ("content_mean", "native_task")
        for profile in (
            ("native_sigma1", "native_sigma2", "native_mean")
            if setting in {"b_native", "f_native"}
            else ("seed1", "seed2", "image_midpoint")
        )
    }
    assert len(decomposition["rows"]) == 48
    assert {
        (r["setting"], r["endpoint"], r["readout"], r["profile"])
        for r in decomposition["rows"]
    } == expected_decompositions
    for row in decomposition["rows"]:
        assert (
            row["points"] == 512 and Path(row["formal_robustness_evidence"]).is_file()
        )
        assert (
            abs(
                row["mean_residual_error_increase_over_native_denominator"]
                + row["centered_residual_error_increase_over_native_denominator"]
                + row["error_reduction_recomputed"]
            )
            < 1e-7
        )
    numerics = {}
    for name in MODELS:
        value = read(root / "audits" / f"cal-batch-layers-{name}.json")
        assert value["complete"] and value["training_updates"] == 0
        assert value["all_formal_flattened_checks_passed"]
        numerics[name] = {
            "rows": len(value["rows"]),
            "local_layer_warnings": sum(
                r["comparison"]["per_layer_below_0p995_count"] for r in value["rows"]
            ),
        }
    runtime = read(root / "runtime-check.json")
    assert runtime["device_count"] == 16 and runtime["arithmetic_sum"] == 512
    assert runtime["torch_npu_imported"] and runtime["cann_environment_present"]
    cleanup = read(root / "resource-cleanup.json")
    assert (
        cleanup["status"] == "verified_stopped" and cleanup["notebook_object_preserved"]
    )
    assert (
        not cleanup["artifacts_deleted"]
        and cleanup["live_responses"]["after-stop"]["data"]["status"] == "STOPPED"
    )
    final_status = read(root / "logs/notebook-final-status.json")
    assert final_status["success"]
    assert final_status["data"]["status"] == "STOPPED"
    for key in (
        "name",
        "created_at",
        "workspace",
        "compute_group",
        "image",
        "resource",
    ):
        assert (
            final_status["data"][key]
            == cleanup["live_responses"]["before-stop"]["data"][key]
        )
    xml = ET.parse(root / "audits/tests-v3-v4-v5.xml").getroot()
    suites = list(xml.iter("testsuite"))
    assert suites and all(
        int(v.attrib[k]) == 0 for v in suites for k in ("errors", "failures", "skipped")
    )
    test_count = sum(int(v.attrib["tests"]) for v in suites)
    assert test_count >= 34
    assert "All checks passed!" in (root / "logs/ruff-v5.log").read_text()
    figures = read(root / "figures/index.json")
    assert not figures["partial_preflight"] and set(figures["settings"]) == set(
        SETTINGS
    )
    expected_figures = {
        f"layers-common-{mode}" for mode in ("centered_euclidean", "unit_sphere")
    }
    expected_figures |= {f"readouts-{s}" for s in SETTINGS if s != "siglip_native"}
    expected_figures |= {
        "mapping-fixed",
        "mapping-dev_selected",
        "mapping-dimensions",
        "coco-fit-size",
        "aro-cosine",
        "aro-distance",
    }
    expected_figures |= {f"robustness-{s}" for s in robust}
    assert {v["name"] for v in figures["figures"]} == expected_figures
    visual = passed(root, "visual-review.json")
    assert set(visual["pngs_inspected"]) == expected_figures
    assert len(visual["pngs_inspected"]) == len(expected_figures)
    pdf_checks = []
    for figure in figures["figures"]:
        with Image.open(figure["png"]) as im:
            assert min(im.size) >= 400
            im.verify()
        with Path(figure["pdf"]).open("rb") as handle:
            assert handle.read(5) == b"%PDF-"
        pdf = subprocess.run(
            ["pdfinfo", figure["pdf"]], capture_output=True, text=True, check=True
        )
        assert not pdf.stderr.strip(), (figure["name"], pdf.stderr)
        assert re.search(r"^Pages:\s+1\s*$", pdf.stdout, re.MULTILINE)
        assert re.search(r"^Encrypted:\s+no\s*$", pdf.stdout, re.MULTILINE)
        pdf_checks.append({"name": figure["name"], "pdfinfo": pdf.stdout})
    numeric_report = root / "RESULTS_ZH.md"
    assert len(numeric_report.read_text()) > 3000 and len(report.read_text()) > 1000
    for marker in ("待完成", "状态：解释报告草稿"):
        assert (
            marker not in report.read_text()
            and marker not in numeric_report.read_text()
        )
    primary_rows = narrative_primary_table(root, report)
    review_path = report.parent / "CROSS_MODEL_GEOMETRY_V5_COMPLETION_AUDIT_20260907.md"
    assert all(f"## 要求 {i}" in review_path.read_text() for i in range(1, 9))
    links = local_links(numeric_report) + local_links(report) + local_links(review_path)
    requirements = [
        {
            "requirement": 1,
            "evidence": [
                "audits/model-sources-and-training.json",
                "audits/features-*.json",
                "audits/b-v4-parity.json",
                "audits/results-*.json",
            ],
            "scope": "all requested models, actual F class/final EMA, frozen source weights",
        },
        {
            "requirement": 2,
            "evidence": ["samples.json", "audits/samples-and-preprocessing.json"],
            "scope": "all six exact dataset counts/identity splits/caption aggregation/ARO original crop registry",
        },
        {
            "requirement": 3,
            "evidence": [
                "comparison-contract.json",
                "adapter-contracts/",
                "audits/features-*.json",
            ],
            "scope": "every actual input/block/final norm and every predeclared readout",
        },
        {
            "requirement": 4,
            "evidence": [
                "adapter-contracts/janusflow.json",
                "adapter-contracts/showo2.json",
                "features/",
                "robustness/",
                "audits/perturbation-error-decomposition.json",
            ],
            "scope": "understanding/generation routes, upstream representations, independent semantic inputs, seed/time controls",
        },
        {
            "requirement": 5,
            "evidence": ["analysis/", "geometry-v5.csv", "audits/summary-tables.json"],
            "scope": "all layer kNN5/10/20 RSA CKA, fit-only orthogonal maps/dimensions/full-width restrictions/variance/rank/condition",
        },
        {
            "requirement": 6,
            "evidence": [
                "analysis/",
                "fitted-maps/",
                "selected-maps/",
                "endpoint-statistics/",
                "audits/results-*.json",
            ],
            "scope": "both held-out domains, three COCO fit sizes, completely frozen cross-domain transfer, strict ARO, all null and negative outcomes",
        },
        {
            "requirement": 7,
            "evidence": [
                "statistics-contract.json",
                "endpoints/",
                "layer-search-null-v5.json",
                "audits/paired-model-differences.json",
            ],
            "scope": "fixed and source-dev endpoints, identity bootstrap, paired differences, conditional/multiplicity caveats",
        },
        {
            "requirement": 8,
            "evidence": [
                "audits/",
                "figures/index.json",
                str(report),
                "RESULTS_ZH.md",
                "resource-cleanup.json",
                "logs/notebook-final-status.json",
                "audits/visual-review.json",
                str(review_path),
            ],
            "scope": "numerical/leakage/provenance/integrity audits, tests, figures, Chinese report, verified STOPPED preserving object/artifacts",
        },
    ]
    value = {
        "schema": "geometry_v5_completion_artifact_audit_1",
        "status": "passed",
        "utc": datetime.now(UTC).isoformat(),
        "requirements": requirements,
        "settings": 14,
        "layer_readout_rows": 1171,
        "endpoint_records": 1640,
        "figure_pairs": len(expected_figures),
        "pdf_parse_checks": pdf_checks,
        "tests_passed": test_count,
        "narrative_primary_table_cells_checked": len(primary_rows) * 4,
        "posthoc_decomposition_rows": len(decomposition["rows"]),
        "parallel_preflight_endpoint_pairs": parallel["real_endpoint_pairs"],
        "final_notebook_status": final_status["data"]["status"],
        "numeric_warnings_preserved": numerics,
        "robustness_rows": robust,
        "local_report_links_checked": links,
        "interpretation": "Artifact scope gate only; primary agent must also inspect all referenced evidence, rendered figures and scientific claims before marking goal complete.",
    }
    write_json(root / "audits/completion-artifacts.json", value)
    emit("v5_complete_artifact_scope_audit_passed", requirements=len(requirements))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/CROSS_MODEL_GEOMETRY_V5_RESULTS_20260907.md"),
    )
    args = parser.parse_args()
    main(args.output_dir, args.report)
