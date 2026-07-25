import json

import pytest

from scripts.archive.flow_head_ablation.dual_stream_flow_head_ablation import (
    EXPECTED_FLOW_HEAD_PARAMETERS,
    POSITION_VARIANTS,
    TRAIN_CELLS,
    build_ablation_config,
    validate_ablation_config,
)
from scripts.archive.flow_head_ablation.summarize_dual_stream_flow_head_ablation import (
    BASELINE_METRICS,
    CELLS,
    summarize,
)


@pytest.mark.parametrize("cell", TRAIN_CELLS)
def test_six_cell_configs_are_closed_and_self_validating(cell):
    architecture, position = cell.split("-")
    config = build_ablation_config(architecture, position)
    validate_ablation_config(config, cell)
    assert config.experiment.ablation_id == cell
    assert config.experiment.architecture_id == architecture
    assert config.experiment.position_id == position
    assert config.experiment.project == (
        f"selfless-flow-dual-{architecture.lower()}-{position.lower()}-s42"
    )
    expected_modes = {
        "FH0": ("additive_2d", "additive_2d", "none"),
        "FH1": ("additive_2d", "additive_2d", "row_col_2d"),
        "FH4": ("none", "none", "row_col_2d"),
    }[position]
    assert (
        config.model.image_flow_query_position_mode,
        config.model.image_flow_context_position_mode,
        config.model.image_flow_rope_mode,
    ) == expected_modes


def test_six_cell_config_fingerprints_are_unique():
    fingerprints = set()
    for cell in TRAIN_CELLS:
        architecture, position = cell.split("-")
        config = build_ablation_config(architecture, position)
        fingerprints.add(str(config.experiment.config_fingerprint))
    assert len(fingerprints) == len(TRAIN_CELLS)


def _metrics_payload(
    cell,
    *,
    fid,
    inception_score,
    wall,
    provenance=None,
):
    architecture, position = cell.split("-")
    return {
        "architecture": {
            "flow_head": {
                "variant": architecture,
                "position_contract": {"variant": position},
            }
        },
        "parameters": {"flow_head": EXPECTED_FLOW_HEAD_PARAMETERS},
        "training_protocol": (
            {
                "dual_stream_flow_head": {
                    "provenance": provenance,
                    "training_runtime": {
                        "world_size": 8,
                        "train_samples_per_second": 10.0,
                        "peak_cuda_allocated_bytes_per_rank": 100,
                    },
                }
            }
            if provenance is not None
            else None
        ),
        "strategies": {
            "spatial_halton": {
                "fid": fid,
                "inception_score_mean": inception_score,
                "inception_score_std": 0.2,
                "generation_wall_seconds": wall,
                "generation_samples_per_second": 10_000 / wall,
                "flow_content_cache_peak_bytes_per_sample": 1024,
                "flow_cfg_content_cache_divergence_by_layer": [0.0] * 8,
            }
        },
    }


def test_matrix_summary_reports_interactions_and_applies_selector(tmp_path):
    pairing = {
        "flow_head_parameter_count": EXPECTED_FLOW_HEAD_PARAMETERS,
        "flow_head_parameter_schema_sha256": "a" * 64,
        "flow_head_initial_state_sha256": "b" * 64,
        "train_order_sha256": "c" * 64,
        "augmentation_sha256": "d" * 64,
        "runtime_source_manifest_sha256": "e" * 64,
    }
    paths = {}
    dynamic_values = {
        "DF1-FH0": (24.4, 61.3, 8000.0),
        "DF1-FH1": (24.3, 61.4, 8200.0),
        "DF1-FH4": (24.2, 61.5, 8300.0),
        "DF2-FH0": (24.1, 61.6, 7900.0),
        "DF2-FH1": (24.0, 61.7, 7800.0),
        # Deliberately much slower than the historical baseline: sampling
        # efficiency is reported, but no longer gates the quality selector.
        "DF2-FH4": (23.7, 61.8, 50_000.0),
    }
    for cell in CELLS:
        path = tmp_path / f"{cell}.json"
        architecture, position = cell.split("-")
        if architecture == "DF0":
            values = BASELINE_METRICS[cell]
            payload = _metrics_payload(
                cell,
                fid=values["fid"],
                inception_score=values["inception_score_mean"],
                wall=7369.137152409181,
            )
        else:
            fid, inception_score, wall = dynamic_values[cell]
            provenance = {
                **pairing,
                "ablation_id": cell,
                "architecture": {
                    "variant": architecture,
                    "position_variant": position,
                },
            }
            payload = _metrics_payload(
                cell,
                fid=fid,
                inception_score=inception_score,
                wall=wall,
                provenance=provenance,
            )
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[cell] = path

    summary = summarize(paths)
    assert summary["decision"]["selected"] == "DF2-FH4"
    assert "sampling_wall_ratio_vs_df0_fh0_max" not in summary["thresholds"]
    assert summary["sampling_efficiency"]["role"] == "reported_diagnostic_not_gate"
    df2_fh4 = next(row for row in summary["rows"] if row["cell_id"] == "DF2-FH4")
    assert df2_fh4["sampling_wall_ratio_vs_df0_fh0"] > 1.20
    assert df2_fh4["passes_screen"] is True
    assert "passes_efficiency_gate" not in df2_fh4
    assert set(summary["estimands"]) == {
        "architecture",
        "position",
        "architecture_position_interaction",
    }
    assert set(summary["estimands"]["architecture"]) == set(POSITION_VARIANTS)
    assert len(summary["rows"]) == 9
