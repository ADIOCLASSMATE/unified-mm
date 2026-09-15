import csv
import json

from scripts.build_evaluation_report import write
from scripts.evaluation_report_unified import export_generation_sweep


def sweep_case():
    study = {"rows": [], "generation_cfg_values": [1.0, 1.5, 2.0, 3.5]}
    for cfg, bf, bi, of, oi in [(1.0, 19, 81, 15, 98), (1.5, 6, 170, None, None),
                                (2.0, 4.2, 222, 4.1, 256), (3.5, 5.2, 272, 6.2, 301)]:
        for key, b, o in (("fid", bf, of), ("is", bi, oi)):
            study["rows"].append({"cfg": cfg, "task": "generation", "key": key,
                                  "baseline": {"value": b, "std": 2, "source": f"b/{cfg}.json"},
                                  "only": {"value": o, "std": 3, "source": f"only/{cfg}.json"} if o else None,
                                  "delta": b - o if o else None})
    return study


def test_partial_sweep_exports_blanks_and_selects_fid_and_is_independently(tmp_path):
    result = export_generation_sweep(tmp_path, sweep_case(), write)
    assert result["completed"] == 3 and not result["complete"]
    assert result["only"]["best_fid"]["cfg"] == 2.0
    assert result["only"]["best_is"]["cfg"] == 3.5
    assert result["paired_wins"] == {"baseline": {"fid": 1, "is": 0}, "only": {"fid": 2, "is": 3}}
    assert result["points"][1]["only"] is None
    with (tmp_path / result["csv"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["only_fid"] == rows[1]["only_is"] == rows[1]["delta_fid"] == ""
    assert float(rows[2]["delta_fid"]) > 0
    assert float(rows[2]["delta_is"]) < 0


def test_new_results_hide_previous_plots_until_they_are_regenerated(tmp_path):
    study = sweep_case()
    result = export_generation_sweep(tmp_path, study, write)
    directory = tmp_path / "comparisons/unified-training-ablation"
    (directory / "cfg-sweep.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (directory / "cfg-plot-data.json").write_text(json.dumps({"points": result["points"],
                                                           "artifacts": {"cfg-sweep_svg": "cfg-sweep.svg"}}))
    assert export_generation_sweep(tmp_path, study, write)["plots"]
    for row in study["rows"]:
        if row["cfg"] == 1.5:
            row["only"] = {"value": 5 if row["key"] == "fid" else 190, "std": 3, "source": "only/1.5.json"}
            row["delta"] = row["baseline"]["value"] - row["only"]["value"]
    refreshed = export_generation_sweep(tmp_path, study, write)
    assert refreshed["complete"] and refreshed["completed"] == 4
    assert refreshed["plots"] == {}
