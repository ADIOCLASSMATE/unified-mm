"""Freeze all relative-depth and readout comparisons before inspecting new scores."""

import argparse
import json
from pathlib import Path

from scripts.geometry_v5_protocol import SETTINGS, readout_pairs, relative_layer_pairs
from scripts.prepare_cross_model_geometry_v5 import V4
from utils.research.geometry_v5_assets import RUN, emit, write_json


def layout(root, component, modality):
    model, route = (
        component.split("_", 1)
        if component.startswith(("janusflow_", "showo2_"))
        else (component, None)
    )
    if model == "siglip_native":
        model = "siglip"
    contract = json.loads((root / "adapter-contracts" / f"{model}.json").read_text())
    if component == "siglip_native":
        return ["native_endpoint"], ["native_endpoint"], contract
    if route is not None:
        return contract["layers"], contract["pools"][route], contract
    return contract["layers"][modality], contract["pools"][modality], contract


def comparison_value(root):
    result = {
        "schema": "geometry_v5_comparison_contract_1",
        "seed": 20260909,
        "depth_matching": "max-depth grid; nearest relative depth with explicit round-half-up, independent final norm; all actual layers covered",
        "common_dimensions": [32, 128, 512],
        "full_dimension": "within-model diagnostic only, requires equal original widths",
        "selection": "source-dev R2 only; common-budget selection excludes full; separately per predeclared readout, geometry mode, and fit size",
        "native_readout_warning": "task-dependent roles differ; native results are not matched common pooling",
        "permutations": 199,
        "permutation_seed": 20260908,
        "bootstrap": 2000,
        "mapping_seed": 20260909,
        "bootstrap_seed": 20260908,
        "rank_tolerances": {
            "source_singular_value_relative": 1e-6,
            "cross_covariance_singular_value_relative": 1e-6,
        },
        "settings": {},
    }
    for setting, spec in SETTINGS.items():
        if spec["kind"] == "v4":
            base = V4 if spec["model"] == "b" else root / "f-v4"
            source = json.loads((base / "protocol.json").read_text())
            lx = ly = source["layers"]
            px = py = source["pools"]
            contracts = {
                "v4_protocol": str(base / "protocol.json"),
                "profile": spec["profile"],
            }
        else:
            lx, px, cx = layout(root, spec["x"], "image")
            ly, py, cy = layout(root, spec["y"], "text")
            contracts = {"image": cx, "text": cy}
        result["settings"][setting] = {
            "specification": spec,
            "layers_x": lx,
            "layers_y": ly,
            "pools_x": px,
            "pools_y": py,
            "layer_pairs": relative_layer_pairs(lx, ly),
            "readouts": readout_pairs(px, py),
            "source_contracts": contracts,
        }
    return result


def freeze(root):
    result = comparison_value(root)
    path = root / "comparison-contract.json"
    if path.exists():
        assert json.loads(path.read_text()) == result, (
            "Comparison contract changed; do not mix scored versions"
        )
    else:
        write_json(path, result)
    emit(
        "v5_comparison_contract_frozen",
        settings=len(result["settings"]),
        layer_readout_combinations=sum(
            len(value["layer_pairs"]) * len(value["readouts"])
            for value in result["settings"].values()
        ),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    freeze(parser.parse_args().output_dir)
