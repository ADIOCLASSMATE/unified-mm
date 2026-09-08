"""Prospective comparison registry and dimension-aware V4-compatible view loading."""

import json

import torch

from scripts.prepare_cross_model_geometry_v5 import V4

SETTINGS = {
    **{
        f"{model}_{profile}": {"model": model, "profile": profile, "kind": "v4"}
        for model in ("b", "f")
        for profile in ("native", "bare", "neutral")
    },
    "dinov2_qwen": {"x": "dinov2", "y": "qwen_text", "kind": "independent"},
    "mae_qwen": {"x": "mae", "y": "qwen_text", "kind": "independent"},
    "siglip": {"x": "siglip", "y": "siglip", "kind": "dual_encoder"},
    "siglip_native": {
        "x": "siglip_native",
        "y": "siglip_native",
        "kind": "native_endpoint",
    },
    **{
        f"{model}_{route}": {
            "x": f"{model}_{route}",
            "y": f"{model}_{route}",
            "kind": "flow",
            "route": route,
        }
        for model in ("janusflow", "showo2")
        for route in ("understanding", "generation")
    },
}


def canonical_mode(mode):
    aliases = {
        "centered_euclidean": "centered_euclidean",
        "unit_sphere": "centered_unit_sphere",
        "centered_unit_sphere": "centered_unit_sphere",
    }
    return aliases[mode]


def relative_layer_pairs(layers_x, layers_y):
    if layers_x == ["native_endpoint"] and layers_y == ["native_endpoint"]:
        return [
            {
                "index_x": 0,
                "index_y": 0,
                "layer_x": layers_x[0],
                "layer_y": layers_y[0],
                "relative_depth": 1.0,
                "kind": "native_endpoint",
            }
        ]
    assert layers_x[-1] == layers_y[-1] == "final_norm"
    dx, dy = len(layers_x) - 2, len(layers_y) - 2
    assert layers_x[:-1] == [str(i) for i in range(dx + 1)]
    assert layers_y[:-1] == [str(i) for i in range(dy + 1)]
    grid = max(dx, dy)
    result = []
    for step in range(grid + 1):
        # Explicit round-half-up, not Python's parity-dependent round-to-even.
        ix, iy = (
            (2 * step * dx + grid) // (2 * grid),
            (2 * step * dy + grid) // (2 * grid),
        )
        result.append(
            {
                "index_x": ix,
                "index_y": iy,
                "layer_x": layers_x[ix],
                "layer_y": layers_y[iy],
                "relative_depth": step / grid,
                "kind": "input" if step == 0 else "block",
            }
        )
    result.append(
        {
            "index_x": dx + 1,
            "index_y": dy + 1,
            "layer_x": "final_norm",
            "layer_y": "final_norm",
            "relative_depth": 1.0,
            "kind": "final_norm",
        }
    )
    assert {row["index_x"] for row in result} == set(range(len(layers_x)))
    assert {row["index_y"] for row in result} == set(range(len(layers_y)))
    return result


def readout_pairs(pools_x, pools_y):
    if pools_x == pools_y == ["native_endpoint"]:
        return {
            "native_endpoint": {
                "pool_x": 0,
                "pool_y": 0,
                "name_x": pools_x[0],
                "name_y": pools_y[0],
            }
        }
    result = {}
    for label, index in (
        ("content_mean", 0),
        ("content_last", 1),
        ("native_task", 2),
        ("generation_center", 3),
    ):
        if index < min(len(pools_x), len(pools_y)):
            result[label] = {
                "pool_x": index,
                "pool_y": index,
                "name_x": pools_x[index],
                "name_y": pools_y[index],
            }
    return result


def load_pair_views(root, setting, family):
    spec = SETTINGS[setting]
    if spec["kind"] == "v4":
        source = V4 if spec["model"] == "b" else root / "f-v4"
        data = dict(
            torch.load(
                source / "views/final_ema" / spec["profile"] / f"{family}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        )
        protocol = json.loads((source / "protocol.json").read_text())
        data["layers_x"] = data["layers_y"] = protocol["layers"]
        data["pools_x"] = data["pools_y"] = protocol["pools"]
        if family == "aro":
            rows = json.loads((root / "samples.json").read_text())["aro_images"]
            indices = torch.tensor([r["v4_index"] for r in rows])
            for key, value in list(data.items()):
                if torch.is_tensor(value) and value.ndim >= 1 and len(value) == 2000:
                    data[key] = value[indices]
                elif key in {"tasks", "categories"}:
                    data[key] = [value[i] for i in indices.tolist()]
            assert data["groups"].tolist() == [r["group"] for r in rows]
        return data
    data = {}
    for axis, modality in (("x", "images"), ("y", "texts")):
        path = root / "component-views" / spec[axis] / f"{family}_{modality}.pt"
        part = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        data[f"layers_{axis}"] = part["layers"]
        data[f"pools_{axis}"] = part["pools"]
        for key, value in part.items():
            if torch.is_tensor(value) and value.ndim in (3, 4):
                renamed = {
                    "test_positive": "test_y_positive",
                    "test_negative": "test_y_negative",
                }.get(key, f"{key}_{axis}")
                data[renamed] = value
            elif key in {
                "groups",
                "fit_indices",
                "dev_indices",
                "test_indices",
                "cal_indices",
                "tasks",
                "categories",
            }:
                if key in data:
                    assert (
                        torch.equal(data[key], value)
                        if torch.is_tensor(value)
                        else data[key] == value
                    )
                data[key] = value
    return data


def layer_values(data, family, pair, readout):
    result = {}
    for key, value in data.items():
        if not torch.is_tensor(value) or value.ndim not in (3, 4):
            continue
        axis = (
            "y"
            if key.endswith("_y") or key in {"test_y_positive", "test_y_negative"}
            else "x"
        )
        layer, pool = pair[f"index_{axis}"], readout[f"pool_{axis}"]
        result[key] = (
            value[:, layer, pool] if value.ndim == 4 else value[layer, pool]
        ).double()
    if family == "imagenet":
        for split in ("fit", "dev", "test"):
            ix = data[f"{split}_indices"]
            result[f"{split}_x"], result[f"{split}_y"] = (
                result["a_15_x"][ix],
                result["a_y"][ix],
            )
        ix = data["test_indices"]
        result["repeat_x"], result["repeat_y"] = result["b_15_x"][ix], result["b_y"][ix]
        for size in (1, 3, 5, 10, 15):
            result[f"test_prototype{size}_x"] = result[f"a_{size}_x"][ix]
            result[f"repeat_prototype{size}_x"] = result[f"b_{size}_x"][ix]
    return result
