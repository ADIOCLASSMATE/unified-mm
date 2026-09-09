"""Aggregate V5 raw independent encodings into V4-identical semantic units."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch

from utils.research.geometry_v5_assets import RUN, emit


def grouped(features, rows, groups, predicate):
    lookup = {group: i for i, group in enumerate(groups)}
    chosen = [
        i for i, row in enumerate(rows) if row["group"] in lookup and predicate(row)
    ]
    indices = torch.tensor([lookup[rows[i]["group"]] for i in chosen], dtype=torch.long)
    counts = torch.bincount(indices, minlength=len(groups))
    assert bool(counts.gt(0).all())
    result = torch.zeros(len(groups), *features.shape[1:], dtype=torch.float32)
    for start in range(0, len(chosen), 128):
        result.index_add_(
            0,
            indices[start : start + 128],
            features[chosen[start : start + 128]].float(),
        )
    return result / counts.reshape(-1, *([1] * (features.ndim - 1)))


def load_shards(directory, dataset, rows, native=False):
    result, metadata, files = None, None, []
    seen = []
    for rank in range(16):
        path = directory / f"{dataset}-rank-{rank:02d}-of-16.pt"
        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        indices = value["indices"]
        assert indices.tolist() == list(range(rank, len(rows), 16)), path
        assert value["schema"] == "cross_model_geometry_v5_features_1"
        assert (
            value["dataset"] == dataset
            and value["rank"] == rank
            and value["world_size"] == 16
        )
        assert (
            value["training_updates"] == 0
            and value["audit"]["opposite_modality_replacement_exact"]
        )
        features = (
            value["native_endpoint"][:, None, None] if native else value["features"]
        )
        expected_dtype = getattr(
            torch, value["contract"].get("storage_dtype", "bfloat16")
        )
        assert features.dtype == expected_dtype and len(features) == len(indices)
        current = {
            "contract": value["contract"],
            "layers": ["native_endpoint"] if native else value["layers"],
            "pools": ["native_endpoint"] if native else value["pools"],
        }
        if result is None:
            result = torch.empty(len(rows), *features.shape[1:], dtype=features.dtype)
            metadata = current
        assert current == metadata and features.shape[1:] == result.shape[1:]
        for start in range(0, len(features), 128):
            chunk = features[start : start + 128]
            assert bool(torch.isfinite(chunk).all())
            result[indices[start : start + 128]] = chunk
        seen.extend(indices.tolist())
        files.append({"path": str(path), "bytes": path.stat().st_size})
    assert sorted(seen) == list(range(len(rows)))
    return result, {**metadata, "source_shards": files}


def aggregate_axis(features, rows, image_rows, family, modality):
    data = {}
    if family == "imagenet":
        groups = list(range(1000))
        assignment = {row["group"]: row["mapping_split"] for row in image_rows}
        for split in ("fit", "dev", "test"):
            data[f"{split}_indices"] = torch.tensor(
                [g for g in groups if assignment[g] == split]
            )
        cal = grouped(features, rows, groups, lambda row: row["split"] == "cal")
        data["cal"] = cal[data["fit_indices"]].mean(0)
        for view in ("a", "b"):
            if modality == "images":
                for size in (1, 3, 5, 10, 15):
                    data[f"{view}_{size}"] = grouped(
                        features,
                        rows,
                        groups,
                        lambda row, v=view, n=size: (
                            row["split"] == v and row["view_index"] < n
                        ),
                    )
            else:
                data[view] = grouped(
                    features, rows, groups, lambda row, v=view: row["split"] == v
                )
    elif family == "coco":
        groups = [row["group"] for row in image_rows]
        for split in ("cal", "fit", "dev", "test"):
            data[f"{split}_indices"] = torch.tensor(
                [i for i, row in enumerate(image_rows) if row["split"] == split]
            )
        means = (
            grouped(features, rows, groups, lambda row: True)
            if modality == "texts"
            else features
        )
        data["cal"] = means[data["cal_indices"]].float().mean(0)
        for split in ("fit", "dev", "test"):
            data[split] = means[data[f"{split}_indices"]]
        if modality == "texts":
            test_groups = [groups[i] for i in data["test_indices"].tolist()]
            for size in (1, 3, 5):
                data[f"test_caption{size}"] = grouped(
                    features,
                    rows,
                    test_groups,
                    lambda row, n=size: row["caption_index"] < n,
                )
            data["test_first2"] = grouped(
                features, rows, test_groups, lambda row: row["caption_index"] < 2
            )
            data["test_remaining"] = grouped(
                features, rows, test_groups, lambda row: row["caption_index"] >= 2
            )
    else:
        groups = [row["group"] for row in image_rows]
        data["tasks"] = [row["task"] for row in image_rows]
        data["categories"] = [row["category"] for row in image_rows]
        if modality == "images":
            data["test"] = features
        else:
            data["test_positive"] = grouped(
                features, rows, groups, lambda row: row["positive"]
            )
            data["test_negative"] = grouped(
                features, rows, groups, lambda row: not row["positive"]
            )
    data["groups"] = torch.tensor(groups)
    return data


def build(task):
    root, component, family, modality = task
    torch.set_num_threads(1)
    native = component == "siglip_native"
    if component.startswith(("janusflow_", "showo2_")):
        model, route = component.split("_", 1)
        directory = root / "features" / model / route / "main"
    else:
        directory = root / "features" / ("siglip" if native else component)
    dataset = f"{family}_{modality}"
    target = root / "component-views" / component / f"{dataset}.pt"
    samples = json.loads((root / "samples.json").read_text())
    if target.exists():
        existing = torch.load(target, map_location="cpu", weights_only=True, mmap=True)
        assert existing["schema"] == "geometry_v5_component_views_1"
        assert existing["component"] == component and existing["dataset"] == dataset
        assert existing["source_rows"] == len(samples[dataset])
        emit("component_view_reused", component=component, dataset=dataset)
        return
    features, metadata = load_shards(directory, dataset, samples[dataset], native)
    data = aggregate_axis(
        features, samples[dataset], samples[f"{family}_images"], family, modality
    )
    data.update(
        {
            **metadata,
            "schema": "geometry_v5_component_views_1",
            "component": component,
            "dataset": dataset,
            "family": family,
            "modality": modality,
            "source_rows": len(features),
            "aggregation": "raw feature arithmetic mean before calibration centering or unit normalization",
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(target)
    emit(
        "component_view_complete",
        component=component,
        dataset=dataset,
        bytes=target.stat().st_size,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument(
        "--components",
        default="qwen_text,dinov2,mae,siglip,siglip_native,janusflow_understanding,janusflow_generation,showo2_understanding,showo2_generation",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    tasks = []
    for component in args.components.split(","):
        modalities = (
            ["texts"]
            if component == "qwen_text"
            else ["images"]
            if component in {"dinov2", "mae"}
            else ["images", "texts"]
        )
        tasks.extend(
            (args.output_dir, component, family, modality)
            for family in ("imagenet", "coco", "aro")
            for modality in modalities
        )
    with mp.get_context("spawn").Pool(args.workers) as pool:
        list(pool.imap_unordered(build, tasks))


if __name__ == "__main__":
    main()
