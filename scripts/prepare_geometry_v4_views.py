"""Aggregate frozen V4 raw features into semantic units before normalization."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_unified_semantics_v2 import load_dataset
from utils.research.representation_protocol import emit


def grouped(features, rows, groups, predicate):
    lookup = {g: i for i, g in enumerate(groups)}
    chosen = [i for i, r in enumerate(rows) if r["group"] in lookup and predicate(r)]
    indices = torch.tensor([lookup[rows[i]["group"]] for i in chosen])
    counts = torch.bincount(indices, minlength=len(groups))
    assert bool(counts.gt(0).all()), "Missing semantic view"
    result = torch.zeros(len(groups), *features.shape[1:], dtype=torch.float32)
    for start in range(0, len(chosen), 128):
        result.index_add_(
            0,
            indices[start : start + 128],
            features[chosen[start : start + 128]].float(),
        )
    return result / counts.reshape(-1, 1, 1, 1)


def build(root, state, profile_name, family):
    path = root / "views" / state / profile_name / f"{family}.pt"
    protocol = json.loads((root / "protocol.json").read_text())
    samples = json.loads((root / "samples.json").read_text())
    profile = protocol["profiles"][profile_name]
    assert profile["subset"] == "all", (
        "Use frozen-native coordinates for perturbation analysis"
    )
    if path.exists():
        saved = torch.load(path, mmap=True, weights_only=True)
        assert saved["state"] == state and saved["profile"] == profile_name
        assert saved["schema"] == "geometry_v4_views_1"
        emit("views_reused_v4", state=state, profile=profile_name, family=family)
        return
    data = {
        "schema": "geometry_v4_views_1",
        "state": state,
        "profile": profile_name,
        "family": family,
    }
    image_rows = samples[f"{family}_images"]
    if family == "imagenet":
        assignment = {r["group"]: r["mapping_split"] for r in image_rows}
        groups = list(range(1000))
        data["groups"] = torch.tensor(groups)
        for split in ("fit", "dev", "test"):
            data[f"{split}_indices"] = torch.tensor(
                [g for g in groups if assignment[g] == split]
            )
    elif family == "coco":
        groups = [r["group"] for r in image_rows]
        data["groups"] = torch.tensor(groups)
        for split in ("cal", "fit", "dev", "test"):
            data[f"{split}_indices"] = torch.tensor(
                [i for i, r in enumerate(image_rows) if r["split"] == split]
            )
    else:
        groups = [r["group"] for r in image_rows]
        data["groups"] = torch.tensor(groups)
        data["tasks"] = [r["task"] for r in image_rows]
        data["categories"] = [r["category"] for r in image_rows]
    checks = None
    for modality, axis in (("images", "x"), ("texts", "y")):
        features, rows, current = load_dataset(
            root / state / profile_name,
            f"{family}_{modality}",
            samples[f"{family}_{modality}"],
            profile,
            state,
        )
        if features is None:
            raise FileNotFoundError(
                f"Missing {family}/{modality} for {state}/{profile_name}"
            )
        if checks is not None:
            assert checks == current, "Different image/text model provenance"
        checks = current
        assert features.shape[1:] == (30, 3, 1024)
        if family == "imagenet":
            cal = grouped(features, rows, groups, lambda r: r["split"] == "cal")
            data[f"cal_{axis}"] = cal[data["fit_indices"]].mean(0)
            del cal
            for view in ("a", "b"):
                if modality == "images":
                    for n in (1, 3, 5, 10, 15):
                        data[f"{view}_{n}_{axis}"] = grouped(
                            features,
                            rows,
                            groups,
                            lambda r, v=view, n=n: (
                                r["split"] == v and r["view_index"] < n
                            ),
                        )
                else:
                    data[f"{view}_{axis}"] = grouped(
                        features, rows, groups, lambda r, v=view: r["split"] == v
                    )
        elif family == "coco":
            means = (
                grouped(features, rows, groups, lambda r: True)
                if modality == "texts"
                else features
            )
            data[f"cal_{axis}"] = means[data["cal_indices"]].float().mean(0)
            for split in ("fit", "dev", "test"):
                data[f"{split}_{axis}"] = means[data[f"{split}_indices"]]
            if modality == "texts":
                test_groups = data["groups"][data["test_indices"]].tolist()
                for n in (1, 3, 5):
                    data[f"test_caption{n}_y"] = grouped(
                        features,
                        rows,
                        test_groups,
                        lambda r, n=n: r["caption_index"] < n,
                    )
                for label, predicate in (
                    ("first2", lambda r: r["caption_index"] < 2),
                    ("remaining", lambda r: r["caption_index"] >= 2),
                ):
                    data[f"test_{label}_y"] = grouped(
                        features, rows, test_groups, predicate
                    )
        elif modality == "images":
            data["test_x"] = features
        else:
            data["test_y_positive"] = grouped(
                features, rows, groups, lambda r: r["positive"]
            )
            data["test_y_negative"] = grouped(
                features, rows, groups, lambda r: not r["positive"]
            )
        del features
        emit(
            "view_modality_ready_v4",
            state=state,
            profile=profile_name,
            family=family,
            modality=modality,
        )
    data["state_checks"] = checks
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)
    emit(
        "views_complete_v4",
        state=state,
        profile=profile_name,
        family=family,
        bytes=path.stat().st_size,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--state", default="final_ema")
    p.add_argument("--profiles", default="native,bare,neutral")
    p.add_argument("--families", default="imagenet,coco,aro")
    a = p.parse_args()
    torch.set_num_threads(1)
    for profile in a.profiles.split(","):
        for family in a.families.split(","):
            build(a.output_dir, a.state, profile, family)


if __name__ == "__main__":
    main()
