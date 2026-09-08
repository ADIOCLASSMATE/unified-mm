"""Extract frozen official flow models on the fixed Ascend development Notebook."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch

from scripts.geometry_v5_flow import FlowAdapter
from scripts.prepare_geometry_v5_assets import RUN, emit, write_json


def eligible_indices(rows, dataset, profile, cal_smoke, limit):
    if cal_smoke:
        return (
            []
            if dataset.startswith("aro")
            else [i for i, r in enumerate(rows) if r["split"] == "cal"][:limit]
        )
    return [i for i, row in enumerate(rows) if profile == "main" or row["robust"]]


def extract(adapter, root, samples, path, profile, args, rank, world):
    for dataset, all_rows in samples.items():
        modality = "image" if dataset.endswith("images") else "text"
        # Midpoint is explicitly an image corruption, not a fictitious text trajectory.
        if profile == "image_midpoint" and modality != "image":
            continue
        indices = eligible_indices(
            all_rows, dataset, profile, args.cal_smoke, args.limit
        )[rank::world]
        if not indices:
            continue
        directory = (
            root
            / ("flow-smoke" if args.cal_smoke else "features")
            / adapter.name
            / path
            / profile
        )
        destination = directory / f"{dataset}-rank-{rank:02d}-of-{world}.pt"
        if destination.exists():
            value = torch.load(
                str(destination), map_location="cpu", weights_only=True, mmap=True
            )
            assert (
                value["contract"] == adapter.contract
                and value["indices"].tolist() == indices
            )
            emit(
                "flow_shard_reused",
                model=adapter.name,
                path=path,
                profile=profile,
                dataset=dataset,
                rank=rank,
            )
            continue
        chunks, upstream_chunks, audit = [], {}, {}
        started = time.monotonic()
        for start in range(0, len(indices), args.batch_size):
            rows = [all_rows[i] for i in indices[start : start + args.batch_size]]
            features, upstream = adapter.forward(rows, modality, path, profile)
            if start == 0:
                replaced = [
                    {
                        **row,
                        **(
                            {"text": "unrelated target caption"}
                            if modality == "image"
                            else {
                                "source_path": "/target-must-never-be-read.png",
                                "image_id": -1,
                            }
                        ),
                    }
                    for row in rows
                ]
                alternative, _ = adapter.forward(replaced, modality, path, profile)
                assert torch.equal(features, alternative), (
                    "Opposite-target invariance failed"
                )
                one, _ = adapter.forward(rows[:1], modality, path, profile)
                cosine = torch.nn.functional.cosine_similarity(
                    features[:1].float().flatten(1), one.float().flatten(1)
                ).item()
                assert cosine > 0.995, (
                    adapter.name,
                    path,
                    profile,
                    "batch invariance",
                    cosine,
                )
                audit = {
                    "opposite_modality_replacement_exact": True,
                    "batch_single_max_abs": float(
                        (features[:1].float() - one.float()).abs().max()
                    ),
                    "batch_single_cosine": cosine,
                }
                if args.cal_smoke and path == "generation":
                    audit["official_generation_reference"] = (
                        adapter.validate_generation_reference(rows[:1], modality)
                    )
            chunks.append(features)
            for key, value in upstream.items():
                assert bool(torch.isfinite(value).all())
                upstream_chunks.setdefault(key, []).append(value)
            if start == 0 or (start // args.batch_size + 1) % 40 == 0:
                emit(
                    "flow_progress",
                    model=adapter.name,
                    path=path,
                    profile=profile,
                    dataset=dataset,
                    rank=rank,
                    done=min(start + args.batch_size, len(indices)),
                    total=len(indices),
                    seconds=time.monotonic() - started,
                )
        payload = {
            "schema": "cross_model_geometry_v5_features_1",
            "model": adapter.name,
            "path": path,
            "profile": profile,
            "dataset": dataset,
            "rank": rank,
            "world_size": world,
            "hostname": socket.gethostname(),
            "indices": torch.tensor(indices),
            "features": torch.cat(chunks),
            "upstream_features": {
                key: torch.cat(values) for key, values in upstream_chunks.items()
            },
            "layers": adapter.collector.names,
            "pools": adapter.contract["pools"][path],
            "contract": adapter.contract,
            "audit": audit,
            "training_updates": 0,
        }
        directory.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        emit(
            "flow_shard_complete",
            model=adapter.name,
            path=path,
            profile=profile,
            dataset=dataset,
            rank=rank,
            shape=list(payload["features"].shape),
            seconds=time.monotonic() - started,
        )


def main():
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument("--model", choices=["janusflow", "showo2"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--paths", default="understanding,generation")
    parser.add_argument("--profiles", default="main")
    parser.add_argument("--cal-smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    assert torch_npu is not None and torch.npu.is_available()
    rank, world = (
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    torch.set_num_threads(2)
    torch.npu.set_device(rank)
    adapter = FlowAdapter(args.model, torch.device("npu", rank))
    root = args.output_dir
    contract_path = root / "adapter-contracts" / f"{args.model}.json"
    if rank == 0:
        if contract_path.exists():
            assert json.loads(contract_path.read_text()) == adapter.contract
        else:
            assert args.cal_smoke
            write_json(contract_path, adapter.contract)
        write_json(root / "model-verification" / f"{args.model}.json", adapter.verify())
        emit("flow_weights_verified", model=args.model)
    samples = json.loads((root / "samples.json").read_text())
    for path in args.paths.split(","):
        assert path in {"understanding", "generation"}
        for profile in args.profiles.split(","):
            assert profile in {"main", "seed1", "seed2", "image_midpoint"}
            if path == "understanding" and profile != "main":
                continue
            extract(adapter, root, samples, path, profile, args, rank, world)
    if rank == 0 and adapter.vae is not None:
        write_json(
            root / "model-verification" / f"{args.model}-vae.json", adapter.verify_vae()
        )
    emit(
        "flow_rank_complete",
        model=args.model,
        rank=rank,
        paths=args.paths,
        profiles=args.profiles,
    )


if __name__ == "__main__":
    main()
