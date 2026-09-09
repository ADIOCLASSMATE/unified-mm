"""All-layer feature extraction for SigLIP, DINOv2, MAE, and original Qwen."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch

from scripts.geometry_v5_encoders import EncoderAdapter, verify_parameters
from utils.research.geometry_v5_assets import RUN, emit, write_json


def main():
    import torch_npu

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN)
    parser.add_argument(
        "--model", choices=["qwen_text", "siglip", "dinov2", "mae"], required=True
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cal-smoke", action="store_true")
    args = parser.parse_args()
    assert torch_npu is not None and torch.npu.is_available()
    rank, world = (
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    torch.set_num_threads(2)
    torch.npu.set_device(rank)
    adapter = EncoderAdapter(args.model, torch.device("npu", rank))
    root = args.output_dir
    contract_path = root / "adapter-contracts" / f"{args.model}.json"
    if rank == 0:
        if contract_path.exists():
            assert json.loads(contract_path.read_text()) == adapter.contract
        else:
            assert args.cal_smoke, (
                "Freeze adapter via calibration smoke before full extraction"
            )
            write_json(contract_path, adapter.contract)
        verification = verify_parameters(adapter.model, adapter.path)
        write_json(
            root / "model-verification" / f"{args.model}.json",
            {**verification, "contract": adapter.contract},
        )
    samples = json.loads((root / "samples.json").read_text())
    for dataset, all_rows in samples.items():
        modality = "image" if dataset.endswith("images") else "text"
        if modality not in adapter.collectors:
            continue
        eligible = list(range(len(all_rows)))
        if args.cal_smoke:
            if dataset.startswith("aro"):
                continue
            eligible = [i for i in eligible if all_rows[i]["split"] == "cal"][:32]
        indices = eligible[rank::world]
        if not indices:
            continue
        directory = (
            root / ("encoder-smoke" if args.cal_smoke else "features") / args.model
        )
        destination = directory / f"{dataset}-rank-{rank:02d}-of-{world}.pt"
        if destination.exists():
            existing = torch.load(
                str(destination), weights_only=True, mmap=True, map_location="cpu"
            )
            assert existing["contract"] == adapter.contract
            assert existing["indices"].tolist() == indices
            emit("encoder_shard_reused", model=args.model, dataset=dataset, rank=rank)
            continue
        chunks, endpoints, truncations = [], [], 0
        started = time.monotonic()
        audit = {}
        for start in range(0, len(indices), args.batch_size):
            rows = [all_rows[i] for i in indices[start : start + args.batch_size]]
            features, endpoint, stats = adapter.forward(rows, modality)
            if start == 0:
                # Deliberately supply wrong opposite-modality fields: never consumed.
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
                alternative, alternative_endpoint, _ = adapter.forward(
                    replaced, modality
                )
                assert torch.equal(features, alternative) and torch.equal(
                    endpoint, alternative_endpoint
                )
                one, one_endpoint, _ = adapter.forward(rows[:1], modality)
                maximum = float((features[:1].float() - one.float()).abs().max())
                cosine = torch.nn.functional.cosine_similarity(
                    features[:1].float().flatten(1), one.float().flatten(1)
                ).item()
                assert cosine > 0.995, (args.model, "batch invariance", cosine)
                audit = {
                    "opposite_modality_replacement_exact": True,
                    "batch_single_max_abs": maximum,
                    "batch_single_cosine": cosine,
                    "native_endpoint_finite": bool(torch.isfinite(one_endpoint).all()),
                }
            chunks.append(features)
            endpoints.append(endpoint)
            truncations += stats["truncated_texts"]
            if start == 0 or (start // args.batch_size + 1) % 40 == 0:
                emit(
                    "encoder_progress",
                    model=args.model,
                    dataset=dataset,
                    rank=rank,
                    done=min(start + args.batch_size, len(indices)),
                    total=len(indices),
                    seconds=time.monotonic() - started,
                )
        payload = {
            "schema": "cross_model_geometry_v5_features_1",
            "model": args.model,
            "dataset": dataset,
            "rank": rank,
            "world_size": world,
            "hostname": socket.gethostname(),
            "indices": torch.tensor(indices),
            "features": torch.cat(chunks),
            "native_endpoint": torch.cat(endpoints),
            "layers": adapter.collectors[modality].names,
            "pools": adapter.pools()[modality],
            "contract": adapter.contract,
            "audit": audit,
            "truncated_texts": truncations,
            "training_updates": 0,
        }
        directory.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        emit(
            "encoder_shard_complete",
            model=args.model,
            dataset=dataset,
            rank=rank,
            shape=list(payload["features"].shape),
            seconds=time.monotonic() - started,
        )


if __name__ == "__main__":
    main()
