"""Expanded frozen geometry extraction on dev-wjx-ascend; reuse audited V2 semantics."""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_unified_representations import emit, write_json
from scripts.probe_unified_semantics_v2 import LAYERS, Collector, load_state, make_batch


@torch.inference_mode()
def extract(args):
    import torch_npu

    from utils.evaluation.multimodal_likelihood import PosteriorCache

    rank, world = (
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    torch.set_num_threads(2)
    if not torch.npu.is_available():
        raise RuntimeError("V4 forwards require the fixed Ascend development Notebook")
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    unit = torch.ones(8, 8, device=device)
    torch.testing.assert_close((unit @ unit).cpu(), torch.full((8, 8), 8.0))
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    samples = json.loads((args.output_dir / "samples.json").read_text())
    if protocol["schema"] != "unified_geometry_v4_frozen_1":
        raise ValueError("Unexpected V4 schema")
    profiles = args.profiles.split(",")
    assert set(profiles) <= set(protocol["states_profiles"][args.state])
    model, tokenizer, checks = load_state(args, protocol, device)
    if rank == 0:
        verification = dict(checks)
        if args.state.startswith("final"):
            from safetensors import safe_open

            count = 0
            actual_state = model.state_dict()
            with safe_open(
                str(Path(checks["path"]) / "model.safetensors"), framework="pt"
            ) as stored:
                stored_keys = stored.keys()
                for key in stored_keys:
                    actual = actual_state[key].detach().cpu()
                    assert torch.equal(
                        actual, stored.get_tensor(key).to(actual.dtype)
                    ), key
                    count += 1
            verification["stored_tensors_exactly_loaded"] = count
        else:
            assert checks["pretrained_backbone_equality_checked"]
        write_json(
            args.output_dir / "model-verification" / f"{args.state}.json", verification
        )
    emit(
        "model_loaded_v4",
        rank=rank,
        hostname=socket.gethostname(),
        device=str(device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        **checks,
    )
    collector, caches = Collector(model.model), {}
    datasets = args.datasets.split(",") if args.datasets else list(samples)
    completed = []
    for profile_name in profiles:
        profile = protocol["profiles"][profile_name]
        for dataset in datasets:
            all_rows = samples[dataset]
            eligible = [
                i
                for i, r in enumerate(all_rows)
                if profile["subset"] == "all" or r["robust"]
            ]
            if not eligible:
                continue
            if args.limit:
                eligible = eligible[: args.limit]
            indices = eligible[rank::world]
            if not indices:
                raise ValueError("Increase smoke limit to populate all ranks")
            filename = f"{dataset}-rank-{rank:02d}-of-{world}.pt"
            if args.limit:
                filename = "smoke-" + filename
            path = args.output_dir / args.state / profile_name / filename
            if path.exists():
                existing = torch.load(
                    path, weights_only=True, mmap=True, map_location="cpu"
                )
                assert existing["indices"].tolist() == indices
                assert (
                    existing["state_checks"] == checks
                    and existing["profile"] == profile
                )
                assert (
                    existing["layers"] == LAYERS
                    and existing["pools"] == protocol["pools"]
                )
                completed.append(str(path))
                emit(
                    "reused_v4",
                    rank=rank,
                    state=args.state,
                    profile=profile_name,
                    dataset=dataset,
                )
                continue
            family = dataset.split("_")[0]
            modality = "image" if dataset.endswith("images") else "text"
            if modality == "image" and family not in caches:
                caches[family] = PosteriorCache(
                    Path(protocol["cache_roots"][family]),
                    expected_image_tokens=256,
                    expected_latent_dim=16,
                    seed=protocol["seed"],
                )
            cache = caches.get(family) if modality == "image" else None
            chunks, started = [], time.monotonic()
            for start in range(0, len(indices), args.batch_size):
                chosen = indices[start : start + args.batch_size]
                rows = [all_rows[i] for i in chosen]
                batch, mask, last, readout = make_batch(
                    rows,
                    modality,
                    tokenizer,
                    model.config,
                    cache,
                    device,
                    protocol["seed"],
                    profile,
                )
                collector.begin(mask, last, readout)
                outputs = model.model(**batch)
                value = collector.finish()
                assert torch.equal(
                    value[:, -1, 2],
                    outputs.last_hidden_state[
                        torch.arange(len(rows), device=device), readout
                    ].cpu(),
                )
                if start == 0:
                    changed = dict(batch)
                    if modality == "text" and profile["prompt"] == "native":
                        changed["image_latents"] = torch.full_like(
                            batch["image_latents"], 0.375
                        )
                        changed["image_latent_mask"] = batch["token_types"].eq(1)
                    else:
                        changed_ids = batch["X0_input_ids"].clone()
                        changed_ids[torch.arange(len(rows), device=device), readout] = (
                            tokenizer.eos_token_id + 1
                        ) % model.config.vocab_size
                        changed["X0_input_ids"] = changed_ids
                    collector.begin(mask, last, readout)
                    model.model(**changed)
                    assert torch.equal(value, collector.finish()), (
                        "Hidden target leaked into readout"
                    )
                    emit(
                        "hidden_target_invariance_v4",
                        rank=rank,
                        state=args.state,
                        profile=profile_name,
                        dataset=dataset,
                        all_layers=True,
                    )
                chunks.append(value)
                del outputs, batch
                if start == 0 or (start // args.batch_size + 1) % 40 == 0:
                    emit(
                        "progress_v4",
                        rank=rank,
                        state=args.state,
                        profile=profile_name,
                        dataset=dataset,
                        done=min(start + args.batch_size, len(indices)),
                        total=len(indices),
                        seconds=round(time.monotonic() - started, 2),
                    )
            payload = {
                "indices": torch.tensor(indices),
                "features": torch.cat(chunks),
                "state_checks": checks,
                "profile": profile,
                "dataset": dataset,
                "rank": rank,
                "world_size": world,
                "layers": LAYERS,
                "pools": protocol["pools"],
                "hostname": socket.gethostname(),
                "schema": "unified_geometry_v4_features_1",
                "hidden_target_invariance": True,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            completed.append(str(path))
            emit(
                "dataset_complete_v4",
                rank=rank,
                state=args.state,
                profile=profile_name,
                dataset=dataset,
                shape=list(payload["features"].shape),
                seconds=time.monotonic() - started,
            )
    write_json(
        args.output_dir / "stage-markers" / args.stage / f"rank-{rank:02d}.json",
        {
            "state": args.state,
            "rank": rank,
            "world_size": world,
            "profiles": profiles,
            "datasets": datasets,
            "limit": args.limit,
            "files": completed,
            "training_updates": 0,
            "device": str(device),
            "hostname": socket.gethostname(),
            "arithmetic_checked": True,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--state", default="final_ema")
    p.add_argument("--profiles", default="native")
    p.add_argument("--datasets", default="")
    p.add_argument("--stage", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    extract(p.parse_args())


if __name__ == "__main__":
    main()
