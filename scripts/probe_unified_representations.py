"""Frozen-representation diagnostics, separate from the formal likelihood suite.

Prepare deterministic samples on CPU, extract independent image/text features
on the development NPU Notebook, then analyze on CPU. No training model weights
are changed. All splits, prompts, layers and ridge strengths are fixed before
seeing any scores. There is no hashing or external model in this experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils.research.representation_protocol import (
    ROOT, FLICKR, IMAGENET, TEMPLATES, PREFIX, SUFFIX, DATASETS,
    emit, write_json, read_jsonl,
)
from utils.research.representation_features import make_batch, FeatureCollector


def prepare(args):
    from scripts.evaluate_cross_dataset_retrieval import load_records
    from utils.evaluation.native_understanding import (
        DEFAULT_CLASSES,
        DEFAULT_CLASSNAMES,
        load_imagenet_records,
        load_openai_clip_class_names,
    )
    from utils.evaluation_model_source import resolve_evaluation_model_source

    if (args.output_dir / "samples.json").exists():
        raise FileExistsError("Refusing to replace an existing sample protocol")
    source = resolve_evaluation_model_source(args.model_source)
    _, flickr = load_records(FLICKR)
    generator = torch.Generator().manual_seed(args.seed)
    fit_indices = set(torch.randperm(len(flickr), generator=generator)[:800].tolist())
    samples = {key: [] for key in DATASETS}
    for row in flickr:
        common = {
            "group": row.image_index,
            "image_id": row.img_id,
            "fit": row.image_index in fit_indices,
        }
        samples["flickr_images"].append({**common, "source_path": row.source_path})
        for caption_index, caption in enumerate(row.captions):
            samples["flickr_texts"].append(
                {**common, "text": caption, "caption_index": caption_index}
            )
    names, _ = load_openai_clip_class_names(ROOT / DEFAULT_CLASSNAMES)
    records = load_imagenet_records(
        IMAGENET / "manifest_val.jsonl", ROOT / DEFAULT_CLASSES
    )
    groups = defaultdict(list)
    for row in records:
        groups[row.class_index].append(row)
    for class_index in range(1000):
        generator = torch.Generator().manual_seed(args.seed + 1009 * class_index)
        indices = torch.randperm(50, generator=generator)[:10].tolist()
        for position, index in enumerate(indices):
            row = groups[class_index][index]
            samples["imagenet_images"].append(
                {
                    "group": class_index,
                    "image_id": row.img_id,
                    "source_path": row.source_path,
                    "fit": position < 6,
                }
            )
        for template_index, template in enumerate(TEMPLATES):
            samples["imagenet_texts"].append(
                {
                    "group": class_index,
                    "text": template.format(name=names[class_index]),
                    "fit": template_index < 6,
                    "template_index": template_index,
                }
            )
    protocol = {
        "schema": "unified_representation_diagnostic_v1",
        "seed": args.seed,
        "model": source.report(),
        "project_formal_protocol": False,
        "runtime_hashing_enabled": False,
        "model_weights_frozen": True,
        "prefix": PREFIX,
        "readout_suffix": SUFFIX,
        "encoding": "independent modalities; no paired-caption access in image forwards",
        "attention": "B random image sigma; strict query, diagonal content",
        "image_posterior": "one deterministic sample per image, native cached scale",
        "pools": ["x0_mean_data_tokens", "xt_shared_suffix_query"],
        "layers": "embedding, every residual block, final trained RMSNorm",
        "flickr_linear_fit": "800 images fit; 200 image-disjoint test, all 5 captions grouped",
        "imagenet_linear_fit": "6 val images/class fit; 4 disjoint val images/class test",
        "imagenet_text_fit": "6 fixed templates/class fit; 2 distinct templates/class test",
        "ridge_relative_strength": 0.1,
        "centering": "full-pool unlabeled means for descriptive full retrieval; fit-only means for probes",
        "counts": {key: len(rows) for key, rows in samples.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "samples.json", samples)
    write_json(args.output_dir / "protocol.json", protocol)
    emit("prepared", **protocol["counts"], model=source.report())


@torch.inference_mode()
def extract(args):
    import torch_npu
    from omegaconf import OmegaConf

    from utils.evaluation.multimodal_likelihood import PosteriorCache
    from utils.evaluation_model_source import (
        configure_model_source,
        resolve_evaluation_model_source,
    )
    from utils.utils import load_model_tokenizer

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.set_num_threads(2)
    if not torch.npu.is_available():
        raise RuntimeError("NPU extraction must run on the development Notebook")
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    torch.manual_seed(args.seed)
    source = resolve_evaluation_model_source(args.model_source)
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    if source.report() != protocol["model"]:
        raise ValueError("model source changed since sample preparation")
    config = OmegaConf.load(args.config)
    configure_model_source(config, source)
    model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)
    model.requires_grad_(False).to(device).eval()
    collector = FeatureCollector(model.model)
    samples = json.loads((args.output_dir / "samples.json").read_text())
    emit(
        "model_loaded",
        rank=rank,
        hostname=socket.gethostname(),
        device=str(device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        state_keys=len(model.state_dict()),
        source=source.report(),
    )
    caches = {}
    selected = args.datasets.split(",") if args.datasets else DATASETS
    for dataset in selected:
        output = (
            args.output_dir
            / "features"
            / f"{dataset}-rank-{rank:02d}-of-{world:02d}.pt"
        )
        if output.exists() and not args.limit:
            emit("existing_features", rank=rank, dataset=dataset)
            continue
        dataset_rows = samples[dataset]
        if args.limit:
            dataset_rows = dataset_rows[: args.limit]
        indices = list(range(rank, len(dataset_rows), world))
        modality = "image" if dataset.endswith("images") else "text"
        family = dataset.split("_")[0]
        cache = None
        if modality == "image":
            if family not in caches:
                cache_root = (
                    FLICKR / "vae_posterior_mar_kl16/shards"
                    if family == "flickr"
                    else IMAGENET / "vae_posterior_mar_kl16/val_shards"
                )
                caches[family] = PosteriorCache(
                    cache_root,
                    expected_image_tokens=256,
                    expected_latent_dim=16,
                    seed=args.seed,
                )
            cache = caches[family]
        chunks = []
        started = time.monotonic()
        for start in range(0, len(indices), args.batch_size):
            selected_indices = indices[start : start + args.batch_size]
            rows = [dataset_rows[index] for index in selected_indices]
            batch, mask, readout = make_batch(
                rows, modality, tokenizer, model, cache, device, args.seed
            )
            collector.begin(mask, readout)
            outputs = model.model(**batch)
            extracted = collector.finish()
            actual_query = outputs.last_hidden_state[
                torch.arange(len(rows), device=device), readout
            ]
            if not torch.equal(extracted[:, -1, 1], actual_query.cpu()):
                raise AssertionError("hook readout does not match the model output")
            # The initial query is a constant mask, independent of sample identity.
            if not torch.equal(
                extracted[:, 0, 1], extracted[:1, 0, 1].expand_as(extracted[:, 0, 1])
            ):
                raise AssertionError("unexpected sample information at the query input")
            if start == 0:
                # The strict query and pooled content cannot read this hidden target.
                changed = dict(batch)
                changed_ids = batch["X0_input_ids"].clone()
                changed_ids[torch.arange(len(rows), device=device), readout] = (
                    int(tokenizer.eos_token_id) + 1
                ) % model.config.vocab_size
                changed["X0_input_ids"] = changed_ids
                collector.begin(mask, readout)
                model.model(**changed)
                if not torch.equal(extracted, collector.finish()):
                    raise AssertionError(
                        "representations leaked the hidden target token"
                    )
                emit("hidden_target_invariance_passed", rank=rank, dataset=dataset)
            chunks.append(extracted)
            del outputs, batch
            if start == 0 or (start // args.batch_size + 1) % 10 == 0:
                emit(
                    "extract_progress",
                    rank=rank,
                    dataset=dataset,
                    done=min(start + args.batch_size, len(indices)),
                    total=len(indices),
                    elapsed=round(time.monotonic() - started, 2),
                )
        if not chunks:
            raise ValueError("empty rank; reduce world size for a smoke run")
        result = {
            "indices": torch.tensor(indices),
            "features": torch.cat(chunks),
            "source": source.report(),
            "dataset": dataset,
            "rank": rank,
            "world_size": world,
            "hostname": socket.gethostname(),
            "layers": [str(value) for value in range(29)] + ["final_norm"],
            "pools": protocol["pools"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.limit:
            output = output.with_name("smoke-" + output.name)
        temporary = output.with_suffix(".tmp")
        torch.save(result, temporary)
        temporary.replace(output)
        emit(
            "dataset_complete",
            rank=rank,
            dataset=dataset,
            shape=list(result["features"].shape),
            elapsed=round(time.monotonic() - started, 2),
        )
    write_json(
        args.output_dir / f"extract-complete-rank-{rank:02d}.json",
        {
            "rank": rank,
            "world_size": world,
            "source": source.report(),
            "smoke_limit": args.limit,
            "hostname": socket.gethostname(),
            "device_count": torch.npu.device_count(),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "extract"))
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/selfless/unified_baseline_100b_ascend_64npu.yaml",
    )
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    (prepare if arguments.action == "prepare" else extract)(arguments)
