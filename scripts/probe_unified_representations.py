"""Frozen-representation diagnostics, separate from the formal likelihood suite.

Prepare deterministic samples on CPU, extract independent image/text features
on the development NPU Notebook, then analyze on CPU. No training model weights
are changed. All splits, prompts, layers and ridge strengths are fixed before
seeing any scores. There is no hashing or external model in this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
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

ROOT = Path(__file__).resolve().parents[1]
FLICKR = ROOT / "public/benchmarks/flickr30k_karpathy_retrieval_v1"
IMAGENET = ROOT / "public/datasets/imagenet_full"
TEMPLATES = (
    "a photo of a {name}.",
    "an image of a {name}.",
    "a picture of a {name}.",
    "this is a {name}.",
    "the subject is a {name}.",
    "a close-up of a {name}.",
    "a photograph showing a {name}.",
    "there is a {name} in the picture.",
)
PREFIX = "Describe this image in one detailed caption:"
SUFFIX = "\nThe main subject is"
DATASETS = ("flickr_images", "flickr_texts", "imagenet_images", "imagenet_texts")


def emit(event, **kwargs):
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def make_batch(rows, modality, tokenizer, model, cache, device, seed):
    from utils.evaluation.multimodal_likelihood import (
        build_attention_masks,
        build_image_sigma,
        image_order_mc_seed,
    )

    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    sequences = []
    for row in rows:
        if modality == "image":
            start = len(prefix) + 1
            body = [model.config.image_mask_token_id] * 256
            ids = (
                prefix
                + [model.config.boi_token_id]
                + body
                + [model.config.eoi_token_id]
            )
            types = [0] * len(prefix) + [2] + [1] * 256 + [2]
            sigma = list(range(len(ids)))
            sigma[-1] = len(prefix) + 1
            order = build_image_sigma(
                256, order="random", seed=image_order_mc_seed(seed, row["image_id"], 0)
            )
            sigma[start : start + 256] = [len(prefix) + 2 + value for value in order]
            data_positions = list(range(start, start + 256))
        else:
            body = tokenizer.encode(row["text"], add_special_tokens=False)
            if not body:
                raise ValueError("empty text")
            start = len(prefix)
            ids = prefix + body
            types = [0] * len(ids)
            sigma = list(range(len(ids)))
            data_positions = list(range(start, len(ids)))
        tail = suffix + [tokenizer.eos_token_id]
        sigma.extend(range(len(ids), len(ids) + len(tail)))
        ids.extend(tail)
        types.extend([0] * len(tail))
        sequences.append((ids, types, sigma, data_positions, start))
    length = math.ceil(max(len(row[0]) for row in sequences) / 64) * 64
    batch = len(rows)
    ids = torch.full((batch, length), tokenizer.eos_token_id, dtype=torch.long)
    types = torch.full_like(ids, 3)
    sigma = torch.zeros_like(ids)
    segments = torch.full_like(ids, -1)
    data_mask = torch.zeros((batch, length), dtype=torch.bool)
    readout = torch.zeros(batch, dtype=torch.long)
    latents = torch.zeros((batch, length, 16), dtype=torch.bfloat16)
    spans = []
    for index, (item, row) in enumerate(zip(sequences, rows)):
        row_ids, row_types, row_sigma, positions, start = item
        size = len(row_ids)
        ids[index, :size] = torch.tensor(row_ids)
        types[index, :size] = torch.tensor(row_types)
        sigma[index, :size] = torch.tensor(row_sigma)
        segments[index, :size] = 0
        data_mask[index, positions] = True
        readout[index] = size - 1
        if modality == "image":
            latents[index, start : start + 256] = cache.sample(row["image_id"])
            spans.append([index, 0, start, start + 256])
    ids, types, sigma, segments = [
        tensor.to(device) for tensor in (ids, types, sigma, segments)
    ]
    query_mask, content_mask = build_attention_masks(
        sigma=sigma,
        segment_ids=segments,
        token_types=types,
        input_ids=ids,
        boi_token_id=model.config.boi_token_id,
        attention_contract=model.config.dual_stream_attention_contract,
        device=device,
    )
    kwargs = {
        "X0_input_ids": ids,
        "token_types": types,
        "flow_sigma": sigma,
        "attention_mask": query_mask,
        "content_attention_mask": content_mask,
        "calculate_likelihood": True,
        "use_cache": False,
        "return_x0_hidden_state": True,
        "image_span_table": torch.tensor(
            spans, device=device, dtype=torch.long
        ).reshape(-1, 4),
    }
    if modality == "image":
        kwargs.update(image_latents=latents.to(device), image_latent_mask=types.eq(1))
    return kwargs, data_mask.to(device), readout.to(device)


class FeatureCollector:
    def __init__(self, backbone):
        self.backbone = backbone
        self.handles = []
        self.handles.append(backbone.layers[0].register_forward_pre_hook(self.before))
        for index, layer in enumerate(backbone.layers):
            self.handles.append(layer.register_forward_hook(self.after(index + 1)))

    def begin(self, mask, readout):
        self.mask = mask
        self.readout = readout
        self.values = []

    def collect(self, x0, xt):
        count = self.mask.sum(dim=1, keepdim=True).float()
        content = (x0.float() * self.mask.unsqueeze(-1)).sum(dim=1) / count
        query = xt[torch.arange(xt.shape[0], device=xt.device), self.readout].float()
        self.values.append(torch.stack([content, query], dim=1))

    def before(self, module, args):
        self.collect(args[0], args[1])

    def after(self, index):
        def hook(module, args, output):
            self.collect(output[0], output[1])
            if index == len(self.backbone.layers):
                self.collect(
                    self.backbone.norm(output[0]), self.backbone.norm(output[1])
                )

        return hook

    def finish(self):
        values = torch.stack(self.values, dim=1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("nonfinite extracted features")
        return values.to(device="cpu", dtype=torch.bfloat16)


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
