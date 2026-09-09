"""Frozen V2 semantics experiment; independent modalities and controlled queries.

This is a representation diagnostic, not the repository's likelihood benchmark.
Prepare the immutable sample manifest before extraction or viewing any scores.
All accelerator forwards must run on the permanent Ascend development Notebook.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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

from scripts.probe_unified_representations import emit, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1"
RESEARCH = ROOT / "output/evaluation/research" / RUN.name
COCO = ROOT / "public/benchmarks/mscoco_karpathy_retrieval_v1"
IMAGENET = ROOT / "public/datasets/imagenet_full"
MULTIMODAL = ROOT / "public/benchmarks/selfless_multimodal_likelihood_v1"
LAYERS = [str(i) for i in range(29)] + ["final_norm"]
POOLS = ["content_mean", "content_last_sigma", "query_native", "query_swapped_mask"]
TEMPLATES = (
    "a photo of a {name}.",
    "a photograph showing a {name}.",
    "this is a {name}.",
    "the subject is a {name}.",
    "an image of a {name}.",
    "there is a {name} in the picture.",
)
PROFILES = {
    "bare": {"prompt": "bare", "sigma": 0, "posterior": "sample", "subset": "all"},
    "native": {"prompt": "native", "sigma": 0, "posterior": "sample", "subset": "all"},
    "neutral": {
        "prompt": "neutral",
        "sigma": 0,
        "posterior": "sample",
        "subset": "all",
    },
    "bare_sigma1": {
        "prompt": "bare",
        "sigma": 1,
        "posterior": "sample",
        "subset": "robust",
    },
    "bare_sigma2": {
        "prompt": "bare",
        "sigma": 2,
        "posterior": "sample",
        "subset": "robust",
    },
    "native_sigma1": {
        "prompt": "native",
        "sigma": 1,
        "posterior": "sample",
        "subset": "robust",
    },
    "native_sigma2": {
        "prompt": "native",
        "sigma": 2,
        "posterior": "sample",
        "subset": "robust",
    },
    "bare_mean": {
        "prompt": "bare",
        "sigma": 0,
        "posterior": "mean",
        "subset": "robust",
    },
}


def prepare(args):
    from scripts.evaluate_cross_dataset_retrieval import load_records
    from utils.evaluation.native_understanding import (
        DEFAULT_CLASSES,
        DEFAULT_CLASSNAMES,
        load_imagenet_records,
        load_openai_clip_class_names,
    )
    from utils.evaluation_model_source import resolve_evaluation_model_source

    if (args.output_dir / "protocol.json").exists():
        raise FileExistsError("An existing V2 protocol must not be overwritten")
    rng = random.Random(args.seed)
    _, coco = load_records(COCO)
    shuffled = list(coco)
    rng.shuffle(shuffled)
    samples = {
        f"{family}_{modality}": []
        for family in ("coco", "imagenet", "hard")
        for modality in ("images", "texts")
    }
    selected_coco_ids = set()
    for index, row in enumerate(shuffled[:2500]):
        split = (
            "cal"
            if index < 500
            else "fit"
            if index < 1000
            else "dev"
            if index < 1500
            else "test"
        )
        selected_coco_ids.add(int(row.source_image_id))
        base = {
            "group": row.image_index,
            "image_id": row.img_id,
            "split": split,
            "robust": split == "cal" or 1500 <= index < 1600,
            "source_image_id": row.source_image_id,
        }
        samples["coco_images"].append({**base, "source_path": row.source_path})
        for j, caption in enumerate(row.captions):
            samples["coco_texts"].append({**base, "text": caption, "caption_index": j})
    # Entirely new image identities relative to V1, with a separate calibration split.
    previous = json.loads(
        (
            RESEARCH / "representation-diagnostic/step-95415-ema-20260907/samples.json"
        ).read_text()
    )
    used = {row["image_id"] for row in previous["imagenet_images"]}
    records = load_imagenet_records(
        IMAGENET / "manifest_val.jsonl", ROOT / DEFAULT_CLASSES
    )
    names, _ = load_openai_clip_class_names(ROOT / DEFAULT_CLASSNAMES)
    groups = defaultdict(list)
    for row in records:
        if row.img_id not in used:
            groups[row.class_index].append(row)
    class_order = list(range(1000))
    rng.shuffle(class_order)
    mapping_fit_classes = set(class_order[:600])
    mapping_dev_classes = set(class_order[600:800])
    for cls in range(1000):
        rows = groups[cls]
        rng.shuffle(rows)
        mapping_split = (
            "fit"
            if cls in mapping_fit_classes
            else "dev"
            if cls in mapping_dev_classes
            else "test"
        )
        for j, row in enumerate(rows[:8]):
            split = "cal" if j < 2 else "fit" if j < 5 else "test"
            samples["imagenet_images"].append(
                {
                    "group": cls,
                    "image_id": row.img_id,
                    "source_path": row.source_path,
                    "split": split,
                    "mapping_split": mapping_split,
                    "robust": False,
                }
            )
        for j, template in enumerate(TEMPLATES):
            samples["imagenet_texts"].append(
                {
                    "group": cls,
                    "image_id": cls,
                    "text": template.format(name=names[cls]),
                    "split": "cal" if j < 2 else "fit" if j < 4 else "test",
                    "mapping_split": mapping_split,
                    "template_index": j,
                    "robust": False,
                }
            )
    hard_groups = defaultdict(list)
    for row in read_jsonl(MULTIMODAL / "tasks/sugarcrepe.jsonl"):
        if int(Path(row["metadata"]["filename"]).stem) not in selected_coco_ids:
            hard_groups[row["category"]].append(row)
    used_hard_images = set()
    for category in sorted(hard_groups):
        rows = hard_groups[category]
        rng.shuffle(rows)
        selected = []
        for row in rows:
            if row["image_id"] not in used_hard_images:
                used_hard_images.add(row["image_id"])
                selected.append(row)
            if len(selected) == 150:
                break
        if len(selected) != 150:
            raise ValueError(f"Insufficient disjoint hard negatives for {category}")
        for j, row in enumerate(selected):
            base = {
                "group": len(samples["hard_images"]),
                "image_id": row["image_id"],
                "split": "cal" if j < 25 else "dev" if j < 50 else "test",
                "category": category,
                "item_id": row["item_id"],
                "robust": False,
            }
            samples["hard_images"].append(base)
            for k, caption in enumerate(row["candidates"]):
                samples["hard_texts"].append(
                    {**base, "text": caption, "positive": k == row["label"]}
                )
    source = resolve_evaluation_model_source(args.model_source)
    protocol = {
        "schema": "unified_semantics_v2_frozen_1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "model": source.report(),
        "model_weights_frozen": True,
        "project_formal_benchmark": False,
        "runtime_hashing_enabled": False,
        "layers": LAYERS,
        "pools": POOLS,
        "profiles": PROFILES,
        "states": {
            "final_ema": {"path": str(args.model_source.resolve()), "init_seed": None},
            "final_raw": {"path": str(RUN / "hf_model-final"), "init_seed": None},
            **{
                f"init{s}": {
                    "path": str(ROOT / "public/models/Qwen--Qwen3-0.6B-Base"),
                    "init_seed": s,
                }
                for s in (42, 43, 44)
            },
        },
        "initialization_claim": "Initialization-procedure references, NOT authenticated historical step-zero weights",
        "counts": {key: len(rows) for key, rows in samples.items()},
        "coco_split": {"cal": 500, "fit": 500, "dev": 500, "test": 1000},
        "robustness": "COCO 500 calibration + 100 predetermined test images, and all their captions; compare identical subsets",
        "imagenet_split": "1000 classes, 2 new images cal / 3 fit / 3 test; 2 text templates in each split",
        "mapping_classes": {"fit": 600, "dev": 200, "test": 200},
        "hard_split": "7 SugarCrepe categories: 25 cal / 25 dev / 100 test each; unique images, excludes all selected COCO identities",
        "centering": "Means fit on calibration images/texts separately, frozen before test; center THEN L2 normalize",
        "query_intervention": "Only initial XT at selected readout changes; X0, positions, token types and visibility fixed",
        "target_image_positions": "Native T2I target permutation is fixed across samples within each sigma replicate; never seeded by class or paired image identity",
        "interpretation": "First native target query encodes the context, not an observed target token; swaps are counterfactual",
        "ridge_relative_strength": 0.1,
        "primary_endpoint": "final_norm; complete layer scan descriptive; choose intermediate layer only on dev",
        "cache_roots": {
            "coco": str(COCO / "vae_posterior_mar_kl16/shards"),
            "imagenet": str(IMAGENET / "vae_posterior_mar_kl16/val_shards"),
            "hard": str(MULTIMODAL / "vae_posterior_mar_kl16_v2/shards"),
        },
    }
    write_json(args.output_dir / "samples.json", samples)
    write_json(args.output_dir / "protocol.json", protocol)
    emit("prepared_v2", **protocol["counts"])


def posterior(cache, image_id, mode):
    if mode == "sample":
        return cache.sample(image_id)
    shard, index = cache._locations[int(image_id)]
    return cache._payloads[shard]["posterior_stats"][index, :, :16]


def make_batch(rows, modality, tokenizer, config, cache, device, seed, profile):
    from utils.evaluation.multimodal_likelihood import (
        build_attention_masks,
        build_image_sigma,
        image_order_mc_seed,
    )

    prompt = profile["prompt"]
    prefix_text = ""
    if prompt == "neutral":
        prefix_text = "Input:"
    elif prompt == "native":
        prefix_text = (
            "Describe this image in one detailed caption:"
            if modality == "image"
            else "Generate an image matching this description:"
        )
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    sequences = []
    for row in rows:
        order_identity = row["image_id"] if modality == "image" else 0
        order = build_image_sigma(
            256,
            order="random",
            seed=image_order_mc_seed(seed, order_identity, profile["sigma"]),
        )
        ids = list(prefix)
        types = [0] * len(ids)
        sigma = list(range(len(ids)))
        image_start = None
        if modality == "image":
            image_start = len(ids) + 1
            ids += (
                [config.boi_token_id]
                + [config.image_mask_token_id] * 256
                + [config.eoi_token_id]
            )
            types += [2] + [1] * 256 + [2]
            sigma += (
                [image_start - 1]
                + [image_start + 1 + int(x) for x in order]
                + [image_start]
            )
            data_positions = list(range(image_start, image_start + 256))
            last = image_start + max(range(256), key=lambda j: int(order[j]))
        else:
            body = tokenizer.encode(row["text"], add_special_tokens=False)
            if not body:
                raise ValueError("Empty text input")
            data_positions = list(range(len(ids), len(ids) + len(body)))
            ids += body
            types += [0] * len(body)
            sigma += list(range(len(sigma), len(ids)))
            last = len(ids) - 1
        native_image_query = modality == "text" and prompt == "native"
        if native_image_query:
            image_start = len(ids) + 1
            ids += (
                [config.boi_token_id]
                + [config.image_mask_token_id] * 256
                + [config.eoi_token_id]
            )
            types += [2] + [1] * 256 + [2]
            sigma += (
                [image_start - 1]
                + [image_start + 1 + int(x) for x in order]
                + [image_start]
            )
            readout = image_start + min(range(256), key=lambda j: int(order[j]))
        else:
            readout = len(ids)
            ids.append(tokenizer.eos_token_id)
            types.append(0)
            sigma.append(max(sigma, default=-1) + 1)
        sequences.append(
            (ids, types, sigma, data_positions, last, readout, image_start)
        )
    width = math.ceil(max(len(row[0]) for row in sequences) / 64) * 64
    n = len(rows)
    ids = torch.full((n, width), tokenizer.eos_token_id, dtype=torch.long)
    types = torch.full_like(ids, 3)
    sigma = torch.zeros_like(ids)
    segments = torch.full_like(ids, -1)
    data_mask = torch.zeros_like(ids, dtype=torch.bool)
    latent_mask = torch.zeros_like(data_mask)
    latents = torch.zeros(n, width, 16, dtype=torch.bfloat16)
    last = torch.zeros(n, dtype=torch.long)
    readout = torch.zeros_like(last)
    spans = []
    for i, (row, values) in enumerate(zip(rows, sequences)):
        ri, rt, rs, positions, end, query, image_start = values
        size = len(ri)
        ids[i, :size] = torch.tensor(ri)
        types[i, :size] = torch.tensor(rt)
        sigma[i, :size] = torch.tensor(rs)
        segments[i, :size] = 0
        data_mask[i, positions] = True
        last[i], readout[i] = end, query
        if image_start is not None:
            spans.append([i, 0, image_start, image_start + 256])
        if modality == "image":
            latents[i, image_start : image_start + 256] = posterior(
                cache, row["image_id"], profile["posterior"]
            )
            latent_mask[i, image_start : image_start + 256] = True
    ids, types, sigma, segments = [x.to(device) for x in (ids, types, sigma, segments)]
    query_mask, content_mask = build_attention_masks(
        sigma=sigma,
        segment_ids=segments,
        token_types=types,
        input_ids=ids,
        boi_token_id=config.boi_token_id,
        attention_contract=config.dual_stream_attention_contract,
        device=device,
    )
    batch = {
        "X0_input_ids": ids,
        "token_types": types,
        "flow_sigma": sigma,
        "attention_mask": query_mask,
        "content_attention_mask": content_mask,
        "calculate_likelihood": True,
        "use_cache": False,
        "return_x0_hidden_state": True,
        "image_span_table": torch.tensor(
            spans, dtype=torch.long, device=device
        ).reshape(-1, 4),
    }
    if spans:
        batch.update(
            image_latents=latents.to(device), image_latent_mask=latent_mask.to(device)
        )
    return batch, data_mask.to(device), last.to(device), readout.to(device)


class Collector:
    def __init__(self, backbone):
        self.backbone = backbone
        self.handles = [backbone.layers[0].register_forward_pre_hook(self.before)]
        for i, layer in enumerate(backbone.layers):
            self.handles.append(layer.register_forward_hook(self.after(i)))

    def begin(self, mask, last, readout, replacement=None, audit=False, reference=None):
        self.mask, self.last, self.readout = mask, last, readout
        self.replacement = replacement
        self.values = []
        self.audit, self.reference = audit, reference
        self.x0_values = []

    def collect(self, x0, xt):
        i = torch.arange(x0.shape[0], device=x0.device)
        mean = (x0.float() * self.mask.unsqueeze(-1)).sum(1) / self.mask.sum(
            1, keepdim=True
        )
        self.values.append(
            torch.stack(
                (mean, x0[i, self.last].float(), xt[i, self.readout].float()), 1
            )
        )
        if self.audit:
            if self.reference is not None:
                if not torch.equal(x0, self.reference[len(self.values) - 1]):
                    raise AssertionError(
                        "Mask-only intervention changed X0 activations"
                    )
            else:
                self.x0_values.append(x0.detach().clone())

    def before(self, module, args):
        x0, xt = args[:2]
        if self.replacement is not None:
            xt = xt.clone()
            xt[torch.arange(xt.shape[0], device=xt.device), self.readout] = (
                self.replacement
            )
            args = (x0, xt, *args[2:])
        self.collect(x0, xt)
        return args

    def after(self, index):
        def hook(module, args, output):
            self.collect(output[0], output[1])
            if index == len(self.backbone.layers) - 1:
                self.collect(
                    self.backbone.norm(output[0]), self.backbone.norm(output[1])
                )

        return hook

    def finish(self):
        features = torch.stack(self.values, 1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Nonfinite representation")
        return features.to(device="cpu", dtype=torch.bfloat16)


def load_state(args, protocol, device):
    from omegaconf import OmegaConf
    from safetensors import safe_open

    from utils.utils import load_model_tokenizer

    state = protocol["states"][args.state]
    cfg = OmegaConf.load(RUN / "config.yaml")
    cfg.model.model_path = state["path"]
    cfg.training.use_gradient_checkpointing = False
    cfg.training.from_scratch = False
    seed = state["init_seed"] or 42
    random.seed(seed)
    torch.manual_seed(seed)
    model, tokenizer = load_model_tokenizer(cfg, model_dtype=torch.bfloat16)
    model.requires_grad_(False).eval().to(device)
    checks = {
        "state": args.state,
        "path": state["path"],
        "init_seed": state["init_seed"],
        "historical_step_zero": False if state["init_seed"] is not None else None,
        "num_state_keys": len(model.state_dict()),
        "text_mask_id": model.config.mask_token_id,
        "image_mask_id": model.config.image_mask_token_id,
    }
    emb = model.model.embed_tokens.weight
    text_mask = emb[model.config.mask_token_id].float()
    image_mask = emb[model.config.image_mask_token_id].float()
    checks.update(
        mask_equal=torch.equal(text_mask, image_mask),
        mask_cosine=float(
            torch.nn.functional.cosine_similarity(text_mask, image_mask, dim=0)
        ),
        projector_norm=float(
            model.model.image_token_embedder.z_proj.weight.float().norm()
        ),
    )
    if state["init_seed"] is not None:
        if not checks["mask_equal"]:
            raise AssertionError("Fresh image mask must be copied from text mask")
        # Check actual pretrained backbone identity, not just a logged source name.
        with safe_open(
            str(Path(state["path"]) / "model.safetensors"), framework="pt"
        ) as f:
            for name in (
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.27.mlp.down_proj.weight",
            ):
                actual = model.state_dict()[name].detach().cpu()
                if not torch.equal(actual, f.get_tensor(name).to(actual.dtype)):
                    raise AssertionError(
                        f"Initialization lost pretrained weights: {name}"
                    )
        checks["pretrained_backbone_equality_checked"] = True
    return model, tokenizer, checks


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
        raise RuntimeError("Run extraction on dev-wjx-ascend, not the local agent host")
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    # Real accelerator arithmetic, not import-only evidence.
    unit = torch.ones(8, 8, device=device)
    if not torch.equal((unit @ unit).cpu(), torch.full((8, 8), 8.0)):
        raise AssertionError("NPU arithmetic check failed")
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    samples = json.loads((args.output_dir / "samples.json").read_text())
    model, tokenizer, state_checks = load_state(args, protocol, device)
    emit(
        "model_loaded_v2",
        rank=rank,
        hostname=socket.gethostname(),
        device=str(device),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        **state_checks,
    )
    collector = Collector(model.model)
    caches = {}
    profiles = args.profiles.split(",")
    datasets = args.datasets.split(",") if args.datasets else list(samples)
    outroot = args.output_dir / args.state
    for profile_name in profiles:
        profile = protocol["profiles"][profile_name]
        for dataset in datasets:
            rows_all = samples[dataset]
            eligible = [
                i
                for i, row in enumerate(rows_all)
                if profile["subset"] == "all" or row["robust"]
            ]
            if not eligible:
                continue
            if args.limit:
                eligible = eligible[: args.limit]
            indices = eligible[rank::world]
            if not indices:
                raise ValueError("Empty rank; increase smoke limit")
            path = outroot / profile_name / f"{dataset}-rank-{rank:02d}-of-{world}.pt"
            if args.limit:
                path = path.with_name("smoke-" + path.name)
            if path.exists():
                existing = torch.load(
                    str(path), weights_only=True, map_location="cpu", mmap=True
                )
                if (
                    existing["indices"].tolist() != indices
                    or existing["state_checks"] != state_checks
                ):
                    raise ValueError("Existing artifact has incompatible identity")
                emit(
                    "reused_v2",
                    rank=rank,
                    state=args.state,
                    profile=profile_name,
                    dataset=dataset,
                )
                continue
            family = dataset.split("_")[0]
            modality = "image" if dataset.endswith("images") else "text"
            cache = None
            if modality == "image":
                if family not in caches:
                    caches[family] = PosteriorCache(
                        Path(protocol["cache_roots"][family]),
                        expected_image_tokens=256,
                        expected_latent_dim=16,
                        seed=protocol["seed"],
                    )
                cache = caches[family]
            do_swap = args.state.startswith("final") and profile["prompt"] in (
                "bare",
                "native",
            )
            chunks = []
            started = time.monotonic()
            for start in range(0, len(indices), args.batch_size):
                chosen = indices[start : start + args.batch_size]
                rows = [rows_all[i] for i in chosen]
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
                audit = start == 0
                collector.begin(mask, last, readout, audit=audit and do_swap)
                outputs = model.model(**batch)
                value = collector.finish()
                reference = collector.x0_values
                if not torch.equal(
                    value[:, -1, 2],
                    outputs.last_hidden_state[
                        torch.arange(len(rows), device=device), readout
                    ].cpu(),
                ):
                    raise AssertionError(
                        "Collected final query differs from backbone output"
                    )
                if audit:
                    # Change only the hidden target while preserving all observable context.
                    changed = dict(batch)
                    target_image = modality == "text" and profile["prompt"] == "native"
                    if target_image:
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
                    if not torch.equal(value, collector.finish()):
                        raise AssertionError(
                            "Observed Content/native query leaked hidden target"
                        )
                    emit(
                        "hidden_target_invariance_v2",
                        rank=rank,
                        state=args.state,
                        profile=profile_name,
                        dataset=dataset,
                        all_layers=True,
                    )
                if do_swap:
                    current_image_query = (
                        modality == "text" and profile["prompt"] == "native"
                    )
                    alternate = (
                        model.config.mask_token_id
                        if current_image_query
                        else model.config.image_mask_token_id
                    )
                    replacement = model.model.embed_tokens.weight[alternate]
                    collector.begin(
                        mask,
                        last,
                        readout,
                        replacement,
                        audit=audit,
                        reference=reference if audit else None,
                    )
                    model.model(**batch)
                    swapped = collector.finish()
                    if not torch.equal(value[:, :, :2], swapped[:, :, :2]):
                        raise AssertionError("Mask swap altered pooled content")
                    value = torch.cat([value, swapped[:, :, 2:3]], dim=2)
                    if audit:
                        emit(
                            "all_x0_mask_swap_invariance_v2",
                            rank=rank,
                            state=args.state,
                            profile=profile_name,
                            dataset=dataset,
                            all_layers=True,
                        )
                elif state_checks["mask_equal"]:
                    # This equality is an initialization contract, not a measured trained swap.
                    value = torch.cat([value, value[:, :, 2:3]], 2)
                chunks.append(value)
                del outputs, reference, batch
                collector.x0_values = []
                if start == 0 or (start // args.batch_size + 1) % 20 == 0:
                    emit(
                        "progress_v2",
                        rank=rank,
                        state=args.state,
                        profile=profile_name,
                        dataset=dataset,
                        done=min(start + args.batch_size, len(indices)),
                        total=len(indices),
                        elapsed=round(time.monotonic() - started, 2),
                    )
            payload = {
                "indices": torch.tensor(indices),
                "features": torch.cat(chunks),
                "state_checks": state_checks,
                "profile": profile,
                "dataset": dataset,
                "rank": rank,
                "world_size": world,
                "layers": LAYERS,
                "pools": POOLS[: chunks[0].shape[2]],
                "hostname": socket.gethostname(),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            emit(
                "dataset_complete_v2",
                rank=rank,
                state=args.state,
                profile=profile_name,
                dataset=dataset,
                shape=list(payload["features"].shape),
                elapsed=round(time.monotonic() - started, 2),
            )
    write_json(
        outroot / f"{'smoke-' if args.limit else ''}complete-rank-{rank:02d}.json",
        {
            "rank": rank,
            "world_size": world,
            "state_checks": state_checks,
            "profiles": profiles,
            "datasets": datasets,
            "smoke_limit": args.limit,
            "hostname": socket.gethostname(),
            "device_count": torch.npu.device_count(),
        },
    )


@torch.inference_mode()
def verify(args):
    import torch_npu
    from safetensors import safe_open

    from utils.evaluation.multimodal_likelihood import PosteriorCache
    from scripts.probe_unified_representations import (
        FLICKR,
        FeatureCollector,
    )
    from scripts.probe_unified_representations import (
        make_batch as legacy_batch,
    )

    torch.set_num_threads(2)
    if not torch.npu.is_available():
        raise RuntimeError("Verification requires the development NPU")
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    protocol = json.loads((args.output_dir / "protocol.json").read_text())
    model, tokenizer, checks = load_state(args, protocol, device)
    checked = 0
    with safe_open(
        str(Path(checks["path"]) / "model.safetensors"), framework="pt"
    ) as handle:
        state = model.state_dict()
        stored_keys = handle.keys()
        for key in stored_keys:
            actual = state[key].detach().cpu()
            if not torch.equal(actual, handle.get_tensor(key).to(actual.dtype)):
                raise AssertionError(f"Loaded checkpoint tensor differs: {key}")
            checked += 1
    emit("checkpoint_equality_passed", stored_tensors=checked)
    legacy_root = RESEARCH / "representation-diagnostic/step-95415-ema-20260907"
    legacy_samples = json.loads((legacy_root / "samples.json").read_text())
    cache = PosteriorCache(
        FLICKR / "vae_posterior_mar_kl16/shards",
        expected_image_tokens=256,
        expected_latent_dim=16,
        seed=424242,
    )
    comparisons = {}
    for dataset in ("flickr_images", "flickr_texts"):
        old = torch.load(
            str(legacy_root / "features" / f"{dataset}-rank-00-of-16.pt"),
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )
        indices = old["indices"][:16].tolist()
        rows = [legacy_samples[dataset][i] for i in indices]
        modality = "image" if dataset.endswith("images") else "text"
        batch, mask, readout = legacy_batch(
            rows, modality, tokenizer, model, cache, device, 424242
        )
        collector = FeatureCollector(model.model)
        collector.begin(mask, readout)
        model.model(**batch)
        current = collector.finish()
        for hook in collector.handles:
            hook.remove()
        saved = old["features"][:16]
        comparison = {
            "bitwise_equal": torch.equal(current, saved),
            "max_absolute_difference": float(
                (current.float() - saved.float()).abs().max()
            ),
            "relative_rms_difference": float(
                (current.float() - saved.float()).square().mean().sqrt()
                / saved.float().square().mean().sqrt().clamp_min(1e-8)
            ),
        }
        if comparison["relative_rms_difference"] > 0.002:
            raise AssertionError(f"Legacy replay mismatch: {dataset}: {comparison}")
        if modality == "image":
            new_batch, new_mask, last, new_readout = make_batch(
                rows,
                modality,
                tokenizer,
                model.config,
                cache,
                device,
                424242,
                PROFILES["native"],
            )
            collector2 = Collector(model.model)
            collector2.begin(new_mask, last, new_readout)
            model.model(**new_batch)
            new_features = collector2.finish()
            for hook in collector2.handles:
                hook.remove()
            comparison["v2_native_image_content_equals_v1"] = torch.equal(
                new_features[:, :, 0], current[:, :, 0]
            )
            comparison["v2_content_max_difference"] = float(
                (new_features[:, :, 0].float() - current[:, :, 0].float()).abs().max()
            )
            if not comparison["v2_native_image_content_equals_v1"]:
                raise AssertionError(
                    "V2 native image Content changed despite identical visible context"
                )
        comparisons[dataset] = comparison
    write_json(
        args.output_dir / "legacy-replay-and-weight-verification.json",
        {
            "state_checks": checks,
            "stored_checkpoint_tensors_exactly_loaded": checked,
            "comparisons": comparisons,
            "torch_npu": torch_npu.__version__,
            "hostname": socket.gethostname(),
        },
    )
    emit(
        "legacy_and_weight_verification_complete",
        checked_tensors=checked,
        comparisons=comparisons,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "extract", "verify"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, default=RUN / "hf_model-final-ema")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--state", default="final_ema")
    parser.add_argument("--profiles", default="bare,native,neutral")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"prepare": prepare, "extract": extract, "verify": verify}[args.action](args)
