#!/usr/bin/env python3
"""Paired, no-hash qualitative T2I/I2T/text generation for final Unified EMAs.

prepare freezes inputs on CPU; run uses one independent worker per NPU;
render produces a portable HTML gallery and checks exact sample coverage.
No model, reference dataset, or old evaluation output is modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import html
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
             "DIFFUSERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

# Import the project attention bootstrap before Transformers (also CPU-safe).
import models.modeling_model.modeling_selfless_flow  # noqa: F401, E402

from omegaconf import OmegaConf
from safetensors import safe_open

from pretrain.train_selfless_flow import _build_i2t_generation_prefix, _generate_i2t_caption_batch
from utils.evaluation_model_source import (
    configure_model_source, load_model_source_weights, resolve_evaluation_model_source,
)
from utils.image_generation_io import decode_latents, load_vae
from utils.experiment_registry import is_temporary_training_run, model_labels, experiment_identity, task_training_labels, read_run_identity
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.imagenet_synthetic_text_index import ImageNetSyntheticTextIndex
from utils.utils import load_model_tokenizer

SCHEMA = "unified_qualitative_generation_v1"
BASE_CONFIG = "configs/selfless/unified_baseline_100b_ascend_64npu.yaml"
PROMPT_CONFIG = "configs/protocols/unified_qualitative_prompts_v1.json"
T2I_PREFIX = "Generate an image matching this description:"
I2T_PREFIX = "Describe this image in one detailed caption:"
MODEL_LABELS = model_labels()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(value)
        temporary = Path(f.name)
    os.replace(temporary, path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def task_training(run):
    return task_training_labels(experiment_identity(run))


def model_inventory(repo):
    paths = {p.parent.parent.name: p.parent
             for p in (repo / "output").glob("unified-*/hf_model-final-ema/config.json")
             if not is_temporary_training_run(p.parent.parent.name)}
    ordered = [name for name in MODEL_LABELS if name in paths]
    ordered += sorted(set(paths) - set(ordered))
    result = []
    for name in ordered:
        run_config_path = paths[name].parent / "config.yaml"
        run_config = OmegaConf.to_container(OmegaConf.load(run_config_path), resolve=True) if run_config_path.is_file() else None
        identity = read_run_identity(paths[name].parent, run_config)
        if identity["purpose"] == "temporary":
            continue
        source = resolve_evaluation_model_source(paths[name])
        if not source.is_hf_final_ema:
            raise ValueError(f"not a final EMA: {paths[name]}")
        saved = read_json(paths[name] / "config.json")
        model_id, label = identity["id"], identity["label"]
        result.append({"id": model_id, "label": label, "run": name,
                       "checkpoint": str(paths[name].resolve()), "source": source.report(),
                       "architecture": saved["architecture_variant"],
                       "backbone_attention": saved["dual_stream_attention_contract"],
                       "flow_attention": saved.get("flow_head_attention_contract", "not_applicable" if saved["architecture_variant"] == "positionwise_flow_head_on_b" else "selfless_strict"),
                       "flow_condition": saved.get("dynamic_xt_flow_condition_contract", saved.get("flow_condition_contract", "legacy_or_architecture_owned")),
                       "image_order": saved.get("training_image_sigma_order", "random"),
                       "task_training": task_training_labels(identity), "experiment_identity": identity})
    if not result:
        raise ValueError("no completed Unified final EMA exports found")
    return result


def fixed_posterior(stats, seed):
    if tuple(stats.shape) != (256, 32) or not bool(torch.isfinite(stats).all()):
        raise ValueError("invalid scaled KL16 posterior")
    mean, std = stats.float().chunk(2, dim=-1)
    if bool((std < 0).any()):
        raise ValueError("negative posterior standard deviation")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return (mean + std * torch.randn(mean.shape, generator=g)).to(stats.dtype)


def prepare(args):
    root, repo = args.output_dir.resolve(), args.repo_root.resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError(f"refusing to replace frozen inputs: {root}")
    custom = read_json(repo / PROMPT_CONFIG)
    assert len(custom["t2i"]) == len(custom["text"]) == 32
    models = model_inventory(repo)
    source_manifest = jsonl(repo / "public/datasets/imagenet_full/manifest_val.jsonl")
    if len(source_manifest) != 50000 or any(x["split"] != "val" for x in source_manifest):
        raise ValueError("expected held-out ImageNet validation manifest")
    by_class = defaultdict(list)
    for i, row in enumerate(source_manifest):
        by_class[row["synset"]].append(i)
    classes = sorted(by_class)
    if len(classes) != 1000:
        raise ValueError("expected 1000 ImageNet classes")
    rng = random.Random(42)
    # 32 spread-out synsets, one seeded validation image from each; no quality selection.
    selected = [rng.choice(by_class[classes[round(i * 999 / 31)]]) for i in range(32)]
    cache_path = repo / "public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_val_fp16.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    if cache["metadata"]["format"] != "imagenet_kl16_scaled_posterior_v1" or cache["metadata"]["stats_layout"] != "scaled_mean_then_scaled_std":
        raise ValueError("unexpected ImageNet posterior contract")
    text_index = ImageNetSyntheticTextIndex(repo / "public/datasets/imagenet1k_synthetic_v1/indexed/val/manifest.json")
    t2i, i2t, latents = [], [], []
    for n, idx in enumerate(selected):
        row = source_manifest[idx]
        caption = text_index.read_caption(idx)
        prompt = text_index.read_t2i(idx)
        if int(cache["img_ids"][idx]) != int(row["img_id"]) or caption["synset"] != row["synset"] or prompt["image_id"] != row["image_id"] or caption["manifest_index"] != idx:
            raise ValueError(f"ImageNet image/caption/cache identity mismatch: {idx}")
        seed = (42 + 97409 * idx + 11) & ((1 << 63) - 1)
        latents.append(fixed_posterior(cache["posterior_stats"][idx], seed))
        sample_id = f"imagenet_{n:02d}"
        i2t.append({"id": sample_id, "group": "imagenet_val", "image_id": row["image_id"],
                    "manifest_index": idx, "synset": row["synset"], "posterior_seed": seed,
                    "reference": caption["captions"][0]["text"], "input_image": f"inputs/{sample_id}.png",
                    "display_kind": "fixed_model_input_vae_reconstruction"})
        t2i.append({"id": sample_id, "group": "imagenet_val", "prompt": prompt["model_result"]["prompts"][0]["prompt"],
                    "image_id": row["image_id"], "manifest_index": idx,
                    "reference_image": f"inputs/{sample_id}.png"})
    text_index.close()
    del cache
    from utils.evaluation.multimodal_likelihood import PosteriorCache, arithmetic_seed
    for dataset in ("mscoco", "flickr30k"):
        assets = repo / f"public/benchmarks/{dataset}_karpathy_retrieval_v1"
        metadata = read_json(assets / "manifest.json")
        if metadata.get("complete") is not True or metadata.get("split") != "karpathy_test":
            raise ValueError(f"incomplete test assets: {assets}")
        rows = jsonl(assets / "retrieval.jsonl")
        chosen = random.Random(42).sample(range(len(rows)), 16)
        posterior = PosteriorCache(assets / "vae_posterior_mar_kl16/shards", expected_image_tokens=256, expected_latent_dim=16, seed=42)
        for n, idx in enumerate(chosen):
            row = rows[idx]
            if row["split"] != "karpathy_test":
                raise ValueError("non-test image selected")
            sample_id = f"{dataset}_{n:02d}"
            latents.append(posterior.sample(row["img_id"]).clone())
            i2t.append({"id": sample_id, "group": f"{dataset}_karpathy_test", "image_id": row["source_image_id"],
                        "manifest_index": idx, "posterior_seed": arithmetic_seed(42, row["img_id"], 11),
                        "reference": row["captions"][0], "reference_captions": row["captions"],
                        "input_image": f"inputs/{sample_id}.png", "display_kind": "fixed_model_input_vae_reconstruction"})
            # Preserve originals as an additional reference, not as the encoded crop.
            raw = Path(row["source_path"])
            if not raw.is_file():
                raise FileNotFoundError(raw)
            original = f"inputs/original-{sample_id}{raw.suffix.lower()}"
            (root / "inputs").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw, root / original)
            i2t[-1]["original_image"] = original
        del posterior
    t2i += custom["t2i"]
    for task_rows in (t2i, i2t, custom["text"]):
        if len({r["id"] for r in task_rows}) != len(task_rows):
            raise ValueError("duplicate sample IDs")
    root.mkdir(parents=True, exist_ok=True)
    torch.save(torch.stack(latents), root / "i2t_latents.pt")
    contract = {"sampling_steps": 10, "flow_solver": "heun", "cfg": 3.5, "cfg_schedule": "constant",
                "flow_temperature": 1.0, "parallel_rate": 1, "t2i_seeds": [42, 43],
                "t2i_noise_seed_formula": "seed + 1000003 * prompt_index; CPU float32 noise indexed by spatial token",
                "i2t_max_new_tokens": 96, "i2t_temperature": 0.0, "text_max_new_tokens": 256,
                "text_temperatures": [0.0, 0.8], "text_sampling": "greedy or full-softmax temperature; no top-k/top-p",
                "text_prompt_style": "base-model continuation; no chat template", "use_cache": True,
                "model_dtype": "bfloat16", "vae_dtype": "float32", "vae_scaling_factor": 0.2325,
                "runtime_hashing_enabled": False, "sample_selection": "frozen before model generation; all outputs retained",
                "i2t_image_order": "checkpoint-native: random fixed per image, except sequential E",
                "i2t_inputs": "fixed scaled posterior samples; no ground-truth caption passed to model"}
    manifest = {"schema": SCHEMA, "created_at": utc_now(), "repo_root": str(repo), "models": models,
                "contract": contract, "samples": {"t2i": t2i, "i2t": i2t, "text": custom["text"]},
                "expected_per_model": {"t2i": 128, "i2t": 64, "text": 64},
                "excluded_without_final_ema": ["unified-a-0p6b-t2i-only-100bphys-s42-r1"],
                "scope": "completed Unified-MM experimental conditions, one final EMA each; historical non-Unified ImageNet-only models and intermediate checkpoints excluded"}
    write_json(root / "manifest.json", manifest)
    print(json.dumps({"prepared": str(root), "models": len(models), "expected_per_model": manifest["expected_per_model"]}, ensure_ascii=False), flush=True)


def build_t2i_item(tokenizer, model, prompt, prompt_index, seed, image_order):
    prefix = torch.tensor(tokenizer.encode(f"{T2I_PREFIX} {prompt}", add_special_tokens=False), dtype=torch.long)
    count, dim = int(model.config.image_tokens_per_img), int(model.config.image_latent_dim)
    ids = torch.cat((prefix, torch.tensor([model.config.boi_token_id]),
                     torch.full((count,), model.config.mask_token_id),
                     torch.tensor([model.config.eoi_token_id, tokenizer.eos_token_id])))
    types = torch.cat((torch.zeros(len(prefix), dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8),
                       torch.ones(count, dtype=torch.uint8), torch.tensor([2, 2], dtype=torch.uint8)))
    start = len(prefix) + 1
    loss_mask = torch.zeros_like(ids, dtype=torch.bool)
    loss_mask[start:start + count] = True
    return {"input_ids": ids, "token_types": types, "labels": torch.full_like(ids, -100),
            "image_loss_mask": loss_mask, "image_latents": torch.zeros(count, dim),
            "prompt_len": len(prefix), "suffix_len": 0, "image_start": start,
            "img_id": prompt_index + 1, "task_mode": "t2i", "reveal_seed": seed,
            "image_sigma_order": image_order}


def noise_for(prompt_index, seed, count=256, dim=16):
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 1000003 * int(prompt_index))
    return torch.randn((count, dim), generator=g, dtype=torch.float32)


def caption_sigmas(tokenizer, model, rows, order):
    _, _, base, start = _build_i2t_generation_prefix(tokenizer, text_prefix=I2T_PREFIX,
        boi_token_id=model.config.boi_token_id, eoi_token_id=model.config.eoi_token_id,
        image_mask_token_id=model.config.image_mask_token_id, image_tokens=model.config.image_tokens_per_img)
    count = int(model.config.image_tokens_per_img)
    values = []
    for row in rows:
        sigma = base.clone()
        if order == "random":
            g = torch.Generator(device="cpu").manual_seed(int(row["posterior_seed"]) + 53)
            sigma[start:start + count] = float(start + 1) + torch.rand(count, generator=g).argsort().float()
        elif order != "sequential":
            raise ValueError(order)
        values.append(sigma)
    return torch.stack(values)


def decode_suffix(tokenizer, suffix, stop_ids):
    ids, reason = [], "max_new_tokens"
    for value in suffix:
        value = int(value)
        if value in stop_ids:
            reason = "eos" if value == tokenizer.eos_token_id else "im_end"
            break
        ids.append(value)
    return {"text": tokenizer.decode(ids, skip_special_tokens=True).strip(), "token_ids": ids,
            "stop_reason": reason, "generated_tokens": len(ids)}


def save_png(tensor, path):
    from PIL import Image
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("non-finite decoded image")
    pixels = tensor.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(pixels).save(temporary)
    os.replace(temporary, path)


def check_loaded_values(model, checkpoint):
    state = model.state_dict()
    count = 0
    with safe_open(str(Path(checkpoint) / "model.safetensors"), framework="pt", device="cpu") as f:
        for name in f.keys():
            expected = f.get_tensor(name)
            if name not in state or not bool(torch.isfinite(expected).all()):
                raise RuntimeError(f"invalid or missing checkpoint tensor: {name}")
            actual = state[name].detach().cpu()
            if not torch.equal(actual, expected.to(actual.dtype)):
                raise RuntimeError(f"post-load checkpoint tensor mismatch: {name}")
            count += 1
    return {"complete": True, "stored_tensors_exact_after_dtype_cast": count, "runtime_hashing_enabled": False}


def event(root, rank, **payload):
    value = {"updated_at": utc_now(), "rank": rank, **payload}
    write_json(root / "progress" / f"rank-{rank:02d}.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def preflight(args):
    root = args.output_dir.resolve()
    manifest = read_json(root / "manifest.json")
    reports = []
    for spec in manifest["models"]:
        config = OmegaConf.load(BASE_CONFIG)
        config.training.runtime_hashing_enabled = False
        source = resolve_evaluation_model_source(spec["checkpoint"])
        configure_model_source(config, source)
        model, tokenizer = load_model_tokenizer(config, model_dtype=torch.bfloat16)
        report = load_model_source_weights(model, source)
        report["full_checkpoint_value_check"] = check_loaded_values(model, spec["checkpoint"])
        max_t2i_length = max(len(tokenizer.encode(f"{T2I_PREFIX} {row['prompt']}", add_special_tokens=False)) + 259
                             for row in manifest["samples"]["t2i"])
        if max_t2i_length > 512:
            raise ValueError("T2I prompt exceeds generation context")
        for row in manifest["samples"]["text"]:
            if not tokenizer.encode(row["prompt"], add_special_tokens=False):
                raise ValueError("empty pure-text input")
        reports.append({"model": spec["id"], "weight_check": report,
                        "max_t2i_sequence_length": max_t2i_length})
        print(json.dumps({"preflight": spec["id"], "exact_tensors": report["full_checkpoint_value_check"]["stored_tensors_exact_after_dtype_cast"]}), flush=True)
        del model, tokenizer
        gc.collect()
    write_json(root / "preflight.json", {"schema": SCHEMA, "complete": True, "cpu_only": True,
        "checked_at": utc_now(), "models": reports, "runtime_hashing_enabled": False})


@torch.inference_mode()
def run(args):
    import torch_npu  # noqa: F401
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU required")
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    root = args.output_dir.resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("unexpected manifest schema")
    input_latents = torch.load(root / "i2t_latents.pt", weights_only=True, map_location="cpu")
    config = OmegaConf.load(BASE_CONFIG)
    config.experiment.validation_vae_module_root = "public/code/mar"
    config.experiment.validation_vae_path = "public/vae/mar-kl16/kl16.ckpt"
    config.experiment.validation_vae_scaling_factor = 0.2325
    vae = load_vae(config, device, "fp32")
    all_i2t = manifest["samples"]["i2t"]
    input_indices = list(range(rank, len(all_i2t), world))
    if args.limit:
        input_indices = input_indices[:args.limit]
    for idx in input_indices:
        image_path = root / all_i2t[idx]["input_image"]
        if not image_path.exists():
            latent = input_latents[idx:idx + 1].reshape(1, 16, 16, 16).permute(0, 3, 1, 2).to(device)
            save_png(decode_latents(vae, latent, 0.2325)[0], image_path)
    for spec in manifest["models"]:
        if args.models and spec["id"] not in args.models.split(","):
            continue
        model_root = root / "models" / spec["id"]
        event(root, rank, model=spec["id"], stage="loading")
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        cfg.training.runtime_hashing_enabled = False
        source = resolve_evaluation_model_source(spec["checkpoint"])
        configure_model_source(cfg, source)
        cfg.model.image_flow_num_sampling_steps = "10"
        model, tokenizer = load_model_tokenizer(cfg, model_dtype=torch.bfloat16)
        weights = load_model_source_weights(model, source)
        if rank == 0:
            weights["full_checkpoint_value_check"] = check_loaded_values(model, spec["checkpoint"])
        write_json(model_root / "load_reports" / f"rank-{rank:02d}.json", weights)
        model.to(device).eval()
        order = str(getattr(model.config, "training_image_sigma_order", "random"))
        if order != spec["image_order"]:
            raise RuntimeError("checkpoint order changed after preparation")
        i2t_indices = list(range(rank, len(all_i2t), world))
        if args.limit:
            i2t_indices = i2t_indices[:args.limit]
        event(root, rank, model=spec["id"], stage="i2t")
        for offset in range(0, len(i2t_indices), 4):
            selected = i2t_indices[offset:offset + 4]
            rows = [all_i2t[i] for i in selected]
            captions, ids, reasons = _generate_i2t_caption_batch(model, tokenizer,
                input_latents[selected].to(device), text_prefix=I2T_PREFIX, max_new_tokens=min(96, args.smoke_token_limit or 96),
                temperature=0.0, base_sigma_batch=caption_sigmas(tokenizer, model, rows, order))
            for row, caption, tokens, reason in zip(rows, captions, ids, reasons):
                write_json(model_root / "i2t" / f"{row['id']}.json", {"sample_id": row["id"], "model": spec["id"],
                    "text": caption, "token_ids": tokens, "stop_reason": reason, "generated_tokens": len(tokens),
                    "temperature": 0.0, "input_image": row["input_image"], "image_order": order,
                    "reference_passed_to_model": False, "rank": rank})
        event(root, rank, model=spec["id"], stage="text")
        text_indices = list(range(rank, len(manifest["samples"]["text"]), world))
        if args.limit:
            text_indices = text_indices[:args.limit]
        for idx in text_indices:
            row = manifest["samples"]["text"][idx]
            prefix_ids = torch.tensor([tokenizer.encode(row["prompt"], add_special_tokens=False)], device=device)
            stops = [int(tokenizer.eos_token_id)]
            im_end = getattr(model.config, "im_end_token_id", None)
            if im_end is not None:
                stops.append(int(im_end))
            for temperature in (0.0, 0.8):
                seed = 424242 + 1000003 * idx
                torch.manual_seed(seed)
                torch.npu.manual_seed(seed)
                output, trace = model.generate("text", input_ids=prefix_ids, max_new_tokens=min(256, args.smoke_token_limit or 256),
                    temperature=temperature, eos_token_id=stops, use_cache=True, return_trace=True)
                if trace.get("backbone_kv_cache_enabled") is not True:
                    raise RuntimeError("text generation cache disabled")
                decoded = decode_suffix(tokenizer, output[0, prefix_ids.shape[1]:].cpu().tolist(), stops)
                mode = "greedy" if temperature == 0 else "sample"
                write_json(model_root / "text" / f"{row['id']}-{mode}.json", {"sample_id": row["id"],
                    "model": spec["id"], "prompt": row["prompt"], "temperature": temperature, "seed": seed,
                    "decoding": mode, "trace": trace, "rank": rank, **decoded})
        event(root, rank, model=spec["id"], stage="t2i")
        pairs = [(i, seed) for i in range(len(manifest["samples"]["t2i"])) for seed in (42, 43)]
        pairs = pairs[rank::world]
        if args.limit:
            pairs = pairs[:args.limit]
        for offset in range(0, len(pairs), 8):
            selected = pairs[offset:offset + 8]
            rows = [manifest["samples"]["t2i"][idx] for idx, _ in selected]
            items = [build_t2i_item(tokenizer, model, row["prompt"], idx, seed, order)
                     for row, (idx, seed) in zip(rows, selected)]
            batch = collate_imagenet_flow_cache(items, pad_to_length=512)
            noise = torch.stack([noise_for(idx, seed) for idx, seed in selected])
            started = time.monotonic()
            latents, trace = model.generate("t2i", input_ids=batch["input_ids"].to(device),
                token_types=batch["token_types"].to(device), sigma=batch["sigma"].to(device),
                spans=[(b, item["image_start"], item["image_start"] + 256) for b, item in enumerate(items)],
                image_latent_dim=16, initial_noise_bank=noise, flow_temperature=1.0, flow_cfg=3.5,
                flow_cfg_schedule="constant", flow_solver="heun", flow_num_steps=10, parallel_rate=1,
                order_strategy="sequential" if order == "sequential" else "spatial_halton",
                use_cache=True, return_trace=True)
            torch.npu.synchronize()
            seconds = time.monotonic() - started
            if latents is None or not bool(torch.isfinite(latents).all()) or trace.get("backbone_kv_cache_enabled") is not True:
                raise RuntimeError("invalid T2I generation or disabled cache")
            for begin in range(0, len(rows), 4):
                decoded = decode_latents(vae, latents[begin:begin + 4].float(), 0.2325)
                for j in range(len(decoded)):
                    at = begin + j
                    row, (idx, seed) = rows[at], selected[at]
                    name = f"{row['id']}-s{seed}"
                    image_path = model_root / "t2i" / f"{name}.png"
                    save_png(decoded[j], image_path)
                    write_json(image_path.with_suffix(".json"), {"sample_id": row["id"], "model": spec["id"],
                        "prompt": row["prompt"], "serialized_prompt": f"{T2I_PREFIX} {row['prompt']}",
                        "seed": seed, "noise_seed": seed + 1000003 * idx, "image": str(image_path.relative_to(root)),
                        "order_strategy": "sequential" if order == "sequential" else "spatial_halton",
                        "generation_seconds_in_batch": seconds, "rank": rank,
                        "latent_std": float(latents[at].float().std().item()),
                        "pixel_std": float(decoded[j].float().std().item())})
            event(root, rank, model=spec["id"], stage="t2i_batch_complete", count=offset + len(selected), seconds=seconds)
        del model, tokenizer
        gc.collect()
        torch.npu.empty_cache()
        event(root, rank, model=spec["id"], stage="model_complete")
    event(root, rank, stage="worker_complete", complete=True, world_size=world,
          limited_run=bool(args.limit or args.smoke_token_limit or args.models))


def render(args):
    root = args.output_dir.resolve()
    manifest = read_json(root / "manifest.json")
    models, samples = manifest["models"], manifest["samples"]
    esc = lambda x: html.escape(str(x), quote=True)
    records, counts, missing = {}, {}, []
    for spec in models:
        mid = spec["id"]
        counts[mid] = {}
        for task in ("t2i", "i2t", "text"):
            rows = [read_json(p) for p in sorted((root / "models" / mid / task).glob("*.json"))]
            key = lambda r: (r["sample_id"], r.get("seed") if task == "t2i" else r.get("decoding") if task == "text" else None)
            mapping = {key(r): r for r in rows}
            if len(mapping) != len(rows):
                raise ValueError("duplicate generated records")
            expected = {(r["id"], mode) for r in samples[task]
                        for mode in ((42, 43) if task == "t2i" else ("greedy", "sample") if task == "text" else (None,))}
            if set(mapping) - expected:
                raise ValueError("unexpected generated sample")
            for r in rows:
                if r["model"] != mid:
                    raise ValueError("record belongs to a different model")
                if task == "t2i" and not (root / r["image"]).is_file():
                    raise FileNotFoundError(root / r["image"])
            missing.extend((mid, task, key) for key in sorted(expected - set(mapping)))
            records[mid, task] = mapping
            counts[mid][task] = len(rows)
    input_complete = all((root / row["input_image"]).is_file() for row in samples["i2t"])
    complete = not missing and input_complete
    summary = {"schema": SCHEMA, "updated_at": utc_now(), "complete": complete, "models": len(models),
               "counts": counts, "missing_records": len(missing), "input_images_complete": input_complete,
               "expected_per_model": manifest["expected_per_model"], "runtime_hashing_enabled": False}
    write_json(root / "summary.json", summary)
    css = """body{font:15px system-ui,sans-serif;margin:24px;color:#17202a;background:#f5f6f8}h1{font-size:26px}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:3;border-bottom:1px solid #ddd}a{color:#1659a5}table{border-collapse:collapse;background:#fff}th,td{border:1px solid #ddd;padding:10px;vertical-align:top}th{background:#e9eef5}td{min-width:256px;max-width:360px}td:first-child,th:first-child{position:sticky;left:0;background:#f3f5f7;min-width:245px;z-index:1}.scroll{overflow:auto;max-height:85vh;margin-bottom:28px}.scroll thead{position:sticky;top:0;z-index:2}img{width:256px;height:256px;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 ui-monospace,monospace;margin:4px 0}small{color:#596575}.pending{color:#999}.warn{color:#924d05}section{margin-top:36px}.prompt{font-weight:600}details{margin:6px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}button{padding:6px 12px}footer{margin:32px 0}select{padding:6px}summary{cursor:pointer}"""
    body = [f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Unified-MM 生成质量对照</title><style>{css}</style>",
            "<h1>Unified-MM · T2I / I2T / 纯文本生成对照</h1>",
            f"<p>{len(models)} 个 final EMA；每模型 128 张 T2I、64 个 I2T、64 条文本续写。状态：{'全部完成' if complete else '生成中，缺项留空'}。</p>",
            "<p>固定样本及成对噪声，所有输出保留，无质量筛选。10 步 Heun / CFG 3.5 / BF16 模型 / FP32 VAE。E 使用 sequential，其余 spatial_halton；顺序是模型合同的一部分。这里只做定性展示，不代表正式 benchmark 分数。</p>",
            "<p>I2T 显示<strong>模型实际输入 latent 的 VAE 重建图</strong>；COCO/Flickr 可展开原图。参考描述仅用于人工对照，未输入模型。纯文本使用基座续写格式、greedy 与温度 0.8（无 top-k/top-p）；中文与组合提示含域外诊断。</p>",
            "<nav><a href='#t2i'>T2I</a> · <a href='#i2t'>I2T</a> · <a href='#text'>纯文本</a> · <a href='#inventory'>模型与进度</a> · <a href='manifest.json'>完整协议/来源</a> · <a href='results.jsonl'>原始结果 JSONL</a></nav>",
            "<div class='controls'><button onclick='selectModels(true)'>全部模型</button><button onclick='selectModels(false)'>只看主对照</button></div><div class='controls'>"]
    main_ids = {"b_x0", "b_flowdiag", "a_x0", "c_on_b", "d_on_b", "e_on_b", "f_on_b"}
    for spec in models:
        body.append(f"<label><input class='model-toggle' type='checkbox' checked data-main='{int(spec['id'] in main_ids)}' value='{esc(spec['id'])}' onchange='toggleModel(this)'>{esc(spec['label'])}</label>")
    body.append("</div>")
    all_results = []
    for task, title in (("t2i", "T2I · 同提示 / 同 seed"), ("i2t", "I2T · 同一图像输入"), ("text", "纯文本 · 同前缀续写")):
        body.append(f"<section id='{task}'><h2>{title}</h2><div class='scroll'><table><thead><tr><th>样本 / 输入</th>")
        for spec in models:
            body.append(f"<th data-model='{esc(spec['id'])}'>{esc(spec['label'])}<br><small class='warn'>{esc(spec['task_training'][task])}</small></th>")
        body.append("</tr></thead><tbody>")
        for row in samples[task]:
            modes = (42, 43) if task == "t2i" else ("greedy", "sample") if task == "text" else (None,)
            for mode in modes:
                body.append(f"<tr><td><small>{esc(row['id'])} · {esc(row['group'])} · {esc(mode or 'greedy')}</small>")
                if task == "i2t":
                    body.append(f"<p><img loading='lazy' src='{esc(row['input_image'])}' alt='模型输入 VAE 重建'></p><details><summary>参考描述（未输入模型）</summary><p>{esc(row['reference'])}</p></details>")
                    if row.get("original_image"):
                        body.append(f"<details><summary>原始图像（与编码裁剪可能不同）</summary><img loading='lazy' src='{esc(row['original_image'])}'></details>")
                else:
                    body.append(f"<pre class='prompt'>{esc(row['prompt'])}</pre>")
                    if row.get("reference_image"):
                        body.append(f"<details><summary>提示来源图像的 VAE 重建</summary><img loading='lazy' src='{esc(row['reference_image'])}'></details>")
                body.append("</td>")
                for spec in models:
                    result = records[spec["id"], task].get((row["id"], mode))
                    body.append(f"<td data-model='{esc(spec['id'])}'>")
                    if result is None:
                        body.append("<span class='pending'> </span>")
                    elif task == "t2i":
                        body.append(f"<a href='{esc(result['image'])}'><img loading='lazy' src='{esc(result['image'])}' alt='{esc(row['prompt'])}'></a>")
                    else:
                        body.append(f"<pre>{esc(result['text'])}</pre><small>{result['generated_tokens']} tokens · {esc(result['stop_reason'])}</small>")
                    if result is not None:
                        all_results.append({"task": task, **result})
                    body.append("</td>")
                body.append("</tr>")
        body.append("</tbody></table></div></section>")
    body.append("<section id='inventory'><h2>模型来源与进度</h2><table><tr><th>模型</th><th>Checkpoint / contract</th><th>已完成数量</th></tr>")
    for spec in models:
        body.append(f"<tr><td>{esc(spec['label'])}</td><td><pre>{esc(spec['checkpoint'])}</pre><small>step {spec['source']['global_step']} · {esc(spec['architecture'])} · backbone {esc(spec['backbone_attention'])} · flow {esc(spec['flow_attention'])}</small></td><td>{esc(counts[spec['id']])}</td></tr>")
    body.append("</table><p>仅纳入已完成的 Unified-MM final EMA；A T2I-only 尚无 final EMA，未纳入。中间 checkpoint 与更早的独立 ImageNet-only 项目不在本轮范围。</p></section>")
    body.append("""<script>function toggleModel(el){document.querySelectorAll('[data-model="'+el.value+'"]').forEach(x=>x.hidden=!el.checked)}function selectModels(all){document.querySelectorAll('.model-toggle').forEach(x=>{x.checked=all||x.dataset.main==='1';toggleModel(x)})}</script></html>""")
    write_text(root / "index.html", "\n".join(body))
    write_text(root / "results.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_results))
    write_text(root / "README.md", f"# Unified-MM 生成质量对照\n\n打开同目录 `index.html`，可按模型隐藏列，查看 T2I、I2T 与纯文本逐样本对照。\n\n状态：{'全部完成' if complete else '生成中'}。共 {len(models)} 个 final EMA，每模型 128 张图、64 个 I2T、64 条文本续写。\n\nI2T 输入为固定 posterior 的 VAE 重建，参考 caption 不进入模型。单模态未训练任务在页面标注；保留所有输出，无筛选。纯文本是基座续写，不是指令聊天。\n\n精确来源及采样协议见 `manifest.json`；逐条原始输出含 token IDs 见 `results.jsonl`；各模型载入校验在 `models/*/load_reports/`。\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.require_complete:
        if not complete:
            raise RuntimeError(f"incomplete qualitative outputs: {len(missing)} missing records; inputs={input_complete}")
        # Decode every image and check provenance/weight gates before publishing completion.
        from PIL import Image
        progress_paths = sorted((root / "progress").glob("rank-*.json"))
        if len(progress_paths) != 16:
            raise RuntimeError("expected all 16 production workers")
        for rank, path in enumerate(progress_paths):
            progress = read_json(path)
            if (progress.get("rank") != rank or progress.get("world_size") != 16
                    or progress.get("stage") != "worker_complete" or progress.get("complete") is not True
                    or progress.get("limited_run") is not False):
                raise RuntimeError("incomplete or smoke-only worker output")
        for spec in models:
            load_dir = root / "models" / spec["id"] / "load_reports"
            load = read_json(load_dir / "rank-00.json")
            if not load.get("full_checkpoint_value_check", {}).get("complete"):
                raise RuntimeError("missing full checkpoint load gate")
            for rank in range(16):
                report = read_json(load_dir / f"rank-{rank:02d}.json")
                if int(report["global_step"]) != int(spec["source"]["global_step"]):
                    raise RuntimeError("generation weight source step mismatch")
                if spec["architecture"] == "dynamic_xt" and not report.get("post_load_validation", {}).get("complete"):
                    raise RuntimeError("missing Dynamic-XT time embedding value gate")
        for path in root.glob("models/*/t2i/*.png"):
            with Image.open(path) as img:
                if img.size != (256, 256):
                    raise ValueError(f"invalid image shape: {path}")
                img.verify()
        write_json(root / "COMPLETED.json", {**summary, "image_decode_audit": "passed", "checkpoint_load_audit": "passed"})
        import zipfile
        archive = root / "unified-generation-gallery.zip"
        temporary = archive.with_suffix(".tmp.zip")
        paths = [root / name for name in ("index.html", "README.md", "manifest.json", "summary.json", "results.jsonl", "COMPLETED.json", "preflight.json")]
        paths += sorted(p for folder in ("models", "inputs") for p in (root / folder).rglob("*") if p.is_file())
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
            for path in paths:
                bundle.write(path, arcname=str(path.relative_to(root)))
        os.replace(temporary, archive)
        print(json.dumps({"portable_gallery": str(archive), "bytes": archive.stat().st_size}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "run", "render"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--models", default="")
    parser.add_argument("--limit", type=int, default=0, help="per-rank bounded generation for a development smoke only")
    parser.add_argument("--smoke-token-limit", type=int, default=0, help="development smoke only; never use in the full run")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.smoke_token_limit < 0:
        parser.error("smoke limits must be non-negative")
    {"prepare": prepare, "preflight": preflight, "run": run, "render": render}[args.action](args)


if __name__ == "__main__":
    main()
