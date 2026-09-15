"""Fixed-prompt EMA images on training validation, sharded across ranks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from utils.atomic_io import atomic_write_text
from utils.image_generation_io import T2I_PREFIX, build_t2i_item, decode_latents, load_vae, noise_for, save_png
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.training_downstream_validation import (
    _coverage_status, _distributed, _local_phase, _reduce, _synchronize, evaluation_state, rank_indices,
)


@dataclass(frozen=True)
class ImageGenerationProfile:
    enabled: bool = False
    samples: int = 16
    seed: int = 42
    prompt_file: str = "configs/protocols/unified_qualitative_prompts_v1.json"
    cfg: float = 3.5
    steps: int = 10
    solver: str = "heun"
    vae_module_root: str = "public/code/mar"
    vae_path: str = "public/vae/mar-kl16/kl16.ckpt"
    vae_scaling_factor: float = 0.2325

    def __post_init__(self):
        if self.samples < 1 or self.seed < 0 or self.steps < 1:
            raise ValueError("invalid validation image count, seed, or steps")
        if self.solver not in {"heun", "euler"} or not math.isfinite(self.cfg) or self.cfg <= 0:
            raise ValueError("validation generation requires Heun/Euler and positive finite CFG")
        if not math.isfinite(self.vae_scaling_factor) or self.vae_scaling_factor <= 0:
            raise ValueError("invalid validation VAE scaling factor")

    @classmethod
    def from_config(cls, config):
        return cls(**dict(config.experiment.get("validation_generation", {})))


def serializable_trace(trace):
    """Keep scalar contracts and actual reveal order, excluding optional tensor diagnostics."""
    result = {key: value for key, value in trace.items()
              if value is None or isinstance(value, (str, bool, int, float))}
    if torch.is_tensor(trace.get("generation_order")):
        result["generation_order"] = trace["generation_order"].detach().cpu().tolist()
    return result


def _write_gallery(directory, rows, *, step, weights, method="Z"):
    from PIL import Image, ImageDraw

    columns = min(4, len(rows))
    cell_width, cell_height = 256, 288
    grid = Image.new("RGB", (columns * cell_width, math.ceil(len(rows) / columns) * cell_height), "white")
    draw = ImageDraw.Draw(grid)
    cards = []
    for index, row in enumerate(rows):
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        with Image.open(directory / row["image"]) as source:
            grid.paste(source.convert("RGB").resize((256, 256)), (x, y))
        draw.text((x + 5, y + 262), f"{index:02d} {row['id']}", fill="black")
        path, prompt = html.escape(row["image"], quote=True), html.escape(row["prompt"])
        cards.append(f'<figure><a href="{path}"><img src="{path}" alt="{prompt}"></a>'
                     f'<figcaption>{index:02d}. {prompt}<br>Seed: {row["noise_seed"]}</figcaption></figure>')
    grid.save(directory / "overview.tmp.png")
    os.replace(directory / "overview.tmp.png", directory / "overview.png")
    atomic_write_text(directory / "index.html",
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(method)} validation images</title>'
        '<style>body{font:16px system-ui;margin:24px}main{display:flex;flex-wrap:wrap}'
        'figure{width:256px;margin:12px}img{width:256px;height:256px}figcaption{line-height:1.4}</style>'
        f'<h1>{html.escape(method)} · step {step} · {html.escape(weights)}</h1>'
        '<p>Fixed prompts and noise seeds. <a href="overview.png">Overview</a> · '
        '<a href="summary.json">Generation settings and traces</a></p><main>' + "".join(cards) + '</main></html>\n')


class TrainingImageGenerator:
    """Keep prompts on CPU; load and release the FP32 VAE only on active ranks."""

    def __init__(self, config):
        self.profile = ImageGenerationProfile.from_config(config)
        self.prompts = ()
        self.prompt_prefix = T2I_PREFIX
        self.pad_to_length = 512
        if self.profile.enabled:
            image = config.dataset.params.image
            self.prompt_prefix = str(image.caption_t2i_prefix)
            self.pad_to_length = int(image.pad_to_length)
            rows = json.loads(Path(self.profile.prompt_file).read_text())["t2i"]
            if self.profile.samples > len(rows):
                raise ValueError("not enough fixed validation prompts")
            self.prompts = tuple(dict(row) for row in rows[:self.profile.samples])
            if len({row["id"] for row in self.prompts}) != len(self.prompts):
                raise ValueError("duplicate validation prompt IDs")
            if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", row["id"]) or not row["prompt"].strip() for row in self.prompts):
                raise ValueError("invalid validation prompt ID or empty prompt")

    def run(self, model, tokenizer, *, device, step, output_dir, ema=None):
        if not self.profile.enabled:
            return None
        profile = self.profile
        joint = getattr(model.config, "architecture_variant", None) == "selfless_joint_dit"
        method = "Z" if joint else "B + S2-single modulation"
        if joint and getattr(model.config, "joint_dit_head_type", "s2") == "b_single_stream":
            method = "Z + B head (single stream)"
        image_order, generation_order = ("joint", "joint") if joint else ("random", "spatial_halton")
        rank, world = (dist.get_rank(), dist.get_world_size()) if _distributed() else (0, 1)
        directory = Path(output_dir) / "validation_generation" / f"step-{step}"
        _synchronize(device)
        started = time.monotonic()
        weights = "ema" if ema is not None else "model"
        summary = dict(schema="training_image_generation_v1", method=method, step=int(step),
            complete=False, weight_source=weights, world_size=world, profile=asdict(profile),
            prompt_prefix=self.prompt_prefix, samples=0, expected_samples=profile.samples,
            overview="overview.png", gallery="index.html", runtime_hashing_enabled=False)
        if ema is not None:
            summary["ema_step"] = ema.global_step
        seen = torch.zeros(profile.samples)
        with _local_phase(device):
            if not joint and not (
                getattr(model.config, "architecture_variant", None) == "selfless_contextual"
                and getattr(model.config, "image_flow_conditioning_mode", None) == "s2_input"
            ):
                raise ValueError("validation_generation requires Z or B + S2 modulation")
            if rank == 0:
                atomic_write_text(directory / "summary.json", json.dumps(summary, indent=2) + "\n")
                print(json.dumps({"event": "training_image_generation_start", "step": step,
                                  "samples": profile.samples, "weights": weights}), flush=True)
        # All ranks enter EMA collectives, including ranks with zero image work.
        # A local failure is agreed collectively before restoring training weights.
        with evaluation_state(model, device, ema):
            with _local_phase(device):
                local = list(rank_indices(profile.samples, rank, world))
                vae = None
                try:
                    if local:
                        vae_config = SimpleNamespace(experiment={
                            "validation_vae_module_root": profile.vae_module_root,
                            "validation_vae_path": profile.vae_path})
                        # load_vae accepts an OmegaConf experiment with get/attribute access.
                        from omegaconf import OmegaConf
                        vae_config.experiment = OmegaConf.create(vae_config.experiment)
                        vae = load_vae(vae_config, device, "fp32")
                    count, dim = int(model.config.image_tokens_per_img), int(model.config.image_latent_dim)
                    for index in local:
                        prompt = self.prompts[index]
                        item = build_t2i_item(tokenizer, model, prompt["prompt"], index, profile.seed, image_order,
                                              prompt_prefix=self.prompt_prefix)
                        if item["input_ids"].numel() > self.pad_to_length:
                            raise ValueError(f"validation prompt exceeds sequence length: {prompt['id']}")
                        batch = collate_imagenet_flow_cache([item], pad_to_length=self.pad_to_length)
                        latents, trace = model.generate("t2i", input_ids=batch["input_ids"].to(device),
                            token_types=batch["token_types"].to(device), sigma=batch["sigma"].to(device),
                            spans=[(0, item["image_start"], item["image_start"] + count)],
                            initial_noise_bank=noise_for(index, profile.seed, count, dim)[None],
                            flow_cfg=profile.cfg, flow_solver=profile.solver, flow_num_steps=profile.steps,
                            flow_temperature=1., flow_cfg_schedule="constant", order_strategy=generation_order,
                            use_cache=not joint, return_trace=True, debug_finite=True)
                        expected_calls = profile.steps * (2 if profile.solver == "heun" else 1)
                        if joint and (trace.get("backbone_calls"), trace.get("flow_head_calls")) != (1, expected_calls):
                            raise RuntimeError("Z validation generation violated its backbone/head call contract")
                        if not joint:
                            if (not trace["backbone_kv_cache_enabled"]
                                or trace["flow_conditioning_mode"] != "s2_input"
                                or trace["flow_solver"] != profile.solver
                                or trace["flow_num_steps"] != profile.steps
                                or trace["flow_content_cache_tokens_committed"] != count - 1):
                                raise RuntimeError("B validation generation violated its conditioning/cache contract")
                            trace = serializable_trace(trace)
                        if tuple(latents.shape) != (1, dim, math.isqrt(count), math.isqrt(count)) or not torch.isfinite(latents).all():
                            raise FloatingPointError("invalid generated validation latents")
                        filename = f"{index:02d}-{prompt['id']}.png"
                        save_png(decode_latents(vae, latents, profile.vae_scaling_factor)[0], directory / filename)
                        row = {**prompt, "index": index, "image": filename, "rank": rank,
                               "serialized_prompt": f"{self.prompt_prefix} {prompt['prompt']}",
                               "noise_seed": profile.seed + 1000003 * index, "trace": trace,
                               "latent_rms": float(latents.float().square().mean().sqrt())}
                        atomic_write_text(directory / f"sample-{index:02d}.json", json.dumps(row, indent=2, allow_nan=False) + "\n")
                        seen[index] = 1
                finally:
                    # Device memory is returned to the allocator before training resumes.
                    del vae
        summary.update(_coverage_status(seen, profile.samples, device))
        with _local_phase(device):
            summary["images"] = [json.loads((directory / f"sample-{i:02d}.json").read_text()) for i in range(profile.samples)]
            if rank == 0:
                _write_gallery(directory, summary["images"], step=step, weights=weights, method=method)
        _synchronize(device)
        summary["wall_seconds"] = float(_reduce([time.monotonic() - started], device, dist.ReduceOp.MAX)[0])
        with _local_phase(device):
            if rank == 0:
                atomic_write_text(directory / "summary.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"event": "training_image_generation_complete", "step": step,
                    "samples": summary["samples"], "wall_seconds": summary["wall_seconds"],
                    "gallery": str(directory / "index.html")}), flush=True)
        return summary
