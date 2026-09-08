"""Frozen, offline HF encoder adapters with explicit layer/readout contracts."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from torchvision import transforms

from scripts.prepare_geometry_v5_assets import MODELS


def verify_parameters(model, path):
    """Compare every loaded parameter to source, allowing only HF base prefixes."""
    checked, aliases = 0, {}
    with safe_open(str(Path(path) / "model.safetensors"), framework="pt") as source:
        names = set(source.keys())
        for name, parameter in model.named_parameters():
            options = [name, "model." + name, "vit." + name]
            matching = [key for key in options if key in names]
            if not matching:
                raise AssertionError(f"Parameter lacks source tensor: {name}")
            expected = source.get_tensor(matching[0]).to(parameter.dtype)
            actual = parameter.detach().cpu()
            if not torch.equal(actual, expected):
                raise AssertionError(f"Loaded parameter differs from source: {name}")
            checked += 1
            aliases[name] = matching[0]
    return {
        "parameters_verified": checked,
        "parameter_elements": sum(p.numel() for p in model.parameters()),
        "source_aliases": aliases,
    }


class LayerCollector:
    def __init__(self, layers, storage_dtype=torch.bfloat16):
        self.layers = layers
        self.storage_dtype = storage_dtype
        self.names = [str(i) for i in range(len(layers) + 1)] + ["final_norm"]
        self.handles = [
            layers[0].register_forward_pre_hook(self.before, with_kwargs=True)
        ]
        self.handles.extend(layer.register_forward_hook(self.after) for layer in layers)

    def begin(self, content_mask, content_last, native_index=None):
        self.mask, self.last, self.native = content_mask, content_last, native_index
        self.values = []

    def collect(self, value):
        indices = torch.arange(value.shape[0], device=value.device)
        assert value.shape[:2] == self.mask.shape
        pooled = (value.float() * self.mask.unsqueeze(-1)).sum(1) / self.mask.sum(
            1, keepdim=True
        )
        values = [pooled, value[indices, self.last].float()]
        if self.native is not None:
            values.append(value[indices, self.native].float())
        self.values.append(torch.stack(values, dim=1))

    def before(self, module, args, kwargs):
        value = args[0] if args else kwargs["hidden_states"]
        self.collect(value)

    def after(self, module, args, output):
        self.collect(output[0] if isinstance(output, tuple) else output)

    def finish(self, final_hidden):
        self.collect(final_hidden)
        assert len(self.values) == len(self.names)
        value = torch.stack(self.values, dim=1)
        assert bool(torch.isfinite(value).all())
        return value.to(device="cpu", dtype=self.storage_dtype)


class PixelLoader:
    """Use exactly the B-cache 256px content crop, then model-required resolution."""

    def __init__(self):
        self.crop = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(256),
            ]
        )
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def one(self, path):
        with Image.open(path) as value:
            return self.crop(value.convert("RGB"))

    def batch(self, rows, processor, resolution):
        values = list(self.pool.map(self.one, [r["source_path"] for r in rows]))
        values = [
            im.resize((resolution, resolution), Image.Resampling.BICUBIC)
            for im in values
        ]
        return processor(
            images=values, do_resize=False, do_center_crop=False, return_tensors="pt"
        )["pixel_values"]


class EncoderAdapter:
    def __init__(self, name, device, dtype=None):
        from transformers import (
            AutoImageProcessor,
            AutoModel,
            AutoTokenizer,
            SiglipModel,
        )

        self.name, self.device = name, device
        self.dtype = (
            dtype
            if dtype is not None
            else torch.float32
            if name == "siglip"
            else torch.bfloat16
        )
        self.pixel_loader = PixelLoader()
        self.models = {
            "qwen_text": "Qwen--Qwen3-0.6B-Base",
            "dinov2": "facebook--dinov2-base",
            "mae": "facebook--vit-mae-base",
            "siglip": "google--siglip-so400m-patch14-384",
        }
        self.path = MODELS / self.models[name]
        cls = SiglipModel if name == "siglip" else AutoModel
        self.model = cls.from_pretrained(
            self.path,
            local_files_only=True,
            dtype=self.dtype,
            attn_implementation="eager",
        )
        self.model.requires_grad_(False).eval().to(device)
        self.collectors = {}
        if name in ("qwen_text", "siglip"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.path, local_files_only=True
            )
        if name != "qwen_text":
            self.processor = AutoImageProcessor.from_pretrained(
                self.path, local_files_only=True
            )
        if name == "qwen_text":
            self.tokenizer.padding_side = "right"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.collectors["text"] = LayerCollector(self.model.layers)
        elif name == "siglip":
            self.collectors["text"] = LayerCollector(
                self.model.text_model.encoder.layers
            )
            self.collectors["image"] = LayerCollector(
                self.model.vision_model.encoder.layers
            )
        else:
            self.collectors["image"] = LayerCollector(self.model.encoder.layer)
            if name == "mae":
                self.model.config.mask_ratio = 0.0
        self.contract = {
            "schema": "geometry_v5_hf_encoder_contract_2"
            if name == "siglip"
            else "geometry_v5_hf_encoder_contract_1",
            "name": name,
            "source_path": str(self.path),
            "precision": str(self.dtype).removeprefix("torch."),
            "attention": "eager",
            "layers": {key: c.names for key, c in self.collectors.items()},
            "pools": self.pools(),
            "pixel_policy": "V4 Resize(shorter-edge=256,BICUBIC)+CenterCrop256; resize this exact RGB content to native encoder resolution; processor normalization only",
            "resolution": None
            if name == "qwen_text"
            else 384
            if name == "siglip"
            else 224,
            "text_policy": "raw observed text; Qwen append EOS and exclude it from content; SigLIP native 64-token padded input and explicit truncation accounting",
            "mae_mask_ratio": 0.0 if name == "mae" else None,
            "mae_patch_order": "deterministic ascending noise ranks, all patches visible"
            if name == "mae"
            else None,
            "paired_target_input": False,
            "training_updates": 0,
            "native_endpoint": "official pooler_output"
            if name == "siglip"
            else "final CLS"
            if name in ("dinov2", "mae")
            else "final appended EOS",
        }

    def pools(self):
        if self.name == "siglip":
            return {
                "image": ["content_mean", "content_last"],
                "text": ["content_mean", "content_last"],
            }
        if self.name == "qwen_text":
            return {"text": ["content_mean", "content_last", "eos_boundary"]}
        return {"image": ["content_mean", "content_last", "cls"]}

    @torch.inference_mode()
    def forward(self, rows, modality):
        if modality == "image":
            return self.forward_image(rows)
        return self.forward_text(rows)

    def forward_image(self, rows):
        size = self.contract["resolution"]
        pixels = self.pixel_loader.batch(rows, self.processor, size).to(
            self.device, self.dtype
        )
        cls = self.name != "siglip"
        patch = 14 if self.name in ("siglip", "dinov2") else 16
        length = (size // patch) ** 2 + int(cls)
        mask = torch.ones(len(rows), length, dtype=torch.bool, device=self.device)
        if cls:
            mask[:, 0] = False
        last = torch.full(
            (len(rows),), length - 1, dtype=torch.long, device=self.device
        )
        native = torch.zeros_like(last) if cls else None
        collector = self.collectors["image"]
        collector.begin(mask, last, native)
        kwargs = {"pixel_values": pixels, "return_dict": True}
        if self.name == "mae":
            kwargs["noise"] = (
                torch.arange(length - 1, device=self.device)
                .float()
                .expand(len(rows), -1)
            )
        output = (
            self.model.vision_model(**kwargs)
            if self.name == "siglip"
            else self.model(**kwargs)
        )
        features = collector.finish(output.last_hidden_state)
        endpoint = (
            output.pooler_output
            if self.name == "siglip"
            else output.last_hidden_state[:, 0]
        )
        return (
            features,
            endpoint.detach().to("cpu", dtype=torch.bfloat16),
            {"truncated_texts": 0},
        )

    def forward_text(self, rows):
        texts = [r["text"] for r in rows]
        if self.name == "siglip":
            raw = self.tokenizer(texts, add_special_tokens=True, truncation=False)[
                "input_ids"
            ]
            truncated = sum(len(ids) > 64 for ids in raw)
            inputs = self.tokenizer(
                texts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            specials = inputs.pop("special_tokens_mask").bool()
            mask = ~specials & inputs["input_ids"].ne(self.tokenizer.pad_token_id)
            # Preserve the official padding-visibility policy; don't invent a new mask.
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            mask = mask.to(self.device)
            last = (
                (torch.arange(mask.shape[1], device=self.device)[None] * mask)
                .max(1)
                .values
            )
            assert bool(mask.any(1).all())
            collector = self.collectors["text"]
            collector.begin(mask, last)
            output = self.model.text_model(**inputs, return_dict=True)
            endpoint = output.pooler_output
        else:
            sequences = [
                self.tokenizer.encode(text, add_special_tokens=False) for text in texts
            ]
            assert all(sequences)
            width = max(map(len, sequences)) + 1
            ids = torch.full(
                (len(rows), width), self.tokenizer.eos_token_id, dtype=torch.long
            )
            attention = torch.zeros_like(ids)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            last = torch.tensor([len(seq) - 1 for seq in sequences], dtype=torch.long)
            native = last + 1
            for i, seq in enumerate(sequences):
                ids[i, : len(seq)] = torch.tensor(seq)
                attention[i, : len(seq) + 1] = 1
                mask[i, : len(seq)] = True
            collector = self.collectors["text"]
            collector.begin(
                mask.to(self.device), last.to(self.device), native.to(self.device)
            )
            output = self.model(
                input_ids=ids.to(self.device),
                attention_mask=attention.to(self.device),
                use_cache=False,
                return_dict=True,
            )
            endpoint = output.last_hidden_state[
                torch.arange(len(rows), device=self.device), native.to(self.device)
            ]
            truncated = 0
        features = collector.finish(output.last_hidden_state)
        return (
            features,
            endpoint.detach().to("cpu", dtype=torch.bfloat16),
            {"truncated_texts": truncated},
        )
