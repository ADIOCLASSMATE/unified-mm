"""
Cache-backed full ImageNet latent dataset for unified image-flow training.

The cache is the merged tensor produced by the VAE encoder:
    {"latents": [N, image_tokens, latent_dim], "img_ids": [N], ...}

Each item is represented as:
    optional text prompt, <|boi|>, image mask tokens, <|eoi|>, <eos>

The real image latent tokens live only in the parallel `image_latents` tensor.
The X0 stream receives those latent embeddings; visibility is controlled by the
selfless attention sigma/order, while the XT stream remains mask queries.

Text, BOI, and EOS positions can be trained with cross-entropy labels. EOI is
kept as context for image tokens but is ignored by the CE loss.
"""

import json
import random
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class ImageNetFlowCacheDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        tokenizer,
        boi_token_id: int,
        eoi_token_id: int,
        mask_token_id: int,
        eos_token_id: int,
        image_tokens_per_img: int = 256,
        image_latent_dim: int = 16,
        manifest_jsonl: Optional[str] = None,
        prompt_template: str = "",
        prompt_templates: Optional[List[str]] = None,
        prompt_templates_path: Optional[str] = None,
        synset_mapping_path: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_samples: int = -1,
        seed: int = 42,
        latent_hflip_prob: float = 0.0,
    ):
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(self.cache_path)

        obj = torch.load(self.cache_path, map_location="cpu")
        self.latents = obj["latents"]
        self.img_ids = obj.get("img_ids", torch.arange(self.latents.shape[0]))
        if max_samples is not None and max_samples > 0:
            self.latents = self.latents[:max_samples]
            self.img_ids = self.img_ids[:max_samples]

        if self.latents.dim() != 3:
            raise ValueError(f"Expected latents [N,T,C], got {tuple(self.latents.shape)}")
        if self.latents.shape[1:] != (image_tokens_per_img, image_latent_dim):
            raise ValueError(
                f"Expected latent shape (*,{image_tokens_per_img},{image_latent_dim}), "
                f"got {tuple(self.latents.shape)}"
            )

        self.tokenizer = tokenizer
        self.boi_id = int(boi_token_id)
        self.eoi_id = int(eoi_token_id)
        self.mask_id = int(mask_token_id)
        self.eos_id = int(eos_token_id)
        self.image_tokens_per_img = int(image_tokens_per_img)
        self.image_latent_dim = int(image_latent_dim)
        self.prompt_template = prompt_template or ""
        self.prompt_template_groups, self.fallback_templates, self.global_contexts = self._load_prompt_template_groups(
            prompt_template=prompt_template,
            prompt_templates=prompt_templates,
            prompt_templates_path=prompt_templates_path,
        )
        self.sequence_templates = self._all_sequence_templates()
        self.max_seq_length = int(max_seq_length) if max_seq_length else None
        self.seed = int(seed)
        self.epoch = 0
        self.latent_hflip_prob = float(latent_hflip_prob)
        if not 0.0 <= self.latent_hflip_prob <= 1.0:
            raise ValueError(f"latent_hflip_prob must be in [0, 1], got {latent_hflip_prob}")
        self.latent_side = int(self.image_tokens_per_img ** 0.5)
        if self.latent_hflip_prob > 0.0 and self.latent_side * self.latent_side != self.image_tokens_per_img:
            raise ValueError(
                "latent_hflip_prob requires image_tokens_per_img to be a square grid, "
                f"got {self.image_tokens_per_img}."
            )
        self.augmentation_train_size: Optional[int] = None
        self.synsets = self._load_synsets(manifest_jsonl)
        self.synset_names = self._load_synset_names(synset_mapping_path)
        self.prompt_cache = self._build_prompt_cache()
        self.sequence_cache = self._build_sequence_cache()
        self.fixed_length = 1 + self.image_tokens_per_img + 2 if not self.sequence_templates else None
        if self.fixed_length is not None:
            self.fixed_input_ids = torch.tensor(
                [self.boi_id] + [self.mask_id] * self.image_tokens_per_img + [self.eoi_id, self.eos_id],
                dtype=torch.long,
            )
            self.fixed_token_types = torch.tensor(
                [2] + [1] * self.image_tokens_per_img + [2, 2],
                dtype=torch.uint8,
            )
            self.fixed_labels = torch.tensor(
                [self.boi_id]
                + [-100] * self.image_tokens_per_img
                + [-100, self.eos_id],
                dtype=torch.long,
            )

    def _normalize_templates(
        self,
        templates: Sequence[str],
        source: str,
    ) -> List[str]:
        normalized = [str(t).strip() for t in templates if str(t).strip()]
        for template in normalized:
            if "{image}" not in template:
                raise ValueError(
                    f"ImageNetFlowCacheDataset prompt template from {source} must contain '{{image}}': {template!r}"
                )
        return normalized

    def _normalize_contexts(self, contexts: Sequence[str], source: str) -> List[str]:
        normalized = [str(t).strip() for t in contexts if str(t).strip()]
        for context in normalized:
            if "{image}" in context:
                raise ValueError(
                    f"ImageNetFlowCacheDataset prompt context from {source} must not contain '{{image}}': {context!r}"
                )
        return normalized

    def _attach_context(self, template: str, context: str) -> str:
        if not context:
            return template
        prefix, suffix = self._split_rendered_template(template)
        if len(prefix) >= len(suffix):
            prefix = f"{prefix} {context}".strip()
        else:
            suffix = f"{context} {suffix}".strip()
        return f"{prefix} {{image}} {suffix}".strip()

    def _expand_templates_with_contexts(self, templates: List[str], contexts: List[str]) -> List[str]:
        if not contexts:
            return templates
        expanded = []
        for template in templates:
            for context in contexts:
                expanded.append(self._attach_context(template, context))
        return self._dedupe_templates(expanded)

    def _dedupe_templates(self, templates: Sequence[str]) -> List[str]:
        seen = set()
        unique_templates = []
        for template in templates:
            if template in seen:
                continue
            seen.add(template)
            unique_templates.append(template)
        return unique_templates

    def _load_prompt_template_groups(
        self,
        prompt_template: str,
        prompt_templates: Optional[List[str]],
        prompt_templates_path: Optional[str],
    ) -> Tuple[List[Dict[str, object]], List[str], List[str]]:
        groups: List[Dict[str, object]] = []
        fallback_templates: List[str] = []
        global_contexts: List[str] = []

        if prompt_templates_path:
            path = Path(prompt_templates_path)
            if not path.exists():
                raise FileNotFoundError(path)
            cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            global_contexts = self._normalize_contexts(
                cfg.get("global_contexts", []),
                str(path),
            )
            fallback_templates = self._normalize_templates(
                cfg.get("fallback_templates", []),
                str(path),
            )
            for group in cfg.get("groups", []):
                templates = self._normalize_templates(group.get("templates", []), f"{path}:{group.get('name', '')}")
                if not templates:
                    continue
                groups.append(
                    {
                        "name": str(group.get("name", "")),
                        "keywords": [str(k).lower() for k in group.get("keywords", [])],
                        "synsets": [str(s) for s in group.get("synsets", [])],
                        "synset_ranges": [
                            (str(r[0]), str(r[1])) for r in group.get("synset_ranges", [])
                        ],
                        "templates": templates,
                    }
                )

        if prompt_templates:
            fallback_templates = self._normalize_templates(prompt_templates, "dataset.params.prompt_templates")
        elif prompt_template:
            fallback_templates = self._normalize_templates(
                [f"{prompt_template.strip()} {{image}}"],
                "dataset.params.prompt_template",
            )

        fallback_templates = self._expand_templates_with_contexts(fallback_templates, global_contexts)
        for group in groups:
            group["templates"] = self._expand_templates_with_contexts(group["templates"], global_contexts)

        return groups, fallback_templates, global_contexts

    def _all_sequence_templates(self) -> List[str]:
        templates = list(self.fallback_templates)
        for group in self.prompt_template_groups:
            templates.extend(group["templates"])
        return self._dedupe_templates(templates)

    def _load_synsets(self, manifest_jsonl: Optional[str]) -> Dict[int, str]:
        if not manifest_jsonl or not self.sequence_templates:
            return {}
        path = Path(manifest_jsonl)
        if not path.exists():
            raise FileNotFoundError(path)
        synsets: Dict[int, str] = {}
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                synsets[int(row["img_id"])] = str(row.get("synset", ""))
        return synsets

    def _load_synset_names(self, synset_mapping_path: Optional[str]) -> Dict[str, Tuple[str, str]]:
        if not synset_mapping_path:
            return {}
        path = Path(synset_mapping_path)
        if not path.exists():
            raise FileNotFoundError(path)
        names: Dict[str, Tuple[str, str]] = {}
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                synset, _, raw_names = line.partition(" ")
                class_names = raw_names.strip()
                class_name = class_names.split(",", 1)[0].strip() if class_names else synset
                names[synset] = (class_name, class_names or class_name)
        return names

    def _template_pool_for_synset(self, synset: str) -> List[str]:
        group = self._template_group_for_synset(synset)
        if group is not None:
            return group["templates"]
        return self.fallback_templates

    def prompt_group_for_synset(self, synset: str) -> str:
        group = self._template_group_for_synset(synset)
        return str(group["name"]) if group is not None else "fallback"

    def _template_group_for_synset(self, synset: str) -> Optional[Dict[str, object]]:
        class_name, class_names = self.synset_names.get(synset, (synset, synset))
        haystack = f"{synset} {class_name} {class_names}".lower()
        synset_num = self._synset_number(synset)

        for group in self.prompt_template_groups:
            if synset in group["synsets"]:
                return group

        for group in self.prompt_template_groups:
            if self._is_catchall_group(group):
                continue
            for start, end in group["synset_ranges"]:
                start_num = self._synset_number(start)
                end_num = self._synset_number(end)
                if start_num is not None and end_num is not None and synset_num is not None:
                    if start_num <= synset_num <= end_num:
                        return group

        for group in self.prompt_template_groups:
            if self._is_catchall_group(group):
                continue
            if any(self._keyword_matches(haystack, keyword) for keyword in group["keywords"]):
                return group

        for group in self.prompt_template_groups:
            if not self._is_catchall_group(group):
                continue
            for start, end in group["synset_ranges"]:
                start_num = self._synset_number(start)
                end_num = self._synset_number(end)
                if start_num is not None and end_num is not None and synset_num is not None:
                    if start_num <= synset_num <= end_num:
                        return group

        for group in self.prompt_template_groups:
            if not self._is_catchall_group(group):
                continue
            if any(self._keyword_matches(haystack, keyword) for keyword in group["keywords"]):
                return group

        return None

    def _is_catchall_group(self, group: Dict[str, object]) -> bool:
        return str(group["name"]).startswith("other_")

    def _keyword_matches(self, haystack: str, keyword: str) -> bool:
        haystack_norm = f" {re.sub(r'[^a-z0-9]+', ' ', haystack.lower()).strip()} "
        keyword_norm = re.sub(r"[^a-z0-9]+", " ", keyword.lower()).strip()
        return bool(keyword_norm) and f" {keyword_norm} " in haystack_norm

    def _synset_number(self, synset: str) -> Optional[int]:
        if len(synset) < 2 or not synset[1:].isdigit():
            return None
        return int(synset[1:])

    def _render_template(self, template: str, img_id: int, synset: str) -> str:
        class_name, class_names = self.synset_names.get(synset, (synset, synset))
        return template.format(
            img_id=int(img_id),
            synset=synset,
            class_name=class_name,
            class_names=class_names,
            image="{image}",
        )

    def _build_prompt_cache(self) -> Dict[str, torch.Tensor]:
        if not self.sequence_templates:
            return {}

        prompt_cache: Dict[str, torch.Tensor] = {}
        if self.synsets:
            unique_synsets = sorted(set(self.synsets.values()))
            for synset in unique_synsets:
                for template in self._template_pool_for_synset(synset):
                    rendered = self._render_template(template, 0, synset)
                    for text in rendered.split("{image}"):
                        text = text.strip()
                        if text and text not in prompt_cache:
                            prompt_cache[text] = torch.tensor(
                                self.tokenizer.encode(text, add_special_tokens=False),
                                dtype=torch.long,
                            )
        return prompt_cache

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_augmentation_train_size(self, train_size: int) -> None:
        self.augmentation_train_size = int(train_size)

    def _augment_latents(self, latents: torch.Tensor, idx: int) -> torch.Tensor:
        if self.latent_hflip_prob <= 0.0:
            return latents
        if self.augmentation_train_size is not None and int(idx) >= self.augmentation_train_size:
            return latents
        rng = random.Random(self.seed + self.epoch * 1_000_003 + int(idx) * 9_176 + 13_579)
        if rng.random() >= self.latent_hflip_prob:
            return latents
        return latents.view(self.latent_side, self.latent_side, self.image_latent_dim).flip(1).reshape_as(latents)

    def _text_ids(self, text: str) -> torch.Tensor:
        text = text.strip()
        if not text:
            return torch.empty(0, dtype=torch.long)
        cached = self.prompt_cache.get(text)
        if cached is None:
            cached = torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
            self.prompt_cache[text] = cached
        return cached

    def _make_sequence_tensors(self, prefix_ids: torch.Tensor, suffix_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        prefix_len = int(prefix_ids.numel())
        suffix_len = int(suffix_ids.numel())
        input_ids = torch.cat(
            [
                prefix_ids,
                torch.tensor(
                    [self.boi_id]
                    + [self.mask_id] * self.image_tokens_per_img
                    + [self.eoi_id],
                    dtype=torch.long,
                ),
                suffix_ids,
                torch.tensor([self.eos_id], dtype=torch.long),
            ]
        )
        token_types = torch.cat(
            [
                torch.zeros(prefix_len, dtype=torch.uint8),
                torch.tensor([2] + [1] * self.image_tokens_per_img + [2], dtype=torch.uint8),
                torch.zeros(suffix_len, dtype=torch.uint8),
                torch.tensor([2], dtype=torch.uint8),
            ]
        )
        labels = torch.cat(
            [
                prefix_ids,
                torch.tensor(
                    [self.boi_id]
                    + [-100] * self.image_tokens_per_img
                    + [-100],
                    dtype=torch.long,
                ),
                suffix_ids,
                torch.tensor([self.eos_id], dtype=torch.long),
            ]
        )
        return {
            "input_ids": input_ids,
            "token_types": token_types,
            "labels": labels,
            "prompt_len": torch.tensor(prefix_len, dtype=torch.long),
            "suffix_len": torch.tensor(suffix_len, dtype=torch.long),
            "image_start": torch.tensor(prefix_len + 1, dtype=torch.long),
        }

    def _build_sequence_cache(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if not self.sequence_templates:
            return {}
        sequence_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.synsets:
            for synset in sorted(set(self.synsets.values())):
                for template in self._template_pool_for_synset(synset):
                    rendered = self._render_template(template, 0, synset)
                    prefix_text, suffix_text = self._split_rendered_template(rendered)
                    sequence_cache[rendered] = self._make_sequence_tensors(
                        self._text_ids(prefix_text),
                        self._text_ids(suffix_text),
                    )
        return sequence_cache

    def _split_rendered_template(self, rendered: str) -> Tuple[str, str]:
        parts = rendered.split("{image}")
        if len(parts) != 2:
            raise ValueError(f"Rendered prompt template must contain exactly one '{{image}}': {rendered!r}")
        return parts[0].strip(), parts[1].strip()

    def _template_for_sample(self, idx: int, synset: str) -> str:
        templates = self._template_pool_for_synset(synset)
        if not templates:
            raise ValueError(
                "No prompt templates available. Provide dataset.params.prompt_templates, "
                "dataset.params.prompt_template, or dataset.params.prompt_templates_path."
            )
        template_idx = (idx * 1103515245 + self.epoch * 1_000_003 + self.seed) % len(templates)
        return templates[template_idx]

    def _sequence_tensors(self, img_id: int, template: str) -> Dict[str, torch.Tensor]:
        synset = self.synsets.get(int(img_id), "")
        rendered = self._render_template(template, int(img_id), synset)
        cached = self.sequence_cache.get(rendered)
        if cached is None:
            prefix_text, suffix_text = self._split_rendered_template(rendered)
            cached = self._make_sequence_tensors(
                self._text_ids(prefix_text),
                self._text_ids(suffix_text),
            )
            self.sequence_cache[rendered] = cached
        return cached

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        latents = self._augment_latents(self.latents[idx], int(idx))
        if self.fixed_length is not None:
            return {
                "input_ids": self.fixed_input_ids,
                "token_types": self.fixed_token_types,
                "labels": self.fixed_labels,
                "image_latents": latents,
                "prompt_len": torch.tensor(0, dtype=torch.long),
                "suffix_len": torch.tensor(0, dtype=torch.long),
                "image_start": torch.tensor(1, dtype=torch.long),
        }

        img_id = int(self.img_ids[idx].item())
        synset = self.synsets.get(int(img_id), "")
        sequence = self._sequence_tensors(img_id, self._template_for_sample(idx, synset))
        length = int(sequence["input_ids"].numel())
        if self.max_seq_length is not None and length > self.max_seq_length:
            raise ValueError(
                f"ImageNetFlowCacheDataset item length {length} exceeds max_seq_length={self.max_seq_length}. "
                "Use a shorter prompt_template or increase dataset.params.max_seq_length."
            )

        return {
            "input_ids": sequence["input_ids"],
            "token_types": sequence["token_types"],
            "labels": sequence["labels"],
            "image_latents": latents,
            "prompt_len": sequence["prompt_len"],
            "suffix_len": sequence["suffix_len"],
            "image_start": sequence["image_start"],
        }


def collate_imagenet_flow_cache(
    batch: List[Dict[str, torch.Tensor]],
    pad_to_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    first = batch[0]
    if first["image_start"].item() == 1 and first["input_ids"].shape[0] == first["image_latents"].shape[0] + 3:
        bsz = len(batch)
        image_tokens = first["image_latents"].shape[0]
        latent_dim = first["image_latents"].shape[-1]
        latent_dtype = first["image_latents"].dtype
        seq_len = image_tokens + 3

        input_ids = first["input_ids"].unsqueeze(0).expand(bsz, seq_len).clone()
        token_types = first["token_types"].unsqueeze(0).expand(bsz, seq_len).clone()
        labels = first["labels"].unsqueeze(0).expand(bsz, seq_len).clone()
        image_latents = torch.zeros(bsz, seq_len, latent_dim, dtype=latent_dtype)
        image_latents[:, 1 : 1 + image_tokens] = torch.stack(
            [item["image_latents"] for item in batch],
            dim=0,
        )

        sigma = torch.empty(bsz, seq_len, dtype=torch.long)
        sigma[:, 0] = 0
        sigma[:, 1 + image_tokens] = 1
        sigma[:, 2 + image_tokens] = image_tokens + 2
        order = torch.rand(bsz, image_tokens).argsort(dim=-1)
        sigma[:, 1 : 1 + image_tokens] = 2 + order

        return {
            "input_ids": input_ids,
            "token_types": token_types,
            "sigma": sigma,
            "labels": labels,
            "image_latents": image_latents,
            "pack_stats": torch.tensor(
                [bsz * seq_len, bsz * image_tokens, 0, seq_len],
                dtype=torch.long,
            ),
        }

    batch_max_len = max(item["input_ids"].shape[0] for item in batch)
    max_len = pad_to_length or batch_max_len
    if pad_to_multiple_of and max_len % pad_to_multiple_of:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    if batch_max_len > max_len:
        raise ValueError(f"Batch max length {batch_max_len} exceeds pad_to_length={max_len}")
    bsz = len(batch)
    latent_dim = batch[0]["image_latents"].shape[-1]
    latent_dtype = batch[0]["image_latents"].dtype
    image_tokens = batch[0]["image_latents"].shape[0]

    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    token_types = torch.full((bsz, max_len), 3, dtype=torch.uint8)
    sigma = torch.full((bsz, max_len), max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    image_latents = torch.zeros(bsz, max_len, latent_dim, dtype=latent_dtype)
    order = torch.rand(bsz, image_tokens).argsort(dim=-1)

    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        prompt_len = int(item["prompt_len"].item())
        suffix_len = int(item["suffix_len"].item())
        image_start = int(item["image_start"].item())
        eoi_pos = image_start + image_tokens
        suffix_start = eoi_pos + 1
        eos_pos = length - 1

        input_ids[i, :length] = item["input_ids"]
        token_types[i, :length] = item["token_types"]
        labels[i, :length] = item["labels"]
        image_latents[i, image_start:eoi_pos] = item["image_latents"]

        if prompt_len:
            sigma[i, :prompt_len] = torch.arange(prompt_len, dtype=torch.long)
        sigma[i, prompt_len] = prompt_len
        sigma[i, eoi_pos] = prompt_len + 1
        sigma[i, image_start:eoi_pos] = prompt_len + 2 + order[i]
        suffix_sigma_start = prompt_len + image_tokens + 2
        if suffix_len:
            sigma[i, suffix_start:eos_pos] = suffix_sigma_start + torch.arange(suffix_len, dtype=torch.long)
        sigma[i, eos_pos] = suffix_sigma_start + suffix_len

    valid_tokens = (token_types != 3).sum()
    image_tokens = (token_types == 1).sum()
    padding_tokens = (token_types == 3).sum()
    return {
        "input_ids": input_ids,
        "token_types": token_types,
        "sigma": sigma,
        "labels": labels,
        "image_latents": image_latents,
        "pack_stats": torch.tensor(
            [valid_tokens, image_tokens, padding_tokens, max_len],
            dtype=torch.long,
        ),
    }


def build_imagenet_flow_cache_dataloaders(config, tokenizer):
    params = config.dataset.params
    dataset = ImageNetFlowCacheDataset(
        cache_path=params.cache_path,
        tokenizer=tokenizer,
        boi_token_id=config.model.boi_token_id,
        eoi_token_id=config.model.eoi_token_id,
        mask_token_id=config.model.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        image_tokens_per_img=params.get("image_tokens_per_img", config.model.image_tokens_per_img),
        image_latent_dim=params.get("image_latent_dim", config.model.image_latent_dim),
        manifest_jsonl=params.get("manifest_jsonl", None),
        prompt_template=params.get("prompt_template", ""),
        prompt_templates=params.get("prompt_templates", None),
        prompt_templates_path=params.get("prompt_templates_path", None),
        synset_mapping_path=params.get("synset_mapping_path", None),
        max_seq_length=params.get("max_seq_length", config.dataset.preprocessing.max_seq_length),
        max_samples=params.get("max_samples", -1),
        seed=config.training.seed,
        latent_hflip_prob=params.get("latent_hflip_prob", 0.0),
    )

    val_ratio = params.get("val_ratio", 0.001)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(0, len(dataset) - val_size)
    dataset.set_augmentation_train_size(train_size)
    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))

    pad_to_length = params.get("pad_to_length", None)
    if params.get("pad_to_max_length", False):
        pad_to_length = params.get("max_seq_length", config.dataset.preprocessing.max_seq_length)
    pad_to_multiple_of = params.get("pad_to_multiple_of", None)

    collate_fn = partial(
        collate_imagenet_flow_cache,
        pad_to_length=pad_to_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=config.training.dataloader_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=config.training.dataloader_workers > 0,
    )
    return train_loader, val_loader
