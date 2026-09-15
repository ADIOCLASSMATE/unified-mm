#!/usr/bin/env python3
"""Evaluate calibrated zero-shot ImageNet-1K classification on all 50K val images.

The 1,000 candidate texts use OpenAI CLIP's curated ImageNet class names and
one frozen template.  The Selfless model itself supplies the length-normalized
conditional token likelihoods; no external CLIP model scores the checkpoint.
The formal score subtracts each class text's log-mean-exp marginal over all
50,000 validation images with a fixed language-prior alpha of one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from torch.nn.attention.flex_attention import create_block_mask
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.evaluation.model_contracts import scoring_contract
from utils.evaluation.multimodal_likelihood import (  # noqa: E402
    CandidateScore,
    LIKELIHOOD_SCORING_CONTRACT,
    LikelihoodExample,
    PosteriorCache,
    atomic_write_text,
    barrier,
    encode_candidate,
    initialize_device,
    score_candidate_requests,
    utc_now,
)
from utils.evaluation.calibration import (  # noqa: E402
    LANGUAGE_PRIOR_ALPHA,
    LANGUAGE_PRIOR_ESTIMATOR,
    language_prior_debiased_scores,
)
from models.modeling_model.image_position_utils import (  # noqa: E402
    build_row_col_position_ids,
)
from models.modeling_model.modeling_selfless_flow import (  # noqa: E402
    SelflessStaticCache,
)
from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
)
from utils.utils import load_model_tokenizer  # noqa: E402


DEFAULT_CONFIG = Path("configs/selfless/unified_baseline_100b_ascend_64npu.yaml")
DEFAULT_MANIFEST = Path("public/datasets/imagenet_full/manifest_val.jsonl")
DEFAULT_CLASSES = Path("public/datasets/imagenet1k_synthetic_v1/t2i/classes.json")
DEFAULT_CLASSNAMES = Path("scripts/assets/imagenet1k_openai_clip_classnames.json")
DEFAULT_CACHE = Path(
    "public/datasets/imagenet_full/vae_posterior_mar_kl16/val_shards"
)
RETRIEVAL_PROMPT = "Describe this image in one detailed caption:"
CLASS_TEXT_TEMPLATE = "a photo of a {class_name}."
CLASSIFICATION_TASK = "imagenet1k_zeroshot_classification"
OPENAI_CLIP_CLASSNAME_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
OPENAI_CLIP_CLASSNAME_NOTEBOOK = "notebooks/Prompt_Engineering_for_ImageNet.ipynb"
CLASS_IDENTITY_CORRECTIONS = (
    (744, "n04008634", "missile", "projectile"),
    (836, "n04355933", "sunglasses", "sunglass"),
)


@dataclass(frozen=True)
class ImageNetRecord:
    index: int
    img_id: int
    source_path: str
    image_id: str
    synset: str
    class_index: int


class CachedTokenizer:
    """Cache the small fixed evaluation candidate pool within each rank."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._encoded: dict[tuple[str, bool], tuple[int, ...]] = {}

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def encode(self, text, add_special_tokens=False):
        key = (str(text), bool(add_special_tokens))
        value = self._encoded.get(key)
        if value is None:
            value = tuple(
                int(token)
                for token in self.tokenizer.encode(
                    str(text), add_special_tokens=bool(add_special_tokens)
                )
            )
            self._encoded[key] = value
        return list(value)

    def __getattr__(self, name: str):
        return getattr(self.tokenizer, name)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_imagenet_records(
    manifest_path: Path,
    classes_path: Path,
) -> list[ImageNetRecord]:
    images = read_jsonl(manifest_path)
    class_payload = json.loads(classes_path.read_text(encoding="utf-8"))
    class_rows = list(class_payload["classes"])
    if len(images) != 50_000:
        raise ValueError("formal ImageNet-val evaluation requires exactly 50,000 rows")
    if len(class_rows) != 1_000:
        raise ValueError("ImageNet class metadata has unexpected cardinality")
    synset_to_class = {
        str(row["synset"]): int(row["class_index"]) for row in class_rows
    }
    if len(synset_to_class) != 1_000:
        raise ValueError("invalid ImageNet class metadata")
    class_to_synset = {
        class_index: synset for synset, class_index in synset_to_class.items()
    }
    for class_index, expected_synset, _, _ in CLASS_IDENTITY_CORRECTIONS:
        if class_to_synset.get(class_index) != expected_synset:
            raise ValueError(
                "ImageNet class metadata does not match the disambiguated "
                f"class-name contract at index {class_index}"
            )

    records: list[ImageNetRecord] = []
    class_counts: dict[int, int] = defaultdict(int)
    for index, image in enumerate(images):
        if (
            int(image["manifest_index"]) != index
            or str(image.get("split")) != "val"
        ):
            raise ValueError(f"ImageNet val alignment mismatch at row {index}")
        synset = str(image["synset"])
        class_index = synset_to_class[synset]
        class_counts[class_index] += 1
        records.append(
            ImageNetRecord(
                index=index,
                img_id=int(image["img_id"]),
                source_path=str(image["source_path"]),
                image_id=str(image["image_id"]),
                synset=synset,
                class_index=class_index,
            )
        )
    if set(class_counts) != set(range(1_000)) or set(class_counts.values()) != {50}:
        raise ValueError("ImageNet val must contain 50 examples for every class")
    if len({record.img_id for record in records}) != len(records):
        raise ValueError("ImageNet val manifest contains duplicate internal image IDs")
    if len({record.image_id for record in records}) != len(records):
        raise ValueError("ImageNet val manifest contains duplicate official image IDs")
    return records


def load_openai_clip_class_names(path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "openai_clip_imagenet1k_classnames_v1":
        raise ValueError("unexpected OpenAI CLIP ImageNet class-name schema")
    expected_provenance = {
        "source_repository": "https://github.com/openai/CLIP",
        "source_commit": OPENAI_CLIP_CLASSNAME_COMMIT,
        "source_notebook": OPENAI_CLIP_CLASSNAME_NOTEBOOK,
        "class_order": "ILSVRC2012 class index 0..999",
    }
    for key, expected in expected_provenance.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"OpenAI CLIP class-name provenance mismatch for {key}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    names = [str(value).strip() for value in payload.get("class_names", [])]
    if len(names) != 1_000 or any(not value for value in names):
        raise ValueError("OpenAI CLIP class-name asset must contain 1,000 names")
    if len(set(names)) != 1_000:
        raise ValueError("OpenAI CLIP class-name asset contains duplicate candidates")
    expected_corrections = [
        {
            "class_index": class_index,
            "synset": synset,
            "official_notebook_value": original,
            "value": replacement,
            "reason": reason,
        }
        for (class_index, synset, original, replacement), reason in zip(
            CLASS_IDENTITY_CORRECTIONS,
            (
                "avoid duplicate with class 657 and preserve WordNet synset identity",
                "avoid duplicate with class 837 and preserve WordNet synset identity",
            ),
        )
    ]
    if payload.get("identity_corrections") != expected_corrections:
        raise ValueError("ImageNet class-name disambiguation contract mismatch")
    for class_index, _, _, replacement in CLASS_IDENTITY_CORRECTIONS:
        if names[class_index] != replacement:
            raise ValueError(
                f"disambiguated class name mismatch at index {class_index}"
            )
    provenance = {
        key: payload[key]
        for key in (
            "schema",
            "source_repository",
            "source_commit",
            "source_notebook",
            "class_order",
            "identity_corrections",
        )
    }
    return names, provenance


def score_text_candidates(
    *,
    model,
    tokenizer,
    cache: PosteriorCache,
    image_id: int,
    item_id: str,
    prompt: str,
    candidates: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    image_sigma_order: str,
    attention_contract: str,
    evaluation_task: str = "imagenet_pretraining_native",
) -> tuple[torch.Tensor, torch.Tensor]:
    example = LikelihoodExample(
        item_index=0,
        item_id=str(item_id),
        task=str(evaluation_task),
        kind="candidate_matrix_row",
        image_id=int(image_id),
        prompt=str(prompt),
        candidates=tuple(str(value) for value in candidates),
        label=None,
        category=None,
        metadata={},
    )
    scores: list[CandidateScore] = []
    for offset in range(0, len(candidates), int(args.request_chunk_size)):
        stop = min(offset + int(args.request_chunk_size), len(candidates))
        requests = [
            encode_candidate(
                tokenizer,
                example,
                candidate_index,
                image_tokens=int(model.config.image_tokens_per_img),
                boi_token_id=int(model.config.boi_token_id),
                eoi_token_id=int(model.config.eoi_token_id),
                image_mask_token_id=int(model.config.image_mask_token_id),
                max_length=int(args.max_length),
                image_sigma_order=image_sigma_order,
                seed=int(args.seed),
            )
            for candidate_index in range(offset, stop)
        ]
        scores.extend(
            score_candidate_requests(
                model,
                requests,
                cache,
                batch_size=int(args.batch_size_per_rank),
                lm_head_chunk_tokens=int(args.lm_head_chunk_tokens),
                attention_contract=attention_contract,
                device=device,
            )
        )
    raw = torch.tensor([value.loglikelihood for value in scores], dtype=torch.float32)
    normalized = torch.tensor(
        [value.normalized_loglikelihood for value in scores], dtype=torch.float32
    )
    return raw, normalized


def cross_sigma_attention_mask(
    *,
    query_sigma: torch.Tensor,
    key_sigma: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
    include_diagonal: bool,
    device: torch.device,
):
    """Build a QxKV Selfless mask for a static prefix cache."""

    if query_sigma.ndim != 2 or key_sigma.ndim != 2:
        raise ValueError("cached sigma tensors must be rank two")
    if query_valid.shape != query_sigma.shape or key_valid.shape != key_sigma.shape:
        raise ValueError("cached sigma/valid shapes must align")
    comparison = (
        key_sigma.unsqueeze(1) <= query_sigma.unsqueeze(-1)
        if include_diagonal
        else key_sigma.unsqueeze(1) < query_sigma.unsqueeze(-1)
    )
    allowed = (
        query_valid.unsqueeze(-1)
        & key_valid.unsqueeze(1)
        & comparison
    )
    if device.type == "npu":
        return (~allowed).unsqueeze(1)

    def mask_mod(batch, head, query_index, key_index):
        del head
        return allowed[batch, query_index, key_index]

    return create_block_mask(
        mask_mod,
        B=int(allowed.shape[0]),
        H=None,
        Q_LEN=int(allowed.shape[1]),
        KV_LEN=int(allowed.shape[2]),
        device=device,
    )


def clone_prefix_cache(
    prefix_cache: SelflessStaticCache,
    *,
    config,
    batch_size: int,
) -> SelflessStaticCache:
    cloned = SelflessStaticCache(config, prefix_cache.get_max_cache_shape())
    for source, destination in zip(prefix_cache.layers, cloned.layers):
        if not source.is_initialized:
            raise RuntimeError("cannot clone an uninitialized prefix cache")
        destination.keys = source.keys.expand(batch_size, -1, -1, -1).clone()
        destination.values = source.values.expand(batch_size, -1, -1, -1).clone()
        destination.is_initialized = True
    for hidden_attr in ("_single_stream_text_ar_last_hidden",):
        cached_text_source = getattr(prefix_cache, hidden_attr, None)
        if cached_text_source is not None:
            setattr(
                cloned,
                hidden_attr,
                cached_text_source.expand(batch_size, -1).clone(),
            )
    if hasattr(prefix_cache, "semantic_state"):
        from models.modeling_model.modeling_selfless_siglip import copy_semantic_cache
        copy_semantic_cache(prefix_cache, cloned, repeats=batch_size)
    return cloned


@torch.inference_mode()
def score_text_candidates_cached_prefix(
    *,
    model,
    tokenizer,
    cache: PosteriorCache,
    image_id: int,
    item_id: str,
    prompt: str,
    candidates: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    image_sigma_order: str,
    attention_contract: str,
    evaluation_task: str = "imagenet_pretraining_native",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score many suffixes while computing the shared image prefix once."""

    example = LikelihoodExample(
        item_index=0,
        item_id=str(item_id),
        task=str(evaluation_task),
        kind="candidate_matrix_row",
        image_id=int(image_id),
        prompt=str(prompt),
        candidates=tuple(str(value) for value in candidates),
        label=None,
        category=None,
        metadata={},
    )
    requests = [
        encode_candidate(
            tokenizer,
            example,
            candidate_index,
            image_tokens=int(model.config.image_tokens_per_img),
            boi_token_id=int(model.config.boi_token_id),
            eoi_token_id=int(model.config.eoi_token_id),
            image_mask_token_id=int(model.config.image_mask_token_id),
            max_length=int(args.max_length),
            image_sigma_order=image_sigma_order,
            seed=int(args.seed),
        )
        for candidate_index in range(len(candidates))
    ]
    if not requests:
        return torch.empty(0), torch.empty(0)
    prefix_length = int(requests[0].target_start)
    prefix_identity = (
        requests[0].input_ids[:prefix_length],
        requests[0].token_types[:prefix_length],
        requests[0].sigma[:prefix_length],
        requests[0].image_start,
    )
    if any(
        (
            request.input_ids[: request.target_start],
            request.token_types[: request.target_start],
            request.sigma[: request.target_start],
            request.image_start,
        )
        != prefix_identity
        for request in requests[1:]
    ):
        # Candidate-dependent prompt truncation destroys the shared prefix.
        return score_text_candidates(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            image_id=image_id,
            item_id=item_id,
            prompt=prompt,
            candidates=candidates,
            args=args,
            device=device,
            image_sigma_order=image_sigma_order,
            attention_contract=attention_contract,
            evaluation_task=evaluation_task,
        )

    max_cache_length = max(len(request.input_ids) for request in requests)
    prefix_input_ids = torch.tensor(
        [requests[0].input_ids[:prefix_length]], device=device, dtype=torch.long
    )
    prefix_token_types = torch.tensor(
        [requests[0].token_types[:prefix_length]],
        device=device,
        dtype=torch.uint8,
    )
    prefix_sigma = torch.tensor(
        [requests[0].sigma[:prefix_length]], device=device, dtype=torch.long
    )
    prefix_valid = torch.ones_like(prefix_sigma, dtype=torch.bool)
    prefix_key_sigma = torch.zeros(
        (1, max_cache_length), device=device, dtype=torch.long
    )
    prefix_key_sigma[:, :prefix_length] = prefix_sigma
    prefix_key_valid = torch.zeros(
        (1, max_cache_length), device=device, dtype=torch.bool
    )
    prefix_key_valid[:, :prefix_length] = True
    prefix_query_mask = cross_sigma_attention_mask(
        query_sigma=prefix_sigma,
        key_sigma=prefix_key_sigma,
        query_valid=prefix_valid,
        key_valid=prefix_key_valid,
        include_diagonal=False,
        device=device,
    )
    prefix_content_mask = (
        cross_sigma_attention_mask(
            query_sigma=prefix_sigma,
            key_sigma=prefix_key_sigma,
            query_valid=prefix_valid,
            key_valid=prefix_key_valid,
            include_diagonal=True,
            device=device,
        )
        if attention_contract == "xlnet_content_diagonal"
        else None
    )
    latent_dim = int(model.config.image_latent_dim)
    image_tokens = int(model.config.image_tokens_per_img)
    image_start = int(requests[0].image_start)
    image_end = image_start + image_tokens
    prefix_latents = torch.zeros(
        (1, prefix_length, latent_dim),
        device=device,
        dtype=cache.sample(image_id).dtype,
    )
    prefix_latents[:, image_start:image_end] = cache.sample(image_id).to(device)
    prefix_latent_mask = torch.zeros(
        (1, prefix_length), device=device, dtype=torch.bool
    )
    prefix_latent_mask[:, image_start:image_end] = True
    prefix_cache = SelflessStaticCache(model.config, max_cache_length)
    model.model(
        X0_input_ids=prefix_input_ids,
        attention_mask=prefix_query_mask,
        content_attention_mask=prefix_content_mask,
        position_ids=build_row_col_position_ids(
            prefix_token_types, image_tokens
        ),
        past_key_values=prefix_cache,
        use_cache=True,
        cache_position=torch.arange(prefix_length, device=device).unsqueeze(0),
        calculate_likelihood=False,
        _text_ar_mode=True,
        token_types=prefix_token_types,
        image_latents=prefix_latents,
        image_latent_mask=prefix_latent_mask,
        image_span_table=torch.tensor(
            [[0, 0, image_start, image_end, int(image_id)]],
            device=device,
            dtype=torch.long,
        ),
    )

    grouped: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for request_index, request in enumerate(requests):
        grouped[len(request.input_ids) - prefix_length].append(
            (request_index, request)
        )
    ordered_scores: list[CandidateScore | None] = [None] * len(requests)
    for candidate_length, group in sorted(grouped.items()):
        for offset in range(0, len(group), int(args.batch_size_per_rank)):
            batch_pairs = group[offset : offset + int(args.batch_size_per_rank)]
            batch_size = len(batch_pairs)
            batch_requests = [request for _, request in batch_pairs]
            input_ids = torch.tensor(
                [request.input_ids[prefix_length:] for request in batch_requests],
                device=device,
                dtype=torch.long,
            )
            token_types = torch.zeros(
                (batch_size, candidate_length), device=device, dtype=torch.uint8
            )
            query_sigma = torch.tensor(
                [request.sigma[prefix_length:] for request in batch_requests],
                device=device,
                dtype=torch.long,
            )
            query_valid = torch.ones_like(query_sigma, dtype=torch.bool)
            key_sigma = torch.zeros(
                (batch_size, max_cache_length), device=device, dtype=torch.long
            )
            key_sigma[:, :prefix_length] = prefix_sigma.expand(batch_size, -1)
            key_sigma[:, prefix_length : prefix_length + candidate_length] = (
                query_sigma
            )
            key_valid = torch.zeros(
                (batch_size, max_cache_length), device=device, dtype=torch.bool
            )
            key_valid[:, : prefix_length + candidate_length] = True
            query_mask = cross_sigma_attention_mask(
                query_sigma=query_sigma,
                key_sigma=key_sigma,
                query_valid=query_valid,
                key_valid=key_valid,
                include_diagonal=False,
                device=device,
            )
            content_mask = (
                cross_sigma_attention_mask(
                    query_sigma=query_sigma,
                    key_sigma=key_sigma,
                    query_valid=query_valid,
                    key_valid=key_valid,
                    include_diagonal=True,
                    device=device,
                )
                if attention_contract == "xlnet_content_diagonal"
                else None
            )
            full_token_types = torch.cat(
                [prefix_token_types.expand(batch_size, -1), token_types], dim=1
            )
            position_ids = build_row_col_position_ids(
                full_token_types, image_tokens
            )[:, :, prefix_length:]
            candidate_cache = clone_prefix_cache(
                prefix_cache, config=model.config, batch_size=batch_size
            )
            outputs = model.model(
                X0_input_ids=input_ids,
                attention_mask=query_mask,
                content_attention_mask=content_mask,
                position_ids=position_ids,
                past_key_values=candidate_cache,
                use_cache=True,
                cache_position=torch.arange(
                    prefix_length,
                    prefix_length + candidate_length,
                    device=device,
                ).unsqueeze(0).expand(batch_size, -1),
                calculate_likelihood=True,
                token_types=token_types,
            )
            hidden = outputs.last_hidden_state.reshape(-1, outputs.last_hidden_state.shape[-1])
            targets = input_ids.reshape(-1)
            token_logprobs: list[torch.Tensor] = []
            token_greedy: list[torch.Tensor] = []
            for chunk_start in range(0, hidden.shape[0], int(args.lm_head_chunk_tokens)):
                chunk_stop = min(
                    chunk_start + int(args.lm_head_chunk_tokens), hidden.shape[0]
                )
                logits = model.lm_head(hidden[chunk_start:chunk_stop])
                chunk_targets = targets[chunk_start:chunk_stop]
                gold = logits.gather(1, chunk_targets.unsqueeze(1)).squeeze(1).float()
                token_logprobs.append(gold - torch.logsumexp(logits.float(), dim=-1))
                token_greedy.append(logits.argmax(dim=-1).eq(chunk_targets))
            logprobs = torch.cat(token_logprobs).view(batch_size, candidate_length)
            greedy = torch.cat(token_greedy).view(batch_size, candidate_length)
            for row_index, (request_index, _) in enumerate(batch_pairs):
                total = float(logprobs[row_index].sum().item())
                ordered_scores[request_index] = CandidateScore(
                    loglikelihood=total,
                    normalized_loglikelihood=total / candidate_length,
                    token_count=candidate_length,
                    greedy=bool(greedy[row_index].all().item()),
                )
            del candidate_cache, outputs, hidden, logits
    if any(value is None for value in ordered_scores):
        raise RuntimeError("cached-prefix candidate scoring is incomplete")
    scores = [value for value in ordered_scores if value is not None]
    return (
        torch.tensor([value.loglikelihood for value in scores], dtype=torch.float32),
        torch.tensor(
            [value.normalized_loglikelihood for value in scores], dtype=torch.float32
        ),
    )


def score_candidates_with_backend(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    args = kwargs["args"]
    if getattr(kwargs["model"].config, "architecture_variant", None) == "selfless_joint_dit":
        kwargs["image_sigma_order"] = "joint"
    if getattr(kwargs["model"].config, "architecture_variant", None) == "showo2_unified":
        kwargs["image_sigma_order"] = "sequential"
        return score_text_candidates(**kwargs)
    if args.scoring_backend == "cached_prefix":
        return score_text_candidates_cached_prefix(**kwargs)
    return score_text_candidates(**kwargs)


def classification_metrics(
    prior_debiased_loglikelihood: torch.Tensor,
    class_indices: torch.Tensor,
) -> dict[str, Any]:
    """Compute only the canonical ImageNet Top-1 and Top-5 accuracies."""

    scores = prior_debiased_loglikelihood
    targets = class_indices.long()
    if scores.ndim != 2 or scores.shape[0] != targets.numel():
        raise ValueError("classification scores and target labels do not align")
    if int(scores.shape[1]) != 1_000:
        raise ValueError("ImageNet-1K classification requires 1,000 class texts")
    if targets.numel() and (
        int(targets.min()) < 0 or int(targets.max()) >= int(scores.shape[1])
    ):
        raise ValueError("ImageNet class target is outside [0, 1000)")
    top5 = torch.topk(scores, k=5, dim=1, largest=True, sorted=True).indices
    top1_correct = top5[:, 0].eq(targets)
    top5_correct = top5.eq(targets.unsqueeze(1)).any(dim=1)
    return {
        "records": int(targets.numel()),
        "classes": int(scores.shape[1]),
        "primary_metric": "top_1_accuracy",
        "top_1_accuracy": float(top1_correct.float().mean().item()),
        "top_5_accuracy": float(top5_correct.float().mean().item()),
    }


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_classification(
    *,
    records: Sequence[ImageNetRecord],
    class_names: Sequence[str],
    model,
    tokenizer,
    cache: PosteriorCache,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    image_sigma_order: str,
    attention_contract: str,
) -> dict[str, Any] | None:
    selected = list(records)
    if int(args.limit) > 0:
        selected = selected[: int(args.limit)]
    candidates = [
        CLASS_TEXT_TEMPLATE.format(class_name=class_name)
        for class_name in class_names
    ]
    local_pairs = list(enumerate(selected))[rank::world_size]
    local_scores = torch.empty((len(local_pairs), len(candidates)), dtype=torch.float32)
    query_indices: list[int] = []
    started = time.monotonic()
    for local_index, (query_index, record) in enumerate(local_pairs):
        _, scores = score_candidates_with_backend(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            image_id=record.img_id,
            item_id=f"{CLASSIFICATION_TASK}/{record.image_id}",
            prompt=RETRIEVAL_PROMPT,
            candidates=candidates,
            args=args,
            device=device,
            image_sigma_order=image_sigma_order,
            attention_contract=attention_contract,
            evaluation_task=CLASSIFICATION_TASK,
        )
        local_scores[local_index] = scores
        query_indices.append(query_index)
        if (
            (local_index + 1) % int(args.progress_every) == 0
            or local_index + 1 == len(local_pairs)
        ):
            print(
                json.dumps(
                    {
                        "event": "imagenet1k_zeroshot_classification_progress",
                        "task": CLASSIFICATION_TASK,
                        "rank": rank,
                        "image_rows": local_index + 1,
                        "image_rows_total": len(local_pairs),
                        "candidate_class_texts": len(candidates),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    shard = args.output_dir / "shards" / (
        f"rank-{rank:05d}-of-{world_size:05d}.pt"
    )
    atomic_torch_save(
        shard,
        {
            "query_indices": torch.tensor(query_indices, dtype=torch.long),
            "conditional_mean_token_loglikelihood": local_scores,
            "scoring_contract": scoring_contract(attention_contract, LIKELIHOOD_SCORING_CONTRACT),
            "dual_stream_attention_contract": attention_contract,
            "runtime_hashing_enabled": False,
        },
    )
    barrier(device)
    if rank != 0:
        return None
    matrix = torch.empty((len(selected), len(candidates)), dtype=torch.float32)
    seen = torch.zeros(len(selected), dtype=torch.bool)
    for shard_rank in range(world_size):
        path = args.output_dir / "shards" / (
            f"rank-{shard_rank:05d}-of-{world_size:05d}.pt"
        )
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        if payload.get("runtime_hashing_enabled", True) is not False:
            raise ValueError("retrieval shard violates the no-hash contract")
        if payload.get("scoring_contract") != scoring_contract(attention_contract, LIKELIHOOD_SCORING_CONTRACT):
            raise ValueError("retrieval shard uses the wrong scoring contract")
        if payload.get("dual_stream_attention_contract") != attention_contract:
            raise ValueError("retrieval shard uses the wrong attention contract")
        indices = payload["query_indices"].long()
        shard_scores = payload["conditional_mean_token_loglikelihood"].float()
        if shard_scores.shape != (indices.numel(), len(candidates)):
            raise ValueError("classification shard shape mismatch")
        if indices.unique().numel() != indices.numel():
            raise ValueError("duplicate classification image rows within a shard")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= len(selected)
        ):
            raise ValueError("classification shard contains out-of-range image rows")
        if bool(seen[indices].any()):
            raise ValueError("duplicate classification image rows across shards")
        matrix[indices] = shard_scores
        seen[indices] = True
    if not bool(seen.all()):
        raise RuntimeError("classification score matrix is incomplete")
    class_indices = torch.tensor(
        [record.class_index for record in selected], dtype=torch.long
    )
    calibrated, class_text_log_prior = language_prior_debiased_scores(matrix)
    summary = classification_metrics(calibrated, class_indices)
    summary.update(
        {
            "schema": "selfless_imagenet1k_zeroshot_classification_summary_v2",
            "task": CLASSIFICATION_TASK,
            "split": "imagenet_val",
            "accuracy_unit": "unit_interval",
            "formal_target_records": 50_000,
            "complete_formal_target": len(selected) == 50_000,
            "class_text_template": CLASS_TEXT_TEMPLATE,
            "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
            "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
            "language_prior_image_count": len(selected),
        }
    )
    atomic_torch_save(
        args.output_dir / "score_matrix.pt",
        {
            "prior_debiased_loglikelihood": calibrated,
            "class_text_log_prior": class_text_log_prior,
            "img_ids": torch.tensor([record.img_id for record in selected]),
            "class_indices": class_indices,
            "class_names": list(class_names),
            "class_text_template": CLASS_TEXT_TEMPLATE,
            "language_prior_alpha": LANGUAGE_PRIOR_ALPHA,
            "language_prior_estimator": LANGUAGE_PRIOR_ESTIMATOR,
            "scoring_contract": scoring_contract(attention_contract, LIKELIHOOD_SCORING_CONTRACT),
            "dual_stream_attention_contract": attention_contract,
            "runtime_hashing_enabled": False,
        },
    )
    return summary


def validate_args(args: argparse.Namespace, source) -> None:
    for path in (
        args.config,
        source.path,
        args.image_manifest,
        args.classes,
        args.class_names,
        args.cache_shard_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for value in (
        args.batch_size_per_rank,
        args.request_chunk_size,
        args.lm_head_chunk_tokens,
        args.max_length,
        args.progress_every,
    ):
        if int(value) <= 0:
            raise ValueError("batch/chunk/length arguments must be positive")
    if int(args.limit) < 0:
        raise ValueError("limit must be non-negative")
    if bool(args.require_formal_protocol) and int(args.limit) != 0:
        raise ValueError("formal ImageNet classification forbids row limiting")
