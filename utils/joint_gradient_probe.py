"""Task-separated gradient diagnostics for joint caption/T2I training."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch

from utils.selfless_flow_optimizer import optimizer_parameter_role


LAMBDA_TEXT_MIN = 0.025
LAMBDA_TEXT_MAX = 0.4
REFERENCE_LAMBDA_TEXT = (0.05, 0.1, 0.2)


def _unique_trainable(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    result: list[torch.nn.Parameter] = []
    for parameter in parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result.append(parameter)
    return result


def gradient_parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """Build probe groups while preserving the tied embedding/head alias."""

    role_parameters: dict[str, list[torch.nn.Parameter]] = {
        "backbone": [],
        "image_projector": [],
        "flow_head": [],
    }
    for name, parameter in model.named_parameters():
        role = optimizer_parameter_role(name)
        if role == "tied_lm_head_embedding":
            continue
        role_parameters[role].append(parameter)

    lm_head = getattr(model, "lm_head", None)
    model_body = getattr(model, "model", None)
    embed_tokens = getattr(model_body, "embed_tokens", None)
    lm_parameter = getattr(lm_head, "weight", None)
    embedding_parameter = getattr(embed_tokens, "weight", None)
    if not isinstance(lm_parameter, torch.nn.Parameter):
        raise RuntimeError("gradient probe requires model.lm_head.weight")
    if not isinstance(embedding_parameter, torch.nn.Parameter):
        raise RuntimeError("gradient probe requires model.model.embed_tokens.weight")

    return {
        "backbone": _unique_trainable(role_parameters["backbone"]),
        "lm_head": _unique_trainable([lm_parameter]),
        "special_token_embedding": _unique_trainable([embedding_parameter]),
        "image_projector": _unique_trainable(
            role_parameters["image_projector"]
        ),
        "flow_head": _unique_trainable(role_parameters["flow_head"]),
    }


def _sum_squares(
    parameters: Sequence[torch.nn.Parameter],
    *,
    row_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    total: torch.Tensor | None = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if row_indices is not None:
            if gradient.ndim < 1:
                raise RuntimeError("special-token gradient must have a row dimension")
            indices = row_indices.to(device=gradient.device, dtype=torch.long)
            gradient = gradient.index_select(0, indices)
        contribution = gradient.detach().float().square().sum()
        total = contribution if total is None else total + contribution
    if total is None:
        device = parameters[0].device if parameters else torch.device("cpu")
        total = torch.zeros((), device=device, dtype=torch.float32)
    return total


def _gradient_norms(
    groups: dict[str, list[torch.nn.Parameter]],
    special_token_ids: Sequence[int],
) -> dict[str, float]:
    special_rows = torch.tensor(tuple(map(int, special_token_ids)), dtype=torch.long)
    result: dict[str, float] = {}
    for name, parameters in groups.items():
        row_indices = special_rows if name == "special_token_embedding" else None
        value = _sum_squares(parameters, row_indices=row_indices).sqrt()
        result[name] = float(value.detach().cpu())
    return result


def _clone_gradients(
    parameters: Sequence[torch.nn.Parameter],
) -> list[tuple[torch.nn.Parameter, torch.Tensor | None]]:
    return [
        (
            parameter,
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else None,
        )
        for parameter in parameters
    ]


def _backbone_pair_metrics(
    text_gradients: Sequence[tuple[torch.nn.Parameter, torch.Tensor | None]],
) -> dict[str, float]:
    text_squared: torch.Tensor | None = None
    image_squared: torch.Tensor | None = None
    dot: torch.Tensor | None = None
    for parameter, text_gradient in text_gradients:
        image_gradient = parameter.grad
        if text_gradient is None and image_gradient is None:
            continue
        reference = text_gradient if text_gradient is not None else image_gradient
        assert reference is not None
        zero = torch.zeros((), device=reference.device, dtype=torch.float32)
        text_float = text_gradient.float() if text_gradient is not None else None
        image_float = (
            image_gradient.detach().float() if image_gradient is not None else None
        )
        text_term = text_float.square().sum() if text_float is not None else zero
        image_term = image_float.square().sum() if image_float is not None else zero
        dot_term = (
            (text_float * image_float).sum()
            if text_float is not None and image_float is not None
            else zero
        )
        text_squared = text_term if text_squared is None else text_squared + text_term
        image_squared = image_term if image_squared is None else image_squared + image_term
        dot = dot_term if dot is None else dot + dot_term
    if text_squared is None or image_squared is None or dot is None:
        raise RuntimeError("shared backbone produced no text gradients")
    text_norm = text_squared.sqrt()
    image_norm = image_squared.sqrt()
    if not bool(torch.isfinite(text_norm).item()) or not bool(
        torch.isfinite(image_norm).item()
    ):
        raise FloatingPointError("non-finite shared-backbone gradient norm")
    if float(text_norm.detach().cpu()) <= 0.0:
        raise RuntimeError("caption loss produced zero shared-backbone gradient")
    if float(image_norm.detach().cpu()) <= 0.0:
        raise RuntimeError("T2I loss produced zero shared-backbone gradient")
    denominator = text_norm * image_norm
    cosine = (dot / denominator).clamp(-1.0, 1.0)
    return {
        "g_text": float(text_norm.detach().cpu()),
        "g_image": float(image_norm.detach().cpu()),
        "ratio_g_image_over_g_text": float((image_norm / text_norm).detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
    }


def measure_gradient_probe_batch(
    model: torch.nn.Module,
    forward_losses: Callable[[], tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    *,
    special_token_ids: Sequence[int],
    reset_seed: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run separate unweighted text/image forward+backward passes for one batch."""

    groups = gradient_parameter_groups(model)
    model.zero_grad(set_to_none=True)
    if reset_seed is not None:
        reset_seed()
    text_losses, text_counts = forward_losses()
    text_loss = text_losses["text_loss"]
    if not text_loss.requires_grad:
        raise RuntimeError("text loss graph is detached")
    text_loss_value = float(text_loss.detach().float().cpu())
    text_loss.backward()
    text_norms = _gradient_norms(groups, special_token_ids)
    text_backbone = _clone_gradients(groups["backbone"])
    del text_losses, text_loss

    model.zero_grad(set_to_none=True)
    if reset_seed is not None:
        reset_seed()
    image_losses, image_counts = forward_losses()
    image_loss = image_losses["image_loss"]
    if not image_loss.requires_grad:
        raise RuntimeError("image loss graph is detached")
    image_loss_value = float(image_loss.detach().float().cpu())
    image_loss.backward()
    image_norms = _gradient_norms(groups, special_token_ids)
    shared = _backbone_pair_metrics(text_backbone)

    counts = {
        "text_tokens": int(text_counts["text_tokens"].detach().cpu()),
        "image_tokens": int(image_counts["image_tokens"].detach().cpu()),
    }
    second_counts = {
        "text_tokens": int(image_counts["text_tokens"].detach().cpu()),
        "image_tokens": int(image_counts["image_tokens"].detach().cpu()),
    }
    if counts != second_counts:
        raise RuntimeError(
            f"task-separated forwards changed target counts: {counts} != {second_counts}"
        )
    if counts["text_tokens"] <= 0 or counts["image_tokens"] <= 0:
        raise RuntimeError(f"probe batch must contain both tasks, got {counts}")

    model.zero_grad(set_to_none=True)
    return {
        "loss_text_unweighted": text_loss_value,
        "loss_image_unweighted": image_loss_value,
        **counts,
        "shared_backbone": shared,
        "gradient_norms": {
            name: {
                "text": float(text_norms[name]),
                "image": float(image_norms[name]),
            }
            for name in groups
        },
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Sequence[float]) -> dict[str, float]:
    finite = [float(value) for value in values]
    if not finite or not all(math.isfinite(value) for value in finite):
        raise ValueError("probe summaries require a non-empty finite sequence")
    return {
        "mean": sum(finite) / len(finite),
        "median": percentile(finite, 0.5),
        "p25": percentile(finite, 0.25),
        "p75": percentile(finite, 0.75),
    }


def build_lambda_text_candidates(
    ratio_median: float,
    *,
    lower: float = LAMBDA_TEXT_MIN,
    upper: float = LAMBDA_TEXT_MAX,
) -> dict[str, Any]:
    if not math.isfinite(ratio_median) or ratio_median <= 0.0:
        raise ValueError(f"gradient ratio median must be positive, got {ratio_median}")
    center = min(max(float(ratio_median), float(lower)), float(upper))
    scaled = {
        "0.5x": min(max(0.5 * center, float(lower)), float(upper)),
        "1.0x": center,
        "2.0x": min(max(2.0 * center, float(lower)), float(upper)),
    }
    candidates = sorted(
        {
            round(value, 8)
            for value in (*scaled.values(), *REFERENCE_LAMBDA_TEXT)
            if lower <= value <= upper
        }
    )
    return {
        "raw_center": float(ratio_median),
        "center": float(center),
        "bounds": [float(lower), float(upper)],
        "scaled": scaled,
        "reference_points": list(REFERENCE_LAMBDA_TEXT),
        "candidates": candidates,
    }


def summarize_probe_batches(batches: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not 16 <= len(batches) <= 32:
        raise ValueError(f"formal probe requires 16-32 batches, got {len(batches)}")
    scalar_paths = {
        "g_text": lambda row: row["shared_backbone"]["g_text"],
        "g_image": lambda row: row["shared_backbone"]["g_image"],
        "ratio_g_image_over_g_text": lambda row: row["shared_backbone"][
            "ratio_g_image_over_g_text"
        ],
        "cosine": lambda row: row["shared_backbone"]["cosine"],
        "loss_text_unweighted": lambda row: row["loss_text_unweighted"],
        "loss_image_unweighted": lambda row: row["loss_image_unweighted"],
    }
    summary = {
        name: summarize([extract(row) for row in batches])
        for name, extract in scalar_paths.items()
    }
    group_names = tuple(batches[0]["gradient_norms"])
    summary["gradient_norms"] = {
        group: {
            task: summarize(
                [row["gradient_norms"][group][task] for row in batches]
            )
            for task in ("text", "image")
        }
        for group in group_names
    }
    cosines = [float(row["shared_backbone"]["cosine"]) for row in batches]
    negative_fraction = sum(value < 0.0 for value in cosines) / len(cosines)
    summary["task_conflict"] = {
        "negative_cosine_batches": sum(value < 0.0 for value in cosines),
        "negative_fraction": negative_fraction,
        "persistent_negative": bool(
            summary["cosine"]["median"] < 0.0 or negative_fraction >= 0.5
        ),
        "interpretation": (
            "shared-backbone task conflict detected in this fixed probe"
            if summary["cosine"]["median"] < 0.0 or negative_fraction >= 0.5
            else "no persistent negative shared-backbone cosine in this fixed probe"
        ),
    }
    summary["lambda_text"] = build_lambda_text_candidates(
        summary["ratio_g_image_over_g_text"]["median"]
    )
    return summary
