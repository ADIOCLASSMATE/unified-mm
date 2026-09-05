#!/usr/bin/env python3
"""Evaluate a Selfless dual-stream model on standard text benchmarks.

The model predicts a token at the same query-stream position rather than with
the usual causal-LM one-token shift.  This runner therefore scores continuation
tokens directly from the Selfless query stream and must not be replaced by the
generic Hugging Face causal-LM adapter.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluation_model_source import (  # noqa: E402
    add_model_source_argument,
    configure_model_source,
    load_model_source_weights,
    model_source_from_args,
    resolve_evaluation_model_source,
)
from utils.utils import get_selfless_mask, load_model_tokenizer  # noqa: E402


DEFAULT_CONFIG = Path("configs/selfless/unified_baseline_100b_ascend_64npu.yaml")
DEFAULT_DATA_ROOT = Path("public/benchmarks/selfless_text_v1")
DEFAULT_TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
    "mmlu",
)
MC_TASKS = DEFAULT_TASKS
CHOICE_LETTERS = ("A", "B", "C", "D")
TEXT_SCORING_CONTRACT = "selfless_same_position_dual_stream_v2"
TEXT_PROTOCOL_SCHEMA = "selfless_text_benchmark_v3"
TEXT_NORMALIZATION = "original_choice_characters"
WINOGRANDE_SCORING = "shared_suffix_given_prefix_and_option"
TEXT_ASSET_SCHEMA = "selfless_text_benchmark_assets_v2"
LM_EVAL_REFERENCE = {
    "repository": "https://github.com/EleutherAI/lm-evaluation-harness",
    "commit": "b954108c9baaaa934b4ad842033b31a97ee30816",
    "role": "task_prompt_split_and_metric_reference",
    "adapter_equivalence": "not_bitwise_causal_lm_equivalent",
}
TASK_PROTOCOLS = {
    "arc_easy": {"split": "test", "fewshot": 0, "primary": "accuracy_normalized"},
    "arc_challenge": {
        "split": "test",
        "fewshot": 0,
        "primary": "accuracy_normalized",
    },
    "hellaswag": {
        "split": "validation",
        "fewshot": 0,
        "primary": "accuracy_normalized",
    },
    "piqa": {"split": "validation", "fewshot": 0, "primary": "accuracy_normalized"},
    "winogrande": {"split": "validation", "fewshot": 0, "primary": "accuracy"},
    "boolq": {"split": "validation", "fewshot": 0, "primary": "accuracy"},
    "openbookqa": {
        "split": "test",
        "fewshot": 0,
        "primary": "accuracy_normalized",
    },
    "mmlu": {"split": "test", "fewshot": 5, "primary": "accuracy_macro"},
}


@dataclass(frozen=True)
class MultipleChoiceExample:
    item_index: int
    item_id: str
    task: str
    context: str
    choices: tuple[str, ...]
    label: int
    category: str | None = None
    # Multiple-input tasks (WinoGrande): choices are contexts, target is shared.
    shared_target: str | None = None


@dataclass(frozen=True)
class ChoiceRequest:
    item_index: int
    choice_index: int
    input_ids: tuple[int, ...]
    target_start: int
    boundary_adjusted: bool
    truncated_context_tokens: int
    normalization_char_count: int

    def __post_init__(self):
        if self.normalization_char_count <= 0:
            raise ValueError("normalization character count must be positive")


@dataclass(frozen=True)
class ChoiceScore:
    loglikelihood: float
    normalized_loglikelihood: float
    token_count: int
    greedy: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_model_source_argument(parser)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--batch_size_per_rank", type=int, default=8)
    parser.add_argument("--lm_head_chunk_tokens", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--progress_every", type=int, default=50)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def initialize_device(kind: str) -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if kind == "npu":
        import torch_npu  # noqa: F401

        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        backend = "hccl"
    elif kind == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def barrier(device: torch.device) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if device.type in {"npu", "cuda"}:
        dist.barrier(device_ids=[int(device.index or 0)])
    else:
        dist.barrier()


def read_checkpoint_step(checkpoint: Path) -> int:
    return resolve_evaluation_model_source(checkpoint).global_step


def validate_args(args: argparse.Namespace, source) -> tuple[str, ...]:
    for path in (args.config, source.path, args.data_root):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = args.data_root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    asset_report = json.loads(manifest.read_text(encoding="utf-8"))
    if asset_report.get("schema") != TEXT_ASSET_SCHEMA:
        raise ValueError(
            "text benchmark assets use an obsolete split protocol; rerun "
            "scripts/prepare_text_benchmark_assets.py"
        )
    if not bool(asset_report.get("complete", False)):
        raise ValueError("text benchmark asset manifest is incomplete")
    if bool(asset_report.get("runtime_hashing_enabled", True)):
        raise ValueError("text benchmark assets must use the no-hash contract")
    for value in (
        args.batch_size_per_rank,
        args.lm_head_chunk_tokens,
        args.max_length,
        args.progress_every,
    ):
        if int(value) <= 0:
            raise ValueError(f"positive integer required, got {value}")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    tasks = tuple(part.strip() for part in str(args.tasks).split(",") if part.strip())
    unknown = sorted(set(tasks) - set(DEFAULT_TASKS))
    if unknown:
        raise ValueError(f"unknown text benchmark tasks: {unknown}")
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def read_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def answer_index(labels: Sequence[Any], answer: Any) -> int:
    normalized = [str(item).strip() for item in labels]
    target = str(answer).strip()
    if target in normalized:
        return normalized.index(target)
    if target.isdigit():
        number = int(target)
        if 1 <= number <= len(normalized):
            return number - 1
    raise ValueError(f"answer {answer!r} is absent from labels {normalized!r}")


def preprocess_hellaswag(text: str) -> str:
    text = str(text).strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def load_arc(task: str, root: Path) -> list[MultipleChoiceExample]:
    rows = read_parquet(root / task / "test.parquet")
    examples = []
    for index, row in enumerate(rows):
        choices = row["choices"]
        texts = tuple(str(value) for value in choices["text"])
        label = answer_index(choices["label"], row["answerKey"])
        examples.append(
            MultipleChoiceExample(
                item_index=index,
                item_id=str(row["id"]),
                task=task,
                context=f"Question: {str(row['question']).strip()}\nAnswer:",
                choices=texts,
                label=label,
            )
        )
    return examples


def load_hellaswag(root: Path) -> list[MultipleChoiceExample]:
    rows = read_parquet(root / "hellaswag" / "validation.parquet")
    examples = []
    for index, row in enumerate(rows):
        context = str(row["ctx_a"]) + " " + str(row["ctx_b"]).capitalize()
        query = preprocess_hellaswag(f"{row['activity_label']}: {context}")
        examples.append(
            MultipleChoiceExample(
                item_index=index,
                item_id=str(row["ind"]),
                task="hellaswag",
                context=query,
                choices=tuple(preprocess_hellaswag(value) for value in row["endings"]),
                label=int(row["label"]),
                category=str(row.get("split_type") or "unknown"),
            )
        )
    return examples


def load_piqa(root: Path) -> list[MultipleChoiceExample]:
    example_path = root / "piqa" / "validation.jsonl"
    label_path = root / "piqa" / "validation-labels.lst"
    rows = [
        json.loads(line)
        for line in example_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [
        int(line)
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(labels):
        raise ValueError("PIQA examples and labels have different row counts")
    return [
        MultipleChoiceExample(
            item_index=index,
            item_id=str(row.get("id", index)),
            task="piqa",
            context=f"Question: {str(row['goal']).strip()}\nAnswer:",
            choices=(str(row["sol1"]), str(row["sol2"])),
            label=int(labels[index]),
        )
        for index, row in enumerate(rows)
    ]


def load_winogrande(root: Path) -> list[MultipleChoiceExample]:
    rows = read_parquet(root / "winogrande" / "validation.parquet")
    examples = []
    for index, row in enumerate(rows):
        sentence = str(row["sentence"])
        if sentence.count("_") != 1:
            raise ValueError(f"unexpected WinoGrande blank count: {sentence!r}")
        prefix, suffix = sentence.split("_", 1)
        examples.append(
            MultipleChoiceExample(
                item_index=index,
                item_id=f"winogrande:{index}",
                task="winogrande",
                context="",
                choices=(
                    prefix + str(row["option1"]),
                    prefix + str(row["option2"]),
                ),
                shared_target=suffix.strip(),
                label=int(str(row["answer"]).strip()) - 1,
            )
        )
    return examples


def load_boolq(root: Path) -> list[MultipleChoiceExample]:
    rows = read_parquet(root / "boolq" / "validation.parquet")
    examples = []
    for index, row in enumerate(rows):
        question = str(row["question"]).strip().rstrip("?") + "?"
        examples.append(
            MultipleChoiceExample(
                item_index=index,
                item_id=f"boolq:{index}",
                task="boolq",
                context=(
                    f"{str(row['passage']).strip()}\nQuestion: {question}\nAnswer:"
                ),
                choices=("no", "yes"),
                label=int(bool(row["answer"])),
            )
        )
    return examples


def load_openbookqa(root: Path) -> list[MultipleChoiceExample]:
    rows = read_parquet(root / "openbookqa" / "test.parquet")
    examples = []
    for index, row in enumerate(rows):
        choices = row["choices"]
        examples.append(
            MultipleChoiceExample(
                item_index=index,
                item_id=str(row["id"]),
                task="openbookqa",
                context=str(row["question_stem"]).strip(),
                choices=tuple(str(value) for value in choices["text"]),
                label=answer_index(choices["label"], row["answerKey"]),
            )
        )
    return examples


def format_mmlu_question(row: dict[str, Any], answer: int | None) -> str:
    choices = [str(value) for value in row["choices"]]
    if len(choices) != 4:
        raise ValueError(f"MMLU item does not have four choices: {choices!r}")
    lines = [str(row["question"]).strip()]
    lines.extend(f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, choices))
    suffix = "Answer:"
    if answer is not None:
        suffix += f" {CHOICE_LETTERS[int(answer)]}"
    lines.append(suffix)
    return "\n".join(lines)


def load_mmlu(root: Path) -> list[MultipleChoiceExample]:
    dev_rows = read_parquet(root / "mmlu" / "dev.parquet")
    test_rows = read_parquet(root / "mmlu" / "test.parquet")
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev_rows:
        by_subject[str(row["subject"])].append(row)
    examples = []
    for index, row in enumerate(test_rows):
        subject = str(row["subject"])
        fewshot = by_subject[subject][:5]
        if len(fewshot) != 5:
            raise ValueError(f"MMLU subject {subject!r} lacks five dev examples")
        description = (
            "The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n"
        )
        demonstrations = "\n\n".join(
            format_mmlu_question(example, int(example["answer"]))
            for example in fewshot
        )
        context = (
            description
            + demonstrations
            + "\n\n"
            + format_mmlu_question(row, None)
        )
        examples.append(
            MultipleChoiceExample(
                item_index=index,
                item_id=f"{subject}:{index}",
                task="mmlu",
                context=context,
                choices=CHOICE_LETTERS,
                label=int(row["answer"]),
                category=subject,
            )
        )
    return examples


def load_multiple_choice_task(
    task: str, root: Path
) -> list[MultipleChoiceExample]:
    if task in {"arc_easy", "arc_challenge"}:
        return load_arc(task, root)
    if task == "hellaswag":
        return load_hellaswag(root)
    if task == "piqa":
        return load_piqa(root)
    if task == "winogrande":
        return load_winogrande(root)
    if task == "boolq":
        return load_boolq(root)
    if task == "openbookqa":
        return load_openbookqa(root)
    if task == "mmlu":
        return load_mmlu(root)
    raise ValueError(task)


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and int(left[index]) == int(right[index]):
        index += 1
    return index


def encode_choice(
    tokenizer,
    example: MultipleChoiceExample,
    choice_index: int,
    max_length: int,
) -> ChoiceRequest:
    choice = str(example.choices[choice_index])
    # lm-eval acc_norm uses len(original choice), before adding a delimiter or
    # tokenizing. For WinoGrande the choice is a candidate-specific context.
    normalization_char_count = len(choice)
    if normalization_char_count == 0:
        raise ValueError("cannot normalize an empty choice")
    context = str(example.context) if example.shared_target is None else choice
    continuation = choice if example.shared_target is None else example.shared_target
    if not continuation:
        raise ValueError("choice has an empty scoring target")
    trailing_count = len(context) - len(context.rstrip())
    if trailing_count:
        continuation = context[-trailing_count:] + continuation
        context = context[:-trailing_count]
    elif not continuation[:1].isspace():
        continuation = " " + continuation
    context_ids = [
        int(value) for value in tokenizer.encode(context, add_special_tokens=False)
    ]
    full_ids = [
        int(value)
        for value in tokenizer.encode(context + continuation, add_special_tokens=False)
    ]
    target_start = common_prefix_length(context_ids, full_ids)
    boundary_adjusted = target_start != len(context_ids)
    if target_start >= len(full_ids):
        raise ValueError(
            f"choice tokenized to an empty continuation for {example.task}/{example.item_id}"
        )
    continuation_ids = full_ids[target_start:]
    context_prefix = full_ids[:target_start]
    eos_id = tokenizer.eos_token_id
    if not context_prefix:
        if eos_id is None:
            raise ValueError("tokenizer needs eos_token_id for empty contexts")
        context_prefix = [int(eos_id)]
    if len(continuation_ids) >= max_length:
        raise ValueError(
            f"continuation has {len(continuation_ids)} tokens, max_length={max_length}"
        )
    retained_context = max_length - len(continuation_ids)
    truncated = max(0, len(context_prefix) - retained_context)
    context_prefix = context_prefix[-retained_context:]
    combined = tuple(context_prefix + continuation_ids)
    return ChoiceRequest(
        item_index=int(example.item_index),
        choice_index=int(choice_index),
        input_ids=combined,
        target_start=len(context_prefix),
        boundary_adjusted=boundary_adjusted,
        truncated_context_tokens=truncated,
        normalization_char_count=normalization_char_count,
    )


@torch.inference_mode()
def score_choice_requests(
    model,
    requests: Sequence[ChoiceRequest],
    *,
    batch_size: int,
    lm_head_chunk_tokens: int,
    device: torch.device,
) -> list[ChoiceScore]:
    if not requests:
        return []
    pad_id = int(getattr(model.config, "eos_token_id", 0) or 0)
    ordered = sorted(
        enumerate(requests), key=lambda pair: len(pair[1].input_ids), reverse=True
    )
    scores: list[ChoiceScore | None] = [None] * len(requests)
    for offset in range(0, len(ordered), batch_size):
        batch_pairs = ordered[offset : offset + batch_size]
        batch = [request for _, request in batch_pairs]
        length = max(len(request.input_ids) for request in batch)
        input_ids = torch.full(
            (len(batch), length),
            pad_id,
            device=device,
            dtype=torch.long,
        )
        segment_ids = torch.full(
            (len(batch), length), -1, device=device, dtype=torch.long
        )
        sigma = torch.zeros(
            (len(batch), length), device=device, dtype=torch.float32
        )
        target_mask = torch.zeros(
            (len(batch), length), device=device, dtype=torch.bool
        )
        for row_index, request in enumerate(batch):
            item_length = len(request.input_ids)
            input_ids[row_index, :item_length] = torch.tensor(
                request.input_ids, device=device, dtype=torch.long
            )
            segment_ids[row_index, :item_length] = 0
            sigma[row_index, :item_length] = torch.arange(
                item_length, device=device, dtype=torch.float32
            )
            target_mask[row_index, request.target_start:item_length] = True
        attention_mask = get_selfless_mask(
            sigma=sigma,
            seq_len=length,
            device=device,
            segment_ids=segment_ids,
        )
        attention_contract = str(
            getattr(
                model.config,
                "dual_stream_attention_contract",
                "selfless_strict",
            )
        ).strip().lower()
        if attention_contract not in {
            "selfless_strict",
            "xlnet_content_diagonal",
        }:
            raise ValueError(
                "unsupported dual-stream attention contract: "
                f"{attention_contract!r}"
            )
        content_attention_mask = (
            get_selfless_mask(
                sigma=sigma,
                seq_len=length,
                device=device,
                segment_ids=segment_ids,
                include_diagonal=True,
            )
            if attention_contract == "xlnet_content_diagonal"
            else None
        )
        forward_kwargs = {
            "X0_input_ids": input_ids,
            "attention_mask": attention_mask,
            "calculate_likelihood": True,
            "return_logits": False,
        }
        if content_attention_mask is not None:
            forward_kwargs["content_attention_mask"] = content_attention_mask
        outputs = model(
            **forward_kwargs,
        )
        hidden = outputs.last_hidden_state
        selected_hidden = hidden[target_mask]
        selected_targets = input_ids[target_mask]
        selected_rows = target_mask.nonzero(as_tuple=False)[:, 0]
        token_logprobs: list[torch.Tensor] = []
        token_greedy: list[torch.Tensor] = []
        for start in range(0, selected_hidden.shape[0], lm_head_chunk_tokens):
            stop = min(start + lm_head_chunk_tokens, selected_hidden.shape[0])
            logits = model.lm_head(selected_hidden[start:stop])
            targets = selected_targets[start:stop]
            gold = logits.gather(1, targets.unsqueeze(1)).squeeze(1).float()
            token_logprobs.append(gold - torch.logsumexp(logits.float(), dim=-1))
            token_greedy.append(logits.argmax(dim=-1).eq(targets))
        logprobs = torch.cat(token_logprobs)
        greedy = torch.cat(token_greedy)
        for row_index, (original_index, request) in enumerate(batch_pairs):
            selection = selected_rows.eq(row_index)
            row_logprobs = logprobs[selection]
            row_greedy = greedy[selection]
            count = int(row_logprobs.numel())
            if count <= 0:
                raise RuntimeError("choice request has no target tokens")
            total = float(row_logprobs.sum().item())
            scores[original_index] = ChoiceScore(
                loglikelihood=total,
                normalized_loglikelihood=total / request.normalization_char_count,
                token_count=count,
                greedy=bool(row_greedy.all().item()),
            )
        del (
            outputs,
            hidden,
            selected_hidden,
            selected_targets,
            attention_mask,
            content_attention_mask,
        )
    if any(score is None for score in scores):
        raise RuntimeError("choice scoring left incomplete results")
    return [score for score in scores if score is not None]


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


def rank_shard_paths(output_dir: Path, task: str, rank: int, world_size: int):
    shard_dir = output_dir / "shards" / task
    suffix = f"rank-{rank:05d}-of-{world_size:05d}"
    return shard_dir / f"{suffix}.jsonl", shard_dir / f"{suffix}.complete.json"


def completed_rank_shard(
    marker_path: Path,
    shard_path: Path,
    *,
    task: str,
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    checkpoint_step: int,
    expected_records: int,
    max_length: int,
    limit: int,
    attention_contract: str,
) -> bool:
    if not marker_path.is_file() or not shard_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "selfless_text_benchmark_rank_complete_v4",
        "complete": True,
        "protocol_schema": TEXT_PROTOCOL_SCHEMA,
        "task_protocol": TASK_PROTOCOLS[task],
        "scoring_contract": TEXT_SCORING_CONTRACT,
        "task": task,
        "rank": rank,
        "world_size": world_size,
        "checkpoint": checkpoint_dir,
        "checkpoint_step": checkpoint_step,
        "records": expected_records,
        "max_length": max_length,
        "limit": limit,
        "dual_stream_attention_contract": attention_contract,
        "query_stream_diagonal": False,
        "content_stream_diagonal": (
            attention_contract == "xlnet_content_diagonal"
        ),
        "runtime_hashing_enabled": False,
    }
    return all(marker.get(key) == value for key, value in expected.items())


def write_rank_shard(
    output_dir: Path,
    task: str,
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    checkpoint_step: int,
    rows: Sequence[dict[str, Any]],
    *,
    max_length: int,
    limit: int,
    attention_contract: str,
    wall_seconds: float,
) -> None:
    shard_path, marker_path = rank_shard_paths(output_dir, task, rank, world_size)
    atomic_write_text(shard_path, jsonl_text(rows))
    marker = {
        "schema": "selfless_text_benchmark_rank_complete_v4",
        "complete": True,
        "protocol_schema": TEXT_PROTOCOL_SCHEMA,
        "task_protocol": TASK_PROTOCOLS[task],
        "scoring_contract": TEXT_SCORING_CONTRACT,
        "runtime_hashing_enabled": False,
        "task": task,
        "rank": rank,
        "world_size": world_size,
        "checkpoint": checkpoint_dir,
        "checkpoint_step": checkpoint_step,
        "records": len(rows),
        "max_length": max_length,
        "limit": limit,
        "dual_stream_attention_contract": attention_contract,
        "query_stream_diagonal": False,
        "content_stream_diagonal": (
            attention_contract == "xlnet_content_diagonal"
        ),
        "wall_seconds": float(wall_seconds),
        "completed_at": utc_now(),
    }
    atomic_write_text(marker_path, json.dumps(marker, indent=2, sort_keys=True) + "\n")


def evaluate_multiple_choice_task(
    model,
    tokenizer,
    examples: Sequence[MultipleChoiceExample],
    *,
    task: str,
    output_dir: Path,
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    checkpoint_step: int,
    max_length: int,
    batch_size: int,
    lm_head_chunk_tokens: int,
    limit: int,
    progress_every: int,
    device: torch.device,
) -> None:
    attention_contract = str(
        getattr(
            model.config,
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if attention_contract not in {
        "selfless_strict",
        "xlnet_content_diagonal",
    }:
        raise ValueError(
            "unsupported dual-stream attention contract: "
            f"{attention_contract!r}"
        )
    selected = list(examples[:limit] if limit else examples)
    local_examples = [example for example in selected if example.item_index % world_size == rank]
    shard_path, marker_path = rank_shard_paths(output_dir, task, rank, world_size)
    if completed_rank_shard(
        marker_path,
        shard_path,
        task=task,
        rank=rank,
        world_size=world_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=checkpoint_step,
        expected_records=len(local_examples),
        max_length=max_length,
        limit=limit,
        attention_contract=attention_contract,
    ):
        print(f"[{task}] rank {rank}: reusing complete shard", flush=True)
        return
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for local_offset in range(0, len(local_examples), progress_every):
        example_chunk = local_examples[local_offset : local_offset + progress_every]
        requests: list[ChoiceRequest] = []
        spans: list[tuple[int, int]] = []
        for example in example_chunk:
            start = len(requests)
            requests.extend(
                encode_choice(tokenizer, example, choice_index, max_length)
                for choice_index in range(len(example.choices))
            )
            spans.append((start, len(requests)))
        choice_scores = score_choice_requests(
            model,
            requests,
            batch_size=batch_size,
            lm_head_chunk_tokens=lm_head_chunk_tokens,
            device=device,
        )
        for example, (start, stop) in zip(example_chunk, spans):
            scores = choice_scores[start:stop]
            raw = [score.loglikelihood for score in scores]
            normalized = [score.normalized_loglikelihood for score in scores]
            prediction = argmax(raw)
            normalized_prediction = argmax(normalized)
            example_requests = requests[start:stop]
            rows.append(
                {
                    "schema": "selfless_text_multiple_choice_sample_v2",
                    "protocol_schema": TEXT_PROTOCOL_SCHEMA,
                    "normalization": TEXT_NORMALIZATION,
                    "task": task,
                    "item_index": int(example.item_index),
                    "item_id": example.item_id,
                    "category": example.category,
                    "label": int(example.label),
                    "prediction": prediction,
                    "prediction_normalized": normalized_prediction,
                    "correct": prediction == int(example.label),
                    "correct_normalized": normalized_prediction == int(example.label),
                    "choice_loglikelihoods": raw,
                    "choice_normalized_loglikelihoods": normalized,
                    "choice_token_counts": [score.token_count for score in scores],
                    "choice_char_counts": [
                        request.normalization_char_count for request in example_requests
                    ],
                    "choice_greedy": [score.greedy for score in scores],
                    "boundary_adjusted": any(
                        request.boundary_adjusted for request in example_requests
                    ),
                    "truncated_context_tokens": max(
                        request.truncated_context_tokens for request in example_requests
                    ),
                }
            )
        completed = min(local_offset + len(example_chunk), len(local_examples))
        elapsed = time.monotonic() - started
        print(
            f"[{task}] rank {rank}: {completed}/{len(local_examples)} items "
            f"({completed / max(elapsed, 1e-6):.3f} items/s)",
            flush=True,
        )
    write_rank_shard(
        output_dir,
        task,
        rank,
        world_size,
        checkpoint_dir,
        checkpoint_step,
        rows,
        max_length=max_length,
        limit=limit,
        attention_contract=attention_contract,
        wall_seconds=time.monotonic() - started,
    )


def standard_error(correct: Sequence[bool]) -> float:
    count = len(correct)
    if count <= 1:
        return 0.0
    mean = sum(bool(value) for value in correct) / count
    return math.sqrt(mean * (1.0 - mean) / (count - 1))


def aggregate_mc_task(
    output_dir: Path,
    task: str,
    world_size: int,
    expected_records: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        shard_path, _ = rank_shard_paths(output_dir, task, rank, world_size)
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        rows.extend(
            json.loads(line)
            for line in shard_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return aggregate_mc_rows(output_dir, task, rows, expected_records)


def aggregate_mc_rows(
    output_dir: Path,
    task: str,
    rows: list[dict[str, Any]],
    expected_records: int,
) -> dict[str, Any]:
    """Reduce either freshly inferred or explicitly migrated sample scores."""
    rows.sort(key=lambda row: int(row["item_index"]))
    observed = [int(row["item_index"]) for row in rows]
    if observed != list(range(expected_records)):
        raise ValueError(f"{task} distributed coverage is incomplete or duplicated")
    if any(
        row.get("schema") != "selfless_text_multiple_choice_sample_v2"
        or row.get("protocol_schema") != TEXT_PROTOCOL_SCHEMA
        or row.get("normalization") != TEXT_NORMALIZATION
        or row.get("task") != task
        for row in rows
    ):
        raise ValueError(f"{task} shards use an obsolete or mixed text protocol")
    correct = [bool(row["correct"]) for row in rows]
    correct_normalized = [bool(row["correct_normalized"]) for row in rows]
    metrics: dict[str, Any] = {
        "schema": "selfless_text_multiple_choice_metrics_v2",
        "protocol_schema": TEXT_PROTOCOL_SCHEMA,
        "normalization": TEXT_NORMALIZATION,
        "complete": True,
        "runtime_hashing_enabled": False,
        "task": task,
        "samples": len(rows),
        "accuracy": sum(correct) / len(rows),
        "accuracy_stderr": standard_error(correct),
        "accuracy_normalized": sum(correct_normalized) / len(rows),
        "accuracy_normalized_stderr": standard_error(correct_normalized),
        "boundary_adjusted_samples": sum(bool(row["boundary_adjusted"]) for row in rows),
        "truncated_context_samples": sum(
            int(row["truncated_context_tokens"]) > 0 for row in rows
        ),
        "completed_at": utc_now(),
    }
    categories = sorted(
        {str(row["category"]) for row in rows if row.get("category") is not None}
    )
    if categories:
        by_category = {}
        for category in categories:
            category_rows = [row for row in rows if str(row.get("category")) == category]
            category_correct = [bool(row["correct"]) for row in category_rows]
            category_normalized = [
                bool(row["correct_normalized"]) for row in category_rows
            ]
            by_category[category] = {
                "samples": len(category_rows),
                "accuracy": sum(category_correct) / len(category_rows),
                "accuracy_normalized": sum(category_normalized)
                / len(category_rows),
            }
        metrics["by_category"] = by_category
        metrics["accuracy_macro"] = sum(
            value["accuracy"] for value in by_category.values()
        ) / len(by_category)
        metrics["accuracy_normalized_macro"] = sum(
            value["accuracy_normalized"] for value in by_category.values()
        ) / len(by_category)
    task_dir = output_dir / "tasks" / task
    atomic_write_text(task_dir / "samples.jsonl", jsonl_text(rows))
    atomic_write_text(
        task_dir / "metrics.json",
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
    )
    return metrics


def primary_metric(task: str, metrics: dict[str, Any]) -> float:
    if task in {
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
        "openbookqa",
    }:
        return float(metrics["accuracy_normalized"])
    if task in {"winogrande", "boolq"}:
        return float(metrics["accuracy"])
    if task == "mmlu":
        return float(metrics["accuracy_macro"])
    raise ValueError(task)


def main() -> None:
    args = parse_args()
    source = model_source_from_args(args)
    tasks = validate_args(args, source)
    rank, world_size, _, device = initialize_device(args.device)
    torch.manual_seed(int(args.seed) + rank)
    checkpoint_step = source.global_step
    checkpoint_dir = str(source.path)
    config = OmegaConf.load(args.config)
    if bool(config.training.get("runtime_hashing_enabled", True)):
        raise ValueError("pure-text evaluation requires runtime_hashing_enabled=false")
    config.training.runtime_hashing_enabled = False
    configure_model_source(config, source)
    attention_contract = str(
        config.model.get(
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if attention_contract not in {
        "selfless_strict",
        "xlnet_content_diagonal",
    }:
        raise ValueError(
            "unsupported dual-stream attention contract: "
            f"{attention_contract!r}"
        )
    model_dtype = torch.bfloat16 if args.model_dtype == "bf16" else torch.float32
    model, tokenizer = load_model_tokenizer(config, model_dtype=model_dtype)
    loaded_attention_contract = str(
        getattr(
            model.config,
            "dual_stream_attention_contract",
            "selfless_strict",
        )
    ).strip().lower()
    if loaded_attention_contract != attention_contract:
        raise ValueError(
            "loaded model attention contract does not match evaluation config: "
            f"model={loaded_attention_contract!r}, config={attention_contract!r}"
        )
    ema_report = load_model_source_weights(model, source)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        run = {
            "schema": "selfless_text_benchmark_run_v4",
            "complete": False,
            "runtime_hashing_enabled": False,
            "checkpoint": str(source.path),
            "checkpoint_step": checkpoint_step,
            "weight_source": source.kind,
            "model_source": str(Path(config.model.model_path).resolve()),
            "ema": ema_report,
            "data_root": str(args.data_root.resolve()),
            "tasks": list(tasks),
            "world_size": world_size,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "dual_stream_attention_contract": attention_contract,
            "scoring_contract": TEXT_SCORING_CONTRACT,
            "protocol_schema": TEXT_PROTOCOL_SCHEMA,
            "max_length": args.max_length,
            "limit": args.limit,
            "seed": args.seed,
            "contamination_check": "disabled",
            "lm_eval_reference": LM_EVAL_REFERENCE,
            "task_protocols": {task: TASK_PROTOCOLS[task] for task in tasks},
            "started_at": utc_now(),
        }
        atomic_write_text(
            args.output_dir / "evaluation_run.json",
            json.dumps(run, indent=2, sort_keys=True) + "\n",
        )
    barrier(device)
    task_sizes: dict[str, int] = {}
    for task in tasks:
        examples = load_multiple_choice_task(task, args.data_root)
        task_sizes[task] = min(len(examples), args.limit) if args.limit else len(examples)
        evaluate_multiple_choice_task(
            model,
            tokenizer,
            examples,
            task=task,
            output_dir=args.output_dir,
            rank=rank,
            world_size=world_size,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=checkpoint_step,
            max_length=args.max_length,
            batch_size=args.batch_size_per_rank,
            lm_head_chunk_tokens=args.lm_head_chunk_tokens,
            limit=args.limit,
            progress_every=args.progress_every,
            device=device,
        )
        barrier(device)
    if rank == 0:
        task_metrics: dict[str, dict[str, Any]] = {}
        for task in tasks:
            task_metrics[task] = aggregate_mc_task(
                args.output_dir, task, world_size, task_sizes[task]
            )
        primary = {
            task: primary_metric(task, metrics)
            for task, metrics in task_metrics.items()
        }
        summary = {
            "schema": "selfless_text_benchmark_summary_v4",
            "complete": True,
            "runtime_hashing_enabled": False,
            "checkpoint": str(source.path),
            "checkpoint_step": checkpoint_step,
            "weight_source": source.kind,
            "world_size": world_size,
            "accuracy_unit": "unit_interval",
            "tasks": task_metrics,
            "primary_metrics": primary,
            "macro_average_primary": sum(primary.values()) / len(primary),
            "macro_average_role": "internal_cross_task_summary_only",
            "protocol": {
                "normalization": TEXT_NORMALIZATION,
                "winogrande_scoring": WINOGRANDE_SCORING,
                "scoring_contract": TEXT_SCORING_CONTRACT,
                "protocol_schema": TEXT_PROTOCOL_SCHEMA,
                "selfless_query_stream_same_position_scoring": True,
                "dual_stream_attention_contract": attention_contract,
                "query_stream_diagonal": False,
                "content_stream_diagonal": (
                    attention_contract == "xlnet_content_diagonal"
                ),
                "max_length": args.max_length,
                "mmlu_fewshot": 5,
                "fewshot_sampler": "first_n",
                "contamination_check": "disabled",
                "lm_eval_reference": LM_EVAL_REFERENCE,
                "task_protocols": {
                    task: TASK_PROTOCOLS[task] for task in tasks
                },
            },
            "completed_at": utc_now(),
        }
        atomic_write_text(
            args.output_dir / "summary.json",
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
        run_path = args.output_dir / "evaluation_run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["complete"] = True
        run["completed_at"] = utc_now()
        run["summary"] = str((args.output_dir / "summary.json").resolve())
        atomic_write_text(run_path, json.dumps(run, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    barrier(device)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
