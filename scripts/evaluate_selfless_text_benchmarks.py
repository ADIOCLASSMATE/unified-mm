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


from utils.evaluation.text_benchmarks import (  # noqa: F401 -- compatibility exports
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_TASKS,
    MC_TASKS,
    CHOICE_LETTERS,
    TEXT_SCORING_CONTRACT,
    TEXT_PROTOCOL_SCHEMA,
    TEXT_NORMALIZATION,
    WINOGRANDE_SCORING,
    TEXT_ASSET_SCHEMA,
    LM_EVAL_REFERENCE,
    TASK_PROTOCOLS,
    MultipleChoiceExample,
    ChoiceRequest,
    ChoiceScore,
    utc_now,
    atomic_write_text,
    jsonl_text,
    initialize_device,
    barrier,
    read_checkpoint_step,
    validate_args,
    read_parquet,
    answer_index,
    preprocess_hellaswag,
    load_arc,
    load_hellaswag,
    load_piqa,
    load_winogrande,
    load_boolq,
    load_openbookqa,
    format_mmlu_question,
    load_mmlu,
    load_multiple_choice_task,
    common_prefix_length,
    encode_choice,
    score_choice_requests,
    argmax,
    rank_shard_paths,
    completed_rank_shard,
    write_rank_shard,
    evaluate_multiple_choice_task,
    standard_error,
    aggregate_mc_task,
    aggregate_mc_rows,
    primary_metric,
)


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
