#!/usr/bin/env python3
"""Read-only preflight for the 100B unified baseline (no hashing)."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from utils.climbmix_online_dataset import ClimbMixOnlineBatchDataset
from utils.combined_dataloaders import BASELINE_SOURCE_SCHEDULE
from utils.selfless_training_runtime import validate_wsd_contract


def _require_file(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _manifest_split(row: dict) -> str:
    split = str(row.get("split", "")).strip().lower()
    if split:
        return split
    parts = {part.lower() for part in Path(str(row["source_path"])).parts}
    if "train" in parts:
        return "train"
    if "val" in parts:
        return "val"
    return ""


def _audit_imagenet_manifest(
    path: Path,
    *,
    expected_split: str,
    expected_records: int,
    forbidden_identities: set[tuple[str, str]] | None = None,
    retain_identities: bool = False,
) -> dict[str, object]:
    identities: set[tuple[str, str]] = set()
    synsets: set[str] = set()
    records = 0
    with path.open(encoding="utf-8") as handle:
        for records, line in enumerate(handle, start=1):
            row = json.loads(line)
            if int(row.get("img_id", -1)) != records:
                raise ValueError(
                    f"{path}:{records} img_id is not contiguous from one"
                )
            split = _manifest_split(row)
            if split != expected_split:
                raise ValueError(
                    f"{path}:{records} split={split!r}, "
                    f"expected={expected_split!r}"
                )
            synset = str(row.get("synset", ""))
            source_path = str(row.get("source_path", ""))
            if not synset or not source_path:
                raise ValueError(f"{path}:{records} has incomplete identity")
            identity = (synset, Path(source_path).name)
            if forbidden_identities and identity in forbidden_identities:
                raise ValueError(
                    "ImageNet train/val image identity overlap: "
                    f"{identity!r}"
                )
            if retain_identities:
                if identity in identities:
                    raise ValueError(f"duplicate image identity in {path}: {identity}")
                identities.add(identity)
            synsets.add(synset)
    if records != expected_records or len(synsets) != 1_000:
        raise ValueError(
            f"{path} count/class mismatch: records={records}, "
            f"classes={len(synsets)}"
        )
    return {
        "split": expected_split,
        "records": records,
        "classes": len(synsets),
        "identities": identities,
    }


def _load_image_cache(path: Path, *, expected_records: int):
    cache = torch.load(
        str(path), map_location="cpu", mmap=True, weights_only=True
    )
    posterior = cache.get("posterior_stats")
    image_ids = cache.get("img_ids")
    if (
        posterior is None
        or tuple(posterior.shape) != (expected_records, 256, 32)
        or image_ids is None
        or not torch.equal(
            image_ids,
            torch.arange(1, expected_records + 1, dtype=torch.int64),
        )
    ):
        raise ValueError(
            f"ImageNet cache contract mismatch: {path}, "
            f"shape={getattr(posterior, 'shape', None)}"
        )
    return cache


def _validate_text_index(path: Path, *, split: str, records: int) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split") != split or int(manifest.get("records", -1)) != records:
        raise ValueError(
            f"synthetic text index contract mismatch: {path}, "
            f"split={manifest.get('split')!r}, records={manifest.get('records')!r}"
        )
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--formal-world-size", type=int, default=64)
    parser.add_argument("--require-npu-count", type=int, default=0)
    parser.add_argument("--tokenizer-probe", action="store_true")
    parser.add_argument("--run-project")
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--flow-lr", type=float)
    parser.add_argument("--save-ema-eval-every", type=int)
    parser.add_argument("--ablation", choices=("b", "c", "d"), default="b")
    parser.add_argument(
        "--dynamic-xt-t2i-gradient-checkpointing",
        choices=("true", "false"),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = OmegaConf.load(args.config)
    validate_wsd_contract(config)
    if str(config.dataset.class_name) != "UnifiedMixedDataset":
        raise ValueError("dataset.class_name must be UnifiedMixedDataset")
    schedule = tuple(str(name) for name in config.dataset.params.schedule)
    if schedule != BASELINE_SOURCE_SCHEDULE:
        raise ValueError(
            f"source schedule must be {BASELINE_SOURCE_SCHEDULE}, got {schedule}"
        )
    if bool(config.training.get("runtime_hashing_enabled", True)):
        raise ValueError("runtime_hashing_enabled must be false")
    image = config.dataset.params.image
    if image.get("validation", None) is None:
        raise ValueError("dataset.params.image.validation is required")
    packing = config.dataset.params.image.get("packing", None)
    if packing is not None and bool(packing.get("audit_manifests", False)):
        raise ValueError("packing audit manifests must be disabled")

    sources = config.dataset.params.sources
    world_size = int(args.formal_world_size)
    if world_size not in {64, 128}:
        raise ValueError(
            "the unified formal token contract supports 64 or 128 NPUs, "
            f"got {world_size}"
        )
    if int(config.training.gradient_accumulation_steps) != len(schedule):
        raise ValueError("gradient_accumulation_steps must equal schedule length")
    text_batch = int(sources.climbmix.micro_batch_size)
    text_length = int(sources.climbmix.sequence_length)
    nominal_targets_per_step = (
        schedule.count("climbmix")
        * text_batch
        * world_size
        * (text_length - 1)
    )
    target_tokens = int(config.training.target_text_tokens)
    expected_steps = math.ceil(target_tokens / nominal_targets_per_step)
    if int(config.training.max_train_steps) != expected_steps:
        raise ValueError(
            f"100B step contract mismatch: {config.training.max_train_steps} "
            f"!= {expected_steps}"
        )
    nominal_field = f"nominal_text_targets_per_step_{world_size}npu"
    if nominal_field not in config.training:
        raise ValueError(f"missing training.{nominal_field}")
    if int(config.training[nominal_field]) != nominal_targets_per_step:
        raise ValueError(f"{nominal_field} mismatch")
    rows_per_step = sum(
        int(sources[name].micro_batch_size) for name in schedule
    ) * world_size
    if int(config.training.total_batch_size) != rows_per_step:
        raise ValueError("total physical-row batch contract mismatch")
    if bool(config.training.from_scratch):
        raise ValueError("baseline must initialize from Qwen pretrained weights")
    if str(config.model.backbone_attention_output_gate) != "none":
        raise ValueError("baseline backbone attention output gate must be none")
    if str(config.model.architecture_variant) != "selfless_contextual":
        raise ValueError("base config must use the baseline-b architecture")
    if str(config.model.training_objective) != "selfless_dual_stream":
        raise ValueError("base config must use the baseline-b training objective")
    if (
        str(config.model.dual_stream_attention_contract)
        != "xlnet_content_diagonal"
    ):
        raise ValueError(
            "base config must use baseline b's xlnet_content_diagonal contract"
        )
    if int(config.model.image_flow_batch_mul) != 4:
        raise ValueError(
            "unified B-based ablations require model.image_flow_batch_mul=4"
        )
    ablation_contracts = {
        "b": (
            "selfless_contextual",
            "selfless_dual_stream",
            "xlnet_content_diagonal",
        ),
        "c": (
            "single_stream_text_ar",
            "selfless_dual_stream",
            "xlnet_content_diagonal",
        ),
        "d": (
            "dynamic_xt",
            "selfless_dual_stream",
            "xlnet_content_diagonal",
        ),
    }
    architecture_variant, training_objective, attention_contract = ablation_contracts[
        args.ablation
    ]
    dynamic_xt_t2i_gradient_checkpointing = (
        args.dynamic_xt_t2i_gradient_checkpointing == "true"
        if args.dynamic_xt_t2i_gradient_checkpointing is not None
        else bool(
            config.model.get(
                "dynamic_xt_t2i_gradient_checkpointing",
                False,
            )
        )
    )
    if bool(config.training.get("use_gradient_checkpointing", False)):
        raise ValueError(
            "unified 0.6B training keeps global gradient checkpointing off"
        )
    if args.ablation == "d" and not dynamic_xt_t2i_gradient_checkpointing:
        raise ValueError(
            "ablation D requires T2I-only Dynamic-XT activation "
            "checkpointing for the formal B16 x image_flow_batch_mul=4 shape"
        )
    if args.ablation != "d" and dynamic_xt_t2i_gradient_checkpointing:
        raise ValueError(
            "T2I-only Dynamic-XT checkpointing is valid only for ablation D"
        )
    backbone_lr = float(
        args.backbone_lr
        if args.backbone_lr is not None
        else config.optimizer.params.backbone_learning_rate
    )
    flow_lr = float(
        args.flow_lr
        if args.flow_lr is not None
        else config.optimizer.params.flow_learning_rate
    )
    if backbone_lr <= 0.0 or flow_lr <= 0.0:
        raise ValueError("backbone and flow learning rates must be positive")

    model_path = Path(config.model.model_path)
    _require_file(str(model_path / "config.json"), "Qwen config")
    _require_file(str(model_path / "model.safetensors"), "Qwen weights")
    _require_file(str(model_path / "tokenizer.json"), "Qwen tokenizer")
    source_model_config = AutoConfig.from_pretrained(
        model_path, local_files_only=True
    )
    if str(source_model_config.model_type) != "qwen3":
        raise ValueError(
            f"expected qwen3 initialization, got {source_model_config.model_type}"
        )

    if str(image.get("image_sigma_order", "")).lower() != "random":
        raise ValueError(
            "baseline b requires random image reveal order"
        )
    if str(image.get("expected_split", "")) != "train":
        raise ValueError("training image dataset must set expected_split=train")
    if int(image.get("expected_records", -1)) != 1_281_167:
        raise ValueError("training image dataset must contain 1,281,167 rows")

    image_steps_per_epoch = {}
    for source_name in ("t2i", "i2t"):
        occurrences = schedule.count(source_name)
        if occurrences != 1:
            raise ValueError(
                "the formal image-epoch contract requires exactly one "
                f"{source_name} microbatch per optimizer step"
            )
        global_images_per_step = (
            int(sources[source_name].micro_batch_size)
            * world_size
            * occurrences
        )
        image_steps_per_epoch[source_name] = (
            int(image.expected_records) // global_images_per_step
        )
    if len(set(image_steps_per_epoch.values())) != 1:
        raise ValueError(
            "T2I/I2T image epochs do not share one optimizer-step cadence: "
            f"{image_steps_per_epoch}"
        )
    optimizer_steps_per_image_epoch = next(
        iter(image_steps_per_epoch.values())
    )
    expected_ema_eval_every = 12_510
    if int(config.experiment.save_ema_eval_every) != expected_ema_eval_every:
        raise ValueError(
            "config must use the fixed paired-model evaluation export cadence: "
            f"save_ema_eval_every={config.experiment.save_ema_eval_every}, "
            f"expected={expected_ema_eval_every}"
        )
    if not bool(config.experiment.get("save_model_with_ema_eval", False)):
        raise ValueError(
            "unified periodic evaluation must export both the current model "
            "and the EMA model"
        )
    runtime_ema_eval_every = int(
        config.experiment.save_ema_eval_every
        if args.save_ema_eval_every is None
        else args.save_ema_eval_every
    )
    if runtime_ema_eval_every not in {0, expected_ema_eval_every}:
        raise ValueError(
            "runtime paired-model evaluation export cadence must be disabled "
            f"or equal {expected_ema_eval_every}, got "
            f"{runtime_ema_eval_every}"
        )
    if str(config.experiment.get("ema_eval_dtype", "")).lower() != "bf16":
        raise ValueError("complete periodic EMA exports must use bf16")
    validation_image = image.validation
    if str(validation_image.get("expected_split", "")) != "val":
        raise ValueError("validation image dataset must set expected_split=val")
    if int(validation_image.get("expected_records", -1)) != 50_000:
        raise ValueError("validation image dataset must contain 50,000 rows")

    cache_path = _require_file(image.cache_path, "ImageNet train KL16 cache")
    train_manifest_path = _require_file(
        image.manifest_jsonl, "ImageNet train manifest"
    )
    _require_file(image.caption_jsonl, "ImageNet train captions")
    train_index_path = _require_file(
        image.synthetic_text_index_manifest,
        "ImageNet train synthetic-text seek index",
    )
    val_cache_path = _require_file(
        validation_image.cache_path, "ImageNet val KL16 cache"
    )
    val_manifest_path = _require_file(
        validation_image.manifest_jsonl, "ImageNet val manifest"
    )
    _require_file(validation_image.caption_jsonl, "ImageNet val captions")
    val_index_path = _require_file(
        validation_image.synthetic_text_index_manifest,
        "ImageNet val synthetic-text seek index",
    )
    train_cache = _load_image_cache(cache_path, expected_records=1_281_167)
    val_cache = _load_image_cache(val_cache_path, expected_records=50_000)
    if val_cache.get("metadata", {}).get("runtime_hashing_enabled") is not False:
        raise ValueError(
            "ImageNet val cache must be prepared under the no-hash contract"
        )
    _validate_text_index(train_index_path, split="train", records=1_281_167)
    val_text_index = _validate_text_index(
        val_index_path, split="val", records=50_000
    )
    if val_text_index.get("runtime_hashing_enabled") is not False:
        raise ValueError("ImageNet val text index must be no-hash")
    val_manifest = _audit_imagenet_manifest(
        val_manifest_path,
        expected_split="val",
        expected_records=50_000,
        retain_identities=True,
    )
    train_manifest = _audit_imagenet_manifest(
        train_manifest_path,
        expected_split="train",
        expected_records=1_281_167,
        forbidden_identities=val_manifest["identities"],
    )

    shard_paths = sorted(glob.glob(str(sources.climbmix.shard_glob)))
    if len(shard_paths) != 100:
        raise ValueError(f"expected 100 ClimbMix shards, got {len(shard_paths)}")
    for path in shard_paths:
        _require_file(path, "ClimbMix shard")

    tokenizer_probe = None
    if args.tokenizer_probe:
        tokenizer = AutoTokenizer.from_pretrained(
            sources.climbmix.tokenizer_path,
            fix_mistral_regex=True,
            local_files_only=True,
        )
        dataset = ClimbMixOnlineBatchDataset(
            shard_paths=shard_paths,
            tokenizer=tokenizer,
            eos_token_id=int(tokenizer.eos_token_id),
            sequence_length=text_length,
            micro_batch_size=text_batch,
            rank=0,
            world_size=world_size,
            seed=int(config.training.seed),
            tokenizer_batch_documents=int(
                sources.climbmix.tokenizer_batch_documents
            ),
            max_document_chars=int(sources.climbmix.max_document_chars),
            rayon_num_threads=int(sources.climbmix.rayon_num_threads),
        )
        batch = next(iter(dataset))
        tokenizer_probe = {
            "shape": list(batch["input_ids"].shape),
            "valid_tokens": int(batch["pack_stats"][0]),
            "supervised_tokens": int(batch["supervised_text_tokens"]),
            "eos_token_id": int(tokenizer.eos_token_id),
        }

    npu = None
    if args.require_npu_count:
        import torch_npu  # noqa: F401

        count = int(torch.npu.device_count())
        available = bool(torch.npu.is_available())
        if not available or count != int(args.require_npu_count):
            raise RuntimeError(
                f"expected {args.require_npu_count} NPUs, "
                f"got available={available}, count={count}"
            )
        npu = {"available": available, "count": count}

    report = {
        "schema": "unified_baseline_preflight_v1",
        "config": str(Path(args.config)),
        "runtime_hashing_enabled": False,
        "initialization": {
            "model_type": source_model_config.model_type,
            "model_path": str(model_path),
            "from_scratch": False,
        },
        "run_project": args.run_project,
        "ablation": {
            "id": args.ablation,
            "architecture_variant": architecture_variant,
            "training_objective": training_objective,
            "dual_stream_attention_contract": attention_contract,
            "text_prediction": (
                "causal_next_token_shift" if args.ablation == "c" else "same_position"
            ),
            "image_path": (
                "identical_to_b"
                if args.ablation == "c"
                else "dynamic_xt_on_b"
                if args.ablation == "d"
                else None
            ),
            "image_flow_batch_mul": int(config.model.image_flow_batch_mul),
            "backbone_condition": (
                "dynamic_xt_every_ode_evaluation"
                if args.ablation == "d"
                else "static_semantic"
            ),
            "dynamic_query_scope": (
                "t2i_only" if args.ablation == "d" else None
            ),
            "global_gradient_checkpointing": False,
            "dynamic_xt_t2i_gradient_checkpointing": (
                dynamic_xt_t2i_gradient_checkpointing
            ),
        },
        "optimizer": {
            "backbone_and_special_lr": backbone_lr,
            "flow_and_projector_lr": flow_lr,
        },
        "schedule": list(schedule),
        "formal_world_size": world_size,
        "gradient_accumulation_steps": len(schedule),
        "text": {
            "sequence_length": text_length,
            "micro_batch_size_per_rank": text_batch,
            "shards": len(shard_paths),
            "target_tokens": target_tokens,
            "nominal_targets_per_step": nominal_targets_per_step,
            "max_train_steps": expected_steps,
            "tokenizer_probe": tokenizer_probe,
        },
        "image": {
            "train": {
                "split": train_manifest["split"],
                "cache_rows": int(train_cache["posterior_stats"].shape[0]),
                "cache_shape": list(train_cache["posterior_stats"].shape),
                "manifest_records": train_manifest["records"],
            },
            "validation": {
                "split": val_manifest["split"],
                "cache_rows": int(val_cache["posterior_stats"].shape[0]),
                "cache_shape": list(val_cache["posterior_stats"].shape),
                "manifest_records": val_manifest["records"],
                "paired_tasks_per_image": ["t2i", "i2t"],
                "runtime_hashing_enabled": False,
            },
            "t2i_micro_batch_size_per_rank": int(
                sources.t2i.micro_batch_size
            ),
            "i2t_micro_batch_size_per_rank": int(
                sources.i2t.micro_batch_size
            ),
            "image_sigma_order": str(image.image_sigma_order),
            "optimizer_steps_per_epoch": optimizer_steps_per_image_epoch,
        },
        "ema_evaluation_export": {
            "enabled": runtime_ema_eval_every > 0,
            "every_optimizer_steps": runtime_ema_eval_every,
            "dtype": "bf16",
            "artifacts": ["current_model", "ema_model", "pair_manifest"],
        },
        "npu": npu,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
