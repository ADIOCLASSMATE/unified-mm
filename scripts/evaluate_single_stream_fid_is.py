#!/usr/bin/env python3
import argparse
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.image_evaluation_metrics import (  # noqa: E402
    FeatureMoments,
    InceptionScoreMoments,
    build_inception_extractor,
    extract_inception_features,
    frechet_distance,
    metric_accumulation_dtype,
)
from scripts.generate_flow_validation_images import (  # noqa: E402
    decode_latents,
    load_adapter,
    load_sharded_ema_checkpoint,
    load_model_state,
    load_vae,
)
from models.modeling_model.image_backbone import (  # noqa: E402
    CANONICAL_IMAGE_GRID_SIDE,
    CANONICAL_IMAGE_LATENT_DIM,
    pure_2d_position_contract,
)
from utils.dataset_utils import get_dataloaders  # noqa: E402
from utils.utils import load_model_tokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INCEPTION_WEIGHTS = (
    REPO_ROOT
    / "output"
    / "cache"
    / "inception"
    / "weights-inception-2015-12-05-6726825d.pth"
)

EVALUATOR_RNG_SEED_MODULUS = 2**63
DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS = 2 * 60 * 60
CANONICAL_NOISE_MANIFEST_SCHEMA = "canonical_image_flow_noise_manifest_v1"
ORDERED_EVAL_SAMPLE_MANIFEST_SCHEMA = "ordered_image_flow_eval_samples_v1"
EVALUATION_PROGRESS_SCHEMA = "single_stream_fid_is_progress_v1"
EVALUATION_RESUME_SCHEMA = "single_stream_fid_is_resume_v1"
EVALUATION_RESUME_COMMIT_SCHEMA = "single_stream_fid_is_resume_commit_v1"
DEFAULT_PROGRESS_LOG_INTERVAL_SAMPLES = 250
DEFAULT_PROGRESS_LOG_INTERVAL_SECONDS = 60.0


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


EVALUATOR_RNG_CONTRACT = {
    "schema": "canonical_image_flow_initial_noise_v1",
    "canonical_shape": [
        CANONICAL_IMAGE_GRID_SIDE,
        CANONICAL_IMAGE_GRID_SIDE,
        CANONICAL_IMAGE_LATENT_DIM,
    ],
    "canonical_layout": "HWC",
    "distribution": "torch.randn_standard_normal",
    "dtype": "torch.float32",
    "device": "cpu",
    "generator": "torch.Generator(device='cpu')",
    "seed_derivation": {
        "formula": "(evaluation_seed + global_sample_index) mod 2**63",
        "modulus": EVALUATOR_RNG_SEED_MODULUS,
    },
    "flattening": "row_major_[16,16,16]_to_[256,16]",
    "temperature_application": (
        "FlowLoss.sample multiplies validated packed initial_noise by temperature"
    ),
    "independence": [
        "architecture",
        "batch_partition",
        "distributed_rank",
        "strategy",
    ],
    "per_sample_digest": (
        "sha256(canonical tensor header JSON + newline + contiguous C-order bytes)"
    ),
    "noise_manifest_schema": CANONICAL_NOISE_MANIFEST_SCHEMA,
    "sample_manifest_schema": ORDERED_EVAL_SAMPLE_MANIFEST_SCHEMA,
}
EVALUATOR_RNG_CONTRACT_SHA256 = canonical_json_sha256(EVALUATOR_RNG_CONTRACT)


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().to(device="cpu").contiguous()
    header = {
        "dtype": str(tensor.dtype),
        "shape": [int(value) for value in tensor.shape],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    digest.update(b"\n")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_image_flow_initial_noise(
    evaluation_seed: int,
    global_sample_index: int,
) -> torch.Tensor:
    global_sample_index = int(global_sample_index)
    if global_sample_index < 0:
        raise ValueError(
            f"global_sample_index must be nonnegative, got {global_sample_index}"
        )
    sample_seed = (
        int(evaluation_seed) + global_sample_index
    ) % EVALUATOR_RNG_SEED_MODULUS
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sample_seed)
    return torch.randn(
        (
            CANONICAL_IMAGE_GRID_SIDE,
            CANONICAL_IMAGE_GRID_SIDE,
            CANONICAL_IMAGE_LATENT_DIM,
        ),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )


def build_canonical_initial_noise_bank(
    global_sample_indices: list[int],
    *,
    evaluation_seed: int,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    if not global_sample_indices:
        raise ValueError("global_sample_indices must be nonempty")
    normalized_indices = [int(value) for value in global_sample_indices]
    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError(
            f"global_sample_indices must be unique, got {normalized_indices}"
        )
    canonical = torch.stack(
        [
            canonical_image_flow_initial_noise(evaluation_seed, global_index)
            for global_index in normalized_indices
        ],
        dim=0,
    )
    records = [
        {
            "global_sample_index": global_index,
            "canonical_noise_sha256": canonical_tensor_sha256(noise),
        }
        for global_index, noise in zip(normalized_indices, canonical)
    ]
    flattened = canonical.reshape(
        int(canonical.shape[0]),
        CANONICAL_IMAGE_GRID_SIDE * CANONICAL_IMAGE_GRID_SIDE,
        CANONICAL_IMAGE_LATENT_DIM,
    )
    return flattened, records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline FID/IS evaluation for selfless single-stream image generation."
    )
    parser.add_argument(
        "--config",
        default="configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml",
    )
    parser.add_argument("--model_path_override", default="")
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--model_state", default="")
    parser.add_argument("--ema_checkpoint", default="")
    parser.add_argument("--output_dir", default="output/single_stream_fid_is")
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--model_dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Floating-point dtype used to load and execute the generation model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help=(
            "Global generation batch. The Ascend default is 256 samples per "
            "rank on 16 ranks (4096 total); sharding happens before "
            "dataset collation."
        ),
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--sampling_steps", default="10")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--cfg_schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--flow_solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--parallel_rate", type=int, default=1)
    parser.add_argument("--strategies", default="spatial_halton")
    parser.add_argument(
        "--disable_backbone_kv_cache",
        action="store_true",
        help=(
            "Disable incremental Qwen backbone KV caching for numerical A/B "
            "checks. Fixed-order image evaluation enables it by default."
        ),
    )
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--vae_decode_batch_size",
        type=int,
        default=0,
        help=(
            "Per-rank VAE decode microbatch. Zero decodes the whole local "
            "generation batch at once."
        ),
    )
    parser.add_argument("--fid_feature", type=int, default=2048)
    parser.add_argument("--is_splits", type=int, default=10)
    parser.add_argument(
        "--inception_weights_path",
        default=os.environ.get(
            "TORCH_FIDELITY_INCEPTION_WEIGHTS",
            str(DEFAULT_INCEPTION_WEIGHTS) if DEFAULT_INCEPTION_WEIGHTS.exists() else "",
        ),
        help="Optional local torch-fidelity InceptionV3 weights path for FID/IS.",
    )
    parser.add_argument(
        "--real_source",
        choices=["vae_decoded_target_latents", "imagenet_original"],
        default="vae_decoded_target_latents",
        help=(
            "Reference distribution for FID. The default compares against VAE-decoded target latents; "
            "imagenet_original loads original ImageNet files from the manifest/source_path mapping."
        ),
    )
    parser.add_argument(
        "--real_stats_path",
        default="",
        help=(
            "Precomputed original-ImageNet Inception moments shared across architectures. "
            "When set, this overrides --real_source and real images are not re-extracted. "
            "Pass 'none' to explicitly disable a path inherited from the config."
        ),
    )
    parser.add_argument(
        "--imagenet_train_dir",
        default="/inspire/dataset/imagenet/v1/ILSVRC/Data/CLS-LOC/train",
        help="Root used to resolve manifest source paths when --real_source=imagenet_original.",
    )
    parser.add_argument(
        "--real_image_size",
        type=int,
        default=256,
        help="Resize/center-crop size for original ImageNet real images.",
    )
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument(
        "--skip_target_decode",
        action="store_true",
        help=(
            "Skip decoding target latents when frozen real-image statistics "
            "already provide the metric reference."
        ),
    )
    parser.add_argument(
        "--allow_sigma_strategies",
        action="store_true",
        help="Allow sigma/sigma_replay strategies. Disabled by default because real generation cannot know training sigma.",
    )
    parser.add_argument(
        "--require_official_protocol",
        action="store_true",
        help=(
            "Fail unless shared real stats, the full matching fake sample count, "
            "and 10 deterministic IS splits are used."
        ),
    )
    parser.add_argument(
        "--allow_nonofficial_fid",
        action="store_true",
        help=(
            "Compute a diagnostic FID against frozen real statistics even "
            "when the generated sample count does not match the real-stat "
            "count. The result is not an official comparable FID."
        ),
    )
    parser.add_argument(
        "--canonical_pairing",
        action="store_true",
        help=(
            "Use per-global-sample canonical CPU initial noise so paired "
            "architecture evaluations receive identical noise independently "
            "of rank and batch partition."
        ),
    )
    parser.add_argument(
        "--debug_finite_generation",
        action="store_true",
        help="Debug only: check backbone, CFG conditions, context, and every flow ODE step for non-finite values.",
    )
    parser.add_argument(
        "--progress_log_interval_samples",
        type=int,
        default=DEFAULT_PROGRESS_LOG_INTERVAL_SAMPLES,
        help=(
            "Emit a newline evaluation-progress log after at least this many "
            "additional global samples."
        ),
    )
    parser.add_argument(
        "--progress_log_interval_seconds",
        type=float,
        default=DEFAULT_PROGRESS_LOG_INTERVAL_SECONDS,
        help=(
            "Emit a newline evaluation-progress log after this many seconds "
            "even if the sample interval has not been reached."
        ),
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help=(
            "Disable tqdm and newline progress logs. The atomic "
            "evaluation_progress.json file is still maintained."
        ),
    )
    parser.add_argument(
        "--resume_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Atomically checkpoint per-rank metric state and resume from the "
            "last globally committed batch after interruption."
        ),
    )
    parser.add_argument(
        "--resume_checkpoint_interval_batches",
        type=int,
        default=1,
        help="Commit resumable evaluator state every N completed batches.",
    )
    return parser.parse_args()


def per_rank_batch_size(global_batch_size: int, world_size: int) -> int:
    global_batch_size = int(global_batch_size)
    world_size = int(world_size)
    if global_batch_size <= 0:
        raise ValueError(
            f"--batch_size must be positive, got {global_batch_size}"
        )
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if global_batch_size % world_size:
        raise ValueError(
            "--batch_size is the global pre-sharding batch and must be "
            f"divisible by world_size; got {global_batch_size} and "
            f"{world_size}."
        )
    return global_batch_size // world_size


class GlobalBatchStrideSampler:
    """Shard every global evaluation batch before dataset decoding/collation."""

    def __init__(
        self,
        *,
        samples: int,
        global_batch_size: int,
        rank: int,
        world_size: int,
    ):
        self.samples = int(samples)
        self.global_batch_size = int(global_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        per_rank_batch_size(self.global_batch_size, self.world_size)
        if self.samples <= 0:
            raise ValueError(f"samples must be positive, got {self.samples}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}"
            )
        if self.samples % self.world_size:
            raise ValueError(
                "distributed evaluation requires --samples divisible by "
                f"world_size; got {self.samples} and {self.world_size}"
            )

    def __iter__(self):
        for start in range(0, self.samples, self.global_batch_size):
            end = min(start + self.global_batch_size, self.samples)
            yield list(range(start + self.rank, end, self.world_size))

    def __len__(self) -> int:
        return math.ceil(self.samples / self.global_batch_size)


def evaluation_process_group_timeout_seconds() -> int:
    timeout_seconds = int(
        os.environ.get(
            "EVAL_PROCESS_GROUP_TIMEOUT_SECONDS",
            DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS,
        )
    )
    if timeout_seconds <= 0:
        raise ValueError(
            "EVAL_PROCESS_GROUP_TIMEOUT_SECONDS must be positive, "
            f"got {timeout_seconds}"
        )
    return timeout_seconds


def npu_is_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def init_distributed(requested_device: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    distributed_env = {
        key: os.environ.get(key)
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        if os.environ.get(key) is not None
    }
    if not distributed and (rank != 0 or local_rank != 0):
        raise RuntimeError(
            "Inconsistent distributed environment: got nonzero RANK/LOCAL_RANK "
            f"but WORLD_SIZE={world_size}. Launch with torchrun/torch.distributed.run "
            f"so every rank receives WORLD_SIZE, or unset rank env vars. env={distributed_env}"
        )

    requested = str(requested_device).lower().strip()
    wants_npu = requested == "npu" or requested.startswith("npu:")
    wants_cuda = requested == "cuda" or requested.startswith("cuda:")
    if requested not in {"auto", "cpu"} and not (wants_npu or wants_cuda):
        raise ValueError(
            "--device must be one of auto, cpu, cuda[:index], or "
            f"npu[:index], got {requested_device!r}"
        )

    npu_available = npu_is_available() if requested in {"auto", "npu"} or wants_npu else False
    if wants_npu and not npu_available:
        raise RuntimeError("--device requests Ascend NPU, but torch_npu/NPU is unavailable")
    if wants_cuda and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA, but CUDA is unavailable")

    if npu_available and (wants_npu or requested == "auto"):
        if distributed or requested in {"auto", "npu"}:
            device = torch.device(f"npu:{local_rank}")
        else:
            device = torch.device(requested_device)
        torch.npu.set_device(device)
    elif torch.cuda.is_available() and (wants_cuda or requested == "auto"):
        if distributed or requested in {"auto", "cuda"}:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(requested_device)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        backend = {
            "cuda": "nccl",
            "npu": "hccl",
        }.get(device.type, "gloo")
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(
                seconds=evaluation_process_group_timeout_seconds()
            ),
        )
    return distributed, rank, world_size, local_rank, device


def token_type_debug(token_types: torch.Tensor, image_tokens_per_img: int) -> dict:
    token_types_cpu = token_types.detach().cpu()
    image_counts = (token_types_cpu == 1).sum(dim=1).tolist()
    unique_values, unique_counts = torch.unique(token_types_cpu, return_counts=True)
    run_lengths = []
    for row in token_types_cpu:
        pos = 0
        while pos < int(row.numel()):
            if int(row[pos].item()) != 1:
                pos += 1
                continue
            start = pos
            while pos < int(row.numel()) and int(row[pos].item()) == 1:
                pos += 1
            run_lengths.append(pos - start)
    return {
        "shape": list(token_types_cpu.shape),
        "unique_token_types": {
            str(int(value.item())): int(count.item())
            for value, count in zip(unique_values, unique_counts)
        },
        "image_token_counts_first8": [int(value) for value in image_counts[:8]],
        "image_token_count_min": int(min(image_counts)) if image_counts else None,
        "image_token_count_max": int(max(image_counts)) if image_counts else None,
        "image_run_lengths_first8": [int(value) for value in run_lengths[:8]],
        "complete_run_count": int(sum(1 for value in run_lengths if value == image_tokens_per_img)),
    }


def distributed_barrier(distributed: bool, device: torch.device):
    if not distributed or not (dist.is_available() and dist.is_initialized()):
        return
    if device.type in {"cuda", "npu"}:
        dist.barrier(device_ids=[int(device.index or 0)])
    else:
        dist.barrier()


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)


def max_memory_allocated_mib(device: torch.device) -> float:
    if device.type == "cuda":
        value = torch.cuda.max_memory_allocated(device)
    elif device.type == "npu":
        value = torch.npu.max_memory_allocated(device)
    else:
        value = 0
    return float(value) / (1024.0**2)


def max_memory_reserved_mib(device: torch.device) -> float:
    if device.type == "cuda":
        value = torch.cuda.max_memory_reserved(device)
    elif device.type == "npu":
        value = torch.npu.max_memory_reserved(device)
    else:
        value = 0
    return float(value) / (1024.0**2)


def is_main_process(rank: int) -> bool:
    return rank == 0


def evaluation_progress_payload(
    *,
    stage: str,
    samples_completed: int,
    samples_total: int,
    elapsed_seconds: float,
    strategies: list[str],
    world_size: int,
    batch_idx: int | None = None,
    completed: bool = False,
    metrics_path: str | None = None,
    updated_at: str | None = None,
) -> dict[str, object]:
    total = int(samples_total)
    if total <= 0:
        raise ValueError(f"samples_total must be positive, got {total}")
    done = min(max(int(samples_completed), 0), total)
    elapsed = max(float(elapsed_seconds), 0.0)
    rate = done / elapsed if done > 0 and elapsed > 0.0 else None
    if done >= total:
        eta_seconds = 0.0
    elif rate is None or rate <= 0.0:
        eta_seconds = None
    else:
        eta_seconds = (total - done) / rate
    return {
        "schema": EVALUATION_PROGRESS_SCHEMA,
        "stage": str(stage),
        "completed": bool(completed),
        "samples_completed": done,
        "samples_total": total,
        "progress_percent": 100.0 * done / total,
        "elapsed_seconds": elapsed,
        "samples_per_second": rate,
        "eta_seconds": eta_seconds,
        "batch_idx": None if batch_idx is None else int(batch_idx),
        "strategies": [str(strategy) for strategy in strategies],
        "world_size": int(world_size),
        "metrics_path": metrics_path,
        "updated_at": (
            updated_at
            if updated_at is not None
            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
    }


def write_json_atomic(path: str | Path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_torch_atomic(path: str | Path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def evaluation_artifact_identity(path: str | Path | None) -> dict[str, object] | None:
    if path is None or not str(path).strip():
        return None
    resolved = Path(path).expanduser().resolve()
    identity: dict[str, object] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        return identity
    if resolved.is_file():
        stat = resolved.stat()
        identity.update(
            {
                "kind": "file",
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
        return identity
    entries = []
    for child in sorted(
        (item for item in resolved.rglob("*") if item.is_file()),
        key=lambda item: str(item.relative_to(resolved)),
    ):
        stat = child.stat()
        entries.append(
            {
                "path": str(child.relative_to(resolved)),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    identity.update(
        {
            "kind": "directory",
            "files": entries,
        }
    )
    return identity


def build_evaluation_resume_contract(
    *,
    args,
    config_path: str | Path,
    model_path: str | Path,
    strategies: list[str],
    world_size: int,
    canonical_pairing_enabled: bool,
    target_latents_are_placeholders: bool,
    real_stats_path: str,
    inception_weights_path: str | None,
    image_tokens: int,
) -> dict[str, object]:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    contract = {
        "schema": "single_stream_fid_is_resume_contract_v1",
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "artifacts": {
            "model_path": evaluation_artifact_identity(model_path),
            "adapter": evaluation_artifact_identity(args.adapter),
            "model_state": evaluation_artifact_identity(args.model_state),
            "ema_checkpoint": evaluation_artifact_identity(args.ema_checkpoint),
            "real_stats": evaluation_artifact_identity(real_stats_path),
            "inception_weights": evaluation_artifact_identity(
                inception_weights_path
            ),
        },
        "distributed": {
            "world_size": int(world_size),
        },
        "dataset": {
            "split": str(args.split),
            "batch_size": int(args.batch_size),
            "vae_decode_batch_size": int(args.vae_decode_batch_size),
            "samples": int(args.samples),
            "image_tokens": int(image_tokens),
            "target_latents_are_placeholders": bool(
                target_latents_are_placeholders
            ),
            "real_source": str(args.real_source),
            "real_image_size": int(args.real_image_size),
            "imagenet_train_dir": str(
                Path(args.imagenet_train_dir).expanduser().resolve()
            ),
        },
        "generation": {
            "seed": int(args.seed),
            "strategies": list(strategies),
            "cfg": float(args.cfg),
            "cfg_schedule": str(args.cfg_schedule),
            "sampling_steps": str(args.sampling_steps),
            "temperature": float(args.temperature),
            "flow_solver": str(args.flow_solver),
            "parallel_rate": int(args.parallel_rate),
            "backbone_kv_cache": not bool(args.disable_backbone_kv_cache),
            "model_dtype": str(args.model_dtype),
            "vae_dtype": str(args.vae_dtype),
            "canonical_pairing_enabled": bool(canonical_pairing_enabled),
        },
        "metrics": {
            "fid_feature": int(args.fid_feature),
            "is_splits": int(args.is_splits),
        },
        "output": {
            "save_images": bool(args.save_images),
            "skip_target_decode": bool(args.skip_target_decode),
        },
    }
    contract["sha256"] = canonical_json_sha256(contract)
    return contract


def feature_moments_state(moments: FeatureMoments | None):
    if moments is None:
        return None
    return {
        "count": moments.count.detach().cpu().clone(),
        "sum": moments.sum.detach().cpu().clone(),
        "outer_sum": moments.outer_sum.detach().cpu().clone(),
    }


def feature_moments_from_state(state, *, device: torch.device):
    if state is None:
        return None
    return FeatureMoments(
        count=torch.as_tensor(state["count"], dtype=torch.long, device=device),
        sum=torch.as_tensor(
            state["sum"],
            dtype=metric_accumulation_dtype(device),
            device=device,
        ),
        outer_sum=torch.as_tensor(
            state["outer_sum"],
            dtype=metric_accumulation_dtype(device),
            device=device,
        ),
    )


def inception_score_moments_state(moments: InceptionScoreMoments | None):
    if moments is None:
        return None
    return {
        "count": moments.count.detach().cpu().clone(),
        "probability_sum": moments.probability_sum.detach().cpu().clone(),
        "probability_log_probability_sum": (
            moments.probability_log_probability_sum.detach().cpu().clone()
        ),
    }


def inception_score_moments_from_state(state, *, device: torch.device):
    if state is None:
        return None
    return InceptionScoreMoments(
        count=torch.as_tensor(state["count"], dtype=torch.long, device=device),
        probability_sum=torch.as_tensor(
            state["probability_sum"],
            dtype=metric_accumulation_dtype(device),
            device=device,
        ),
        probability_log_probability_sum=torch.as_tensor(
            state["probability_log_probability_sum"],
            dtype=metric_accumulation_dtype(device),
            device=device,
        ),
    )


def evaluation_metrics_state(metrics: Mapping) -> dict[str, object]:
    serialized = {}
    for strategy, state in metrics.items():
        serialized[str(strategy)] = {
            "fake_moments": feature_moments_state(state["fake_moments"]),
            "score_moments": inception_score_moments_state(
                state["score_moments"]
            ),
            "latent_mse_sum": float(state["latent_mse_sum"]),
            "latent_rms_sum": float(state["latent_rms_sum"]),
            "count": int(state["count"]),
            "generation_wall_seconds": float(
                state["generation_wall_seconds"]
            ),
            "generation_step_max": (
                None
                if state.get("generation_step_max") is None
                else float(state["generation_step_max"])
            ),
            "flow_content_cache_peak_bytes_per_sample": float(
                state["flow_content_cache_peak_bytes_per_sample"]
            ),
            "backbone_kv_cache_peak_bytes": float(
                state.get("backbone_kv_cache_peak_bytes", 0.0)
            ),
            "flow_cfg_cache_divergence_sum": (
                None
                if state["flow_cfg_cache_divergence_sum"] is None
                else [
                    float(value)
                    for value in state["flow_cfg_cache_divergence_sum"]
                ]
            ),
            "flow_cfg_cache_divergence_count": int(
                state["flow_cfg_cache_divergence_count"]
            ),
        }
    return serialized


def evaluation_metrics_from_state(
    state: Mapping,
    *,
    strategies: list[str],
    device: torch.device,
) -> dict[str, object]:
    if set(state) != set(strategies):
        raise ValueError(
            "resume metric strategies do not match current evaluation: "
            f"checkpoint={sorted(state)}, current={sorted(strategies)}"
        )
    restored = {}
    for strategy in strategies:
        item = state[strategy]
        restored[strategy] = {
            "fake_moments": feature_moments_from_state(
                item["fake_moments"],
                device=device,
            ),
            "score_moments": inception_score_moments_from_state(
                item["score_moments"],
                device=device,
            ),
            "latent_mse_sum": float(item["latent_mse_sum"]),
            "latent_rms_sum": float(item["latent_rms_sum"]),
            "count": int(item["count"]),
            "generation_wall_seconds": float(
                item["generation_wall_seconds"]
            ),
            "flow_content_cache_peak_bytes_per_sample": float(
                item["flow_content_cache_peak_bytes_per_sample"]
            ),
            "backbone_kv_cache_peak_bytes": float(
                item.get("backbone_kv_cache_peak_bytes", 0.0)
            ),
            "flow_cfg_cache_divergence_sum": (
                None
                if item["flow_cfg_cache_divergence_sum"] is None
                else [
                    float(value)
                    for value in item["flow_cfg_cache_divergence_sum"]
                ]
            ),
            "flow_cfg_cache_divergence_count": int(
                item["flow_cfg_cache_divergence_count"]
            ),
        }
        if item.get("generation_step_max") is not None:
            restored[strategy]["generation_step_max"] = float(
                item["generation_step_max"]
            )
    return restored


def save_evaluation_resume_checkpoint(
    output_dir: str | Path,
    *,
    contract: Mapping,
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
    state: Mapping,
) -> Path:
    next_batch_idx = int(state["next_batch_idx"])
    if next_batch_idx < 0:
        raise ValueError(
            f"next_batch_idx must be nonnegative, got {next_batch_idx}"
        )
    contract_sha256 = str(contract["sha256"])
    root = Path(output_dir) / "resume_state"
    batch_dir = root / f"batch-{next_batch_idx:08d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    rank_path = batch_dir / f"rank-{int(rank):05d}.pt"
    payload = {
        "schema": EVALUATION_RESUME_SCHEMA,
        "contract_sha256": contract_sha256,
        "rank": int(rank),
        "world_size": int(world_size),
        **dict(state),
    }
    write_torch_atomic(rank_path, payload)
    distributed_barrier(distributed, device)
    if is_main_process(rank):
        rank_files = []
        for expected_rank in range(int(world_size)):
            expected_path = (
                batch_dir / f"rank-{int(expected_rank):05d}.pt"
            )
            if not expected_path.is_file():
                raise RuntimeError(
                    "refusing to commit incomplete resume checkpoint; "
                    f"missing {expected_path}"
                )
            rank_files.append(
                {
                    "rank": int(expected_rank),
                    "path": expected_path.name,
                    "size": int(expected_path.stat().st_size),
                }
            )
        write_json_atomic(
            root / "commit.json",
            {
                "schema": EVALUATION_RESUME_COMMIT_SCHEMA,
                "contract_sha256": contract_sha256,
                "world_size": int(world_size),
                "next_batch_idx": next_batch_idx,
                "batch_directory": batch_dir.name,
                "rank_files": rank_files,
                "updated_at": datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            },
        )
    distributed_barrier(distributed, device)
    if is_main_process(rank):
        for stale_dir in root.glob("batch-*"):
            if stale_dir != batch_dir and stale_dir.is_dir():
                shutil.rmtree(stale_dir)
    distributed_barrier(distributed, device)
    return rank_path


def load_evaluation_resume_checkpoint(
    output_dir: str | Path,
    *,
    contract: Mapping,
    rank: int,
    world_size: int,
) -> dict[str, object] | None:
    root = Path(output_dir) / "resume_state"
    commit_path = root / "commit.json"
    if not commit_path.is_file():
        return None
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if commit.get("schema") != EVALUATION_RESUME_COMMIT_SCHEMA:
        raise ValueError(
            f"unsupported evaluator resume commit schema: {commit.get('schema')!r}"
        )
    expected_contract_sha256 = str(contract["sha256"])
    if commit.get("contract_sha256") != expected_contract_sha256:
        raise ValueError(
            "stale evaluator resume checkpoint contract: "
            f"checkpoint={commit.get('contract_sha256')!r}, "
            f"current={expected_contract_sha256!r}"
        )
    if int(commit.get("world_size", -1)) != int(world_size):
        raise ValueError(
            "evaluator resume world size mismatch: "
            f"checkpoint={commit.get('world_size')}, current={world_size}"
        )
    batch_directory = str(commit.get("batch_directory", ""))
    expected_directory = (
        f"batch-{int(commit.get('next_batch_idx', -1)):08d}"
    )
    if batch_directory != expected_directory:
        raise ValueError(
            "evaluator resume commit has inconsistent batch directory: "
            f"{batch_directory!r} != {expected_directory!r}"
        )
    rank_path = root / batch_directory / f"rank-{int(rank):05d}.pt"
    if not rank_path.is_file():
        raise FileNotFoundError(
            f"committed evaluator resume rank state is missing: {rank_path}"
        )
    payload = torch.load(rank_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != EVALUATION_RESUME_SCHEMA:
        raise ValueError(
            f"unsupported evaluator resume schema: {payload.get('schema')!r}"
        )
    expected_fields = {
        "contract_sha256": expected_contract_sha256,
        "rank": int(rank),
        "world_size": int(world_size),
        "next_batch_idx": int(commit["next_batch_idx"]),
    }
    mismatches = {
        key: {"checkpoint": payload.get(key), "expected": value}
        for key, value in expected_fields.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "evaluator resume rank state does not match commit: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return payload


def emit_evaluation_progress(
    output_dir: str | Path,
    payload: Mapping,
    *,
    log_to_console: bool = True,
) -> None:
    write_json_atomic(
        Path(output_dir) / "evaluation_progress.json",
        payload,
    )
    if log_to_console:
        print(
            "[EvalProgress] "
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(
        float(value),
        device=device,
        dtype=metric_accumulation_dtype(device),
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def reduce_max(value: float | None, device: torch.device) -> float | None:
    raw_value = -float("inf") if value is None else float(value)
    tensor = torch.tensor(
        raw_value,
        device=device,
        dtype=metric_accumulation_dtype(device),
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    reduced = float(tensor.item())
    return None if reduced == -float("inf") else reduced


def image_spans(token_types: torch.Tensor, image_tokens_per_img: int):
    spans = []
    bsz, seq_len = token_types.shape
    for b in range(bsz):
        pos = 0
        while pos < seq_len:
            if int(token_types[b, pos].item()) != 1:
                pos += 1
                continue
            start = pos
            while pos < seq_len and int(token_types[b, pos].item()) == 1:
                pos += 1
            end = pos
            if end - start == image_tokens_per_img:
                spans.append((b, start, end))
    return spans


def shard_unpacked_batch_rows(
    tensors: Mapping[str, torch.Tensor],
    spans: list[tuple[int, int, int]],
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int, int]]]:
    """Select one unpacked image row per local rank and rebase span row ids."""

    source_rows = [int(batch_row) for batch_row, _, _ in spans]
    if len(source_rows) != len(set(source_rows)):
        raise ValueError(
            "distributed formal evaluation requires validation to remain "
            "unpacked with at most one complete image span per row"
        )
    if not source_rows:
        return dict(tensors), []

    sharded: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if tensor.ndim < 1:
            raise ValueError(f"{name} must have a batch dimension")
        if max(source_rows) >= int(tensor.shape[0]):
            raise ValueError(
                f"{name} batch has {tensor.shape[0]} rows but span selects "
                f"row {max(source_rows)}"
            )
        row_indices = torch.tensor(
            source_rows,
            dtype=torch.long,
            device=tensor.device,
        )
        sharded[name] = tensor.index_select(0, row_indices)
    rebased_spans = [
        (local_row, int(start), int(end))
        for local_row, (_, start, end) in enumerate(spans)
    ]
    return sharded, rebased_spans


def decode_latents_in_microbatches(
    vae,
    latents: torch.Tensor,
    scaling_factor: float,
    *,
    batch_size: int,
) -> torch.Tensor:
    local_batch = int(latents.shape[0])
    decode_batch = int(batch_size)
    if decode_batch < 0:
        raise ValueError(
            f"VAE decode batch size must be nonnegative, got {decode_batch}"
        )
    if decode_batch == 0 or decode_batch >= local_batch:
        return decode_latents(vae, latents, scaling_factor)
    return torch.cat(
        [
            decode_latents(
                vae,
                latents[start : start + decode_batch],
                scaling_factor,
            )
            for start in range(0, local_batch, decode_batch)
        ],
        dim=0,
    )


def extract_inception_features_in_microbatches(
    inception,
    images: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract metric tensors without materializing full-batch activations."""

    local_batch = int(images.shape[0])
    metric_batch = int(batch_size)
    if metric_batch < 0:
        raise ValueError(
            "Inception batch size must be nonnegative, "
            f"got {metric_batch}"
        )
    if metric_batch == 0 or metric_batch >= local_batch:
        return extract_inception_features(inception, images)

    features = []
    logits = []
    for start in range(0, local_batch, metric_batch):
        batch_features, batch_logits = extract_inception_features(
            inception,
            images[start : start + metric_batch],
        )
        features.append(batch_features)
        logits.append(batch_logits)
    return torch.cat(features, dim=0), torch.cat(logits, dim=0)


def span_latents_to_chw(
    image_latents: torch.Tensor,
    spans: list[tuple[int, int, int]],
    side: int,
) -> torch.Tensor:
    latents = []
    for b, start, end in spans:
        latents.append(image_latents[b, start:end].view(side, side, -1).permute(2, 0, 1))
    return torch.stack(latents)


def metric_images(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().clamp(0.0, 1.0)


def save_batch_images(images: torch.Tensor, directory: Path, offset: int):
    directory.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        save_image(image, directory / f"{offset + idx:08d}.png")


def save_indexed_images(images: torch.Tensor, directory: Path, indices: list[int]):
    directory.mkdir(parents=True, exist_ok=True)
    for image, index in zip(images, indices):
        save_image(image, directory / f"{int(index):08d}.png")


def get_base_dataset_and_indices(loader_dataset):
    if hasattr(loader_dataset, "dataset") and hasattr(loader_dataset, "indices"):
        return loader_dataset.dataset, list(loader_dataset.indices)
    return loader_dataset, None


def ordered_eval_sample_records(
    loader_dataset,
    *,
    loader_rows: list[int],
    global_sample_indices: list[int],
) -> list[dict[str, int]]:
    if len(loader_rows) != len(global_sample_indices):
        raise ValueError(
            "loader_rows and global_sample_indices must have the same length, "
            f"got {len(loader_rows)} and {len(global_sample_indices)}"
        )
    base_dataset, subset_indices = get_base_dataset_and_indices(loader_dataset)
    if not hasattr(base_dataset, "img_ids"):
        raise ValueError(
            "canonical evaluation pairing requires a dataset with img_ids"
        )

    records = []
    for loader_row, global_sample_index in zip(
        loader_rows,
        global_sample_indices,
    ):
        base_row = (
            int(subset_indices[int(loader_row)])
            if subset_indices is not None
            else int(loader_row)
        )
        if base_row < 0 or base_row >= len(base_dataset):
            raise IndexError(
                f"evaluation dataset row {base_row} is outside [0, {len(base_dataset)})"
            )
        image_id = int(base_dataset.img_ids[base_row].item())
        records.append(
            {
                "global_sample_index": int(global_sample_index),
                "image_id": image_id,
            }
        )
    return records


def _gather_object_records(local_records: list[dict[str, object]]) -> list[dict[str, object]]:
    if not (dist.is_available() and dist.is_initialized()):
        return list(local_records)
    gathered: list[list[dict[str, object]] | None] = [
        None for _ in range(dist.get_world_size())
    ]
    dist.all_gather_object(gathered, local_records)
    return [record for rank_records in gathered for record in (rank_records or [])]


def evaluation_pairing_manifest_hashes(
    *,
    local_noise_records: list[dict[str, object]],
    local_sample_records: list[dict[str, object]],
    evaluation_seed: int,
    expected_samples: int,
) -> dict[str, object]:
    expected_indices = list(range(int(expected_samples)))

    noise_records = sorted(
        _gather_object_records(local_noise_records),
        key=lambda record: int(record["global_sample_index"]),
    )
    sample_records = sorted(
        _gather_object_records(local_sample_records),
        key=lambda record: int(record["global_sample_index"]),
    )
    for label, records in (
        ("canonical noise", noise_records),
        ("ordered evaluation sample", sample_records),
    ):
        actual_indices = [int(record["global_sample_index"]) for record in records]
        if actual_indices != expected_indices:
            raise RuntimeError(
                f"{label} manifest must contain each global sample index exactly once; "
                f"expected {len(expected_indices)} ordered indices, got "
                f"{len(actual_indices)} records with prefix={actual_indices[:16]}"
            )

    noise_payload = {
        "schema": CANONICAL_NOISE_MANIFEST_SCHEMA,
        "evaluation_seed": int(evaluation_seed),
        "records": noise_records,
    }
    sample_payload = {
        "schema": ORDERED_EVAL_SAMPLE_MANIFEST_SCHEMA,
        "records": sample_records,
    }
    return {
        "canonical_noise_manifest_schema": CANONICAL_NOISE_MANIFEST_SCHEMA,
        "canonical_noise_manifest_sha256": canonical_json_sha256(noise_payload),
        "ordered_eval_sample_manifest_schema": ORDERED_EVAL_SAMPLE_MANIFEST_SCHEMA,
        "ordered_eval_sample_manifest_sha256": canonical_json_sha256(sample_payload),
        "paired_sample_count": len(expected_indices),
    }


def source_paths_for_rows(
    loader_dataset,
    loader_rows: list[int],
    imagenet_train_dir: Path,
):
    base_dataset, subset_indices = get_base_dataset_and_indices(loader_dataset)
    if not hasattr(base_dataset, "img_ids") or not hasattr(base_dataset, "source_paths"):
        raise ValueError("--real_source=imagenet_original requires an ImageNetFlowCacheDataset or Subset of it.")

    paths = []
    for loader_row in loader_rows:
        dataset_row = int(loader_row)
        if subset_indices is not None:
            dataset_row = int(subset_indices[dataset_row])
        img_id = int(base_dataset.img_ids[dataset_row].item())
        source_path = base_dataset.source_paths.get(img_id)
        if not source_path:
            raise KeyError(f"No source_path in manifest for img_id={img_id}")
        path = Path(source_path)
        if not path.is_absolute():
            path = imagenet_train_dir / path
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def build_real_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
        ]
    )


def load_real_images(paths, transform, device):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images, dim=0).to(device=device, dtype=torch.float32)


def require_finite_metric_scalar(value: float, *, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(
            f"non-finite formal evaluation scalar invalidates FID/IS: {label}={value}"
        )
    return value


def require_finite_metric_tensor(tensor: torch.Tensor, *, label: str) -> None:
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    finite_values = tensor[finite].float()
    finite_range = (
        [float(finite_values.min().item()), float(finite_values.max().item())]
        if finite_values.numel()
        else None
    )
    raise FloatingPointError(
        "non-finite tensor invalidates formal FID/IS: "
        f"label={label!r}, shape={tuple(tensor.shape)}, "
        f"nonfinite={int((~finite).sum().item())}/{tensor.numel()}, "
        f"finite_range={finite_range}"
    )


def require_finite_generated_latents(
    latents: torch.Tensor,
    *,
    strategy: str,
    rank: int,
    batch_idx: int,
    global_indices: list[int],
) -> None:
    finite = torch.isfinite(latents)
    if bool(finite.all()):
        return
    nonfinite = int((~finite).sum().item())
    total = int(latents.numel())
    finite_values = latents[finite].float()
    finite_range = (
        [float(finite_values.min().item()), float(finite_values.max().item())]
        if finite_values.numel()
        else None
    )
    raise FloatingPointError(
        "non-finite generated image latents invalidate formal FID/IS: "
        f"strategy={strategy!r}, rank={rank}, batch_idx={batch_idx}, "
        f"global_indices={global_indices}, nonfinite={nonfinite}/{total}, "
        f"finite_range={finite_range}"
    )


def checkpoint_weight_dtypes(model_path: str | Path) -> list[str]:
    weights_path = Path(model_path) / "model.safetensors"
    dtype_names = {
        "BF16": "bf16",
        "F32": "fp32",
        "F16": "fp16",
    }
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        return sorted(
            {
                dtype_names.get(
                    str(handle.get_slice(key).get_dtype()),
                    str(handle.get_slice(key).get_dtype()).lower(),
                )
                for key in handle.keys()
            }
        )


def is_official_flow_protocol(
    *,
    shared_real_count: int | None,
    samples: int,
    is_splits: int,
    parallel_rate: int,
) -> bool:
    return bool(
        shared_real_count is not None
        and int(samples) == int(shared_real_count)
        and int(is_splits) == 10
        and int(parallel_rate) == 1
    )


def validate_strategies(strategies: list[str], allow_sigma_strategies: bool) -> None:
    if allow_sigma_strategies:
        return
    forbidden = {"sigma", "sigma_replay", "causal_sigma"}
    found = sorted({strategy.lower() for strategy in strategies} & forbidden)
    if found:
        raise ValueError(
            "Refusing sigma-based generation strategies for real evaluation: "
            f"{found}. These strategies require training/data sigma. "
            "Pass --allow_sigma_strategies only for debugging."
        )


def resolve_inception_weights_path(path: str) -> str | None:
    if not path:
        return None
    weights_path = Path(path).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"--inception_weights_path does not exist: {weights_path}")
    return str(weights_path)


def shared_feature_moments(payload, *, feature: int, device) -> FeatureMoments:
    stats = payload["stats"]
    count = int(stats["count"])
    feature_sum = torch.as_tensor(stats["sum"], dtype=torch.float64, device=device)
    outer_sum = torch.as_tensor(
        stats["outer_sum"],
        dtype=torch.float64,
        device=device,
    )
    if tuple(feature_sum.shape) != (int(feature),):
        raise ValueError(
            f"shared real feature sum shape={tuple(feature_sum.shape)}; "
            f"expected={(int(feature),)}"
        )
    if tuple(outer_sum.shape) != (int(feature), int(feature)):
        raise ValueError(
            f"shared real outer-sum shape={tuple(outer_sum.shape)}; "
            f"expected={(int(feature), int(feature))}"
        )
    moments = FeatureMoments.zeros(int(feature), device)
    moments.count.fill_(count)
    moments.sum.copy_(feature_sum)
    moments.outer_sum.copy_(outer_sum)
    return moments


def load_shared_original_real_stats(
    path: str,
    *,
    config,
    fid_feature: int,
    real_image_size: int,
    inception_weights_path: str,
):
    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    payload = torch.load(stats_path, map_location="cpu")
    del config, real_image_size
    if not isinstance(payload, Mapping) or "stats" not in payload:
        raise ValueError(
            f"{stats_path} must contain a mapping with a 'stats' entry."
        )
    shared_feature_moments(
        payload,
        feature=int(fid_feature),
        device=torch.device("cpu"),
    )
    count = int(payload["stats"]["count"])
    if count < 2:
        raise ValueError(
            f"shared real stats require at least two images, found {count}"
        )
    metadata = payload.get("metadata", {})
    feature_metadata = (
        metadata.get("feature", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    recorded_feature = feature_metadata.get("feature")
    if (
        recorded_feature is not None
        and int(recorded_feature) != int(fid_feature)
    ):
        raise ValueError(
            "shared real stats feature dimension mismatch: "
            f"cache={recorded_feature}, requested={fid_feature}."
        )
    recorded_weights_sha256 = feature_metadata.get("weights_sha256")
    if recorded_weights_sha256 and inception_weights_path:
        actual_weights_sha256 = file_sha256(inception_weights_path)
        if str(recorded_weights_sha256) != actual_weights_sha256:
            raise ValueError(
                "shared real stats were produced with different Inception "
                "weights."
            )
    return payload


def warm_inception_cache_if_needed(
    args,
    distributed: bool,
    rank: int,
    device: torch.device,
    weights_path: str | None,
):
    if weights_path is not None:
        return
    if not distributed or is_main_process(rank):
        extractor = build_inception_extractor(
            int(args.fid_feature),
            weights_path,
            device,
        )
        del extractor
    distributed_barrier(distributed, device)


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    distributed, rank, world_size, local_rank, device = init_distributed(args.device)
    local_batch_size = per_rank_batch_size(args.batch_size, world_size)
    if int(args.samples) < int(world_size):
        raise ValueError(
            f"--samples={args.samples} must be at least world_size={world_size}"
        )
    if int(args.samples) % int(world_size):
        raise ValueError(
            f"--samples={args.samples} must be divisible by "
            f"world_size={world_size}"
        )
    if int(args.samples) < int(args.is_splits):
        raise ValueError(
            f"--samples={args.samples} must be at least --is_splits={args.is_splits}"
        )
    if int(args.progress_log_interval_samples) <= 0:
        raise ValueError(
            "--progress_log_interval_samples must be positive, got "
            f"{args.progress_log_interval_samples}"
        )
    if float(args.progress_log_interval_seconds) <= 0.0:
        raise ValueError(
            "--progress_log_interval_seconds must be positive, got "
            f"{args.progress_log_interval_seconds}"
        )
    if (
        args.resume_progress
        and int(args.resume_checkpoint_interval_batches) <= 0
    ):
        raise ValueError(
            "--resume_checkpoint_interval_batches must be positive, got "
            f"{args.resume_checkpoint_interval_batches}"
        )
    progress = not args.no_progress and is_main_process(rank)
    out_dir = Path(args.output_dir)
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(distributed, device)

    torch.manual_seed(args.seed + rank * 100_003)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if not strategies:
        raise ValueError("--strategies must contain at least one strategy")
    validate_strategies(strategies, args.allow_sigma_strategies)
    config = OmegaConf.load(args.config)
    canonical_pairing_enabled = bool(args.canonical_pairing)
    if args.model_path_override:
        config.model.model_path = args.model_path_override
    config.training.batch_size = int(local_batch_size)
    config.training.dataloader_workers = 0
    config.model.image_flow_num_sampling_steps = str(args.sampling_steps)
    target_latents_are_placeholders = bool(
        config.dataset.params.get(
            "target_latents_are_placeholders",
            False,
        )
    )

    if is_main_process(rank):
        print(
            f"Distributed evaluation: world_size={world_size}, "
            f"device={device}, global_batch_size={args.batch_size}, "
            f"batch_size_per_rank={local_batch_size}, "
            f"strategies={strategies}"
        )
        print("Loading model/tokenizer...")
    requested_model_dtype = {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.model_dtype]
    stored_checkpoint_dtypes = checkpoint_weight_dtypes(config.model.model_path)
    model, tokenizer = load_model_tokenizer(
        config,
        model_dtype=requested_model_dtype,
    )
    if is_main_process(rank):
        print(f"Loading adapter: {args.adapter}")
    adapter_report = load_adapter(model, args.adapter)
    if is_main_process(rank):
        print(f"Loading model state: {args.model_state or 'none'}")
    model_state_report = load_model_state(model, args.model_state)
    if is_main_process(rank):
        print(f"Loading sharded EMA checkpoint: {args.ema_checkpoint or 'none'}")
    ema_checkpoint_report = load_sharded_ema_checkpoint(
        model, args.ema_checkpoint
    )
    model = model.to(device).eval()
    if hasattr(model.image_flow_head, "reset_guidance_diagnostics"):
        model.image_flow_head.reset_guidance_diagnostics()
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    image_embedder_parameter_count = sum(
        parameter.numel() for parameter in model.image_token_embedder.parameters()
    )
    flow_head_parameter_count = sum(
        parameter.numel() for parameter in model.image_flow_head.parameters()
    )
    backbone_attention_gate_parameter_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".attn_output_gate_proj." in name
    )
    parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()}
    )
    expected_parameter_dtype = str(requested_model_dtype)
    if parameter_dtypes != [expected_parameter_dtype]:
        raise RuntimeError(
            "generation model contains unexpected floating parameter dtypes: "
            f"requested={expected_parameter_dtype}, actual={parameter_dtypes}"
        )

    if is_main_process(rank):
        print("Loading KL16 VAE...")
    vae = load_vae(config, device, args.vae_dtype)
    scaling_factor = float(config.experiment.validation_vae_scaling_factor)
    if is_main_process(rank):
        print(f"Loading {args.split} dataloader...")
    train_loader, val_loader = get_dataloaders(config, tokenizer)
    source_loader = val_loader if args.split == "val" else train_loader
    if len(source_loader.dataset) < int(args.samples):
        raise ValueError(
            f"{args.split} dataset has {len(source_loader.dataset)} rows, "
            f"fewer than --samples={args.samples}"
        )
    loader = DataLoader(
        source_loader.dataset,
        batch_sampler=GlobalBatchStrideSampler(
            samples=int(args.samples),
            global_batch_size=int(args.batch_size),
            rank=rank,
            world_size=world_size,
        ),
        num_workers=0,
        pin_memory=True,
        # Evaluation is always one logical sample per physical row, even
        # when reading the training split.
        collate_fn=val_loader.collate_fn,
    )
    if args.real_source == "imagenet_original" and args.split != "val":
        raise ValueError("--real_source=imagenet_original currently expects --split val because train loader is shuffled.")
    real_transform = build_real_image_transform(args.real_image_size)
    imagenet_train_dir = Path(args.imagenet_train_dir)

    image_tokens = int(config.model.image_tokens_per_img)
    side = int(image_tokens ** 0.5)
    if side * side != image_tokens:
        raise ValueError(f"image_tokens_per_img={image_tokens} is not a square grid")
    if canonical_pairing_enabled:
        canonical_shape = (
            CANONICAL_IMAGE_GRID_SIDE,
            CANONICAL_IMAGE_GRID_SIDE,
            CANONICAL_IMAGE_LATENT_DIM,
        )
        expected_canonical_shape = tuple(
            EVALUATOR_RNG_CONTRACT["canonical_shape"]
        )
        if canonical_shape != expected_canonical_shape:
            raise ValueError(
                "canonical evaluation requires image-flow noise shape "
                f"{expected_canonical_shape}, got {canonical_shape}"
            )

    inception_weights_path = resolve_inception_weights_path(args.inception_weights_path)
    requested_real_stats_path = str(args.real_stats_path).strip()
    real_stats_path = (
        ""
        if requested_real_stats_path.lower() in {"none", "null"}
        else str(
            requested_real_stats_path
            or config.get("evaluation", {}).get("real_stats_path", "")
        )
    )
    shared_real_payload = None
    if real_stats_path:
        if inception_weights_path is None:
            raise ValueError(
                "shared original-image stats require an explicit local "
                "--inception_weights_path so its content hash can be verified"
            )
        shared_real_payload = load_shared_original_real_stats(
            real_stats_path,
            config=config,
            fid_feature=int(args.fid_feature),
            real_image_size=int(args.real_image_size),
            inception_weights_path=inception_weights_path,
        )
    shared_real_count = (
        int(shared_real_payload["stats"]["count"])
        if shared_real_payload is not None
        else None
    )
    if args.skip_target_decode and shared_real_payload is None:
        raise ValueError(
            "--skip_target_decode requires frozen --real_stats_path"
        )
    if (
        target_latents_are_placeholders
        and shared_real_payload is None
        and args.real_source == "vae_decoded_target_latents"
    ):
        raise ValueError(
            "dataset target latents are declared as placeholders and cannot "
            "serve as real images; provide frozen --real_stats_path or use "
            "--real_source=imagenet_original"
        )
    official_protocol = is_official_flow_protocol(
        shared_real_count=shared_real_count,
        samples=int(args.samples),
        is_splits=int(args.is_splits),
        parallel_rate=int(args.parallel_rate),
    )
    if args.require_official_protocol and not official_protocol:
        raise ValueError(
            "Official flow FID/IS protocol requires shared original-ImageNet "
            f"stats, exactly its {shared_real_count} fake samples, and "
            "--is_splits=10 with --parallel_rate=1; "
            f"got samples={args.samples}, is_splits={args.is_splits}, "
            f"parallel_rate={args.parallel_rate}, real_stats_path={real_stats_path!r}"
        )
    if is_main_process(rank):
        print(
            "Inception weights: "
            f"{inception_weights_path if inception_weights_path is not None else 'torch-fidelity default cache/download'}"
        )
        if shared_real_payload is not None:
            print(f"Shared original-ImageNet real stats: {Path(real_stats_path).resolve()}")
    warm_inception_cache_if_needed(
        args,
        distributed,
        rank,
        device,
        inception_weights_path,
    )
    inception = build_inception_extractor(
        int(args.fid_feature),
        inception_weights_path,
        device,
    )
    resume_contract = build_evaluation_resume_contract(
        args=args,
        config_path=args.config,
        model_path=str(config.model.model_path),
        strategies=strategies,
        world_size=world_size,
        canonical_pairing_enabled=canonical_pairing_enabled,
        target_latents_are_placeholders=target_latents_are_placeholders,
        real_stats_path=real_stats_path,
        inception_weights_path=inception_weights_path,
        image_tokens=image_tokens,
    )

    metrics = {
        strategy: {
            "fake_moments": FeatureMoments.zeros(int(args.fid_feature), device),
            "score_moments": None,
            "latent_mse_sum": 0.0,
            "latent_rms_sum": 0.0,
            "count": 0,
            "generation_wall_seconds": 0.0,
            "flow_content_cache_peak_bytes_per_sample": 0.0,
            "backbone_kv_cache_peak_bytes": 0.0,
            "flow_cfg_cache_divergence_sum": None,
            "flow_cfg_cache_divergence_count": 0,
        }
        for strategy in strategies
    }
    real_moments = (
        None
        if shared_real_payload is not None
        else FeatureMoments.zeros(int(args.fid_feature), device)
    )
    reset_peak_memory_stats(device)

    generated = 0
    seen_complete_spans = 0
    batches_seen = 0
    batches_with_complete_spans = 0
    selected_span_batches = 0
    local_noise_manifest_records: list[dict[str, object]] = []
    local_eval_sample_manifest_records: list[dict[str, object]] = []
    first_batch_debug = None
    first_complete_span_debug = None
    batch_offset = 0
    resume_next_batch_idx = 0
    restored_elapsed_seconds = 0.0
    restored_cuda_peak_allocated_mib = 0.0
    restored_cuda_peak_reserved_mib = 0.0
    resume_payload = (
        load_evaluation_resume_checkpoint(
            out_dir,
            contract=resume_contract,
            rank=rank,
            world_size=world_size,
        )
        if args.resume_progress
        else None
    )
    if resume_payload is not None:
        metrics = evaluation_metrics_from_state(
            resume_payload["metrics"],
            strategies=strategies,
            device=device,
        )
        real_moments = feature_moments_from_state(
            resume_payload["real_moments"],
            device=device,
        )
        resume_next_batch_idx = int(resume_payload["next_batch_idx"])
        generated = int(resume_payload["generated"])
        seen_complete_spans = int(
            resume_payload["seen_complete_spans"]
        )
        batches_seen = int(resume_payload["batches_seen"])
        batches_with_complete_spans = int(
            resume_payload["batches_with_complete_spans"]
        )
        selected_span_batches = int(
            resume_payload["selected_span_batches"]
        )
        batch_offset = int(resume_payload["batch_offset"])
        local_noise_manifest_records = list(
            resume_payload["local_noise_manifest_records"]
        )
        local_eval_sample_manifest_records = list(
            resume_payload["local_eval_sample_manifest_records"]
        )
        first_batch_debug = resume_payload["first_batch_debug"]
        first_complete_span_debug = resume_payload[
            "first_complete_span_debug"
        ]
        restored_elapsed_seconds = float(
            resume_payload["elapsed_seconds"]
        )
        restored_cuda_peak_allocated_mib = float(
            resume_payload.get("cuda_peak_allocated_mib", 0.0)
        )
        restored_cuda_peak_reserved_mib = float(
            resume_payload.get("cuda_peak_reserved_mib", 0.0)
        )
        if seen_complete_spans < 0:
            raise ValueError(
                "invalid seen_complete_spans in evaluator resume state: "
                f"{seen_complete_spans}"
            )
        if generated < 0:
            raise ValueError(
                f"invalid generated count in evaluator resume state: {generated}"
            )
        if is_main_process(rank):
            print(
                "Resuming evaluation from globally committed "
                f"batch {resume_next_batch_idx} "
                f"({min(seen_complete_spans, int(args.samples))}/"
                f"{int(args.samples)} samples)."
            )
    progress_started = time.perf_counter() - restored_elapsed_seconds
    progress_last_logged_at = time.perf_counter()
    progress_last_logged_samples = min(
        seen_complete_spans,
        int(args.samples),
    )

    def report_evaluation_progress(
        *,
        stage: str,
        samples_completed: int,
        batch_idx: int | None = None,
        force: bool = False,
        completed: bool = False,
        metrics_path: str | None = None,
    ) -> None:
        nonlocal progress_last_logged_at, progress_last_logged_samples
        if not is_main_process(rank):
            return
        now = time.perf_counter()
        sample_delta = int(samples_completed) - progress_last_logged_samples
        time_delta = now - progress_last_logged_at
        if (
            not force
            and sample_delta < int(args.progress_log_interval_samples)
            and time_delta < float(args.progress_log_interval_seconds)
        ):
            return
        payload = evaluation_progress_payload(
            stage=stage,
            samples_completed=samples_completed,
            samples_total=int(args.samples),
            elapsed_seconds=now - progress_started,
            strategies=strategies,
            world_size=world_size,
            batch_idx=batch_idx,
            completed=completed,
            metrics_path=metrics_path,
        )
        emit_evaluation_progress(
            out_dir,
            payload,
            log_to_console=not args.no_progress,
        )
        progress_last_logged_at = now
        progress_last_logged_samples = int(samples_completed)

    def commit_evaluation_resume(
        *,
        next_batch_idx: int,
        force: bool = False,
    ) -> None:
        if not args.resume_progress:
            return
        if (
            not force
            and int(next_batch_idx)
            % int(args.resume_checkpoint_interval_batches)
            != 0
        ):
            return
        current_allocated_mib = max_memory_allocated_mib(device)
        current_reserved_mib = max_memory_reserved_mib(device)
        save_evaluation_resume_checkpoint(
            out_dir,
            contract=resume_contract,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            device=device,
            state={
                "next_batch_idx": int(next_batch_idx),
                "generated": int(generated),
                "seen_complete_spans": int(seen_complete_spans),
                "batches_seen": int(batches_seen),
                "batches_with_complete_spans": int(
                    batches_with_complete_spans
                ),
                "selected_span_batches": int(selected_span_batches),
                "batch_offset": int(batch_offset),
                "local_noise_manifest_records": list(
                    local_noise_manifest_records
                ),
                "local_eval_sample_manifest_records": list(
                    local_eval_sample_manifest_records
                ),
                "first_batch_debug": first_batch_debug,
                "first_complete_span_debug": first_complete_span_debug,
                "elapsed_seconds": float(
                    time.perf_counter() - progress_started
                ),
                "cuda_peak_allocated_mib": max(
                    restored_cuda_peak_allocated_mib,
                    current_allocated_mib,
                ),
                "cuda_peak_reserved_mib": max(
                    restored_cuda_peak_reserved_mib,
                    current_reserved_mib,
                ),
                "metrics": evaluation_metrics_state(metrics),
                "real_moments": feature_moments_state(real_moments),
            },
        )

    report_evaluation_progress(
        stage=("resumed" if resume_payload is not None else "generating"),
        samples_completed=min(seen_complete_spans, int(args.samples)),
        batch_idx=(
            resume_next_batch_idx - 1
            if resume_next_batch_idx > 0
            else None
        ),
        force=True,
    )
    iterator = tqdm(loader, desc="single-stream FID/IS", dynamic_ncols=True, disable=not progress)
    for batch_idx, batch in enumerate(iterator):
        if batch_idx < resume_next_batch_idx:
            continue
        if seen_complete_spans >= args.samples:
            break

        input_ids = batch["input_ids"]
        token_types = batch["token_types"]
        sigma = batch["sigma"]
        image_latents = batch["image_latents"]
        current_batch_size = int(input_ids.shape[0])
        global_batch_start = int(batch_idx) * int(args.batch_size)
        selected_global_indices = list(
            range(
                global_batch_start + rank,
                min(
                    global_batch_start + int(args.batch_size),
                    int(args.samples),
                ),
                world_size,
            )
        )
        if current_batch_size != len(selected_global_indices):
            raise RuntimeError(
                "rank-local dataloader batch does not match the global stride "
                f"contract: rows={current_batch_size}, "
                f"indices={len(selected_global_indices)}"
            )
        batch_offset = global_batch_start
        all_spans = image_spans(token_types, image_tokens)
        batches_seen += 1
        if first_batch_debug is None:
            first_batch_debug = token_type_debug(token_types, image_tokens)
        if len(all_spans) != current_batch_size:
            raise RuntimeError(
                "formal evaluation requires exactly one complete image span "
                f"per unpacked row; got {len(all_spans)} spans in "
                f"{current_batch_size} rows"
            )
        batches_with_complete_spans += 1
        if first_complete_span_debug is None:
            first_complete_span_debug = {
                "batch_idx": int(batch_idx),
                "batch_offset": int(batch_offset),
                "complete_spans_in_batch": int(len(all_spans)),
                "first_spans": [
                    [int(batch_row), int(start), int(end)]
                    for batch_row, start, end in all_spans[:8]
                ],
            }

        source_spans = list(all_spans)
        local_batch, spans = shard_unpacked_batch_rows(
            {
                "input_ids": input_ids,
                "token_types": token_types,
                "sigma": sigma,
                "image_latents": image_latents,
            },
            source_spans,
        )
        seen_complete_spans = min(
            global_batch_start + int(args.batch_size),
            int(args.samples),
        )
        input_ids = local_batch["input_ids"].to(device)
        token_types = local_batch["token_types"].to(device)
        sigma = local_batch["sigma"].to(device)
        image_latents = local_batch["image_latents"].to(device)
        selected_span_batches += 1

        initial_noise_bank = None
        if canonical_pairing_enabled:
            initial_noise_bank, noise_records = build_canonical_initial_noise_bank(
                selected_global_indices,
                evaluation_seed=int(args.seed),
            )
            local_noise_manifest_records.extend(noise_records)
            local_eval_sample_manifest_records.extend(
                ordered_eval_sample_records(
                    loader.dataset,
                    loader_rows=selected_global_indices,
                    global_sample_indices=selected_global_indices,
                )
            )

        target_latents = span_latents_to_chw(image_latents, spans, side)
        target_images = None
        if (
            not target_latents_are_placeholders
            and not args.skip_target_decode
        ):
            decoded_target_images = decode_latents_in_microbatches(
                vae,
                target_latents.float(),
                scaling_factor,
                batch_size=int(args.vae_decode_batch_size),
            )
            require_finite_metric_tensor(
                decoded_target_images,
                label=f"decoded_target_images.rank{rank}.batch{batch_idx}",
            )
            target_images = metric_images(decoded_target_images)
        if shared_real_payload is not None:
            real_images = None
        elif args.real_source == "imagenet_original":
            real_paths = source_paths_for_rows(
                loader.dataset,
                selected_global_indices,
                imagenet_train_dir,
            )
            real_images = load_real_images(real_paths, real_transform, device)
        else:
            real_images = target_images
        if real_moments is not None:
            require_finite_metric_tensor(
                real_images,
                label=f"real_images.rank{rank}.batch{batch_idx}",
            )
            real_features, _ = extract_inception_features_in_microbatches(
                inception,
                real_images,
                batch_size=int(args.vae_decode_batch_size),
            )
            require_finite_metric_tensor(
                real_features,
                label=f"real_features.rank{rank}.batch{batch_idx}",
            )
            real_moments.update(real_features)
        if args.save_images and target_images is not None:
            save_indexed_images(target_images.cpu(), out_dir / "target_decoded", selected_global_indices)
        if (
            args.save_images
            and shared_real_payload is None
            and args.real_source == "imagenet_original"
        ):
            save_indexed_images(real_images.cpu(), out_dir / "imagenet_original_real", selected_global_indices)

        for strategy_idx, strategy in enumerate(strategies):
            torch.manual_seed(int(args.seed) + batch_idx * 1009 + strategy_idx * 131071 + rank * 1_000_003)
            synchronize_device(device)
            generation_started = time.perf_counter()
            single_latents, trace = model.sample_image_latents_single_stream(
                input_ids=input_ids,
                token_types=token_types,
                sigma=sigma,
                spans=spans,
                image_latent_dim=image_latents.shape[-1],
                initial_noise_bank=initial_noise_bank,
                flow_temperature=float(args.temperature),
                flow_cfg=float(args.cfg),
                flow_cfg_schedule=str(args.cfg_schedule),
                flow_solver=args.flow_solver,
                parallel_rate=int(args.parallel_rate),
                order_strategy=str(strategy),
                use_backbone_cache=not bool(
                    args.disable_backbone_kv_cache
                ),
                return_trace=True,
                debug_finite=bool(args.debug_finite_generation),
            )
            require_finite_generated_latents(
                single_latents,
                strategy=str(strategy),
                rank=rank,
                batch_idx=batch_idx,
                global_indices=selected_global_indices,
            )
            synchronize_device(device)
            generation_elapsed = time.perf_counter() - generation_started
            decoded_generated_images = decode_latents_in_microbatches(
                vae,
                single_latents.float(),
                scaling_factor,
                batch_size=int(args.vae_decode_batch_size),
            )
            require_finite_metric_tensor(
                decoded_generated_images,
                label=(
                    f"decoded_generated_images.{strategy}.rank{rank}.batch{batch_idx}"
                ),
            )
            generated_images = metric_images(decoded_generated_images)
            state = metrics[strategy]
            fake_features, fake_logits = extract_inception_features_in_microbatches(
                inception,
                generated_images,
                batch_size=int(args.vae_decode_batch_size),
            )
            require_finite_metric_tensor(
                fake_features,
                label=f"fake_features.{strategy}.rank{rank}.batch{batch_idx}",
            )
            require_finite_metric_tensor(
                fake_logits,
                label=f"fake_logits.{strategy}.rank{rank}.batch{batch_idx}",
            )
            state["fake_moments"].update(fake_features)
            if state["score_moments"] is None:
                state["score_moments"] = InceptionScoreMoments.zeros(
                    int(args.is_splits),
                    int(fake_logits.shape[-1]),
                    device,
                )
            state["score_moments"].update(
                fake_logits,
                selected_global_indices,
                int(args.samples),
            )
            count = int(generated_images.shape[0])
            if not target_latents_are_placeholders:
                state["latent_mse_sum"] += float(
                    F.mse_loss(
                        single_latents.float(),
                        target_latents.float(),
                    ).item()
                ) * count
            state["latent_rms_sum"] += float(single_latents.float().pow(2).mean().sqrt().item()) * count
            state["count"] += count
            state["generation_wall_seconds"] += generation_elapsed
            if trace and isinstance(trace.get("generation_step"), torch.Tensor):
                state["generation_step_max"] = float(trace["generation_step"].float().max().item())
            if trace:
                state["flow_content_cache_peak_bytes_per_sample"] = max(
                    float(state["flow_content_cache_peak_bytes_per_sample"]),
                    float(
                        trace.get(
                            "flow_content_cache_peak_bytes_per_sample", 0
                        )
                    ),
                )
                state["backbone_kv_cache_peak_bytes"] = max(
                    float(state["backbone_kv_cache_peak_bytes"]),
                    float(trace.get("backbone_kv_cache_peak_bytes", 0)),
                )
                cache_divergence = trace.get(
                    "flow_cfg_content_cache_divergence_by_layer"
                )
                if isinstance(cache_divergence, list):
                    if state["flow_cfg_cache_divergence_sum"] is None:
                        state["flow_cfg_cache_divergence_sum"] = [
                            0.0
                        ] * len(cache_divergence)
                    for layer_idx, value in enumerate(cache_divergence):
                        state["flow_cfg_cache_divergence_sum"][layer_idx] += (
                            float(value) * count
                        )
                    state["flow_cfg_cache_divergence_count"] += count
            if args.save_images:
                save_indexed_images(generated_images.cpu(), out_dir / str(strategy), selected_global_indices)

        generated += len(spans)
        batch_offset = seen_complete_spans
        if progress:
            iterator.set_postfix_str(
                f"rank0 {generated}; seen {min(seen_complete_spans, args.samples)}/{args.samples}",
                refresh=False,
            )
        report_evaluation_progress(
            stage="generating",
            samples_completed=min(seen_complete_spans, int(args.samples)),
            batch_idx=batch_idx,
        )
        commit_evaluation_resume(
            next_batch_idx=batch_idx + 1,
            force=seen_complete_spans >= int(args.samples),
        )

    total_generated = int(reduce_sum(float(generated), device))
    if total_generated != int(args.samples):
        debug = {
            "rank": int(rank),
            "world_size": int(world_size),
            "local_rank": int(local_rank),
            "distributed": bool(distributed),
            "device": str(device),
            "config": str(args.config),
            "model_path": str(config.model.model_path),
            "split": str(args.split),
            "samples_requested": int(args.samples),
            "batch_size": int(args.batch_size),
            "loader_batches": int(len(loader)) if hasattr(loader, "__len__") else None,
            "loader_dataset_len": int(len(loader.dataset)) if hasattr(loader, "dataset") else None,
            "image_tokens_per_img": int(image_tokens),
            "batches_seen": int(batches_seen),
            "batches_with_complete_spans": int(batches_with_complete_spans),
            "seen_complete_spans": int(seen_complete_spans),
            "selected_spans_local": int(generated),
            "selected_span_batches": int(selected_span_batches),
            "first_batch": first_batch_debug,
            "first_complete_span_batch": first_complete_span_debug,
            "distributed_env": {
                key: os.environ.get(key)
                for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
                if os.environ.get(key) is not None
            },
        }
        raise RuntimeError(
            f"Expected exactly {args.samples} generated samples across all ranks, "
            f"found {total_generated}. Debug: "
            + json.dumps(debug, sort_keys=True)
        )
    report_evaluation_progress(
        stage="reducing_metrics",
        samples_completed=total_generated,
        batch_idx=batches_seen - 1,
        force=True,
    )

    pairing_manifests = None
    if canonical_pairing_enabled:
        pairing_manifests = evaluation_pairing_manifest_hashes(
            local_noise_records=local_noise_manifest_records,
            local_sample_records=local_eval_sample_manifest_records,
            evaluation_seed=int(args.seed),
            expected_samples=int(args.samples),
        )
    guidance_diagnostics = {}

    if real_moments is not None:
        real_moments.all_reduce_()
        if int(real_moments.count.item()) != int(args.samples):
            raise RuntimeError(
                f"distributed real feature count={int(real_moments.count.item())}; "
                f"expected={args.samples}"
            )
    for strategy, state in metrics.items():
        if state["score_moments"] is None:
            raise RuntimeError(f"strategy={strategy!r} generated no Inception logits")
        state["fake_moments"].all_reduce_()
        state["score_moments"].all_reduce_()
        if int(state["fake_moments"].count.item()) != int(args.samples):
            raise RuntimeError(
                f"strategy={strategy!r} distributed generated feature count="
                f"{int(state['fake_moments'].count.item())}; expected={args.samples}"
            )
        if int(state["score_moments"].count.sum().item()) != int(args.samples):
            raise RuntimeError(
                f"strategy={strategy!r} distributed Inception Score count="
                f"{int(state['score_moments'].count.sum().item())}; "
                f"expected={args.samples}"
            )

    compute_fid = bool(
        shared_real_payload is None
        or int(args.samples) == int(shared_real_count)
        or args.allow_nonofficial_fid
    )
    if is_main_process(rank) and compute_fid:
        real_reference_moments = (
            shared_feature_moments(
                shared_real_payload,
                feature=int(args.fid_feature),
                device=torch.device("cpu"),
            )
            if shared_real_payload is not None
            else real_moments
        )
        real_mean, real_cov = real_reference_moments.mean_cov()
    peak_device_allocated_mib = reduce_max(
        (
            max(
                restored_cuda_peak_allocated_mib,
                max_memory_allocated_mib(device),
            )
            if device.type in {"cuda", "npu"}
            else None
        ),
        device,
    )
    peak_device_reserved_mib = reduce_max(
        (
            max(
                restored_cuda_peak_reserved_mib,
                max_memory_reserved_mib(device),
            )
            if device.type in {"cuda", "npu"}
            else None
        ),
        device,
    )

    results = {
        "official_protocol": official_protocol,
        "implementation_contracts": {
            "evaluator_rng_contract": EVALUATOR_RNG_CONTRACT,
            "evaluator_rng_contract_sha256": EVALUATOR_RNG_CONTRACT_SHA256,
            "canonical_initial_noise_enabled": bool(canonical_pairing_enabled),
            "evaluation_resume": {
                "schema": EVALUATION_RESUME_SCHEMA,
                "commit_schema": EVALUATION_RESUME_COMMIT_SCHEMA,
                "enabled": bool(args.resume_progress),
                "resumed": resume_payload is not None,
                "resumed_from_next_batch_idx": (
                    int(resume_next_batch_idx)
                    if resume_payload is not None
                    else None
                ),
                "checkpoint_interval_batches": int(
                    args.resume_checkpoint_interval_batches
                ),
                "contract_sha256": str(resume_contract["sha256"]),
            },
            **(pairing_manifests or {}),
        },
        "metric_protocol": {
            "fid_reducer": "symmetric_eigendecomposition",
            "fid_computed": bool(compute_fid),
            "nonofficial_fid_enabled": bool(args.allow_nonofficial_fid),
            "is_split_assignment": "contiguous_by_global_sample_index",
            "is_std": "population",
            "is_splits": int(args.is_splits),
        },
        "mechanism_diagnostics": {
            "conditional_unconditional_velocity_delta_rms_by_context": (
                guidance_diagnostics
            ),
            "generated_latent_finite_rate": 1.0,
        },
        "config": args.config,
        "model_path": str(config.model.model_path),
        "architecture": {
            "position_contract": pure_2d_position_contract(),
            "image_layout": (
                f"{int(side)}x{int(side)}x{int(config.model.image_latent_dim)}"
            ),
            "image_grid_side": int(side),
            "image_tokens_per_img": int(image_tokens),
            "image_latent_dim": int(config.model.image_latent_dim),
            "backbone_attention_output_gate": {
                "mode": str(
                    getattr(
                        model.config,
                        "backbone_attention_output_gate",
                        "none",
                    )
                ),
                "stream_sharing": True,
                "identity_scale": 2.0,
                "parameter_count": int(
                    backbone_attention_gate_parameter_count
                ),
            },
            "padded_sequence_length": (
                int(config.dataset.params.pad_to_length)
                if config.dataset.params.get("pad_to_length", None) is not None
                else None
            ),
            "flow_head": {
                "architecture": "dynamic_dual_stream",
                "depth": int(config.model.get("image_flow_depth", 8)),
                "width": int(config.model.get("image_flow_width", 1280)),
                "mlp_ratio": 1.0,
                "attention_heads": 8,
                "attention_dropout": 0.0,
                "adaln_zero_init": True,
                "position_contract": (
                    model.image_flow_head.net.position_contract()
                    if hasattr(model.image_flow_head.net, "position_contract")
                    else None
                ),
                "cache_contract": (
                    model.image_flow_head.net.cache_contract()
                    if hasattr(model.image_flow_head.net, "cache_contract")
                    else None
                ),
            },
        },
        "parameters": {
            "total": int(total_parameter_count),
            "trainable": int(trainable_parameter_count),
            "image_embedder": int(image_embedder_parameter_count),
            "flow_head": int(flow_head_parameter_count),
            "backbone_attention_output_gate": int(
                backbone_attention_gate_parameter_count
            ),
        },
        "precision_protocol": {
            "schema": "flow_eval_precision_v1",
            "model_dtype": str(args.model_dtype),
            "model_parameter_dtypes": parameter_dtypes,
            "checkpoint_weight_dtypes": stored_checkpoint_dtypes,
            "vae_dtype": str(args.vae_dtype),
            "vae_decode_batch_size": int(args.vae_decode_batch_size),
            "flow_integrator_dtype": "fp32",
            "autocast_enabled": False,
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "metric_accumulation_dtype": str(
                metric_accumulation_dtype(device)
            ),
        },
        "adapter": adapter_report,
        "model_state": model_state_report,
        "ema_checkpoint": ema_checkpoint_report,
        "split": args.split,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "samples_requested": int(args.samples),
        "samples_evaluated": int(total_generated),
        "distributed": {
            "enabled": bool(distributed),
            "world_size": int(world_size),
            "rank": int(rank),
            "local_rank": int(local_rank),
            "batch_size_global": int(args.batch_size),
            "batch_size_per_rank": int(local_batch_size),
            "dataloader_sharding": "pre_collation_global_stride",
            "device_type": str(device.type),
            "distributed_backend": (
                dist.get_backend()
                if dist.is_available() and dist.is_initialized()
                else None
            ),
            "peak_device_allocated_mib": peak_device_allocated_mib,
            "peak_device_reserved_mib": peak_device_reserved_mib,
            "peak_cuda_allocated_mib": (
                peak_device_allocated_mib
                if device.type == "cuda"
                else None
            ),
            "peak_cuda_reserved_mib": (
                peak_device_reserved_mib
                if device.type == "cuda"
                else None
            ),
        },
        "real_source": (
            "cached_original_imagenet"
            if shared_real_payload is not None
            else str(args.real_source)
        ),
        "real_stats_path": (
            str(Path(real_stats_path).resolve())
            if shared_real_payload is not None
            else None
        ),
        "real_stats_metadata": (
            shared_real_payload["metadata"]
            if shared_real_payload is not None
            else None
        ),
        "target_latents_are_placeholders": (
            target_latents_are_placeholders
        ),
        "target_decode_skipped": bool(args.skip_target_decode),
        "imagenet_train_dir": (
            str(imagenet_train_dir)
            if shared_real_payload is None and args.real_source == "imagenet_original"
            else None
        ),
        "real_image_size": (
            int(args.real_image_size)
            if shared_real_payload is not None or args.real_source == "imagenet_original"
            else None
        ),
        "cfg": float(args.cfg),
        "cfg_schedule": str(args.cfg_schedule),
        "sampling_steps": str(args.sampling_steps),
        "temperature": float(args.temperature),
        "flow_solver": str(args.flow_solver),
        "parallel_rate": int(args.parallel_rate),
        "backbone_kv_cache": not bool(args.disable_backbone_kv_cache),
        "inception_weights_path": inception_weights_path,
        "strategies": {},
    }

    for strategy, state in metrics.items():
        global_count = int(reduce_sum(float(state["count"]), device))
        global_latent_mse_sum = (
            None
            if target_latents_are_placeholders
            else reduce_sum(state["latent_mse_sum"], device)
        )
        global_latent_rms_sum = reduce_sum(state["latent_rms_sum"], device)
        global_generation_step_max = reduce_max(state.get("generation_step_max"), device)
        global_generation_wall_seconds = reduce_max(
            state.get("generation_wall_seconds"),
            device,
        )
        global_flow_cache_peak_bytes = reduce_max(
            state.get("flow_content_cache_peak_bytes_per_sample", 0.0),
            device,
        )
        global_backbone_cache_peak_bytes = reduce_max(
            state.get("backbone_kv_cache_peak_bytes", 0.0),
            device,
        )
        local_cache_divergence = state.get("flow_cfg_cache_divergence_sum")
        global_cache_divergence = (
            [
                reduce_sum(value, device)
                for value in local_cache_divergence
            ]
            if isinstance(local_cache_divergence, list)
            else None
        )
        global_cache_divergence_count = reduce_sum(
            float(state.get("flow_cfg_cache_divergence_count", 0)),
            device,
        )
        if global_count != int(args.samples):
            raise RuntimeError(
                f"strategy={strategy!r} generated count={global_count}; "
                f"expected={args.samples}"
            )
        if is_main_process(rank):
            fake_mean, fake_cov = state["fake_moments"].mean_cov()
            fid_value = (
                frechet_distance(
                    real_mean,
                    real_cov,
                    fake_mean,
                    fake_cov,
                )
                if compute_fid
                else None
            )
            is_mean, is_std, is_per_split = state["score_moments"].compute()
            fid_metric = (
                require_finite_metric_scalar(
                    fid_value,
                    label=f"strategies.{strategy}.fid",
                )
                if fid_value is not None
                else None
            )
            is_metric = require_finite_metric_scalar(
                is_mean,
                label=f"strategies.{strategy}.inception_score_mean",
            )
            is_std_metric = require_finite_metric_scalar(
                is_std,
                label=f"strategies.{strategy}.inception_score_std",
            )
            is_split_metrics = [
                require_finite_metric_scalar(
                    value,
                    label=f"strategies.{strategy}.inception_score_splits[{index}]",
                )
                for index, value in enumerate(is_per_split)
            ]
            latent_mse_metric = (
                None
                if global_latent_mse_sum is None
                else require_finite_metric_scalar(
                    global_latent_mse_sum / global_count,
                    label=f"strategies.{strategy}.latent_mse_to_target",
                )
            )
            latent_rms_metric = require_finite_metric_scalar(
                global_latent_rms_sum / global_count,
                label=f"strategies.{strategy}.latent_rms",
            )
            wall_metric = require_finite_metric_scalar(
                global_generation_wall_seconds,
                label=f"strategies.{strategy}.generation_wall_seconds",
            )
            if wall_metric <= 0.0:
                raise FloatingPointError(
                    f"formal evaluation wall time must be positive, got {wall_metric}"
                )
            results["strategies"][strategy] = {
                "count": int(global_count),
                "fid": fid_metric,
                "inception_score_mean": is_metric,
                "inception_score_std": is_std_metric,
                "inception_score_splits": is_split_metrics,
                "latent_mse_to_target": latent_mse_metric,
                "latent_rms": latent_rms_metric,
                "generation_step_max": global_generation_step_max,
                "generation_wall_seconds": wall_metric,
                "generation_samples_per_second": global_count / wall_metric,
                "flow_content_cache_peak_bytes_per_sample": int(
                    global_flow_cache_peak_bytes
                ),
                "flow_content_cache_peak_mib_per_sample": (
                    global_flow_cache_peak_bytes / (1024.0 * 1024.0)
                ),
                "backbone_kv_cache_peak_bytes_per_rank": int(
                    global_backbone_cache_peak_bytes
                ),
                "backbone_kv_cache_peak_mib_per_rank": (
                    global_backbone_cache_peak_bytes / (1024.0 * 1024.0)
                ),
                "flow_cfg_content_cache_divergence_by_layer": (
                    [
                        value / max(global_cache_divergence_count, 1.0)
                        for value in global_cache_divergence
                    ]
                    if global_cache_divergence is not None
                    else None
                ),
            }

    if is_main_process(rank):
        metrics_path = out_dir / "metrics.json"
        report_evaluation_progress(
            stage="writing_metrics",
            samples_completed=total_generated,
            batch_idx=batches_seen - 1,
            force=True,
            metrics_path=str(metrics_path.resolve()),
        )
        metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results["strategies"], indent=2))
        print(f"Saved metrics: {metrics_path}")
        report_evaluation_progress(
            stage="completed",
            samples_completed=total_generated,
            batch_idx=batches_seen - 1,
            force=True,
            completed=True,
            metrics_path=str(metrics_path.resolve()),
        )
    if distributed:
        distributed_barrier(distributed, device)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
