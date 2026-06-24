"""
Standalone latent-image dataset for pure flow-head pretraining.

Unlike the packed multimodal dataset, this loader returns one VAE latent grid per
sample. The training script flattens the grid tokens and trains an unconditional
flow objective over the image latent distribution.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


class ImageLatentFlowDataset(Dataset):
    def __init__(
        self,
        latent_dir: str,
        docs_jsonl: Optional[str] = None,
        image_tokens_per_img: int = 256,
        image_latent_dim: int = 16,
        latent_key: str = "latent",
        max_samples: int = -1,
        deduplicate_image_ids: bool = True,
        cache_path: Optional[str] = None,
        cache_mode: str = "none",
        cache_dtype: str = "float16",
        cache_mmap: bool = True,
        cache_build_wait_seconds: int = 86400,
        return_dtype: str = "float16",
    ):
        self.latent_dir = Path(latent_dir)
        self.docs_jsonl = Path(docs_jsonl) if docs_jsonl else None
        self.image_tokens_per_img = image_tokens_per_img
        self.image_latent_dim = image_latent_dim
        self.latent_key = latent_key
        self.return_dtype = _resolve_dtype(return_dtype)
        self.cache_tensor = None

        if not self.latent_dir.exists():
            raise FileNotFoundError(self.latent_dir)

        self.paths = self._index_paths(
            max_samples=max_samples,
            deduplicate_image_ids=deduplicate_image_ids,
        )
        self.cache_tensor = self._maybe_load_or_build_cache(
            cache_path=cache_path,
            cache_mode=cache_mode,
            cache_dtype=cache_dtype,
            cache_mmap=cache_mmap,
            cache_build_wait_seconds=cache_build_wait_seconds,
        )
        if self.cache_tensor is not None and not self.paths:
            self.paths = [None] * int(self.cache_tensor.shape[0])
        if not self.paths:
            raise ValueError(f"No latent files found in {self.latent_dir} and no usable cache was loaded")

    def _index_paths(self, max_samples: int, deduplicate_image_ids: bool) -> List[Path]:
        if self.docs_jsonl is None:
            paths = sorted(
                path for path in self.latent_dir.glob("*.pt")
                if path.stem.isdigit() and len(path.stem) == 12
            )
            return paths[:max_samples] if max_samples > 0 else paths

        if not self.docs_jsonl.exists():
            raise FileNotFoundError(self.docs_jsonl)

        paths: List[Path] = []
        seen = set()
        skipped = 0
        with self.docs_jsonl.open() as f:
            for line in tqdm(f, desc="Indexing flow latent JSONL", unit="docs"):
                if max_samples > 0 and len(paths) >= max_samples:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                for img_id in row.get("img_ids", []):
                    img_id = int(img_id)
                    if deduplicate_image_ids and img_id in seen:
                        continue
                    seen.add(img_id)
                    path = self.latent_dir / f"{img_id:012d}.pt"
                    if not path.exists():
                        skipped += 1
                        continue
                    paths.append(path)
                    if max_samples > 0 and len(paths) >= max_samples:
                        break

        if skipped:
            print(f"Skipped {skipped} missing latent files while indexing {self.docs_jsonl}.")
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_tensor is not None:
            tokens = self.cache_tensor[idx]
            if self.return_dtype is not None:
                tokens = tokens.to(dtype=self.return_dtype)
            return {"latents": tokens, "path_id": torch.tensor(idx, dtype=torch.long)}

        path = self.paths[idx]
        tokens = self._load_tokens(path)
        if self.return_dtype is not None:
            tokens = tokens.to(dtype=self.return_dtype)
        return {"latents": tokens, "path_id": torch.tensor(idx, dtype=torch.long)}

    def _load_tokens(self, path: Path) -> torch.Tensor:
        obj = torch.load(path, map_location="cpu")
        latent = obj[self.latent_key]
        if latent.dim() != 3:
            raise ValueError(f"{path}: expected latent [C,H,W], got {tuple(latent.shape)}")
        c, h, w = latent.shape
        tokens = latent.permute(1, 2, 0).reshape(h * w, c)
        expected = (self.image_tokens_per_img, self.image_latent_dim)
        if tuple(tokens.shape) != expected:
            raise ValueError(f"{path}: expected {expected}, got {tuple(tokens.shape)}")
        return tokens

    def _maybe_load_or_build_cache(
        self,
        cache_path: Optional[str],
        cache_mode: str,
        cache_dtype: str,
        cache_mmap: bool,
        cache_build_wait_seconds: int,
    ) -> Optional[torch.Tensor]:
        cache_mode = str(cache_mode or "none").lower()
        if cache_mode in {"none", "false", "null", ""}:
            return None
        if cache_path is None:
            raise ValueError("dataset.params.cache_path is required when cache_mode is enabled")

        path = Path(cache_path)
        if path.exists() and cache_mode != "rebuild":
            return self._load_cache_tensor(path, cache_mmap=cache_mmap)
        if cache_mode == "readonly":
            raise FileNotFoundError(f"Missing readonly latent cache: {path}")
        if cache_mode not in {"auto", "build", "rebuild"}:
            raise ValueError("cache_mode must be one of: none, readonly, auto, build, rebuild")

        self._build_cache_with_lock(
            path=path,
            dtype=_resolve_dtype(cache_dtype),
            cache_build_wait_seconds=cache_build_wait_seconds,
        )
        return self._load_cache_tensor(path, cache_mmap=cache_mmap)

    def _load_cache_tensor(self, path: Path, cache_mmap: bool) -> torch.Tensor:
        payload = torch.load(path, map_location="cpu", mmap=cache_mmap)
        tensor = payload["latents"] if isinstance(payload, dict) else payload
        expected_tail = (self.image_tokens_per_img, self.image_latent_dim)
        if tuple(tensor.shape[1:]) != expected_tail:
            raise ValueError(f"{path}: expected cached latent tail {expected_tail}, got {tuple(tensor.shape)}")
        if self.paths and tensor.shape[0] != len(self.paths):
            raise ValueError(f"{path}: expected {len(self.paths)} cached latents, got {tensor.shape[0]}")
        return tensor

    def _build_cache_with_lock(self, path: Path, dtype: torch.dtype, cache_build_wait_seconds: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        start = time.time()
        owns_lock = False
        while not owns_lock:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                owns_lock = True
            except FileExistsError:
                if path.exists():
                    return
                if time.time() - start > cache_build_wait_seconds:
                    raise TimeoutError(f"Timed out waiting for latent cache build lock: {lock_path}")
                time.sleep(5)

        try:
            if path.exists():
                return
            tensor = torch.empty(
                (len(self.paths), self.image_tokens_per_img, self.image_latent_dim),
                dtype=dtype,
            )
            for idx, latent_path in enumerate(tqdm(self.paths, desc=f"Building latent cache {path.name}", unit="img")):
                tensor[idx].copy_(self._load_tokens(latent_path).to(dtype=dtype))

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            torch.save(
                {
                    "latents": tensor,
                    "metadata": {
                        "num_images": len(self.paths),
                        "image_tokens_per_img": self.image_tokens_per_img,
                        "image_latent_dim": self.image_latent_dim,
                        "dtype": str(dtype),
                        "latent_dir": str(self.latent_dir),
                        "docs_jsonl": str(self.docs_jsonl) if self.docs_jsonl else None,
                    },
                },
                tmp_path,
            )
            os.replace(tmp_path, path)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def collate_flow_latents(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "latents": torch.stack([item["latents"] for item in batch], dim=0),
        "path_ids": torch.stack([item["path_id"] for item in batch], dim=0),
    }


def build_flow_latent_dataloaders(config):
    params = config.dataset.params
    dataset = ImageLatentFlowDataset(
        docs_jsonl=params.get("docs_jsonl", None),
        latent_dir=params.latent_dir,
        image_tokens_per_img=params.get("image_tokens_per_img", config.model.image_tokens_per_img),
        image_latent_dim=params.get("image_latent_dim", config.model.image_latent_dim),
        latent_key=params.get("latent_key", "latent"),
        max_samples=params.get("max_samples", -1),
        deduplicate_image_ids=params.get("deduplicate_image_ids", True),
        cache_path=params.get("cache_path", None),
        cache_mode=params.get("cache_mode", "none"),
        cache_dtype=params.get("cache_dtype", "float16"),
        cache_mmap=params.get("cache_mmap", True),
        cache_build_wait_seconds=params.get("cache_build_wait_seconds", 86400),
        return_dtype=params.get("return_dtype", "float16"),
    )

    val_ratio = params.get("val_ratio", 0.001)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        train_size = len(dataset) - val_size

    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))

    common_loader_kwargs = {
        "num_workers": config.training.dataloader_workers,
        "pin_memory": config.training.get("pin_memory", True),
        "collate_fn": collate_flow_latents,
    }
    if int(config.training.dataloader_workers) > 0:
        common_loader_kwargs["persistent_workers"] = config.training.get("persistent_workers", True)
        common_loader_kwargs["prefetch_factor"] = config.training.get("prefetch_factor", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader


def _resolve_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None
    name = str(dtype_name).lower()
    if name in {"none", "null", "keep"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]
