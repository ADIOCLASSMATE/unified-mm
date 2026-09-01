"""Verify latent validation on every rank and VAE decode only on rank zero."""

from __future__ import annotations

import os
import shutil
import tempfile
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from omegaconf import OmegaConf

import pretrain.train_selfless_flow as training


class _TinyModel(torch.nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.config = SimpleNamespace(
            image_tokens_per_img=4,
            image_flow_solver="euler",
        )
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def generate(self, task, **kwargs):
        assert task == "t2i"
        assert kwargs["use_cache"] is True
        count = len(kwargs["spans"])
        # Rank-dependent output proves every rank contributes to the reduced
        # metrics instead of waiting for rank 0 to run all image work.
        latents = torch.full(
            (count, 4, 2, 2),
            float(self.rank),
            device=kwargs["input_ids"].device,
        )
        return latents, {
            "attention_contract": "selfless_strict",
            "single_stream_content_self_diagonal": False,
            "backbone_kv_cache_enabled": True,
            "backbone_kv_cache_peak_bytes": 64,
            "generation_step": torch.ones(
                count,
                2,
                2,
                device=latents.device,
            ),
        }


class _TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.decode_calls = 0

    def decode(self, latents):
        self.decode_calls += 1
        return latents[:, :3] + self.anchor


class _DistributedAccelerator:
    def __init__(self, rank: int, device: torch.device):
        self.process_index = rank
        self.num_processes = dist.get_world_size()
        self.is_main_process = rank == 0
        self.device = device
        self.distributed_type = DistributedType.MULTI_GPU
        self.logged = []

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def gather(value):
        gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, value)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def reduce(value, reduction="sum"):
        if value.dtype == torch.float64:
            raise AssertionError(
                "distributed validation metrics must not use float64 reductions"
            )
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        if reduction == "mean":
            result /= dist.get_world_size()
        elif reduction != "sum":
            raise ValueError(reduction)
        return result

    def log(self, values, step):
        self.logged.append((step, values))


def main() -> None:
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    output_directory = [
        tempfile.mkdtemp(prefix="unified-mm-validation-") if rank == 0 else None
    ]
    dist.broadcast_object_list(output_directory, src=0)
    config = OmegaConf.create(
        {
            "experiment": {
                "output_dir": output_directory[0],
                "val_every": 1,
                "validation_image_every": 1,
                "validation_image_samples": 1,
                "validation_flow_temperature": 1.0,
                "validation_flow_cfg": 1.0,
                "validation_flow_cfg_schedule": "constant",
                "validation_flow_solver": "euler",
                "validation_vae_dtype": "fp32",
                "validation_vae_scaling_factor": 1.0,
                "validation_single_stream_images": True,
                "validation_single_stream_order_strategies": ["spatial_halton"],
                "validation_release_vae_gpu": True,
            },
            "model": {
                "image_tokens_per_img": 4,
                "image_flow_solver": "euler",
            },
        }
    )
    accelerator = _DistributedAccelerator(rank, device)
    model = _TinyModel(rank).to(device=device, dtype=torch.bfloat16)
    generator = torch.Generator(device=device).manual_seed(100 + rank)
    input_ids = torch.tensor(
        [[21, 11, 8, 8, 8, 8, 12, 2]],
        device=device,
        dtype=torch.long,
    )
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]],
        device=device,
        dtype=torch.uint8,
    )
    sigma = torch.tensor(
        [[0.0, 0.0, 0.75, 0.25, 1.0, 0.5, 0.0, 1.0]],
        device=device,
    )
    image_latents = torch.zeros(1, 8, 4, device=device)
    image_latents[:, 2:6] = torch.randn(
        1,
        4,
        4,
        device=device,
        generator=generator,
    )
    output = SimpleNamespace(
        last_hidden_state=torch.randn(
            1,
            8,
            8,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
    )

    training._VAE_CACHE = _TinyVAE()
    training._log_wandb_validation_images = lambda *args, **kwargs: None
    training.logger = logging.getLogger("distributed-validation-smoke")
    training._save_validation_flow_images(
        model=model,
        output=output,
        input_ids=input_ids,
        token_types=token_types,
        sigma=sigma,
        image_span_table=torch.tensor([[0, 0, 2, 6, rank]]),
        image_latents=image_latents,
        accelerator=accelerator,
        global_step=1,
        config=config,
    )

    completed = torch.ones((), device=device, dtype=torch.int32)
    dist.all_reduce(completed)
    decode_calls = torch.tensor(
        [training._VAE_CACHE.decode_calls],
        device=device,
        dtype=torch.int32,
    )
    gathered_decode_calls = [
        torch.empty_like(decode_calls) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_decode_calls, decode_calls)
    if rank == 0:
        images = list(
            (Path(output_directory[0]) / "validation_flow_images").glob(
                "*.png"
            )
        )
        assert int(completed.item()) == dist.get_world_size()
        assert len(images) == 4
        assert accelerator.logged
        assert int(gathered_decode_calls[0].item()) > 0
        assert all(
            int(value.item()) == 0 for value in gathered_decode_calls[1:]
        )
        print(
            f"parallel_ranks={int(completed.item())} "
            f"image_files={len(images)} numeric_logs={len(accelerator.logged)}"
        )
        shutil.rmtree(output_directory[0])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
