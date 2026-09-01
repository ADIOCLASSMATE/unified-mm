from types import SimpleNamespace

import torch
from accelerate.utils import DistributedType
from omegaconf import OmegaConf

import pretrain.train_selfless_flow as training


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            image_tokens_per_img=4,
            image_flow_solver="euler",
        )
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.generate_calls = 0

    def generate(self, task, **kwargs):
        assert task == "t2i"
        assert kwargs["use_cache"] is True
        self.generate_calls += 1
        count = len(kwargs["spans"])
        latents = torch.zeros(
            count,
            4,
            2,
            2,
            device=kwargs["input_ids"].device,
        )
        return latents, {
            "attention_contract": "selfless_strict",
            "single_stream_content_self_diagonal": False,
            "backbone_kv_cache_enabled": True,
            "backbone_kv_cache_peak_bytes": 64,
            "generation_step": torch.ones(count, 2, 2),
        }


class _NonMainAccelerator:
    device = torch.device("cpu")
    distributed_type = DistributedType.NO
    is_main_process = False

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def gather(value):
        return value

    @staticmethod
    def reduce(value, reduction="sum"):
        assert reduction == "mean"
        return value

    @staticmethod
    def log(values, step):
        raise AssertionError((values, step))


def test_non_main_validation_keeps_vae_off_device(monkeypatch, tmp_path):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(training, "_load_vae_decoder", fail_if_loaded)
    config = OmegaConf.create(
        {
            "experiment": {
                "output_dir": str(tmp_path),
                "val_every": 1,
                "validation_image_every": 1,
                "validation_image_samples": 1,
                "validation_flow_temperature": 1.0,
                "validation_flow_cfg": 1.0,
                "validation_flow_cfg_schedule": "constant",
                "validation_flow_solver": "euler",
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
    model = _TinyModel()
    input_ids = torch.tensor([[21, 11, 8, 8, 8, 8, 12, 2]])
    token_types = torch.tensor(
        [[0, 2, 1, 1, 1, 1, 2, 0]], dtype=torch.uint8
    )
    sigma = torch.tensor(
        [[0.0, 0.0, 0.75, 0.25, 1.0, 0.5, 0.0, 1.0]]
    )
    image_latents = torch.zeros(1, 8, 4)
    image_latents[:, 2:6] = torch.randn(1, 4, 4)
    output = SimpleNamespace(last_hidden_state=torch.randn(1, 8, 8))

    training._save_validation_flow_images(
        model=model,
        output=output,
        input_ids=input_ids,
        token_types=token_types,
        sigma=sigma,
        image_span_table=torch.tensor([[0, 0, 2, 6, 123]]),
        image_latents=image_latents,
        accelerator=_NonMainAccelerator(),
        global_step=1,
        config=config,
    )

    assert model.generate_calls == 1
