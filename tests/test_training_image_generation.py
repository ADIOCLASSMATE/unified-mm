import copy
import json
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_joint_dit import tiny_model
from test_training_unified_loss_validation import Tokenizer
from utils import training_image_generation as generation
from utils.sharded_ema import RankShardedEMA, build_sharded_ema_layout


def config(directory, samples=2):
    return OmegaConf.create({"experiment": {"validation_generation": {
        "enabled": True, "samples": samples, "prompt_file": str(Path(directory) / "prompts.json")}},
        "dataset": {"params": {"image": {"caption_t2i_prefix": "Generate an image matching this description:",
                                         "pad_to_length": 64}}}})


def write_prompts(directory):
    (Path(directory) / "prompts.json").write_text(json.dumps({"t2i": [
        {"id": "cat", "prompt": "A cat on a blue cushion."},
        {"id": "dog", "prompt": "A dog beside a red ball."},
    ]}))


class TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.rand(()), requires_grad=False)

    def decode(self, latents):
        return latents[:, :3].tanh()


def rng_snapshot():
    return random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state()


def assert_rng(states):
    actual = rng_snapshot()
    assert actual[0] == states[0] and np.array_equal(actual[1], states[1])
    assert torch.equal(actual[2], states[2])


def test_real_dit_fixed_noise_images_and_training_state_survive_two_validations(monkeypatch, tmp_path):
    write_prompts(tmp_path)
    model = tiny_model().train()
    model.image_flow_head.net.layers[0].eval()
    modes = [module.training for module in model.modules()]
    ema = RankShardedEMA(build_sharded_ema_layout(model, world_size=1), rank=0, decay=.9, update_after_step=0)
    ema.bind(model)
    ema.initialize_from_model(global_step=2)
    with torch.no_grad():
        model.image_flow_head.net.output_proj.weight.add_(1)
    original = copy.deepcopy(model.state_dict())
    model.model.embed_tokens.weight.grad = torch.ones_like(model.model.embed_tokens.weight)
    vae_calls, model_calls = [], []
    def load(*args):
        vae_calls.append(1)
        random.random(), np.random.rand(), torch.rand(1)
        return TinyVAE()
    monkeypatch.setattr(generation, "load_vae", load)
    handle = model.model.register_forward_pre_hook(lambda *args: model_calls.append(1))
    runner = generation.TrainingImageGenerator(config(tmp_path))
    states = rng_snapshot()
    for step in (2, 4):
        report = runner.run(model, Tokenizer(), device=torch.device("cpu"), step=step, output_dir=tmp_path, ema=ema)
        assert report["complete"] and report["samples"] == 2 and report["weight_source"] == "ema"
        assert [row["noise_seed"] for row in report["images"]] == [42, 1000045]
        for row in report["images"]:
            assert row["trace"]["backbone_calls"] == 1 and row["trace"]["flow_head_calls"] == 20
        assert_rng(states)
        assert [module.training for module in model.modules()] == modes
        assert torch.equal(model.model.embed_tokens.weight.grad, torch.ones_like(model.model.embed_tokens.weight))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    handle.remove()
    assert len(vae_calls) == 2 and len(model_calls) == 4
    for index, name in enumerate(("cat", "dog")):
        a, b = (tmp_path / "validation_generation" / f"step-{s}" / f"{index:02d}-{name}.png" for s in (2, 4))
        assert a.read_bytes() == b.read_bytes()
    directory = tmp_path / "validation_generation/step-4"
    assert Image.open(directory / "overview.png").size == (512, 288)
    assert "A dog beside a red ball." in (directory / "index.html").read_text()


def test_coordinator_generates_on_every_validation_even_when_downstream_times_out(monkeypatch, tmp_path):
    from utils import training_validation as coordinator

    write_prompts(tmp_path)
    cfg = config(tmp_path)
    cfg.experiment.loss_validation = {"enabled": False}
    monkeypatch.setattr(generation, "load_vae", lambda *args: TinyVAE())
    steps = []
    def downstream(*args, **kwargs):
        step = kwargs["step"]
        assert (tmp_path / f"validation_generation/step-{step}/overview.png").is_file()
        steps.append(step)
        return dict(complete=False, wall_seconds=.01, within_time_budget=True,
                    prepare_seconds=.001, prepare_cache_hit=False, tasks={})
    monkeypatch.setattr(coordinator, "run_downstream_validation", downstream)
    runner = coordinator.TrainingValidator(cfg)
    model = tiny_model().train()
    for step in (2, 4):
        report = runner.run(model, Tokenizer(), device=torch.device("cpu"), step=step,
                            output_dir=tmp_path, forward_batch=None)
        assert not report["complete"] and report["generation"]["complete"]
        assert report["metrics"]["val/generation_images"] == 2
        assert (tmp_path / report["generation"]["gallery"]).is_file()
    assert steps == [2, 4] and model.training


def _distributed_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = tiny_model().train()
        ema = RankShardedEMA(build_sharded_ema_layout(model, world_size=2), rank=rank, decay=.9, update_after_step=0)
        ema.bind(model)
        ema.initialize_from_model(global_step=2)
        with torch.no_grad():
            model.image_flow_head.net.output_proj.weight.add_(.25)
        live = model.image_flow_head.net.output_proj.weight.clone()
        states = rng_snapshot()
        runner = generation.TrainingImageGenerator(config(directory, samples=1))
        loads = []
        with pytest.MonkeyPatch.context() as patch:
            def load(*args):
                loads.append(rank)
                assert rank == 0  # Rank 1 has no image work but must join EMA collectives.
                return TinyVAE()
            patch.setattr(generation, "load_vae", load)
            result = runner.run(model, Tokenizer(), device=torch.device("cpu"), step=2, output_dir=directory, ema=ema)
            assert result["samples"] == 1 and result["images"][0]["rank"] == 0
            assert len(loads) == int(rank == 0)
            def fail(*args):
                raise OSError("simulated decoder failure")
            patch.setattr(generation, "load_vae", fail)
            with pytest.raises(RuntimeError, match="failed on a training rank"):
                runner.run(model, Tokenizer(), device=torch.device("cpu"), step=4, output_dir=directory, ema=ema)
        assert_rng(states)
        assert model.training and torch.equal(model.image_flow_head.net.output_proj.weight, live)
        assert not json.loads((Path(directory) / "validation_generation/step-4/summary.json").read_text())["complete"]
    finally:
        dist.destroy_process_group()


def test_empty_rank_and_decoder_failure_restore_sharded_ema_without_hanging(tmp_path):
    write_prompts(tmp_path)
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)


def test_default_disabled_and_invalid_configuration(tmp_path):
    disabled = generation.TrainingImageGenerator(OmegaConf.create({"experiment": {}}))
    assert disabled.run(None, None, device=torch.device("cpu"), step=2, output_dir=tmp_path) is None
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="Heun/Euler"):
        generation.ImageGenerationProfile(solver="invalid")
    write_prompts(tmp_path)
    with pytest.raises(ValueError, match="not enough"):
        generation.TrainingImageGenerator(config(tmp_path, samples=3))
