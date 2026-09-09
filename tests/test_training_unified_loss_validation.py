import json
import random
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.training_unified_loss_validation import UnifiedLossValidator, validation_metrics
from utils.unified_loss_protocol import loss_protocol


class Tokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return [int(s) if s.isdigit() else 3 + len(s) % 5 for s in text.split()]


def config_for_text(path, schedule=None):
    return OmegaConf.create({
        "dataset": {"class_name": "UnifiedMixedDataset", "params": {
            "schedule": schedule or ["climbmix"] * 8,
            "sources": {s: {"micro_batch_size": 1, "sequence_length": 8} for s in ("climbmix", "t2i", "i2t")}}},
        "model": {"lambda_text": .05, "lambda_image": 1.},
        "training": {"seed": 42, "gradient_accumulation_steps": len(schedule or [0] * 8)},
        "experiment": {"climbmix_validation": {"jsonl": str(path)}},
    })


def test_weighted_microbatch_mean_matches_training_not_token_mean():
    cfg = config_for_text("unused", ["climbmix", "t2i", "climbmix", "i2t"])
    # Pure text batches have losses 2 / 8 with 1 / 9 targets. Their
    # token mean is 7.4 but their mean training objective is .05 * 5.
    values = [[74, 10, .5, 2], [12, 4, 6, 2], [10, 5, .1, 1]]
    metrics = validation_metrics(values, loss_protocol(cfg))
    assert metrics["val/loss_climbmix"] == pytest.approx(7.4)
    assert metrics["val/weighted_contribution_climbmix"] == pytest.approx(.125)
    assert metrics["val/weighted_contribution_t2i"] == pytest.approx(.75)
    assert metrics["val/weighted_contribution_i2t"] == pytest.approx(.025)
    assert metrics["val/loss"] == pytest.approx(.9)
    assert metrics["val/loss"] != pytest.approx(.25 * 3 + .0125 * 2 + .025 * 7.4)


@pytest.mark.parametrize("source,weight", [("t2i", 1.), ("i2t", .05), ("climbmix", .05)])
def test_repeated_single_task_schedule_has_full_not_divided_weight(source, weight):
    cfg = config_for_text("unused", [source] * 8)
    m = validation_metrics([[35, 7, weight * (2 + 8), 2]], loss_protocol(cfg))
    assert m[f"val/loss_{source}"] == 5
    assert m["val/loss"] == pytest.approx(weight * 5)


def test_missing_source_or_mismatched_accumulation_cannot_publish_total():
    cfg = config_for_text("unused", ["climbmix", "t2i", "climbmix", "i2t"])
    with pytest.raises(ValueError, match="every active"):
        validation_metrics([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], loss_protocol(cfg))
    cfg.training.gradient_accumulation_steps = 8
    with pytest.raises(ValueError, match="complete source schedule"):
        loss_protocol(cfg)


class CountingModel(torch.nn.Module):
    lambda_text = .05
    lambda_image = 1.

    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, batch, source):
        assert self.training and not torch.is_grad_enabled()
        targets = batch["labels"].ne(-100)
        mean = batch["labels"][targets].float().mean()
        # Prove loss RNG resets per global microbatch, across ranks and steps.
        mean = mean + torch.rand(()) + random.random() + np.random.rand()
        return SimpleNamespace(loss=.05 * mean, per_modality_loss={"text_loss": mean},
                               per_modality_count={"text_tokens": targets.sum()})


def forward(model, batch, source):
    return model(batch, source)


def write_text(path):
    path.write_text(''.join(json.dumps({"text": text}) + '\n' for text in ("1 2", "3 4 5 6")))
    return path


def test_validation_keeps_rng_weights_gradients_mode_and_training_batch_groups(tmp_path):
    cfg = config_for_text(write_text(tmp_path / "text.jsonl"))
    validator = UnifiedLossValidator(cfg)
    model = CountingModel().eval()
    model.parameter.grad = torch.tensor(7.)
    random.seed(12); np.random.seed(12); torch.manual_seed(12)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = validator.run(model, Tokenizer(), device=torch.device("cpu"), step=10,
                           output_dir=tmp_path, forward_batch=forward)
    assert not model.training and model.parameter == 1 and model.parameter.grad == 7
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert result["model_mode"] == "train_no_grad" and result["model_weights"] == "current"
    assert result["metrics"]["val/climbmix_target_tokens"] == 6
    assert result["pure_text"]["independence"] == "may_have_been_seen_in_training"
    assert json.loads((tmp_path / "validation_unified_loss_metrics_step_10.json").read_text()) == result
    again = validator.run(model, Tokenizer(), device=torch.device("cpu"), step=20,
                          output_dir=tmp_path, forward_batch=forward)
    assert again["metrics"] == result["metrics"]


def test_failed_source_restores_state_without_publishing_partial_loss(tmp_path):
    cfg = config_for_text(write_text(tmp_path / "text.jsonl"))
    model = CountingModel().eval()
    state = torch.get_rng_state()
    def fail(*args):
        torch.rand(1)
        raise ValueError("broken batch")
    with pytest.raises(RuntimeError, match="failed on a training rank"):
        UnifiedLossValidator(cfg).run(model, Tokenizer(), device=torch.device("cpu"), step=10,
                                     output_dir=tmp_path, forward_batch=fail)
    assert not model.training and torch.equal(state, torch.get_rng_state())
    assert not list(tmp_path.glob("validation_unified_loss_metrics_step_*.json"))


def _distributed_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3)
    try:
        cfg = config_for_text(f"{directory}/text.jsonl")
        result = UnifiedLossValidator(cfg).run(CountingModel(), Tokenizer(), device=torch.device("cpu"),
            step=30, output_dir=directory, forward_batch=forward)
        expected = json.loads((Path(directory) / "single.json").read_text())
        assert result["metrics"] == pytest.approx(expected)
        assert result["coverage"]["climbmix"]["complete"]
        def fail_on_one_rank(model, batch, source):
            if rank == 0:
                raise ValueError("rank-local forward failure")
            return forward(model, batch, source)
        with pytest.raises(RuntimeError, match="failed on a training rank"):
            UnifiedLossValidator(cfg).run(CountingModel(), Tokenizer(), device=torch.device("cpu"),
                step=31, output_dir=directory, forward_batch=fail_on_one_rank)
        assert not (Path(directory) / "validation_unified_loss_metrics_step_31.json").exists()
    finally:
        dist.destroy_process_group()


def test_distributed_global_batches_and_empty_rank_match_single_process(tmp_path):
    cfg = config_for_text(write_text(tmp_path / "text.jsonl"))
    single = UnifiedLossValidator(cfg).run(CountingModel(), Tokenizer(), device=torch.device("cpu"),
        step=20, output_dir=tmp_path, forward_batch=forward)
    (tmp_path / "single.json").write_text(json.dumps(single["metrics"]))
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=3, join=True)


@pytest.mark.parametrize("contract", ["selfless_strict", "xlnet_content_diagonal"])
def test_real_model_shared_forward_and_all_three_validation_tasks(contract, tmp_path, monkeypatch):
    from test_imagenet_independent_validation import _config
    from test_pure_text_baseline_forward import _tiny_model
    from pretrain.train_selfless_flow import _prepare_loss_forward_batch
    from utils.combined_dataloaders import build_unified_image_validation_dataloader
    from utils.training_downstream_validation import ImageNetValidationSubset, ValidationProfile
    from scripts.evaluate_imagenet_pretraining_native import ImageNetRecord

    cfg = config_for_text(write_text(tmp_path / "text.jsonl"))
    image = _config(tmp_path, val_names=("val_1", "val_2")).dataset.params
    for cache in (image.cache_path, image.validation.cache_path):
        data = torch.load(cache, weights_only=False)
        stats = data["posterior_stats"]
        data["posterior_stats"] = torch.cat((stats[..., :1].expand(-1, -1, 4), stats[..., 1:].expand(-1, -1, 4)), dim=-1)
        torch.save(data, cache)
    image.image_latent_dim = 4
    cfg.dataset.params.image = image
    cfg.dataset.params.schedule = ["climbmix", "t2i", "climbmix", "i2t"]
    cfg.dataset.preprocessing = {"max_seq_length": 16}
    cfg.training.gradient_accumulation_steps = 4
    cfg.model.update({"boi_token_id": 11, "eoi_token_id": 12, "mask_token_id": 7,
                      "image_tokens_per_img": 4, "image_latent_dim": 4,
                      "training_objective": "selfless_dual_stream", "dual_stream_attention_contract": contract,
                      "image_uncond_prob": .1})
    loader = build_unified_image_validation_dataloader(cfg, Tokenizer(), num_workers=0)
    model = _tiny_model()
    model.config.dual_stream_attention_contract = contract
    model.model.image_input_noise_strength = .01
    batch = loader.collate_fn([loader.dataset[0]])
    # The validation forward must evaluate the same stochastic objective as a
    # real differentiable training forward, including image input noise.
    with torch.random.fork_rng():
        torch.manual_seed(77)
        kwargs, _ = _prepare_loss_forward_batch(batch, config=cfg, device=torch.device("cpu"), source_name="t2i")
        train_loss = model(**kwargs).loss
        train_loss.backward()
        assert any(p.grad is not None for p in model.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(77)
        with torch.no_grad():
            kwargs, _ = _prepare_loss_forward_batch(batch, config=cfg, device=torch.device("cpu"), source_name="t2i")
            validation_loss = model(**kwargs).loss
        torch.testing.assert_close(train_loss.detach(), validation_loss)
    calls = []
    def real_forward(current, batch, source):
        kwargs, _ = _prepare_loss_forward_batch(batch, config=cfg, device=torch.device("cpu"), source_name=source)
        assert kwargs["compute_image_loss"] == (source == "t2i")
        assert kwargs["compute_text_loss"] == (source != "t2i")
        calls.append(source)
        return current(**kwargs)
    profile = ValidationProfile.from_config(cfg)
    shared = ImageNetValidationSubset(tuple(ImageNetRecord(i - 1, i, f"/dataset/val/n00000001/val_{i}.JPEG",
        f"val/val_{i}", "n00000001", 0) for i in (2, 1)), profile.seed, profile.imagenet_per_class,
        profile.image_manifest, profile.image_classes)
    validator = UnifiedLossValidator(cfg, image_loader=loader, imagenet_subset=shared)
    result = validator.run(model, Tokenizer(), device=torch.device("cpu"),
        step=40, output_dir=tmp_path, forward_batch=real_forward)
    assert set(calls) == {"t2i", "i2t", "climbmix"}
    assert all(result["metrics"][f"val/loss_{s}"] > 0 for s in set(calls))
    assert result["metrics"]["val/loss"] == pytest.approx(sum(result["metrics"][f"val/weighted_contribution_{s}"] for s in set(calls)))
    assert model.training and all(p.grad is None for p in model.parameters())
    assert result["imagenet_subset"]["sample_ids"] == ["val/val_2", "val/val_1"]
    assert result["subsets"]["t2i"]["indices"] == result["subsets"]["i2t"]["indices"] == [1, 0]
    def forbidden(*args):
        raise AssertionError("warm validation must reuse the fixed CPU image rows")
    monkeypatch.setattr(type(loader.dataset.datasets["t2i"]), "__getitem__", forbidden)
    warm = validator.run(model, Tokenizer(), device=torch.device("cpu"),
        step=41, output_dir=tmp_path, forward_batch=real_forward)
    assert warm["metrics"] == result["metrics"]


def test_nonfinite_loss_cannot_publish_even_with_deferred_device_check(tmp_path):
    cfg = config_for_text(write_text(tmp_path / "text.jsonl"))
    model = CountingModel().eval()
    def invalid(model, batch, source):
        result = forward(model, batch, source)
        result.loss *= float("nan")
        return result
    with pytest.raises(RuntimeError, match="failed on a training rank"):
        UnifiedLossValidator(cfg).run(model, Tokenizer(), device=torch.device("cpu"), step=1,
                                     output_dir=tmp_path, forward_batch=invalid)
    assert not model.training
    assert not list(tmp_path.glob("validation_unified_loss_metrics_step_*.json"))
