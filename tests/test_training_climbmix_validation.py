import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.prepare_climbmix_validation import prepare
from utils.climbmix_online_dataset import ClimbMixOnlineBatchDataset
from utils.climbmix_validation_data import load_validation_manifest, validation_documents
from utils.training_climbmix_validation import ClimbMixLossValidator, ClimbMixValidationProfile, encode_documents


class Tokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [int(token) for token in text.split()]


class CountingModel(torch.nn.Module):
    lambda_text = .05

    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, **batch):
        assert not self.training and not torch.is_grad_enabled()
        assert batch["compute_text_loss"] and not batch["compute_image_loss"]
        assert batch["image_span_table"].shape == (0, 5)
        assert not batch["image_loss_mask"].any()
        assert batch["labels"][:, 0].eq(-100).all()
        assert batch["labels"][batch["token_types"] == 3].eq(-100).all()
        random.random(), np.random.rand(), torch.rand(1)
        valid = batch["labels"] != -100
        return SimpleNamespace(per_modality_loss={"text_loss": batch["labels"][valid].float().mean()},
                               per_modality_count={"text_tokens": valid.sum()})


def mask_builder(**kwargs):
    return None, None


def _jsonl(path):
    path.write_text('\n'.join(json.dumps({"text": text}) for text in ("1 2", "3 4 5 6")) + '\n')
    return path


def test_fixed_text_loss_weights_true_targets_restores_rng_and_publishes_distinct_metric(tmp_path):
    profile = ClimbMixValidationProfile(jsonl=str(_jsonl(tmp_path / "validation.jsonl")), sequence_length=8, batch_size=1)
    model = CountingModel().train()
    model.parameter.grad = torch.tensor(7.)
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    validator = ClimbMixLossValidator(profile)
    result = validator.run(model, Tokenizer(), device=torch.device("cpu"), step=10,
                           output_dir=tmp_path, mask_builder=mask_builder, training_seed=42)
    metrics = result["metrics"]
    assert metrics["val/climbmix_target_tokens"] == 6
    assert metrics["val/loss_climbmix"] == pytest.approx((2 + 9 + 4 + 5 + 6 + 9) / 6)
    assert "val/loss_text" not in metrics and "val/loss_i2t" not in metrics and "val/loss" not in metrics
    assert result["independence"] == "may_have_been_seen_in_training"
    assert model.training and model.parameter.item() == 1 and model.parameter.grad.item() == 7
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert json.loads((tmp_path / "validation_climbmix_metrics_step_10.json").read_text()) == result
    repeat = validator.run(model, Tokenizer(), device=torch.device("cpu"), step=20,
                           output_dir=tmp_path, mask_builder=mask_builder, training_seed=42, text_only=True)
    assert repeat["metrics"]["val/loss_climbmix"] == metrics["val/loss_climbmix"]
    assert repeat["metrics"]["val/loss"] == pytest.approx(.05 * metrics["val/loss_climbmix"])


def test_empty_or_training_jsonl_is_rejected_without_publishing_loss(tmp_path):
    path = _jsonl(tmp_path / "validation.jsonl")
    validator = ClimbMixLossValidator(ClimbMixValidationProfile(jsonl=str(path), sequence_length=8))
    with pytest.raises(RuntimeError, match="failed on a training rank"):
        validator.run(CountingModel(), Tokenizer(), device=torch.device("cpu"), step=1,
                      output_dir=tmp_path, mask_builder=mask_builder, training_seed=42, training_shards=[path])
    assert not list(tmp_path.glob("validation_climbmix_metrics_step_*.json"))
    with pytest.raises(ValueError, match="empty"):
        encode_documents([], Tokenizer(), ClimbMixValidationProfile())


def test_manifest_exclusion_is_effective_and_exact_resume_cannot_change_holdout(tmp_path):
    shard = tmp_path / "train.jsonl"
    shard.write_text(''.join(json.dumps({"text": f"{10+i} {10+i} {10+i}"}) + '\n' for i in range(20)))
    path = tmp_path / "subset/manifest.json"
    data = prepare(str(shard), path, per_shard=2, seed=123)
    assert prepare(str(shard), path, per_shard=2, seed=123) == data
    documents, contract = validation_documents(path)
    reserved = {int(row["text"].split()[0]) for row in documents}
    args = dict(shard_paths=[shard], tokenizer=Tokenizer(), eos_token_id=9,
                sequence_length=8, micro_batch_size=1, tokenizer_batch_documents=2)
    loader = ClimbMixOnlineBatchDataset(**args, validation_exclusion_manifest=path)
    iterator = iter(loader)
    for _ in range(40):
        batch = next(iterator)
        assert not reserved.intersection(batch["input_ids"].flatten().tolist())
    saved = batch["stream_state"]
    expected = next(iterator)
    resumed = next(iter(ClimbMixOnlineBatchDataset(**args, validation_exclusion_manifest=path, resume_state=saved)))
    assert resumed["stream_state"] == expected["stream_state"]
    assert torch.equal(resumed["input_ids"], expected["input_ids"])
    assert saved["validation_exclusion"] == contract
    with pytest.raises(ValueError, match="exclusions differ"):
        next(iter(ClimbMixOnlineBatchDataset(**args, resume_state=saved)))
    old = next(iter(ClimbMixOnlineBatchDataset(**args)))["stream_state"]
    old.pop("validation_exclusion")  # Existing checkpoints predate this field.
    next(iter(ClimbMixOnlineBatchDataset(**args, resume_state=old)))
    with pytest.raises(ValueError, match="exclusions differ"):
        next(iter(ClimbMixOnlineBatchDataset(**args, validation_exclusion_manifest=path, resume_state=old)))
    shard.write_text(shard.read_text() + '{"text":"new record"}\n')
    with pytest.raises(ValueError, match="source changed"):
        load_validation_manifest(path)


def _distributed_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3)
    try:
        profile = ClimbMixValidationProfile(jsonl=f"{directory}/validation.jsonl", sequence_length=8, batch_size=1)
        result = ClimbMixLossValidator(profile).run(CountingModel(), Tokenizer(), device=torch.device("cpu"),
            step=3, output_dir=directory, mask_builder=mask_builder, training_seed=42)
        # Two documents over three ranks: an empty rank and unequal token counts.
        assert result["samples"] == 2
        assert result["metrics"]["val/climbmix_target_tokens"] == 6
        assert result["metrics"]["val/loss_climbmix"] == pytest.approx(35 / 6)
    finally:
        dist.destroy_process_group()


def test_distributed_empty_rank_matches_global_token_weighted_loss(tmp_path):
    _jsonl(tmp_path / "validation.jsonl")
    mp.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=3, join=True)


@pytest.mark.parametrize("contract", ["selfless_strict", "xlnet_content_diagonal"])
def test_real_selfless_model_evaluates_padded_text_without_flow(contract, tmp_path):
    from test_pure_text_baseline_forward import _tiny_model
    from pretrain.train_selfless_flow import _build_backbone_attention_masks
    from omegaconf import OmegaConf

    model = _tiny_model()
    model.config.dual_stream_attention_contract = contract
    config = OmegaConf.create({"model": {"boi_token_id": 11, "training_objective": "selfless_dual_stream",
                                          "dual_stream_attention_contract": contract}})
    def forbidden(*args, **kwargs):
        raise AssertionError("pure-text validation must not execute the image flow head")
    model.image_flow_head.forward = forbidden
    profile = ClimbMixValidationProfile(jsonl=str(_jsonl(tmp_path / "validation.jsonl")), sequence_length=8)
    result = ClimbMixLossValidator(profile).run(model, Tokenizer(), device=torch.device("cpu"), step=5,
        output_dir=tmp_path, mask_builder=lambda **kw: _build_backbone_attention_masks(config=config, **kw), training_seed=42)
    assert result["metrics"]["val/climbmix_target_tokens"] == 6
    assert 0 < result["metrics"]["val/loss_climbmix"] < 10
    assert model.training and all(p.grad is None for p in model.parameters())
