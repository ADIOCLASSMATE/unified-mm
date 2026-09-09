"""Shared identities, warm data caches, and distributed validation failures."""
from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset

from utils.evaluation.native_understanding import ImageNetRecord
from utils import training_downstream_validation as downstream
from utils.training_unified_loss_validation import UnifiedLossValidationProfile


def tiny_subset(profile=None):
    profile = profile or downstream.ValidationProfile()
    records = tuple(ImageNetRecord(i, img_id, f"/val/n00000001/image_{img_id}.JPEG",
        f"val/image_{img_id}", "n00000001", 0) for i, img_id in enumerate((11, 33)))
    return downstream.ImageNetValidationSubset(records, profile.seed, profile.imagenet_per_class,
                                                profile.image_manifest, profile.image_classes)


def test_shared_selection_preserves_historical_downstream_ids_order_and_balance(monkeypatch):
    from utils.evaluation import native_understanding as image_eval

    records = [ImageNetRecord(c * 50 + i, c * 50 + i + 1, "unused",
        f"val/image_{c:04d}_{i:02d}", f"n{c:08d}", c) for c in range(1000) for i in range(50)]
    monkeypatch.setattr(image_eval, "load_imagenet_records", lambda *args: records)
    profile = downstream.ValidationProfile()
    shared = downstream.prepare_imagenet_subset(profile)
    historical = downstream.stratified_sample(records, 2000, category=lambda r: r.class_index,
                                               identity=lambda r: r.image_id, seed=profile.seed)
    assert shared.records == tuple(historical)
    assert Counter(r.class_index for r in shared.records) == {i: 2 for i in range(1000)}
    assert shared.metadata()["sample_ids"] == [r.image_id for r in historical]
    assert shared.metadata()["samples"] == 2000
    records.reverse()
    assert downstream.prepare_imagenet_subset(profile) == shared
    # Loss sharding groups global microbatches before distributing to ranks.
    batches = [shared.records[i:i + 16] for i in range(0, 2000, 16)]
    for world in (16, 64, 256):
        actual = [r.image_id for rank in range(world)
                  for i in downstream.rank_indices(len(batches), rank, world) for r in batches[i]]
        assert Counter(actual) == Counter(r.image_id for r in historical)


def test_shared_ids_map_to_cache_and_subset_offsets_with_identity_checks():
    shared = tiny_subset()
    dataset = SimpleNamespace(dataset_split="val", img_ids=torch.tensor([33, 22, 11]),
        source_paths_full={i: f"/val/n00000001/image_{i}.JPEG" for i in (11, 22, 33)},
        synsets={i: "n00000001" for i in (11, 22, 33)}, _is_training_index=lambda _: False)
    assert shared.indices_for(Subset(dataset, [1, 0, 2])) == [2, 1]
    assert shared.indices_for(Subset(dataset, [2, 1, 0])) == [0, 2]
    with pytest.raises(ValueError, match="absent"):
        shared.indices_for(Subset(dataset, [0]))
    with pytest.raises(ValueError, match="duplicate"):
        shared.indices_for(Subset(dataset, [0, 0, 2]))
    dataset.synsets[11] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        shared.indices_for(Subset(dataset, [0, 1, 2]))
    dataset.synsets[11] = "n00000001"
    dataset.source_paths_full[11] = "/val/n00000001/different_image.JPEG"
    with pytest.raises(ValueError, match="identity mismatch"):
        shared.indices_for(Subset(dataset, [0, 1, 2]))
    dataset.source_paths_full[11] = "/val/n00000001/image_11.JPEG"
    dataset._is_training_index = lambda _: True
    with pytest.raises(ValueError, match="fixed validation serialization"):
        shared.indices_for(Subset(dataset, [0, 1, 2]))


def test_sample_selection_has_one_configuration_source():
    profile = downstream.ValidationProfile()
    shared = tiny_subset(profile)
    shared.validate_profile(profile)
    for modified in (replace(profile, seed=123), replace(profile, imagenet_per_class=3),
                     replace(profile, image_manifest="other.jsonl")):
        with pytest.raises(ValueError, match="configuration"):
            shared.validate_profile(modified)
    cfg = SimpleNamespace(experiment={"loss_validation": {"image_samples": 1024}})
    with pytest.raises(ValueError, match="downstream_validation.imagenet_per_class"):
        UnifiedLossValidationProfile.from_config(cfg)
    with pytest.raises(ValueError, match="balanced"):
        replace(shared, records=shared.records[:1])
    with pytest.raises(ValueError, match="duplicate"):
        replace(shared, records=(shared.records[0], shared.records[0]))


class CountingTokenizer:
    def __init__(self):
        self.calls = 0

    def encode(self, text, add_special_tokens=False):
        self.calls += 1
        return [1, 2]


def small_model():
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(image_tokens_per_img=4, image_latent_dim=1)
    return model.train()


def fake_prepared(shared):
    ids = {task: [] for task in (*downstream.TEXT_TASKS, *downstream.GROUNDING_TASKS)}
    ids["imagenet"] = [r.image_id for r in shared.records]
    return ({task: [] for task in downstream.TEXT_TASKS}, shared.records, ["class"],
            {task: [] for task in downstream.GROUNDING_TASKS}, None, None, [], ids)


def test_warm_runner_reuses_only_cpu_preparation_and_recomputes_scores(monkeypatch, tmp_path):
    profile = downstream.ValidationProfile()
    shared = tiny_subset(profile)
    state = downstream.DownstreamValidationState(profile, imagenet_subset=shared)
    prepared_calls, score_calls = [], []
    def prepare(model, profile, subset):
        assert subset is shared
        prepared_calls.append(1)
        return fake_prepared(subset)
    def score(model, tokenizer, *args):
        assert not model.training and not torch.is_grad_enabled()
        score_calls.append(1)
        tokenizer.encode("fixed candidates")
        return {"complete": True, "samples": 2, "expected_samples": 2, "primary": model.weight.item()}
    monkeypatch.setattr(downstream, "_prepare", prepare)
    for name in ("_text_task", "_imagenet_task", "_grounding_task"):
        monkeypatch.setattr(downstream, name, score)
    model, tokenizer = small_model(), CountingTokenizer()
    cold = downstream.run_downstream_validation(model, tokenizer, device=torch.device("cpu"),
        output_dir=tmp_path / "cold", state=state)
    with torch.no_grad():
        model.weight.add_(1)
    warm = downstream.run_downstream_validation(model, tokenizer, device=torch.device("cpu"),
        output_dir=tmp_path / "warm", state=state)
    assert not cold["prepare_cache_hit"] and warm["prepare_cache_hit"]
    assert len(prepared_calls) == 1 and len(score_calls) == 22 and tokenizer.calls == 1
    assert warm["tasks"]["imagenet"]["primary"] == pytest.approx(cold["tasks"]["imagenet"]["primary"] + 1)
    assert model.training
    manifest = json.loads((tmp_path / "warm" / "subset.json").read_text())
    assert manifest["sample_ids"]["imagenet"] == shared.metadata()["sample_ids"]
    assert manifest["imagenet_subset"] == shared.metadata()
    with pytest.raises(ValueError, match="tokenizer"):
        state.prepare(model, CountingTokenizer())
    with pytest.raises(RuntimeError, match="failed on a training rank"):
        downstream.run_downstream_validation(model, tokenizer, device=torch.device("cpu"),
            output_dir=tmp_path / "invalid", state=state, profile=replace(profile, seed=123))
    assert model.training


def _write_failure_worker(rank, rendezvous, directory):
    from utils.evaluation import multimodal_likelihood as mm_eval

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = small_model()
        state = downstream.DownstreamValidationState(downstream.ValidationProfile(), imagenet_subset=tiny_subset())
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(downstream, "_prepare", lambda model, profile, subset: fake_prepared(subset))
            def fail(*args):
                raise OSError("rank zero could not write validation results")
            patch.setattr(mm_eval, "atomic_write_text", fail)
            rng = torch.get_rng_state()
            with pytest.raises(RuntimeError, match="failed on a training rank"):
                downstream.run_downstream_validation(model, CountingTokenizer(), device=torch.device("cpu"),
                    output_dir=directory, state=state)
            assert model.training and torch.equal(rng, torch.get_rng_state())
    finally:
        dist.destroy_process_group()


def test_rank_zero_write_failure_propagates_without_hanging_other_ranks(tmp_path):
    mp.spawn(_write_failure_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)


def test_training_runner_shares_selection_and_reports_whole_validation_time(monkeypatch, tmp_path):
    from test_training_unified_loss_validation import config_for_text
    from utils import training_validation as coordinator

    shared = tiny_subset()
    events = []
    class Loss:
        def __init__(self, config, **kwargs):
            self.imagenet_subset = kwargs["imagenet_subset"]
        def run(self, *args, **kwargs):
            events.append("loss")
            return {"complete": True, "metrics": {"val/loss": .1}, "wall_seconds": .01,
                    "timings": {"t2i": {"wall_seconds": .005}}}
    ema = object()
    def score(*args, **kwargs):
        assert kwargs["state"].imagenet_subset is shared and kwargs["ema"] is ema
        assert events == ["loss"]
        events.append("downstream")
        # A timed-out downstream task does not discard completed loss results.
        return {"complete": False, "wall_seconds": .01, "within_time_budget": True,
                "prepare_seconds": .001, "prepare_cache_hit": True,
                "tasks": {"imagenet": {"complete": False}}}
    monkeypatch.setattr(coordinator, "UnifiedLossValidator", Loss)
    monkeypatch.setattr(coordinator, "run_downstream_validation", score)
    runner = coordinator.TrainingValidator(config_for_text("unused"), imagenet_subset=shared)
    result = runner.run(small_model(), CountingTokenizer(), device=torch.device("cpu"), step=10,
                        output_dir=tmp_path, forward_batch=None, ema=ema)
    assert events == ["loss", "downstream"]
    assert not result["complete"] and result["loss"]["complete"]
    assert result["metrics"]["val/loss"] == .1
    assert result["metrics"]["val/validation_seconds"] == result["wall_seconds"]
    assert result["metrics"]["val/downstream_prepare_cache_hit"] == 1
    assert "val/downstream/imagenet" not in result["metrics"]
    assert json.loads((tmp_path / "validation_summary_step_10.json").read_text()) == result
