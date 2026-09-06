import copy
import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.sharded_ema import RankShardedEMA, build_sharded_ema_layout
from utils.training_downstream_validation import (
    Deadline, ValidationProfile, _coverage_status, _local_phase, _text_task,
    evaluation_state, rank_indices, stratified_sample,
)


def test_subset_is_balanced_reproducible_and_world_size_independent():
    records = [(category, row) for category in range(1000) for row in range(50)]
    kwargs = dict(category=lambda r: r[0], identity=lambda r: r, seed=424242)
    subset = stratified_sample(records, 2000, **kwargs)
    assert subset == stratified_sample(list(reversed(records)), 2000, **kwargs)
    assert Counter(row[0] for row in subset) == {c: 2 for c in range(1000)}
    assert any(row[1] > 2 for row in subset)
    for world in (16, 64, 256):
        shards = [[subset[i] for i in rank_indices(len(subset), rank, world)] for rank in range(world)]
        assert Counter(row for shard in shards for row in shard) == Counter(subset)
        assert max(map(len, shards)) - min(map(len, shards)) <= 1
        assert sum(len(rank_indices(3, rank, world)) for rank in range(world)) == 3


def test_grounding_strata_and_rank_independent_image_noise():
    from scripts.evaluate_multimodal_likelihood_benchmarks import image_order_mc_seed

    records = [("large", i) for i in range(97)] + [("small", i) for i in range(3)]
    selected = stratified_sample(records, 20, category=lambda r: r[0], identity=lambda r: r, seed=9)
    assert len(selected) == 20
    assert {row[0] for row in selected} == {"large", "small"}
    reference = {i: [image_order_mc_seed(424242, i, j) for j in range(16)] for i in range(512)}
    for world in (16, 64, 256):
        actual = {i: [image_order_mc_seed(424242, i, j) for j in range(16)]
                  for rank in range(world) for i in rank_indices(512, rank, world)}
        assert reference == actual


def test_timeout_does_not_publish_a_prefix_score(monkeypatch):
    from scripts import evaluate_selfless_text_benchmarks as scorer

    def forbidden(*args, **kwargs):
        raise AssertionError("an expired task must not start a model forward")

    monkeypatch.setattr(scorer, "score_choice_requests", forbidden)
    rows = [SimpleNamespace(category=None)] * 5
    result = _text_task(None, None, rows, "arc_easy", ValidationProfile(), torch.device("cpu"), 0, 1, Deadline(-1))
    assert result == {"complete": False, "samples": 0, "expected_samples": 5, "status": "time_budget_exhausted"}


def test_timeout_after_some_batches_keeps_count_but_withholds_accuracy(monkeypatch):
    from scripts import evaluate_selfless_text_benchmarks as scorer

    monkeypatch.setattr(scorer, "encode_choice", lambda tokenizer, example, choice, length: choice)
    monkeypatch.setattr(scorer, "score_choice_requests", lambda model, requests, **kwargs: [
        SimpleNamespace(loglikelihood=-j, normalized_loglikelihood=-j) for j in requests
    ])
    rows = [SimpleNamespace(category=None, choices=("yes", "no"), label=0)] * 35
    available = iter([True, False])
    deadline = SimpleNamespace(available=lambda: next(available))
    result = _text_task(None, None, rows, "arc_easy", ValidationProfile(), torch.device("cpu"), 0, 1, deadline)
    assert result["samples"] == 32 and result["expected_samples"] == 35
    assert not result["complete"] and "metrics" not in result and "primary" not in result


class TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(7, 3)
        self.head = torch.nn.Linear(3, 7, bias=False)
        self.head.weight = self.embed.weight
        self.register_buffer("counter", torch.tensor(3))
        self.register_buffer("noncontiguous", torch.arange(6.).view(2, 3).t())


def _ema_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        model = TiedModel().train()
        model.head.eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model.embed.weight.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        layout = build_sharded_ema_layout(model, world_size=2, chunk_numel=8)
        ema = RankShardedEMA(layout, rank=rank, decay=0.9, update_after_step=0)
        ema.bind(model)
        ema.initialize_from_model()
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            model.embed.weight.add_(2)
            model.counter.fill_(42)
            model.noncontiguous.add_(7)
        live = {k: v.clone() for k, v in model.state_dict().items()}
        control = copy.deepcopy(model)
        control_optimizer = torch.optim.AdamW(control.parameters(), lr=0.01)
        control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        shards = {k: v.clone() for k, v in ema.shards.items()}
        random.seed(42)
        np.random.seed(42)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        with pytest.raises(ValueError, match="simulate evaluation failure"):
            with evaluation_state(model, torch.device("cpu"), ema):
                assert not model.training
                for name, value in model.state_dict().items():
                    assert torch.equal(value, expected[name])
                random.random(), np.random.rand(), torch.rand(7)
                raise ValueError("simulate evaluation failure")
        assert model.training and not model.head.training
        assert model.embed.weight is model.head.weight
        for name, value in model.state_dict().items():
            assert torch.equal(value, live[name])
        assert random.getstate() == rng[0]
        assert np.array_equal(np.random.get_state()[1], rng[1][1])
        assert torch.equal(torch.get_rng_state(), rng[2])
        assert all(torch.equal(value, shards[key]) for key, value in ema.shards.items())
        # Verify the next optimizer update agrees with a control that never validated.
        for current, opt in ((model, optimizer), (control, control_optimizer)):
            current.embed.weight.square().sum().backward()
            opt.step()
        assert torch.equal(model.embed.weight, control.embed.weight)
        # Uneven and empty rank shards reduce true counts without padding.
        for count in (1, 5):
            seen = torch.zeros(count)
            seen[list(rank_indices(count, rank, 2))] = 1
            assert _coverage_status(seen, count, torch.device("cpu"))["samples"] == count
        with pytest.raises(RuntimeError, match="failed on a training rank"):
            with _local_phase(torch.device("cpu")):
                if rank == 1:
                    raise ValueError("one rank could not read its local data")
    finally:
        dist.destroy_process_group()


def test_distributed_ema_restores_optimizer_buffers_rng_and_tied_storage(tmp_path):
    mp.spawn(_ema_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
