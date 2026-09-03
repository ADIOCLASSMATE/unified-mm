import copy
import json
from types import SimpleNamespace

import torch
import pytest
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

import utils.combined_dataloaders as combined_dataloaders
from utils.combined_dataloaders import (
    ScheduledCombinedLoader,
    _clone_image_validation_loader_for_joint,
)
from utils.imagenet_flow_dataloaders import ScheduledPadCollator
from pretrain.train_selfless_flow import (
    _append_training_metrics_jsonl,
    _debug_nonfinite_loss_trace_details,
    _gradient_accumulation_plugin,
    _single_source_global_physical_token_budget,
    _source_loss_metric_payload,
    _source_task_loss_and_count,
    _validate_source_physical_token_budget,
)


class FakeImageDataset(Dataset):
    def __init__(self, source):
        self.source = source
        self.epoch = 0

    def __len__(self):
        return 4

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        return {"item": int(index), "dataset_epoch": self.epoch}


class FakeCaptionValidationDataset(FakeImageDataset):
    def __init__(self):
        super().__init__("t2i")
        self.caption_sequence_modes = ("t2i",)
        self._epoch_state = torch.zeros((), dtype=torch.int64).share_memory_()
        self.text_cache = {"old": torch.tensor([1])}
        self.sequence_cache = {"old": {}}
        self.synthetic_text_index = None

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["task_mode"] = self.caption_sequence_modes[0]
        return item


class FakeScheduledImageDataset(FakeImageDataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        length = 8
        prompt_len = 2
        image_start = 3
        image_tokens = 1
        token_types = torch.tensor(
            [0, 0, 2, 1, 2, 0, 0, 2], dtype=torch.uint8
        )
        item = super().__getitem__(index)
        item.update(
            {
                "input_ids": torch.arange(length, dtype=torch.long),
                "token_types": token_types,
                "labels": torch.arange(length, dtype=torch.long),
                "image_loss_mask": token_types.eq(1),
                "image_latents": torch.ones(image_tokens, 1),
                "image_start": torch.tensor(image_start),
                "prompt_len": torch.tensor(prompt_len),
                "suffix_len": torch.tensor(2),
                "img_id": torch.tensor(int(index) + 1),
                "reveal_seed": torch.tensor(int(index) + 10),
                "task_mode": "t2i",
            }
        )
        return item


class FakeTextDataset(IterableDataset):
    def __init__(self):
        self.cursor = 0

    def set_resume_state(self, state):
        self.cursor = 0 if state is None else int(state["cursor"])

    def __iter__(self):
        while True:
            self.cursor += 1
            yield {
                "source_name": "climbmix",
                "text_item": self.cursor,
                "stream_state": {"cursor": self.cursor},
            }


class FakeAccelerator:
    process_index = 0
    num_processes = 1

    def wait_for_everyone(self):
        return None

    def prepare_data_loader(self, loader):
        return loader


class FakeLookaheadAccelerator(FakeAccelerator):
    def prepare_data_loader(self, loader):
        return SimpleNamespace(base_dataloader=loader)


class FakeTokenizer:
    eos_token_id = 9


def _config():
    return OmegaConf.create(
        {
            "training": {
                "max_train_steps": 8,
                "seed": 42,
                "dataloader_shuffle_seed": 42,
            },
            "dataset": {
                "params": {
                    "schedule": ["climbmix", "t2i", "climbmix", "i2t"],
                    "sources": {
                        "climbmix": {
                            "dataloader_workers": 0,
                            "micro_batch_size": 1,
                        },
                        "t2i": {
                            "dataloader_workers": 0,
                            "micro_batch_size": 1,
                        },
                        "i2t": {
                            "dataloader_workers": 0,
                            "micro_batch_size": 1,
                        },
                    },
                }
            },
        }
    )


def _loader():
    loader = ScheduledCombinedLoader(
        config=_config(),
        tokenizer=FakeTokenizer(),
        image_loaders={
            "t2i": DataLoader(FakeImageDataset("t2i"), batch_size=None),
            "i2t": DataLoader(FakeImageDataset("i2t"), batch_size=None),
        },
    )
    text_dataset = FakeTextDataset()
    loader._text_dataset = text_dataset
    loader._text_loader = DataLoader(text_dataset, batch_size=None)
    loader._prepared = True
    return loader


def _single_source_loader(source_name, *, repetitions=4):
    config = _config()
    config.dataset.params.schedule = [source_name] * int(repetitions)
    if source_name == "climbmix":
        image_loaders = {}
    else:
        image_loaders = {
            source_name: DataLoader(
                FakeImageDataset(source_name), batch_size=None
            )
        }
    loader = ScheduledCombinedLoader(
        config=config,
        tokenizer=FakeTokenizer(),
        image_loaders=image_loaders,
    )
    if source_name == "climbmix":
        text_dataset = FakeTextDataset()
        loader._text_dataset = text_dataset
        loader._text_loader = DataLoader(text_dataset, batch_size=None)
    loader._prepared = True
    return loader


def test_fixed_schedule_and_rank_local_resume(tmp_path):
    loader = _loader()
    iterator = iter(loader)
    first_update = [next(iterator) for _ in range(4)]
    assert [batch["source_name"] for batch in first_update] == [
        "climbmix",
        "t2i",
        "climbmix",
        "i2t",
    ]
    assert [batch["source_schedule_position"] for batch in first_update] == [
        0,
        1,
        2,
        3,
    ]
    assert first_update[0]["text_item"] == 1
    assert first_update[2]["text_item"] == 2

    accelerator = FakeAccelerator()
    loader.save_state(tmp_path, accelerator, global_step=1)

    resumed = _loader()
    resumed.load_state(tmp_path, accelerator, global_step=1)
    resumed_iterator = iter(resumed)
    next_update = [next(resumed_iterator) for _ in range(4)]
    assert [batch["source_name"] for batch in next_update] == [
        "climbmix",
        "t2i",
        "climbmix",
        "i2t",
    ]
    assert next_update[0]["text_item"] == 3
    assert next_update[1]["item"] == 1
    assert next_update[3]["item"] == 1

    state = copy.deepcopy(resumed.state_dict(global_step=2))
    assert state["schedule_position"] == 0
    assert state["climbmix"] == {"cursor": 4}


@pytest.mark.parametrize("source_name", ["t2i", "i2t"])
def test_repeated_single_image_source_saves_only_active_cursor(
    tmp_path, source_name
):
    loader = _single_source_loader(source_name)
    iterator = iter(loader)
    first_update = [next(iterator) for _ in range(4)]
    assert [batch["source_name"] for batch in first_update] == [
        source_name
    ] * 4
    assert [
        batch["source_schedule_position"] for batch in first_update
    ] == [0, 1, 2, 3]

    accelerator = FakeAccelerator()
    loader.save_state(tmp_path, accelerator, global_step=1)
    state = torch.load(
        tmp_path / "data_state_rank_00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert set(state["image_sources"]) == {source_name}
    assert "climbmix" not in state

    resumed = _single_source_loader(source_name)
    resumed.load_state(tmp_path, accelerator, global_step=1)
    resumed_batch = next(iter(resumed))
    assert resumed_batch["source_name"] == source_name
    assert resumed_batch["dataset_epoch"] == 1
    assert resumed_batch["item"] == 0


def test_repeated_climbmix_saves_no_inactive_image_cursor(tmp_path):
    loader = _single_source_loader("climbmix")
    iterator = iter(loader)
    update = [next(iterator) for _ in range(4)]
    assert [batch["text_item"] for batch in update] == [1, 2, 3, 4]

    accelerator = FakeAccelerator()
    loader.save_state(tmp_path, accelerator, global_step=1)
    state = torch.load(
        tmp_path / "data_state_rank_00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert state["image_sources"] == {}
    assert state["climbmix"] == {"cursor": 4}

    resumed = _single_source_loader("climbmix")
    resumed.load_state(tmp_path, accelerator, global_step=1)
    assert next(iter(resumed))["text_item"] == 5


def test_nonbaseline_multi_source_schedule_is_rejected():
    config = _config()
    config.dataset.params.schedule = ["t2i", "i2t"]
    with pytest.raises(ValueError, match="frozen combined baseline"):
        ScheduledCombinedLoader(
            config=config,
            tokenizer=FakeTokenizer(),
            image_loaders={
                "t2i": DataLoader(
                    FakeImageDataset("t2i"), batch_size=None
                ),
                "i2t": DataLoader(
                    FakeImageDataset("i2t"), batch_size=None
                ),
            },
        )


def test_scheduled_pad_collator_cycles_and_fails_without_advancing():
    dataset = FakeScheduledImageDataset("t2i")
    collator = ScheduledPadCollator(
        [8, 12],
        pad_to_multiple_of=4,
    )

    first = collator([dataset[0]])
    second = collator([dataset[1]])
    third = collator([dataset[2]])
    assert [
        first["input_ids"].shape[1],
        second["input_ids"].shape[1],
        third["input_ids"].shape[1],
    ] == [8, 12, 8]
    assert [
        first["pad_schedule_position"],
        second["pad_schedule_position"],
        third["pad_schedule_position"],
    ] == [0, 1, 0]
    assert collator.position == 1

    collator.reset(0)
    oversize = dataset[0]
    oversize["input_ids"] = torch.arange(9)
    oversize["token_types"] = torch.cat(
        (oversize["token_types"], torch.tensor([2], dtype=torch.uint8))
    )
    oversize["labels"] = torch.arange(9)
    oversize["image_loss_mask"] = torch.cat(
        (oversize["image_loss_mask"], torch.tensor([False]))
    )
    oversize["suffix_len"] = torch.tensor(3)
    with pytest.raises(ValueError, match="exceeds pad_to_length=8"):
        collator([oversize])
    assert collator.position == 0


def _scheduled_image_loader():
    config = _config()
    config.dataset.params.schedule = ["t2i", "t2i"]
    config.dataset.params.sources.t2i.pad_to_length_schedule = [8, 12]
    config.dataset.params.sources.t2i.dataloader_workers = 0
    collator = ScheduledPadCollator(
        [8, 12],
        pad_to_multiple_of=4,
    )
    loader = ScheduledCombinedLoader(
        config=config,
        tokenizer=FakeTokenizer(),
        image_loaders={
            "t2i": DataLoader(
                FakeScheduledImageDataset("t2i"),
                batch_size=1,
                shuffle=False,
                num_workers=0,
                drop_last=True,
                collate_fn=collator,
            )
        },
    )
    loader._prepared = True
    return loader


def test_scheduled_padding_resume_resets_after_epoch_offset_skip(tmp_path):
    loader = _scheduled_image_loader()
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(4)]
    assert [batch["input_ids"].shape[1] for batch in batches] == [
        8,
        12,
        8,
        12,
    ]
    assert int(batches[-1]["image_span_table"][0, -1]) == 1

    accelerator = FakeAccelerator()
    loader.save_state(tmp_path, accelerator, global_step=2)
    state = torch.load(
        tmp_path / "data_state_rank_00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert state["image_sources"]["t2i"] == {
        "epoch": 1,
        "batches_consumed": 1,
    }
    assert state["pad_schedule_positions"] == {"t2i": 0}
    uninterrupted_batch = next(iterator)

    resumed = _scheduled_image_loader()
    resumed.load_state(tmp_path, accelerator, global_step=2)
    resumed_batch = next(iter(resumed))
    assert resumed_batch["input_ids"].shape[1] == 8
    assert resumed_batch["pad_schedule_position"] == 0
    assert int(resumed_batch["image_span_table"][0, -1]) == 2
    torch.testing.assert_close(
        resumed_batch["input_ids"],
        uninterrupted_batch["input_ids"],
    )
    torch.testing.assert_close(
        resumed_batch["image_span_table"],
        uninterrupted_batch["image_span_table"],
    )


def test_scheduled_padding_preparation_uses_sharded_base_without_lookahead():
    loader = _scheduled_image_loader()
    loader._prepared = False
    loader.prepare_with_accelerator(FakeLookaheadAccelerator())

    iterator = iter(loader)
    first = next(iterator)
    second = next(iterator)
    assert first["pad_schedule_position"] == 0
    assert second["pad_schedule_position"] == 1
    assert loader._pad_collators["t2i"].position == 0
    assert isinstance(loader.image_loaders["t2i"], DataLoader)


def test_real_accelerate_preparation_keeps_width_cycle_at_boundary():
    loader = _scheduled_image_loader()
    loader._prepared = False
    accelerator = Accelerator(cpu=True)
    loader.prepare_with_accelerator(accelerator)

    iterator = iter(loader)
    first = next(iterator)
    second = next(iterator)
    assert [
        first["pad_schedule_position"],
        second["pad_schedule_position"],
    ] == [0, 1]
    assert loader._pad_collators["t2i"].position == 0
    assert isinstance(loader.image_loaders["t2i"], DataLoader)
    accelerator.free_memory()


def test_scheduled_padding_checkpoint_rejects_midcycle_state():
    loader = _scheduled_image_loader()
    collator = loader._pad_collators["t2i"]
    collator([FakeScheduledImageDataset("t2i")[0]])
    with pytest.raises(RuntimeError, match="width cycle boundary"):
        loader.state_dict(global_step=0)


@pytest.mark.parametrize("source_name", ["t2i", "i2t"])
def test_builder_constructs_and_prepares_only_active_image_source(
    monkeypatch, source_name
):
    calls = []
    train = DataLoader(FakeImageDataset(source_name), batch_size=None)
    validation = DataLoader(FakeImageDataset(source_name), batch_size=None)

    def fake_build(config, tokenizer, requested_source):
        del config, tokenizer
        calls.append(requested_source)
        return train, validation

    monkeypatch.setattr(
        combined_dataloaders, "_build_image_source", fake_build
    )
    config = _config()
    config.dataset.params.schedule = [source_name] * 4
    loader, val_loader = (
        combined_dataloaders.build_unified_mixed_dataloaders(
            config, FakeTokenizer()
        )
    )

    assert calls == [source_name]
    assert set(loader.image_loaders) == {source_name}
    assert val_loader is validation
    loader.prepare_with_accelerator(FakeAccelerator())
    assert set(loader.image_loaders) == {source_name}


def test_climbmix_builder_does_not_construct_image_sources(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(
        combined_dataloaders, "_build_image_source", fail_if_called
    )
    config = _config()
    config.dataset.params.schedule = ["climbmix"] * 4
    loader, val_loader = (
        combined_dataloaders.build_unified_mixed_dataloaders(
            config, FakeTokenizer()
        )
    )

    assert loader.active_sources == ("climbmix",)
    assert loader.image_loaders == {}
    assert val_loader is None


def test_image_source_propagates_its_scheduled_widths(monkeypatch):
    captured = {}

    def fake_build(image_config, tokenizer):
        del tokenizer
        captured["config"] = image_config
        return "train", "validation"

    monkeypatch.setattr(
        combined_dataloaders,
        "build_imagenet_flow_cache_dataloaders",
        fake_build,
    )
    config = _config()
    config.dataset.preprocessing = {"max_seq_length": 512}
    config.dataset.params.image = {
        "pad_to_length": 512,
        "pad_to_length_schedule": [999],
    }
    config.dataset.params.sources.t2i.pad_to_length_schedule = [
        384,
        384,
        384,
        384,
        512,
    ]

    result = combined_dataloaders._build_image_source(
        config,
        FakeTokenizer(),
        "t2i",
    )

    assert result == ("train", "validation")
    image_config = captured["config"]
    assert list(image_config.dataset.params.pad_to_length_schedule) == [
        384,
        384,
        384,
        384,
        512,
    ]
    assert list(image_config.dataset.params.caption_sequence_modes) == [
        "t2i"
    ]
    assert image_config.training.batch_size == 1
    assert image_config.training.dataloader_workers == 0


def test_joint_validation_clone_covers_t2i_and_i2t_without_mutating_source():
    source = FakeCaptionValidationDataset()
    loader = DataLoader(
        Subset(source, [0, 1, 2]),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    joint = _clone_image_validation_loader_for_joint(loader)

    assert source.caption_sequence_modes == ("t2i",)
    assert len(joint.dataset) == 6
    assert joint.dataset.source_indices == [0, 1, 2]
    assert joint.dataset.datasets["t2i"] is not source
    assert joint.dataset.datasets["i2t"] is not source
    assert joint.dataset.datasets["t2i"].caption_sequence_modes == ("t2i",)
    assert joint.dataset.datasets["i2t"].caption_sequence_modes == ("i2t",)
    assert joint.dataset.datasets["t2i"].text_cache == {}
    assert joint.dataset.datasets["i2t"].sequence_cache == {}
    rows = [joint.dataset[index] for index in range(len(joint.dataset))]
    assert [row["task_mode"] for row in rows] == [
        "t2i",
        "i2t",
        "t2i",
        "i2t",
        "t2i",
        "i2t",
    ]
    assert [row["item"] for row in rows] == [0, 0, 1, 1, 2, 2]


def test_mixed_loader_accumulation_ignores_inner_dataloader_epoch_boundaries():
    mixed = _gradient_accumulation_plugin(
        gradient_accumulation_steps=4,
        mixed_source_training=True,
    )
    ordinary = _gradient_accumulation_plugin(
        gradient_accumulation_steps=4,
        mixed_source_training=False,
    )

    assert mixed.num_steps == 4
    assert mixed.sync_with_dataloader is False
    assert ordinary.num_steps == 4
    assert ordinary.sync_with_dataloader is True


def test_source_task_losses_and_weighted_contributions_are_separate():
    per_modality_loss = {
        "text_loss": torch.tensor(7.0),
        "image_loss": torch.tensor(3.0),
    }
    text_count = torch.tensor(11.0)
    image_count = torch.tensor(13.0)

    for source in ("climbmix", "i2t"):
        loss, count = _source_task_loss_and_count(
            source,
            per_modality_loss=per_modality_loss,
            text_count=text_count,
            image_count=image_count,
        )
        assert loss is per_modality_loss["text_loss"]
        assert count is text_count
    loss, count = _source_task_loss_and_count(
        "t2i",
        per_modality_loss=per_modality_loss,
        text_count=text_count,
        image_count=image_count,
    )
    assert loss is per_modality_loss["image_loss"]
    assert count is image_count

    # Rows are: loss*targets, targets, sum of already weighted microbatch loss.
    reduced = torch.tensor(
        [100.0, 10.0, 8.0, 40.0, 20.0, 16.0, 30.0, 5.0, 4.0]
    )
    logs, display = _source_loss_metric_payload(
        reduced,
        num_processes=2,
        gradient_accumulation_steps=4,
    )
    assert logs["train/loss_climbmix"] == 10.0
    assert logs["train/loss_t2i"] == 2.0
    assert logs["train/loss_i2t"] == 6.0
    assert logs["train/weighted_contribution_climbmix"] == 1.0
    assert logs["train/weighted_contribution_t2i"] == 2.0
    assert logs["train/weighted_contribution_i2t"] == 0.5
    assert sum(value[1] for value in display.values()) == 3.5


def test_nonfinite_loss_trace_identifies_rank_step_slot_and_source():
    trace = torch.ones(2, 8, 3)
    trace[1, 5] = torch.tensor([float("nan"), 7.0, float("inf")])

    details = _debug_nonfinite_loss_trace_details(
        trace,
        ending_global_step=12,
        gradient_accumulation_steps=4,
        source_schedule=("climbmix", "t2i", "climbmix", "i2t"),
    )

    assert len(details) == 1
    assert "rank=1,step=12,slot=2,source='t2i'" in details[0]
    assert "'weighted': nan" in details[0]
    assert "'image_loss': inf" in details[0]


def test_nonfinite_loss_trace_rejects_partial_optimizer_step():
    with pytest.raises(ValueError, match="complete optimizer steps"):
        _debug_nonfinite_loss_trace_details(
            torch.ones(2, 5, 3),
            ending_global_step=2,
            gradient_accumulation_steps=4,
            source_schedule=("climbmix", "t2i", "climbmix", "i2t"),
        )


def test_single_source_metrics_require_only_the_active_target():
    logs, display = _source_loss_metric_payload(
        torch.tensor([30.0, 5.0, 4.0]),
        num_processes=2,
        gradient_accumulation_steps=4,
        active_sources=("i2t",),
    )
    assert logs == {
        "train/loss_i2t": 6.0,
        "train/weighted_contribution_i2t": 0.5,
        "train/i2t_target_tokens": 5.0,
    }
    assert display == {"i2t": (6.0, 0.5)}

    with pytest.raises(RuntimeError, match="produced no optimization targets"):
        _source_loss_metric_payload(
            torch.zeros(3),
            num_processes=1,
            gradient_accumulation_steps=4,
            active_sources=("t2i",),
        )


def test_single_source_physical_token_budget_is_explicit_and_exact():
    config = _config()
    config.dataset.params.schedule = ["t2i"] * 5
    with pytest.raises(
        ValueError,
        match="expected_global_physical_tokens_per_optimizer_step",
    ):
        _single_source_global_physical_token_budget(config, ("t2i",))

    config.dataset.params.sources.t2i[
        "expected_global_physical_tokens_per_optimizer_step"
    ] = 524_288
    config.training.physical_tokens_per_optimizer_step = 524_288
    config.training.target_physical_tokens = (
        int(config.training.max_train_steps) * 524_288
    )
    assert (
        _single_source_global_physical_token_budget(config, ("t2i",))
        == 524_288
    )
    config.training.target_physical_tokens += 1
    with pytest.raises(ValueError, match="horizon does not match"):
        _single_source_global_physical_token_budget(config, ("t2i",))
    config.training.target_physical_tokens -= 1
    assert (
        _validate_source_physical_token_budget(
            [32_768] * 16,
            expected_global=524_288,
            source_name="t2i",
        )
        == 524_288
    )
    with pytest.raises(RuntimeError, match="refusing optimizer.step"):
        _validate_source_physical_token_budget(
            [32_768] * 15 + [32_767],
            expected_global=524_288,
            source_name="t2i",
        )


def test_unified_step_metrics_append_to_readable_jsonl(tmp_path):
    config = OmegaConf.create(
        {"experiment": {"output_dir": str(tmp_path)}}
    )
    for step in (4, 5):
        _append_training_metrics_jsonl(
            config,
            global_step=step,
            logs={
                "step_loss": 1.0 / step,
                "train/loss_climbmix": 2.0,
                "train/weighted_contribution_climbmix": 0.05,
            },
        )

    rows = [
        json.loads(line)
        for line in (tmp_path / "training_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["global_step"] for row in rows] == [4, 5]
    assert all(
        row["schema"] == "unified_training_step_metrics_v1"
        for row in rows
    )
    assert rows[-1]["metrics"]["train/loss_climbmix"] == 2.0
    assert "hash" not in (tmp_path / "training_metrics.jsonl").read_text(
        encoding="utf-8"
    ).lower()
