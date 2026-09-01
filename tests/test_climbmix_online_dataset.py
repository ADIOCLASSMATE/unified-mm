import json

import torch

from utils.climbmix_online_dataset import ClimbMixOnlineBatchDataset


class TinyBatchTokenizer:
    def __call__(
        self,
        texts,
        *,
        add_special_tokens,
        padding,
        truncation,
    ):
        assert not add_special_tokens
        assert not padding
        assert not truncation
        return {
            "input_ids": [
                [int(piece.removeprefix("w")) + 10 for piece in text.split()]
                for text in texts
            ]
        }


def _write_shards(tmp_path):
    paths = []
    value = 0
    for shard_index in range(2):
        path = tmp_path / f"part_{shard_index}.jsonl"
        rows = []
        for _ in range(8):
            words = []
            for _ in range(5):
                words.append(f"w{value}")
                value += 1
            rows.append(json.dumps({"text": " ".join(words)}) + "\n")
        path.write_text("".join(rows), encoding="utf-8")
        paths.append(path)
    return paths


def _dataset(paths, resume_state=None):
    return ClimbMixOnlineBatchDataset(
        shard_paths=paths,
        tokenizer=TinyBatchTokenizer(),
        eos_token_id=9,
        sequence_length=8,
        micro_batch_size=2,
        rank=0,
        world_size=1,
        seed=17,
        tokenizer_batch_documents=2,
        max_document_chars=4096,
        resume_state=resume_state,
    )


def test_online_batches_are_fixed_shape_and_segment_isolated(tmp_path):
    batch = next(iter(_dataset(_write_shards(tmp_path))))

    assert batch["source_name"] == "climbmix"
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    assert batch["position_ids"].shape == (2, 2, 8)
    assert batch["image_span_table"].shape == (0, 5)
    assert not batch["image_loss_mask"].any()
    assert batch["token_types"].eq(1).sum() == 0
    assert batch["token_types"].eq(2).any()

    for row in range(2):
        segment_ids = batch["segment_ids"][row]
        for segment_id in segment_ids.unique().tolist():
            if segment_id < 0:
                continue
            first = int(torch.nonzero(segment_ids == segment_id)[0].item())
            assert batch["labels"][row, first].item() == -100
            assert batch["sigma"][row, first].item() == 0


def test_consumed_stream_state_resumes_at_the_exact_next_batch(tmp_path):
    paths = _write_shards(tmp_path)
    iterator = iter(_dataset(paths))
    first = next(iterator)
    expected = next(iterator)

    resumed = next(iter(_dataset(paths, first["stream_state"])))
    for key in (
        "input_ids",
        "labels",
        "token_types",
        "sigma",
        "segment_ids",
        "position_ids",
    ):
        torch.testing.assert_close(resumed[key], expected[key])
    assert resumed["stream_state"] == expected["stream_state"]
    assert "state_sha256" not in resumed["stream_state"]
    assert "content_hash" not in resumed["stream_state"]
