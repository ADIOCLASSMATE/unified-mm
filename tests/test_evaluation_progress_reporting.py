import json
from types import SimpleNamespace

import pytest
import torch

import scripts.evaluate_single_stream_fid_is as evaluator
from scripts.image_evaluation_metrics import (
    FeatureMoments,
    InceptionScoreMoments,
)
from scripts.evaluate_single_stream_fid_is import (
    EVALUATION_PROGRESS_SCHEMA,
    decode_latents_in_microbatches,
    emit_evaluation_progress,
    build_canonical_initial_noise_bank,
    build_evaluation_resume_contract,
    evaluation_pairing_manifests,
    evaluation_metrics_from_state,
    evaluation_metrics_state,
    evaluation_progress_payload,
    extract_inception_features_in_microbatches,
    load_evaluation_resume_checkpoint,
    save_evaluation_resume_checkpoint,
    shard_unpacked_batch_rows,
)


def test_evaluation_contract_uses_only_readable_identity(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training: {}\n", encoding="utf-8")

    noise, noise_records = build_canonical_initial_noise_bank(
        [0, 1],
        evaluation_seed=42,
    )
    assert tuple(noise.shape) == (2, 256, 16)
    assert noise_records == [
        {"global_sample_index": 0, "canonical_noise_seed": 42},
        {"global_sample_index": 1, "canonical_noise_seed": 43},
    ]
    pairing = evaluation_pairing_manifests(
        local_noise_records=noise_records,
        local_sample_records=[
            {"global_sample_index": 0, "image_id": 10},
            {"global_sample_index": 1, "image_id": 11},
        ],
        evaluation_seed=42,
        expected_samples=2,
    )
    assert pairing["runtime_hashing_enabled"] is False
    assert not any("sha256" in key for key in pairing)

    args = SimpleNamespace(
        adapter="none",
        model_state="",
        ema_checkpoint="",
        caption_sequence_mode="t2i",
        batch_size=32,
        vae_decode_batch_size=2,
        samples=32,
        seed=42,
        cfg=3.5,
        cfg_schedule="constant",
        sampling_steps="10",
        temperature=1.0,
        flow_solver="heun",
        parallel_rate=1,
        disable_backbone_kv_cache=False,
        model_dtype="bf16",
        vae_dtype="fp32",
        fid_feature=2048,
        is_splits=8,
        save_images=False,
    )
    contract = build_evaluation_resume_contract(
        args=args,
        config_path=config_path,
        model_path=tmp_path,
        strategies=["spatial_halton"],
        world_size=16,
        canonical_pairing_enabled=True,
        target_latents_are_placeholders=False,
        real_stats_path="",
        inception_weights_path=None,
        image_tokens=256,
        attention_contract="xlnet_content_diagonal",
    )
    assert contract["runtime_hashing_enabled"] is False
    assert "sha256" not in contract
    assert (
        contract["generation"]["dual_stream_attention_contract"]
        == "xlnet_content_diagonal"
    )
    assert contract["generation"]["single_stream_visible_content_diagonal"]
    assert not contract["generation"]["single_stream_current_query_diagonal"]


def test_evaluation_progress_payload_reports_rate_and_eta():
    payload = evaluation_progress_payload(
        stage="generating",
        samples_completed=2500,
        samples_total=10000,
        elapsed_seconds=5000.0,
        strategies=["spatial_halton"],
        world_size=8,
        batch_idx=12,
        updated_at="2026-07-26T00:00:00Z",
    )
    assert payload == {
        "schema": EVALUATION_PROGRESS_SCHEMA,
        "stage": "generating",
        "completed": False,
        "samples_completed": 2500,
        "samples_total": 10000,
        "progress_percent": 25.0,
        "elapsed_seconds": 5000.0,
        "samples_per_second": 0.5,
        "eta_seconds": 15000.0,
        "batch_idx": 12,
        "strategies": ["spatial_halton"],
        "world_size": 8,
        "metrics_path": None,
        "updated_at": "2026-07-26T00:00:00Z",
    }


def test_distributed_batch_row_shard_rebases_unpacked_spans():
    tensors = {
        "input_ids": torch.arange(5 * 4).view(5, 4),
        "token_types": torch.arange(5 * 4).view(5, 4) + 100,
        "sigma": torch.arange(5 * 4).view(5, 4) + 200,
        "image_latents": torch.arange(5 * 4 * 2).view(5, 4, 2),
    }
    sharded, spans = shard_unpacked_batch_rows(
        tensors,
        [(1, 1, 3), (4, 1, 3)],
    )

    assert spans == [(0, 1, 3), (1, 1, 3)]
    for name, tensor in tensors.items():
        assert torch.equal(sharded[name], tensor[[1, 4]])


def test_distributed_batch_row_shard_rejects_packed_rows():
    with pytest.raises(ValueError, match="validation to remain unpacked"):
        shard_unpacked_batch_rows(
            {"input_ids": torch.zeros(2, 8, dtype=torch.long)},
            [(0, 0, 2), (0, 4, 6)],
        )


def test_vae_decode_microbatch_preserves_order(monkeypatch):
    calls = []

    def fake_decode(_vae, latents, _scaling_factor):
        calls.append(int(latents.shape[0]))
        return latents + 1

    monkeypatch.setattr(evaluator, "decode_latents", fake_decode)
    latents = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1)
    decoded = decode_latents_in_microbatches(
        object(),
        latents,
        1.0,
        batch_size=2,
    )

    assert calls == [2, 2, 1]
    assert torch.equal(decoded, latents + 1)


def test_inception_microbatch_preserves_features_and_logits(monkeypatch):
    calls = []

    def fake_extract(_inception, images):
        calls.append(int(images.shape[0]))
        values = images.reshape(images.shape[0], -1)
        return values + 1, values + 2

    monkeypatch.setattr(
        evaluator,
        "extract_inception_features",
        fake_extract,
    )
    images = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1)
    features, logits = extract_inception_features_in_microbatches(
        object(),
        images,
        batch_size=2,
    )

    assert calls == [2, 2, 1]
    assert torch.equal(features, images.reshape(5, 1) + 1)
    assert torch.equal(logits, images.reshape(5, 1) + 2)


@pytest.mark.parametrize(
    ("batch_size", "expected_calls"),
    [(0, [5]), (5, [5]), (8, [5])],
)
def test_inception_microbatch_full_batch_modes(
    monkeypatch,
    batch_size,
    expected_calls,
):
    calls = []

    def fake_extract(_inception, images):
        calls.append(int(images.shape[0]))
        return images, images

    monkeypatch.setattr(
        evaluator,
        "extract_inception_features",
        fake_extract,
    )
    images = torch.zeros(5, 1, 1, 1)
    extract_inception_features_in_microbatches(
        object(),
        images,
        batch_size=batch_size,
    )

    assert calls == expected_calls


def test_inception_microbatch_rejects_negative_batch_size():
    with pytest.raises(ValueError, match="must be nonnegative"):
        extract_inception_features_in_microbatches(
            object(),
            torch.zeros(1, 1, 1, 1),
            batch_size=-1,
        )


def test_emit_evaluation_progress_writes_atomic_json_and_newline_log(
    tmp_path,
    capsys,
):
    payload = evaluation_progress_payload(
        stage="completed",
        samples_completed=10000,
        samples_total=10000,
        elapsed_seconds=20000.0,
        strategies=["spatial_halton"],
        world_size=8,
        completed=True,
        metrics_path="/tmp/metrics.json",
        updated_at="2026-07-26T00:00:00Z",
    )
    emit_evaluation_progress(tmp_path, payload)

    progress_path = tmp_path / "evaluation_progress.json"
    assert json.loads(progress_path.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob(".evaluation_progress.json.*.tmp")) == []
    output = capsys.readouterr().out
    assert output.startswith("[EvalProgress] {")
    assert '"samples_completed":10000' in output
    assert output.endswith("\n")


def _metric_state_for_resume_test():
    device = torch.device("cpu")
    fake_moments = FeatureMoments.zeros(2, device)
    fake_moments.update(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        )
    )
    score_moments = InceptionScoreMoments.zeros(2, 3, device)
    score_moments.update(
        torch.tensor(
            [[1.0, 0.0, -1.0], [0.5, 1.5, -0.5]],
            dtype=torch.float64,
        ),
        [0, 1],
    )
    return {
        "spatial_halton": {
            "fake_moments": fake_moments,
            "score_moments": score_moments,
            "latent_mse_sum": 1.25,
            "latent_rms_sum": 2.5,
            "count": 2,
            "generation_wall_seconds": 4.0,
            "generation_step_max": 1.0,
            "flow_content_cache_peak_bytes_per_sample": 2048.0,
            "flow_cfg_cache_divergence_sum": [0.25, 0.5],
            "flow_cfg_cache_divergence_count": 2,
        }
    }


def test_evaluation_metric_state_round_trip():
    original = _metric_state_for_resume_test()
    serialized = evaluation_metrics_state(original)
    restored = evaluation_metrics_from_state(
        serialized,
        strategies=["spatial_halton"],
        device=torch.device("cpu"),
    )

    original_state = original["spatial_halton"]
    restored_state = restored["spatial_halton"]
    assert torch.equal(
        restored_state["fake_moments"].count,
        original_state["fake_moments"].count,
    )
    assert torch.equal(
        restored_state["fake_moments"].sum,
        original_state["fake_moments"].sum,
    )
    assert torch.equal(
        restored_state["fake_moments"].outer_sum,
        original_state["fake_moments"].outer_sum,
    )
    assert torch.equal(
        restored_state["score_moments"].count,
        original_state["score_moments"].count,
    )
    assert torch.equal(
        restored_state["score_moments"].probability_sum,
        original_state["score_moments"].probability_sum,
    )
    assert restored_state["flow_cfg_cache_divergence_sum"] == [0.25, 0.5]
    assert restored_state["generation_step_max"] == 1.0


def test_evaluation_resume_checkpoint_commits_atomically_and_rejects_stale_contract(
    tmp_path,
):
    device = torch.device("cpu")
    contract = {"name": "contract-a"}
    state = {
        "next_batch_idx": 3,
        "generated": 2,
        "metrics": evaluation_metrics_state(
            _metric_state_for_resume_test()
        ),
    }
    save_evaluation_resume_checkpoint(
        tmp_path,
        contract=contract,
        rank=0,
        world_size=1,
        distributed=False,
        device=device,
        state=state,
    )

    restored = load_evaluation_resume_checkpoint(
        tmp_path,
        contract=contract,
        rank=0,
        world_size=1,
    )
    assert restored["next_batch_idx"] == 3
    assert restored["generated"] == 2
    commit = json.loads(
        (tmp_path / "resume_state" / "commit.json").read_text(
            encoding="utf-8"
        )
    )
    assert commit["batch_directory"] == "batch-00000003"
    assert list(
        (tmp_path / "resume_state" / "batch-00000003").glob(".*.tmp")
    ) == []

    with pytest.raises(ValueError, match="stale evaluator resume"):
        load_evaluation_resume_checkpoint(
            tmp_path,
            contract={"name": "contract-b"},
            rank=0,
            world_size=1,
        )
