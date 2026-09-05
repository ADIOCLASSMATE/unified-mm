import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.validate_unified_single_source as preflight


PROTOCOL = Path(
    "configs/protocols/unified_single_source_0p6b_100b_ascend16.yaml"
)


def test_single_source_protocol_and_configs_use_main_baseline_b():
    protocol = OmegaConf.load(PROTOCOL)
    assert protocol.schema == "unified_single_source_0p6b_100b_v2"
    assert protocol.dual_stream_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert protocol.flow_head_attention_contract == (
        "xlnet_content_diagonal"
    )
    assert protocol.flow_condition_contract == (
        "backbone_xt_shared_query_content"
    )
    for source in preflight.SUPPORTED_SOURCES:
        config = OmegaConf.load(protocol.single_source_runs[source].config)
        assert config.model.architecture_variant == "selfless_contextual"
        assert config.model.training_objective == "selfless_dual_stream"
        assert config.model.dual_stream_attention_contract == (
            "xlnet_content_diagonal"
        )
        assert config.model.flow_head_attention_contract == (
            "xlnet_content_diagonal"
        )
        assert config.model.flow_condition_contract == (
            "backbone_xt_shared_query_content"
        )
        assert str(config.experiment.project).startswith("unified-b-")
        assert str(protocol.single_source_runs[source].output_root).startswith(
            "output/unified-b-"
        )


def _materialize_protocol(tmp_path, mutate=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    protocol = OmegaConf.load(PROTOCOL)
    protocol.base_config = str(Path(protocol.base_config).resolve())
    configs = {}
    for source in preflight.SUPPORTED_SOURCES:
        config = OmegaConf.load(protocol.single_source_runs[source].config)
        configs[source] = config
    if mutate is not None:
        mutate(protocol, configs)
    for source, config in configs.items():
        config_path = tmp_path / f"{source}.yaml"
        OmegaConf.save(config, config_path)
        protocol.single_source_runs[source].config = str(config_path)
    protocol_path = tmp_path / "protocol.yaml"
    OmegaConf.save(protocol, protocol_path)
    return protocol_path, {
        source: Path(protocol.single_source_runs[source].config)
        for source in preflight.SUPPORTED_SOURCES
    }


def test_all_three_configs_have_exact_source_matched_token_contracts():
    report = preflight.validate_preflight(
        config_paths=[
            "configs/selfless/unified_single_text_0p6b_100b_ascend16.yaml",
            "configs/selfless/unified_single_caption_0p6b_100b_ascend16.yaml",
            "configs/selfless/unified_single_t2i_0p6b_100b_ascend16.yaml",
        ],
        run_project="test-run-id-not-the-platform-project",
        audit_assets=False,
    )

    assert report["schema"] == "unified_single_source_preflight_v1"
    assert report["world_size"] == 16
    assert report["nodes"] == 1
    assert report["npu_per_node"] == 16
    assert report["target_physical_tokens_per_run"] == 100_000_595_968
    assert report["run_project"] == "test-run-id-not-the-platform-project"
    assert set(report["selected_sources"]) == {"climbmix", "i2t", "t2i"}

    runs = report["runs"]
    assert runs["climbmix"]["batch_contract"] == {
        "schedule": ["climbmix"] * 8,
        "pad_to_length_schedule": [2048] * 8,
        "micro_batch_size_per_rank": 4,
        "logical_rows_per_optimizer_step": 512,
        "physical_tokens_per_rank_per_optimizer_step": 65_536,
        "physical_tokens_per_optimizer_step": 1_048_576,
    }
    assert runs["i2t"]["batch_contract"]["schedule"] == ["i2t"] * 2
    assert runs["i2t"]["batch_contract"]["micro_batch_size_per_rank"] == 32
    assert runs["i2t"]["batch_contract"]["pad_to_length_schedule"] == [512] * 2
    assert runs["i2t"]["batch_contract"]["logical_rows_per_optimizer_step"] == 1024
    assert runs["t2i"]["batch_contract"]["schedule"] == ["t2i"] * 4
    assert runs["t2i"]["batch_contract"]["micro_batch_size_per_rank"] == 16
    assert runs["t2i"]["batch_contract"]["pad_to_length_schedule"] == [
        512,
        512,
        512,
        512,
    ]
    assert runs["t2i"]["batch_contract"]["logical_rows_per_optimizer_step"] == 1024
    for source in preflight.SUPPORTED_SOURCES:
        assert runs[source]["ema_evaluation_export"] == {
            "enabled": True,
            "every_optimizer_steps": 25_020,
            "dtype": "bf16",
            "artifacts": ["current_model", "ema_model", "pair_manifest"],
        }
    for source in preflight.SUPPORTED_SOURCES:
        run = runs[source]
        assert (
            run["max_train_steps"]
            * run["batch_contract"]["physical_tokens_per_optimizer_step"]
            == 100_000_595_968
        )
    json.dumps(report)


@pytest.mark.parametrize("source", preflight.SUPPORTED_SOURCES)
def test_rejects_nonfixed_paired_model_export_cadence(
    tmp_path,
    source,
):
    def mutate(_protocol, configs):
        configs[source].experiment.save_ema_eval_every = 25_021

    protocol_path, paths = _materialize_protocol(tmp_path, mutate)
    with pytest.raises(
        ValueError,
        match="paired-model evaluation export cadence",
    ):
        preflight.validate_preflight(
            protocol_path=protocol_path,
            config_paths=[paths[source]],
            source=source,
            audit_assets=False,
        )


def test_rejects_training_source_protocol_token_disagreement(tmp_path):
    def mutate(_protocol, configs):
        configs["i2t"].dataset.params.sources.i2t[
            "expected_global_physical_tokens_per_optimizer_step"
        ] = 524_287

    protocol_path, paths = _materialize_protocol(tmp_path, mutate)
    with pytest.raises(ValueError, match="i2t source global tokens"):
        preflight.validate_preflight(
            protocol_path=protocol_path,
            config_paths=[paths["i2t"]],
            source="i2t",
            audit_assets=False,
        )


def test_rejects_common_target_or_formal_stop_drift(tmp_path):
    def mutate_target(_protocol, configs):
        configs["climbmix"].training.target_physical_tokens -= 1

    protocol_path, paths = _materialize_protocol(tmp_path / "target", mutate_target)
    with pytest.raises(ValueError, match="climbmix target physical tokens"):
        preflight.validate_preflight(
            protocol_path=protocol_path,
            config_paths=[paths["climbmix"]],
            source="climbmix",
            audit_assets=False,
        )

    def mutate_stop(_protocol, configs):
        configs["t2i"].training.stop_after_steps -= 1

    protocol_path, paths = _materialize_protocol(tmp_path / "stop", mutate_stop)
    with pytest.raises(ValueError, match="t2i formal stop"):
        preflight.validate_preflight(
            protocol_path=protocol_path,
            config_paths=[paths["t2i"]],
            source="t2i",
            audit_assets=False,
        )


def test_rejects_audited_image_overflow_before_token_sum_check(tmp_path):
    # Validation reaches the no-truncation gate before the token-sum gate.
    unsafe_widths = [320, 512, 512, 512]

    def mutate(protocol, configs):
        protocol.single_source_runs.t2i.pad_to_length_schedule = unsafe_widths
        configs["t2i"].dataset.params.sources.t2i.pad_to_length_schedule = unsafe_widths

    protocol_path, paths = _materialize_protocol(tmp_path, mutate)
    with pytest.raises(ValueError, match="audited full-corpus max=372"):
        preflight.validate_preflight(
            protocol_path=protocol_path,
            config_paths=[paths["t2i"]],
            source="t2i",
            audit_assets=False,
        )


@pytest.mark.parametrize("source", ["i2t", "t2i"])
def test_rejects_image_workers_or_segment_packing(tmp_path, source):
    def mutate_workers(_protocol, configs):
        configs[source].dataset.params.sources[source].dataloader_workers = 1

    protocol_path, paths = _materialize_protocol(
        tmp_path / "workers", mutate_workers
    )
    with pytest.raises(ValueError, match="source dataloader workers"):
        preflight.validate_preflight(
            protocol_path=protocol_path,
            config_paths=[paths[source]],
            source=source,
            audit_assets=False,
        )

    def mutate_packing(_protocol, configs):
        configs[source].dataset.params.image.packing = {"enabled": True}

    protocol_path, paths = _materialize_protocol(
        tmp_path / "packing", mutate_packing
    )
    with pytest.raises(ValueError, match="segment packing must be disabled"):
        preflight.validate_preflight(
            protocol_path=protocol_path,
            config_paths=[paths[source]],
            source=source,
            audit_assets=False,
        )


def test_heavy_asset_audit_is_dispatched_only_for_active_source(monkeypatch):
    calls = []
    monkeypatch.setattr(
        preflight,
        "_audit_model_assets",
        lambda _config: {"model_type": "qwen3"},
    )
    monkeypatch.setattr(
        preflight,
        "_audit_climbmix_assets",
        lambda _config: calls.append("climbmix") or {"kind": "climbmix"},
    )
    monkeypatch.setattr(
        preflight,
        "_audit_imagenet_assets",
        lambda _config: calls.append("imagenet") or {"kind": "imagenet"},
    )

    preflight.validate_preflight(
        config_paths=[
            "configs/selfless/unified_single_text_0p6b_100b_ascend16.yaml"
        ],
        source="climbmix",
        audit_assets=True,
    )
    assert calls == ["climbmix"]

    calls.clear()
    preflight.validate_preflight(
        config_paths=[
            "configs/selfless/unified_single_t2i_0p6b_100b_ascend16.yaml"
        ],
        source="t2i",
        audit_assets=True,
    )
    assert calls == ["imagenet"]


def test_source_must_match_config_and_npu_count_must_match_formal_world():
    with pytest.raises(ValueError, match="does not match selected configs"):
        preflight.validate_preflight(
            config_paths=[
                "configs/selfless/unified_single_text_0p6b_100b_ascend16.yaml"
            ],
            source="t2i",
            audit_assets=False,
        )


def test_formal_world_and_lr_overrides_are_audited():
    config = "configs/selfless/unified_single_text_0p6b_100b_ascend16.yaml"
    report = preflight.validate_preflight(
        config_paths=[config],
        source="climbmix",
        formal_world_size=16,
        backbone_lr=3.0e-4,
        flow_lr=5.0e-5,
        audit_assets=False,
    )
    assert report["optimizer_overrides"] == {
        "backbone_and_special_lr": 3.0e-4,
        "flow_and_projector_lr": 5.0e-5,
    }
    with pytest.raises(ValueError, match="formal world size"):
        preflight.validate_preflight(
            config_paths=[config],
            source="climbmix",
            formal_world_size=64,
            audit_assets=False,
        )
    with pytest.raises(ValueError, match="requested backbone LR/protocol"):
        preflight.validate_preflight(
            config_paths=[config],
            source="climbmix",
            backbone_lr=2.0e-4,
            audit_assets=False,
        )
    with pytest.raises(ValueError, match="required NPU count/formal world size"):
        preflight.validate_preflight(
            config_paths=[
                "configs/selfless/unified_single_text_0p6b_100b_ascend16.yaml"
            ],
            source="climbmix",
            require_npu_count=8,
            audit_assets=False,
        )
