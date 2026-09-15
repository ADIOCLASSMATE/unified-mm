"""Conditioning-placement ablation: B visibility, caches, gradients and exports."""
import copy
import json
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

import test_dual_stream_flow_head as head_tests
import test_static_x0_content_contract as model_tests
from models.modeling_model.image_flow_loss import FlowLoss
from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.b_s2_modulation_protocol import CONFIG, expected_config, validate_b_s2_modulation_config
from utils.selfless_flow_adapter import validate_adapter_conditioning


def tiny_model():
    config = model_tests._config("xlnet_content_diagonal",
                                condition_contract=model_tests.X0_CONTENT_FLOW_CONDITION_CONTRACT)
    config.architecture_variant = "selfless_contextual"
    config.image_flow_conditioning_mode = "s2_input"
    model = Qwen3ForCausalLM(config)
    model_tests._randomize_flow_output(model)
    return model


def test_only_condition_placement_changes_parameters_and_initialization(tmp_path):
    config = tiny_model().config
    torch.manual_seed(721)
    actual = Qwen3ForCausalLM(config)
    legacy = copy.deepcopy(config)
    del legacy.image_flow_conditioning_mode
    torch.manual_seed(721)
    reference = Qwen3ForCausalLM(legacy)
    assert actual.image_flow_head.net.conditioning_mode == "s2_input"
    assert reference.image_flow_head.net.conditioning_mode == "adaln"
    assert actual.state_dict().keys() == reference.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    actual.save_pretrained(tmp_path)
    restored = Qwen3ForCausalLM.from_pretrained(tmp_path)
    assert restored.image_flow_head.net.conditioning_mode == "s2_input"
    with pytest.raises(ValueError, match="conditioning_mode"):
        FlowLoss(4, 8, 1, 32, 10, conditioning_mode="single_stream")


def test_backbone_conditions_change_stream_inputs_but_not_adaln():
    flow = head_tests._flow()
    flow.net.conditioning_mode = "s2_input"
    content, query, condition, times, positions, _, mask = head_tests._inputs()
    observed = []
    handle = flow.net.blocks[0].register_forward_pre_hook(
        lambda _, args: observed.append((args[0].detach().clone(), args[1].detach().clone())))
    def evaluate(query_condition, content_condition):
        return flow.velocity(query, times, query_condition, context_latents=content,
            context_conditions=content_condition, context_mask=mask,
            query_positions=positions, context_positions=positions)
    query_condition = condition.clone().requires_grad_()
    content_condition = (condition + .4).requires_grad_()
    result = evaluate(query_condition, content_condition)
    original = observed.copy()
    observed.clear()
    changed = evaluate(query_condition + .7, content_condition - .9)
    handle.remove()
    assert len(original) == len(observed) == 2  # content and query remain separate
    for (hidden, modulation), (changed_hidden, changed_modulation) in zip(original, observed):
        assert not torch.allclose(hidden, changed_hidden)
        torch.testing.assert_close(modulation, changed_modulation, rtol=0, atol=0)
    torch.testing.assert_close(original[0][1], flow.net._shape_time(torch.full_like(times, 1000), times.shape))
    torch.testing.assert_close(original[1][1], flow.net._shape_time(times * 1000, times.shape))
    assert not torch.allclose(result, changed)
    result.square().sum().backward()
    for condition_input in (query_condition, content_condition):
        assert torch.isfinite(condition_input.grad).all() and condition_input.grad.abs().sum() > 0


@pytest.mark.parametrize("case,kwargs", [
    ("test_incremental_cache_matches_full_sequence_last_query", {"attention_contract": "xlnet_content_diagonal"}),
    ("test_pending_content_fusion_matches_sequential_reference", {"attention_contract": "xlnet_content_diagonal", "capacity": None}),
    ("test_pending_content_fusion_matches_sequential_reference", {"attention_contract": "xlnet_content_diagonal", "capacity": 4}),
    ("test_fixed_capacity_content_cache_matches_growing_cache", {}),
    ("test_stacked_cfg_cache_matches_separate_branches", {}),
    ("test_cfg_sampling_duplicates_unpaired_pending_content", {}),
    ("test_dual_stream_checkpointing_matches_forward_and_gradients_exactly", {}),
])
def test_s2_conditioning_retains_b_cache_and_gradient_equivalence(monkeypatch, case, kwargs):
    original = head_tests._flow
    def make_flow(**options):
        flow = original(**options)
        flow.net.conditioning_mode = "s2_input"
        return flow
    monkeypatch.setattr(head_tests, "_flow", make_flow)
    getattr(head_tests, case)(**kwargs)


@pytest.mark.parametrize("case,kwargs", [
    ("test_shared_content_rf_batch_preserves_loss_and_all_gradients", {"checkpointing": False}),
    ("test_shared_content_rf_batch_preserves_loss_and_all_gradients", {"checkpointing": True}),
    ("test_training_routes_xt_query_and_x0_content_conditions", {}),
    ("test_current_x0_token_cannot_leak_into_current_velocity", {}),
    ("test_cached_generation_commits_fused_x0_and_matches_full_recompute", {}),
])
def test_s2_conditioning_retains_b_training_and_generation(monkeypatch, case, kwargs):
    original = model_tests._config
    def make_config(*args, **options):
        config = original(*args, **options)
        config.image_flow_conditioning_mode = "s2_input"
        return config
    monkeypatch.setattr(model_tests, "_config", make_config)
    if case != "test_current_x0_token_cannot_leak_into_current_velocity":
        kwargs = {**kwargs, "monkeypatch": monkeypatch}
    getattr(model_tests, case)(attention_contract="xlnet_content_diagonal", **kwargs)


def test_caches_and_adapters_cannot_cross_conditioning_modes(tmp_path):
    legacy = head_tests._flow()
    new = head_tests._flow()
    new.net.conditioning_mode = "s2_input"
    for owner, consumer in ((legacy, new), (new, legacy)):
        with pytest.raises(ValueError, match="cache contract mismatch"):
            consumer.net._validate_latent_mixer_cache(owner.empty_latent_mixer_cache())
    model = SimpleNamespace(config=SimpleNamespace(image_flow_conditioning_mode="s2_input"))
    with pytest.raises(ValueError, match="conditioning mismatch"):
        validate_adapter_conditioning(model, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({"image_flow_conditioning_mode": "s2_input"}))
    validate_adapter_conditioning(model, tmp_path / "model.safetensors")


@pytest.mark.parametrize("saved_mode", [None, "adaln", "s2_input"])
def test_checkpoint_owns_conditioning_even_with_different_evaluation_yaml(tmp_path, saved_mode):
    from test_evaluation_model_source import _evaluation_config, _text_ar_model_contract, _write_sharded_source
    from utils.evaluation_model_source import configure_model_source, resolve_evaluation_model_source
    contract = _text_ar_model_contract()
    contract.update(architecture_variant="selfless_contextual",
                    flow_condition_contract="backbone_xt_query_backbone_x0_content")
    if saved_mode is not None:
        contract["image_flow_conditioning_mode"] = saved_mode
    checkpoint = tmp_path / "checkpoint-10"
    _write_sharded_source(checkpoint, model_contract=contract)
    config = _evaluation_config()
    config.model.image_flow_conditioning_mode = "adaln" if saved_mode == "s2_input" else "s2_input"
    configure_model_source(config, resolve_evaluation_model_source(checkpoint))
    assert config.model.image_flow_conditioning_mode == (saved_mode or "adaln")


def test_formal_config_preserves_b_recipe_and_dual_streams():
    report = validate_b_s2_modulation_config(OmegaConf.load(CONFIG))
    assert report["head_streams"] == report["backbone_streams"] == 2
    config = expected_config()
    config.model.image_input_noise_strength = 0
    with pytest.raises(ValueError, match="B-aligned"):
        validate_b_s2_modulation_config(config)


def test_b_validation_images_use_ema_and_restore_rng(monkeypatch, tmp_path):
    import test_training_image_generation as image_tests
    from test_training_unified_loss_validation import Tokenizer
    from utils import training_image_generation as generation
    from utils.sharded_ema import RankShardedEMA, build_sharded_ema_layout
    image_tests.write_prompts(tmp_path)
    model = tiny_model().train()
    ema = RankShardedEMA(build_sharded_ema_layout(model, world_size=1), rank=0, decay=.9, update_after_step=0)
    ema.bind(model)
    ema.initialize_from_model(global_step=2)
    original = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(generation, "load_vae", lambda *args: image_tests.TinyVAE())
    runner = generation.TrainingImageGenerator(image_tests.config(tmp_path))
    rng = image_tests.rng_snapshot()
    for step in (2, 4):
        report = runner.run(model, Tokenizer(), device=torch.device("cpu"), step=step, output_dir=tmp_path, ema=ema)
        assert report["complete"] and report["samples"] == 2 and report["weight_source"] == "ema"
        for row in report["images"]:
            assert row["trace"]["flow_conditioning_mode"] == "s2_input"
            assert row["trace"]["flow_solver"] == "heun"
        image_tests.assert_rng(rng)
        assert model.training
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for index, name in enumerate(("cat", "dog")):
        a, b = (tmp_path / "validation_generation" / f"step-{s}" / f"{index:02d}-{name}.png" for s in (2, 4))
        assert a.read_bytes() == b.read_bytes()
