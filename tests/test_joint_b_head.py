import copy

from omegaconf import OmegaConf
import pytest
import torch

import test_joint_dit as z_tests
from test_static_x0_content_contract import _randomize_flow_output
from models.modeling_model.image_flow_loss import ContextualFlowTransformerHead
from models.modeling_model.image_flow_loss_joint_b import JointSingleStreamBHead
from models.modeling_model.modeling_joint_dit import JointDiTForCausalLM


def tiny_model(image_tokens=4):
    config = z_tests.tiny_model(image_tokens).config
    config.joint_dit_head_type = "b_single_stream"
    model = JointDiTForCausalLM(config)
    _randomize_flow_output(model)
    return model


def test_single_stream_uses_exact_b_modules_parameters_and_initialization():
    config = tiny_model().config
    torch.manual_seed(753)
    actual = JointSingleStreamBHead(config)
    torch.manual_seed(753)
    expected = ContextualFlowTransformerHead(4, 32, 4, 32, 2,
        image_tokens_per_img=4, flow_head_attention_contract="xlnet_content_diagonal")
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
    assert actual.num_heads == 8 and actual.mlp_ratio == 1
    assert all(type(a) is type(b) for a, b in zip(actual.modules(), expected.modules()) if a is not actual)
    assert actual.cache_contract()["head_kv_source"] == "current_noisy_stream"


def test_condition_and_time_only_enter_adaln_and_kv_reads_same_stream(monkeypatch):
    net = tiny_model().image_flow_head.net.eval()
    x, h, t = torch.randn(2, 4, 4), torch.randn(2, 4, 32), torch.tensor([.2, .7])
    states, caches, finals = [], [], []
    handles = [net.blocks[0].register_forward_pre_hook(lambda _, args: states.append(tuple(v.detach().clone() for v in args[:2]))),
               net.final_layer.register_forward_pre_hook(lambda _, args: finals.append(args[1].detach().clone()))]
    original = net.blocks[0].prepare_cross_cache
    def capture(hidden, *args, **kwargs):
        caches.append(hidden.detach().clone())
        return original(hidden, *args, **kwargs)
    monkeypatch.setattr(net.blocks[0], "prepare_cross_cache", capture)
    first = net(x, t, h)
    changed = net(x, t, h + .7)
    net(x, t + .1, h)
    assert not torch.allclose(first, changed)
    for index in range(3):
        torch.testing.assert_close(states[index][0], net.input_proj(x), rtol=0, atol=0)
        torch.testing.assert_close(caches[index], states[index][0], rtol=0, atol=0)
        torch.testing.assert_close(finals[index], states[index][1], rtol=0, atol=0)
    torch.testing.assert_close(states[0][1], net.cond_embed(h) + net.time_embed(t * 1000)[:, None], rtol=0, atol=0)
    assert not torch.allclose(states[0][1], states[1][1]) and not torch.allclose(states[0][1], states[2][1])
    for handle in handles:
        handle.remove()


def test_training_keeps_one_backbone_pass_rf4_and_gradients():
    model = tiny_model().train()
    b = z_tests.batch()
    calls, observed = [], []
    handles = [model.model.register_forward_pre_hook(lambda *args: calls.append(1)),
        model.image_flow_head.net.register_forward_pre_hook(lambda _, args: observed.append(args))]
    result = model(**b)
    result.loss.backward()
    assert len(calls) == len(observed) == 1
    noisy, times, condition = observed[0]
    assert noisy.shape == (4, 4, 4) and times.shape == (4,) and times.unique().numel() > 1
    torch.testing.assert_close(condition, condition[:1].expand_as(condition), rtol=0, atol=0)
    for parameter in (model.model.layers[0].self_attn.q_proj.weight,
        model.image_flow_head.net.input_proj.weight, model.image_flow_head.net.cond_embed.weight,
        model.image_flow_head.net.time_embed.mlp[0].weight):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert not any("time" in name for name, _ in model.model.named_parameters())
    for handle in handles:
        handle.remove()


def test_checkpointed_single_stream_matches_values_and_gradients():
    model = tiny_model().image_flow_head.net.train()
    checkpointed = copy.deepcopy(model)
    checkpointed.grad_checkpointing = True
    x, h, t = torch.randn(2, 4, 4), torch.randn(2, 4, 32), torch.tensor([.3, .8])
    expected, actual = model(x, t, h), checkpointed(x, t, h)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, parameter), (_, other) in zip(model.named_parameters(), checkpointed.named_parameters()):
        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("case,kwargs", [
    ("test_dit_noise_flows_bidirectionally_and_time_is_shared", {}),
    ("test_generation_reuses_one_backbone_pass_and_reports_solver_head_calls", {"cfg": 1., "solver": "heun", "head_calls": 20}),
    ("test_generation_reuses_one_backbone_pass_and_reports_solver_head_calls", {"cfg": 3.5, "solver": "heun", "head_calls": 20}),
    ("test_cfg_unconditional_branch_cannot_read_prompt", {"solver": "heun", "head_calls": 2}),
    ("test_complete_256_latent_grid_generation", {}),
    ("test_clean_context_image_is_bidirectional_and_conditions_later_target", {}),
    ("test_cached_and_full_caption_generation_match_full_image_context", {}),
])
def test_b_head_retains_z_joint_generation_and_visibility(monkeypatch, case, kwargs):
    original = z_tests.tiny_model
    def make_model(image_tokens=4):
        config = original(image_tokens).config
        config.joint_dit_head_type = "b_single_stream"
        model = JointDiTForCausalLM(config)
        _randomize_flow_output(model)
        return model
    monkeypatch.setattr(z_tests, "tiny_model", make_model)
    getattr(z_tests, case)(**kwargs)


@pytest.mark.parametrize("saved_type", [None, "s2", "b_single_stream"])
def test_checkpoint_owns_head_type_and_reloads_exact_generation(tmp_path, saved_type):
    from utils.evaluation_model_source import _apply_checkpoint_model_contract
    model = tiny_model().eval() if saved_type == "b_single_stream" else z_tests.tiny_model().eval()
    if saved_type is not None:
        model.config.joint_dit_head_type = saved_type
    model.save_pretrained(tmp_path)
    restored = JointDiTForCausalLM.from_pretrained(tmp_path).eval()
    b = z_tests.batch()
    args = dict(input_ids=b["X0_input_ids"], token_types=b["token_types"], sigma=b["flow_sigma"],
                spans=[(0, 2, 6)], initial_noise_bank=torch.randn(1, 4, 4), flow_cfg=3.5)
    torch.testing.assert_close(model.generate("t2i", **args), restored.generate("t2i", **args), rtol=0, atol=0)
    config = OmegaConf.create({"model": {"joint_dit_head_type": "s2" if saved_type == "b_single_stream" else "b_single_stream"}})
    _apply_checkpoint_model_contract(config, model.config.to_dict(), label="head export")
    assert config.model.joint_dit_head_type == (saved_type or "s2")
    assert isinstance(restored.image_flow_head.net, JointSingleStreamBHead) == (saved_type == "b_single_stream")


def test_formal_config_parameter_budget_and_launch_keep_z_controls():
    from utils.joint_b_protocol import CONFIG, RUN, validate_joint_dit_config, parameter_report
    from scripts.launch_z import launch_plan
    config = OmegaConf.load(CONFIG)
    report = validate_joint_dit_config(config)
    assert report["backbone_streams"] == 2 and report["head_streams"] == 1
    assert report["head_adaln"] == "proj(h) + time_embed(t)"
    assert parameter_report(config)["roles"]["flow_head"] == 164072976
    plan = launch_plan(smoke=False, label="r1", steps=12, experiment="z-b", environment={
        "PET_NODE_RANK": "0", "PET_NNODES": "4", "PET_NPROC_PER_NODE": "0",
        "PET_MASTER_ADDR": "10.0.0.1", "PET_MASTER_PORT": "29500"})
    assert plan["world_size"] == 64 and plan["output_root"].endswith(RUN)
    assert f"config={CONFIG}" in plan["command"]
    assert plan["preflight"][-2:] == ["--experiment", "z-b"] or "z-b" in plan["preflight"]
    config.model.image_input_noise_strength = 0
    with pytest.raises(ValueError, match="Z-aligned"):
        validate_joint_dit_config(config)


def test_validation_generates_with_b_head_and_restores_rng(monkeypatch, tmp_path):
    import test_training_image_generation as images
    from test_training_unified_loss_validation import Tokenizer
    from utils import training_image_generation as generation
    from utils.sharded_ema import RankShardedEMA, build_sharded_ema_layout
    images.write_prompts(tmp_path)
    model = tiny_model().train()
    ema = RankShardedEMA(build_sharded_ema_layout(model, world_size=1), rank=0, decay=.9, update_after_step=0)
    ema.bind(model)
    ema.initialize_from_model(global_step=2)
    original = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(generation, "load_vae", lambda *args: images.TinyVAE())
    runner = generation.TrainingImageGenerator(images.config(tmp_path))
    rng = images.rng_snapshot()
    for step in (2, 4):
        result = runner.run(model, Tokenizer(), device=torch.device("cpu"), step=step, output_dir=tmp_path, ema=ema)
        assert result["complete"] and result["method"] == "Z + B head (single stream)"
        for row in result["images"]:
            assert row["trace"]["flow_head_type"] == "b_single_stream"
            assert row["trace"]["flow_head_calls"] == 20 and row["trace"]["backbone_calls"] == 1
        images.assert_rng(rng)
        assert model.training
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
