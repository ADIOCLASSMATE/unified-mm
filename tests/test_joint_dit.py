import copy
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import pytest
import torch
from omegaconf import OmegaConf

from models.modeling_model.modeling_joint_dit import (
    JointDiTConfig, JointDiTForCausalLM, joint_backbone_masks, joint_image_sigma,
)
from models.modeling_model.modeling_single_stream_text_ar import _materialize_allowed_mask


def tiny_model(image_tokens=4):
    torch.manual_seed(42)
    cfg = JointDiTConfig(vocab_size=40, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=512, tie_word_embeddings=True, pad_token_id=0,
        eos_token_id=9, mask_token_id=7, image_mask_token_id=8, boi_token_id=11, eoi_token_id=12,
        architecture_variant="selfless_joint_dit", training_objective="selfless_dual_stream",
        dual_stream_attention_contract="xlnet_content_diagonal",
        flow_head_attention_contract="joint_bidirectional", flow_condition_contract="backbone_xt_fixed",
        training_image_sigma_order="joint", image_tokens_per_img=image_tokens, image_latent_dim=4,
        image_flow_width=32, image_flow_depth=2, image_flow_batch_mul=4,
        image_flow_num_sampling_steps="10", image_flow_solver="heun",
        image_flow_time_sampling="uniform", image_flow_time_uniform_mix=0.,
        image_input_noise_strength=0., image_flow_grad_checkpointing=False,
        joint_dit_head_dim=8, joint_dit_intermediate=48, lambda_text=.05, lambda_image=1.,
        use_flex_attention=True, use_cache=False)
    model = JointDiTForCausalLM(cfg)
    # AdaLN-zero deliberately starts with zero velocities. Nonzero weights
    # make dependency and leakage checks exercise the complete Transformer.
    with torch.no_grad():
        torch.nn.init.normal_(model.image_flow_head.net.output_proj.weight, std=.1)
        for layer in model.image_flow_head.net.layers:
            torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, std=.1)
    return model


def batch():
    ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 5, 9]])
    types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 0, 0]])
    return dict(X0_input_ids=ids, token_types=types,
        flow_sigma=torch.tensor([[0, 1, 3, 6, 5, 4, 2, 7, 8]]),
        image_latents=torch.randn(1, 9, 4), image_span_table=torch.tensor([[0, 1, 2, 6]]),
        labels=ids.masked_fill(types.eq(1), -100), image_loss_mask=types.eq(1),
        compute_text_loss=True, compute_image_loss=True)


def dense(mask, length):
    return _materialize_allowed_mask(mask, batch_size=1, query_length=length, device=torch.device("cpu"))[0]


def test_tied_sigma_preserves_dual_stream_visibility_and_packed_segments():
    b = batch()
    sigma = joint_image_sigma(b["token_types"], b["flow_sigma"])
    assert sigma[0, 2:6].tolist() == [3] * 4
    q, c = joint_backbone_masks(b["X0_input_ids"], b["token_types"], sigma, boi_token_id=11)
    q, c = dense(q, 9), dense(c, 9)
    assert not q[2:6, 2:6].any()
    assert c[2:6, 2:6].all()
    assert q[2:6, [0, 1, 6]].all()
    assert not c[[0, 1, 6]][:, 2:6].any()  # BOI/EOI cannot relay the target.
    assert q[7:, 2:6].all()
    assert not q.diagonal().any() and c.diagonal().all()
    segments = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, -1]])
    q, c = joint_backbone_masks(b["X0_input_ids"], b["token_types"], sigma,
                                segment_ids=segments, boi_token_id=11)
    for mask in (q, c):
        visible = dense(mask, 9)
        assert not visible[2:, :2].any() and not visible[8].any() and not visible[:, 8].any()


def test_joint_dataloader_never_draws_a_reveal_order(monkeypatch):
    from utils.imagenet_flow_sequence import build_selfless_sigma
    def fail(*args, **kwargs):
        raise AssertionError("Joint image ordering sampled randomness")
    monkeypatch.setattr(torch, "rand", fail)
    item = dict(input_ids=torch.zeros(9), prompt_len=1, suffix_len=1, image_start=2,
                reveal_seed=42, image_sigma_order="joint")
    # prefix, BOI, four latents, EOI, suffix, EOS
    sigma = build_selfless_sigma(item, image_tokens=4)
    assert sigma.tolist() == [0, 1, 3, 3, 3, 3, 2, 7, 8]


def test_training_one_dual_backbone_pass_four_shared_times_and_no_target_leakage():
    model = tiny_model().train()
    b = batch()
    calls, observations = [], []
    handle = model.model.register_forward_hook(lambda module, args, output: calls.append(output.last_hidden_state.detach()))
    hook = model.image_flow_head.net.register_forward_pre_hook(
        lambda module, args: observations.append(tuple(value.detach().clone() for value in args)))
    torch.manual_seed(13)
    output = model(**b)
    output.loss.backward()
    assert len(calls) == len(observations) == 1
    noisy, times, condition = observations[0]
    assert noisy.shape == (4, 4, 4) and times.shape == (4,)
    assert times.unique().numel() > 1
    torch.testing.assert_close(condition, condition[:1].expand_as(condition), rtol=0, atol=0)
    assert model.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.model.embed_tokens.weight.grad[8].abs().sum() > 0
    assert model.image_flow_head.net.latent_proj.weight.grad.abs().sum() > 0
    assert model.image_flow_head.net.time_embedder.mlp[0].weight.grad.abs().sum() > 0
    changed = copy.deepcopy(b)
    changed["image_latents"][:, 2:6] += 50
    changed["X0_input_ids"][:, 7:] = torch.tensor([21, 22])
    model(**changed)
    torch.testing.assert_close(calls[0][:, 2:6], calls[1][:, 2:6], rtol=0, atol=0)
    torch.testing.assert_close(condition, observations[1][2], rtol=0, atol=0)
    assert not any("time" in name for name, _ in model.model.named_parameters())
    handle.remove(); hook.remove()


def test_dit_noise_flows_bidirectionally_and_time_is_shared():
    net = tiny_model().image_flow_head.net.eval()
    x, z, t = torch.randn(1, 4, 4), torch.randn(1, 4, 32), torch.tensor([.4])
    original = net(x, t, z)
    changed = x.clone(); changed[:, -1] += 5
    assert not torch.allclose(original[:, 0], net(changed, t, z)[:, 0])
    changed = x.clone(); changed[:, 0] += 5
    assert not torch.allclose(original[:, -1], net(changed, t, z)[:, -1])
    with pytest.raises(ValueError, match="one shared time"):
        net(x, torch.rand(1, 4), z)


@pytest.mark.parametrize("cfg", [1., 3.5])
@pytest.mark.parametrize("solver,head_calls", [("heun", 20), ("euler", 10)])
def test_generation_reuses_one_backbone_pass_and_reports_solver_head_calls(cfg, solver, head_calls):
    model = tiny_model().eval(); b = batch()
    calls = {"backbone": 0, "dit": 0}
    def count(name):
        def hook(module, args):
            calls[name] += 1
        return hook
    hooks = [model.model.register_forward_pre_hook(count("backbone")),
             model.image_flow_head.net.register_forward_pre_hook(count("dit"))]
    noise = torch.randn(1, 4, 4)
    args = dict(input_ids=b["X0_input_ids"], token_types=b["token_types"], sigma=b["flow_sigma"],
        spans=[(0, 2, 6)], initial_noise_bank=noise, flow_cfg=cfg, flow_solver=solver, return_trace=True)
    result, trace = model.generate("t2i", **args)
    assert result.shape == (1, 4, 2, 2) and torch.isfinite(result).all()
    assert calls == {"backbone": 1, "dit": head_calls}
    assert trace["backbone_calls"] == 1 and trace["flow_head_calls"] == head_calls
    assert trace["ode_function_evals"] == head_calls and trace["solver"] == solver
    assert trace["backbone_streams"] == 2 and trace["flow_head_streams"] == 1
    replay, _ = model.generate("t2i", initial_image_latents=b["image_latents"] + 100, **args)
    torch.testing.assert_close(result, replay, rtol=0, atol=0)
    for hook in hooks:
        hook.remove()
    with pytest.raises(ValueError, match="Heun/Euler"):
        model.generate("t2i", **{**args, "flow_solver": "invalid"})


@pytest.mark.parametrize("solver,head_calls", [("heun", 2), ("euler", 1)])
def test_cfg_unconditional_branch_cannot_read_prompt(solver, head_calls):
    model = tiny_model().eval(); b = batch()
    seen = []
    hook = model.image_flow_head.net.register_forward_pre_hook(lambda module, args: seen.append(args[2].detach().clone()))
    args = dict(input_ids=b["X0_input_ids"], token_types=b["token_types"], sigma=b["flow_sigma"],
        spans=[(0, 2, 6)], flow_cfg=3.5, flow_num_steps=1, flow_solver=solver)
    model.generate("t2i", **args)
    first = seen[:]
    seen.clear()
    args["input_ids"] = args["input_ids"].clone(); args["input_ids"][:, 0] = 25
    model.generate("t2i", **args)
    assert len(first) == len(seen) == head_calls
    for before, after in zip(first, seen):
        torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
        assert not torch.allclose(before[0], after[0])
        torch.testing.assert_close(before, first[0], rtol=0, atol=0)
        torch.testing.assert_close(after, seen[0], rtol=0, atol=0)
    hook.remove()


def test_cached_and_full_caption_generation_match_full_image_context():
    model = tiny_model().eval(); b = batch()
    args = dict(input_ids=b["X0_input_ids"][:, :7], token_types=b["token_types"][:, :7],
        image_latents=b["image_latents"][:, :7], sigma=b["flow_sigma"][:, :7],
        max_new_tokens=4, eos_token_id=[], temperature=0.)
    logits = []
    hook = model.lm_head.register_forward_hook(lambda module, inputs, output: logits.append(output.detach().clone()))
    cached = model.generate("i2t", **args, use_cache=True)
    full = model.generate("i2t", **args, use_cache=False)
    torch.testing.assert_close(cached, full)
    for cached_logits, full_logits in zip(logits[:4], logits[4:]):
        torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-6)
    hook.remove()


def test_checkpoint_reload_preserves_joint_conditions_and_generation(tmp_path):
    model = tiny_model().eval(); b = batch()
    model.save_pretrained(tmp_path)
    restored = JointDiTForCausalLM.from_pretrained(tmp_path).eval()
    args = dict(input_ids=b["X0_input_ids"], token_types=b["token_types"], sigma=b["flow_sigma"],
        spans=[(0, 2, 6)], initial_noise_bank=torch.randn(1, 4, 4), flow_cfg=3.5)
    torch.testing.assert_close(model.generate("t2i", **args), restored.generate("t2i", **args), rtol=0, atol=0)
    from utils.evaluation_model_source import _apply_checkpoint_model_contract
    cfg = OmegaConf.create({"model": {}})
    _apply_checkpoint_model_contract(cfg, restored.config.to_dict(), label="joint export")
    assert cfg.model.architecture_variant == "selfless_joint_dit"
    assert cfg.model.training_image_sigma_order == "joint"
    assert cfg.model.joint_dit_head_dim == 8


def test_formal_recipe_preserves_b_budget_and_parameter_count():
    from utils.joint_dit_protocol import ROOT, CONFIG, validate_joint_dit_config, parameter_report
    config = OmegaConf.load(ROOT / CONFIG)
    report = validate_joint_dit_config(config)
    assert config.experiment.identity.id == "z" and config.experiment.identity.label == "Z"
    assert config.experiment.validation_generation.enabled and config.experiment.validation_generation.samples == 16
    assert report["method"] == "Z"
    baseline = OmegaConf.load(ROOT / "configs/selfless/unified_baseline_100b_ascend_64npu.yaml")
    assert config.model.image_input_noise_strength == baseline.model.image_input_noise_strength == .01
    assert config.model.image_flow_solver == baseline.model.image_flow_solver == "heun"
    assert config.model.image_flow_num_sampling_steps == baseline.model.image_flow_num_sampling_steps == "10"
    assert config.evaluation.flow_solver == "heun" and config.experiment.validation_generation.solver == "heun"
    assert report["head_calls_per_image_batch"] == 20
    assert report["world_size"] == 64 and report["optimizer_steps"] == 95415
    assert abs(parameter_report(config)["flow_parameter_relative_difference"]) < .005
    config.model.image_flow_batch_mul = 1
    with pytest.raises(ValueError, match="B-aligned"):
        validate_joint_dit_config(config)


def test_complete_256_latent_grid_generation():
    model = tiny_model(image_tokens=256).eval()
    ids = torch.tensor([[3, 11] + [8] * 256 + [12]])
    types = torch.tensor([[0, 2] + [1] * 256 + [2]])
    out, trace = model.generate("t2i", input_ids=ids, token_types=types, sigma=None,
        spans=[(0, 2, 258)], flow_cfg=3.5, return_trace=True)
    assert out.shape == (1, 4, 16, 16) and torch.isfinite(out).all()
    assert trace["backbone_calls"] == 1 and trace["flow_head_calls"] == 20


def test_clean_context_image_is_bidirectional_and_conditions_later_target():
    model = tiny_model().eval()
    ids = torch.tensor([[3, 11, 8, 8, 8, 8, 12, 5, 11, 8, 8, 8, 8, 12]])
    types = torch.tensor([[0, 2, 1, 1, 1, 1, 2, 0, 2, 1, 1, 1, 1, 2]])
    latents = torch.randn(1, 14, 4)
    query, content = joint_backbone_masks(ids, types)
    def encode(values):
        return model.model(X0_input_ids=ids, token_types=types, image_latents=values,
            attention_mask=query, content_attention_mask=content,
            calculate_likelihood=True, return_x0_hidden_state=True)
    original = encode(latents)
    changed = latents.clone(); changed[:, 5] += 5
    other = encode(changed)
    # The first context-image content token reads its last token, and the
    # later target's query reads that clean context through the content stream.
    assert not torch.allclose(original.x0_last_hidden_state[:, 2], other.x0_last_hidden_state[:, 2])
    assert not torch.allclose(original.last_hidden_state[:, 9:13], other.last_hidden_state[:, 9:13])
    changed = latents.clone(); changed[:, 9:13] += 50
    torch.testing.assert_close(original.last_hidden_state[:, 9:13], encode(changed).last_hidden_state[:, 9:13], rtol=0, atol=0)
    result = model.generate("t2i", input_ids=ids, token_types=types, sigma=None,
        spans=[(0, 9, 13)], initial_image_latents=latents, flow_cfg=1., flow_num_steps=1)
    assert result.shape == (1, 4, 2, 2)


def test_training_batch_adapter_keeps_cfg_dropout_and_joint_order():
    from pretrain.train_selfless_flow import _prepare_loss_forward_batch
    model = tiny_model().eval(); b = batch()
    b["input_ids"] = b.pop("X0_input_ids")
    b["sigma"] = joint_image_sigma(b["token_types"], b.pop("flow_sigma"))
    b["segment_ids"] = torch.zeros_like(b["input_ids"])
    b["image_uncond_mask"] = b["token_types"].eq(1)
    config = OmegaConf.create({"model": model.config.to_dict()})
    kwargs, _ = _prepare_loss_forward_batch(b, config=config, device=torch.device("cpu"), source_name="t2i")
    a = model(**kwargs).last_hidden_state
    kwargs["X0_input_ids"] = kwargs["X0_input_ids"].clone(); kwargs["X0_input_ids"][:, 0] = 26
    torch.testing.assert_close(a[:, 2:6], model(**kwargs).last_hidden_state[:, 2:6], rtol=0, atol=0)


def test_joint_formal_scoring_has_one_exact_image_order():
    from utils.evaluation.model_contracts import validate_formal_image_order_scoring
    contract = dict(architecture_variant="selfless_joint_dit", dual_stream_attention_contract="xlnet_content_diagonal",
        image_sigma_order="joint", image_order_mc_contract="not_applicable_joint_image", mc_samples=1)
    assert validate_formal_image_order_scoring(contract, contract) == 1
    with pytest.raises(ValueError, match="fixed whole-image"):
        validate_formal_image_order_scoring(contract, {**contract, "image_sigma_order": "random"})


@pytest.mark.parametrize("image_tokens", [4, 256])
def test_joint_classification_reuses_bidirectional_image_prefix_with_identical_scores(image_tokens):
    from types import SimpleNamespace
    from test_training_unified_loss_validation import Tokenizer
    from utils.evaluation.native_understanding import score_text_candidates, score_candidates_with_backend

    model = tiny_model(image_tokens=image_tokens).eval()
    latents = torch.randn(image_tokens, 4)
    kwargs = dict(model=model, tokenizer=Tokenizer(), cache=SimpleNamespace(sample=lambda _: latents),
        image_id=1, item_id="z-cached-classification", prompt="Describe this image:",
        candidates=["a cat", "a brown dog", "a bird"], device=torch.device("cpu"),
        image_sigma_order="joint", attention_contract="xlnet_content_diagonal",
        args=SimpleNamespace(request_chunk_size=8, batch_size_per_rank=2, lm_head_chunk_tokens=16,
                             max_length=512, seed=42, scoring_backend="cached_prefix"))
    with torch.no_grad():
        reference = score_text_candidates(**kwargs)
        calls = []
        handle = model.model.register_forward_pre_hook(
            lambda module, args, keywords: calls.append(dict(keywords)), with_kwargs=True)
        cached = score_candidates_with_backend(**kwargs)
        handle.remove()
    assert sum(bool(row.get("_text_ar_mode")) for row in calls) == 1
    assert all(row.get("use_cache") for row in calls)
    assert sum(bool(row.get("token_types").eq(1).any()) for row in calls) == 1
    for actual, expected in zip(cached, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-5)


def test_heun_uses_predicted_endpoint_and_shared_time_for_the_corrector(monkeypatch):
    head = tiny_model().image_flow_head
    observed_times = []
    def velocity(x, t, condition):
        observed_times.append(t.clone())
        return x  # dx/dt=x; each Heun step multiplies x by 1+dt+dt^2/2.
    monkeypatch.setattr(head, "velocity", velocity)
    noise = torch.ones(1, 4, 4)
    result, trace = head.sample(torch.zeros(1, 4, 32), num_steps=2, initial_noise=noise, return_trace=True)
    torch.testing.assert_close(result, torch.full_like(noise, 1.625 ** 2), rtol=0, atol=0)
    assert [t.item() for t in observed_times] == [0., .5, .5, 1.]
    assert all(t.shape == (1,) for t in observed_times)
    assert trace == {"solver": "heun", "steps": 2, "flow_head_calls": 4}
    assert torch.equal(noise, torch.ones_like(noise))


def test_heun_validation_report_cannot_claim_only_ten_head_calls():
    from utils.evaluation.model_contracts import validate_image_generation_report
    report = dict(architecture_variant="selfless_joint_dit", generation_entry="model.generate", use_cache=False,
        strategies={"joint": dict(backbone_kv_cache_enabled=False, generation_mode="joint_dit_full_image_flow",
                                  solver="heun", steps=10, backbone_calls=1, flow_head_calls=20)})
    assert validate_image_generation_report(report, "xlnet_content_diagonal") is False
    report["strategies"]["joint"]["flow_head_calls"] = 10
    with pytest.raises(ValueError, match="solver-specific"):
        validate_image_generation_report(report, "xlnet_content_diagonal")
