import copy

import pytest
import torch
from torch.nn import functional as F

from models.modeling_model.modeling_showo2_unified import (
    Showo2UnifiedConfig, Showo2UnifiedForCausalLM, omni_allowed_mask,
)


def tiny_config(dual=False):
    return Showo2UnifiedConfig(vocab_size=40, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=True,
        architecture_variant="showo2_unified", training_objective="showo2_full_image_flow",
        dual_stream_attention_contract="showo2_omni_attention", flow_head_attention_contract="showo2_omni_attention",
        flow_condition_contract="backbone_noisy_image_hidden", mask_token_id=7,
        boi_token_id=11, eoi_token_id=12, image_mask_token_id=8, image_tokens_per_img=4,
        image_latent_dim=4, image_flow_width=32, image_flow_depth=2, image_flow_batch_mul=4,
        image_flow_num_sampling_steps=2, s2_flow_head_dim=8, s2_flow_intermediate=48,
        image_flow_time_sampling="logit_normal", image_flow_logit_mean=0., image_flow_logit_std=1.,
        image_flow_time_uniform_mix=.1, image_flow_time_eps=1e-5, image_flow_time_scale=1000.,
        image_flow_solver="heun", image_input_noise_strength=0., image_uncond_prob=.1,
        lambda_text=.05, lambda_image=1., s2_use_siglip=dual, s2_semantic_width=16,
        s2_semantic_intermediate=32, s2_semantic_depth=2, s2_semantic_heads=2,
        image_flow_grad_checkpointing=False, s2_full_prediction_checkpointing=False)


def tiny_model(dual=False, nonzero=True):
    torch.manual_seed(42)
    model = Showo2UnifiedForCausalLM(tiny_config(dual))
    if nonzero:
        with torch.no_grad():
            torch.nn.init.normal_(model.image_flow_head.output_proj.weight, std=.05)
            for layer in model.image_flow_head.layers:
                torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, std=.05)
    return model


def image_batch(batch=1):
    ids = torch.tensor([[5, 6, 11, 8, 8, 8, 8, 12]]).repeat(batch, 1)
    types = torch.tensor([[0, 0, 2, 1, 1, 1, 1, 2]]).repeat(batch, 1)
    spans = torch.tensor([[row, 0, 3, 7] for row in range(batch)])
    return dict(X0_input_ids=ids, token_types=types, image_latents=torch.randn(batch, 8, 4),
        image_span_table=spans, labels=torch.full_like(ids, -100), image_loss_mask=types.eq(1),
        compute_text_loss=False, compute_image_loss=True)


def test_omni_truth_table_and_cfg_time_carrier():
    b = image_batch()
    ids, types = b["X0_input_ids"], b["token_types"]
    allowed = omni_allowed_mask(ids, types, boi_token_id=11)[0]
    assert allowed[3:7, 2:7].all()
    assert not allowed[:2, 2:].any()
    assert not allowed[2:7, 7].any()
    assert allowed[7].all()
    dropped = omni_allowed_mask(ids, types, boi_token_id=11, image_uncond_rows=torch.tensor([True]))[0]
    assert not dropped[2:7, :2].any()
    assert dropped[2:7, 2:7].all()
    segments = torch.tensor([[0, 0, 1, 1, 1, 1, 1, -1]])
    packed = omni_allowed_mask(ids, types, boi_token_id=11, segment_ids=segments)[0]
    assert not packed[2:, :2].any() and not packed[7].any()


@pytest.mark.parametrize("dual", [False, True])
def test_ar_shift_no_current_or_future_text_leakage(dual):
    model = tiny_model(dual).eval()
    ids = torch.tensor([[3, 4, 5, 6, 9]])
    original = model(X0_input_ids=ids).logits
    changed = ids.clone(); changed[:, 2:] = torch.tensor([20, 21, 22])
    other = model(X0_input_ids=changed).logits
    torch.testing.assert_close(original[:, :3], other[:, :3], rtol=0, atol=0)
    raw = model.model(X0_input_ids=ids, calculate_likelihood=False).last_hidden_state
    torch.testing.assert_close(original[:, 1:], model.lm_head(raw[:, :-1]))


def test_shared_parameters_do_not_depend_on_extra_branch_rng():
    a, b = tiny_model(False, False), tiny_model(True, False)
    bp = dict(b.named_parameters())
    for name, p in a.named_parameters():
        assert name in bp
        torch.testing.assert_close(p, bp[name], rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("dual", [False, True])
def test_candidate_scorer_matches_explicit_ar_shift(dual):
    from types import SimpleNamespace
    from utils.evaluation.multimodal_likelihood import CandidateRequest, score_candidate_requests

    model = tiny_model(dual).eval()
    ids = torch.tensor([[11, 8, 8, 8, 8, 12, 4, 5, 6]])
    types = torch.tensor([[2, 1, 1, 1, 1, 2, 0, 0, 0]])
    latent = torch.randn(4, 4)
    request = CandidateRequest(0, 0, 7, tuple(ids[0].tolist()), tuple(types[0].tolist()),
        tuple(range(9)), 7, 1, 0)
    score = score_candidate_requests(model, [request], SimpleNamespace(sample=lambda _: latent),
        batch_size=2, lm_head_chunk_tokens=1, attention_contract="showo2_omni_attention",
        device=torch.device("cpu"))[0]
    dense = torch.zeros(1, 9, 4)
    dense[:, 1:5] = latent
    with torch.no_grad():
        raw = model.model(X0_input_ids=ids, token_types=types, image_latents=dense,
            image_span_table=torch.tensor([[0, 0, 1, 5]]), calculate_likelihood=False).last_hidden_state
        expected = F.log_softmax(model.lm_head(raw[:, 6:8]).float(), -1).gather(-1, ids[:, 7:, None]).sum()
    assert score.token_count == 2
    assert score.loglikelihood == pytest.approx(expected.item(), abs=1e-6)
    assert score.normalized_loglikelihood == pytest.approx(expected.item() / 2, abs=1e-6)


def test_formal_contract_rejects_b_exposure_or_flow_sampling_drift():
    from omegaconf import OmegaConf
    from utils.showo2_unified_protocol import ROOT, config_path, validate_s2_config
    from utils.evaluation.model_contracts import image_order_for_scoring

    for variant in ("single", "dual-siglip"):
        config = OmegaConf.load(ROOT / config_path(variant))
        report = validate_s2_config(config)
        assert report["world_size"] == 64 and report["optimizer_steps"] == 95415
        assert image_order_for_scoring(tiny_config(variant == "dual-siglip"), "random") == "sequential"
        for field, value in (("model.image_flow_batch_mul", 1), ("training.gradient_accumulation_steps", 8)):
            changed = copy.deepcopy(config)
            OmegaConf.update(changed, field, value)
            with pytest.raises(ValueError, match="frozen contract"):
                validate_s2_config(changed)


@pytest.mark.parametrize("dual", [False, True])
def test_cfg_blocks_all_caption_paths_and_image_has_full_attention(dual):
    model = tiny_model(dual).eval(); batch = image_batch()
    ids, types, latent = (batch[k] for k in ("X0_input_ids", "token_types", "image_latents"))
    kw = dict(image_span_table=batch["image_span_table"], image_uncond_rows=torch.tensor([True]))
    t = torch.tensor([.3])
    first = model.predict_velocity(ids, types, latent, t, **kw)
    changed = ids.clone(); changed[:, :2] = torch.tensor([22, 23])
    second = model.predict_velocity(changed, types, latent, t, **kw)
    torch.testing.assert_close(first[:, 3:7], second[:, 3:7], rtol=0, atol=0)
    future = latent.clone(); future[:, 6] += 5
    other = model.predict_velocity(ids, types, future, t, **kw)
    assert not torch.equal(first[:, 3], other[:, 3])


@pytest.mark.parametrize("dual", [False, True])
def test_four_flow_draws_loss_and_checkpoint_gradients(dual):
    model = tiny_model(dual).train(); other = copy.deepcopy(model)
    batch = image_batch(2)
    times = torch.tensor([[.1, .2], [.3, .4], [.5, .6], [.7, .8]])
    xt = torch.randn(4, 2, 8, 4); target = torch.randn_like(xt)
    state = dict(times=times, x_t=xt, velocity_target=target)
    out = model(**batch, s2_flow_state=state)
    expected = sum((model.predict_velocity(batch["X0_input_ids"], batch["token_types"], xt[i], times[i],
        image_span_table=batch["image_span_table"])[:, 3:7].float() - target[i, :, 3:7]).square().mean()
        for i in range(4)) / 4
    torch.testing.assert_close(out.loss, expected)
    out.loss.backward()
    other.config.s2_full_prediction_checkpointing = True
    other.image_flow_head.gradient_checkpointing = True
    copied = other(**batch, s2_flow_state=state); copied.loss.backward()
    torch.testing.assert_close(out.loss, copied.loss)
    for (n, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        assert p.grad is not None and q.grad is not None, n
        assert torch.isfinite(p.grad).all(), n
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=2e-6, msg=n)
    # Clean labels can change independently of fixed noisy inputs/velocity targets.
    changed = dict(batch, image_latents=torch.randn_like(batch["image_latents"]) * 100)
    torch.testing.assert_close(model(**changed, s2_flow_state=state).loss, out.loss)


@pytest.mark.parametrize("dual", [False, True])
def test_text_step_connects_every_parameter_and_hf_reload(tmp_path, dual):
    model = tiny_model(dual).train(); ids = torch.tensor([[3, 4, 5, 6]])
    out = model(X0_input_ids=ids, labels=ids, token_types=torch.zeros_like(ids))
    out.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval().save_pretrained(tmp_path)
    restored = Showo2UnifiedForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(model(X0_input_ids=ids).logits, restored(X0_input_ids=ids).logits)
    generated, trace = restored.generate_text(ids[:, :2], max_new_tokens=2, return_trace=True)
    assert generated.shape == (1, 4) and trace["backbone_calls"] == 2
    b = image_batch()
    image, trace = restored.generate_image(b["X0_input_ids"], b["token_types"], None, [(0, 3, 7)],
        flow_num_steps=2, flow_cfg=2, return_trace=True, debug_finite=True)
    assert image.shape == (1, 4, 2, 2)
    assert torch.isfinite(image).all() and trace["backbone_calls"] == 8


@pytest.mark.parametrize("dual", [False, True])
def test_ema_source_uses_s2_module_contract(tmp_path, dual):
    import json
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from utils.evaluation_model_source import configure_model_source, resolve_evaluation_model_source
    model = tiny_model(dual)
    model.save_pretrained(tmp_path)
    (tmp_path / "tokenizer.json").write_text("{}")
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as f:
        key_count = len(f.keys())
    (tmp_path / "ema_export_metadata.json").write_text(json.dumps({
        "schema": "selfless_ema_hf_export_v1", "floating_dtype": "float32", "source_global_step": 6,
        "source_world_size": 16, "state_key_count": len(model.state_dict()),
        "stored_weight_key_count": key_count, "export_kind": "training"}))
    source = resolve_evaluation_model_source(tmp_path)
    config = OmegaConf.create({"model": {"architecture_variant": "selfless_contextual",
        "training_objective": "selfless_dual_stream"}, "training": {"from_scratch": False,
        "use_gradient_checkpointing": False}})
    configure_model_source(config, source)
    assert config.model.training_objective == "showo2_full_image_flow"
    assert config.model.s2_use_siglip is dual
    assert config.model.flow_condition_contract == "backbone_noisy_image_hidden"


def test_semantic_block_matches_official_siglip_equations():
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
    from models.modeling_model.modeling_showo2_unified import SiglipBlock, attention_from_allowed
    cfg = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_attention_heads=2,
        layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh")
    cfg._attn_implementation = "eager"
    reference = SiglipEncoderLayer(cfg).eval()
    ours = SiglipBlock(16, 32, 2).eval()
    ours.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 4, 16)
    expected = reference(x, attention_mask=None)
    if isinstance(expected, tuple):
        expected = expected[0]
    mask = attention_from_allowed(torch.ones(2, 4, 4, dtype=torch.bool))
    torch.testing.assert_close(ours(x, mask), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("chunk", [2, 4])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_mc_batching_preserves_masked_loss_all_gradients_and_rng(dual, chunk, checkpointed):
    from models.modeling_model.image_position_utils import build_row_col_position_ids
    original = tiny_model(dual).train()
    batched = copy.deepcopy(original)
    batched.config.s2_mc_batch_size = chunk
    batched.config.s2_full_prediction_checkpointing = checkpointed
    batch = image_batch(2)
    batch["position_ids"] = build_row_col_position_ids(batch["token_types"], 4)
    batch["_text_segment_ids"] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 1, 1, 1, 1, 1, 1]])
    batch["s2_image_uncond_rows"] = torch.tensor([True, False])
    batch["s2_image_uncond_mask"] = batch["token_types"].eq(1)
    batch["s2_image_uncond_mask"][1] = False
    batch["image_loss_mask"][1, 4] = False
    rng = torch.get_rng_state()
    first = original(**batch)
    first.loss.backward()
    rng_after = torch.get_rng_state()
    torch.set_rng_state(rng)
    second = batched(**batch)
    second.loss.backward()
    torch.testing.assert_close(first.loss, second.loss, rtol=1e-6, atol=1e-7)
    assert torch.equal(rng_after, torch.get_rng_state())
    for (name, p), (_, q) in zip(original.named_parameters(), batched.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=3e-6, msg=name)
    assert batched.image_flow_head.last_forward_stats["backbone_calls"] == 4 // chunk


def test_s2_infra_resume_is_explicit_and_rejects_numerical_or_cross_arm_changes():
    from utils.selfless_training_runtime import (RESUME_SCHEMA, RESUME_CONTRACT_VERSION,
        build_resume_contract, validate_resume_contract)
    from utils.showo2_unified_protocol import ROOT, config_path
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / config_path("single"))
    contract = build_resume_contract(cfg, world_size=64, gradient_accumulation_steps=4)
    for key in ("s2_mc_batch_size", "s2_backbone_checkpoint_every", "s2_semantic_gradient_checkpointing"):
        contract["model"].pop(key, None)
    contract["model"]["s2_full_prediction_checkpointing"] = True
    contract["model"]["image_flow_grad_checkpointing"] = True
    contract["training"]["use_gradient_checkpointing"] = True
    metadata = {"schema": RESUME_SCHEMA, "config_contract_version": RESUME_CONTRACT_VERSION,
                "config_contract": contract}
    other = copy.deepcopy(contract)
    other["model"]["s2_mc_batch_size"] = 4
    other["model"]["s2_full_prediction_checkpointing"] = False
    other["model"]["s2_backbone_checkpoint_every"] = 2
    with pytest.raises(RuntimeError, match="configuration differs"):
        validate_resume_contract(metadata, current_contract=other)
    changes = validate_resume_contract(metadata, current_contract=other, allow_s2_infra_migration=True)
    assert {x["field"] for x in changes} == {"model.s2_mc_batch_size", "model.s2_full_prediction_checkpointing",
                                             "model.s2_backbone_checkpoint_every"}
    for section, key, value in [("model", "s2_use_siglip", True), ("model", "image_flow_batch_mul", 2),
                                ("training", "seed", 99), ("training", "max_train_steps", 120000)]:
        invalid = copy.deepcopy(other); invalid[section][key] = value
        with pytest.raises(RuntimeError, match="outside"):
            validate_resume_contract(metadata, current_contract=invalid, allow_s2_infra_migration=True)
    invalid = copy.deepcopy(other); invalid["world_size"] = 16
    with pytest.raises(RuntimeError, match="outside"):
        validate_resume_contract(metadata, current_contract=invalid, allow_s2_infra_migration=True)
