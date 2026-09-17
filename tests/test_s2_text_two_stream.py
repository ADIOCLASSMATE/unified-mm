"""Text position alignment, causal isolation and unchanged S2 flow behavior."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from models.modeling_model.modeling_showo2_unified import (
    Showo2UnifiedForCausalLM, omni_allowed_mask, text_query_allowed_mask,
)
from test_showo2_unified import tiny_config, image_batch


def model_pair():
    torch.manual_seed(42)
    config = tiny_config()
    config.num_hidden_layers = 3  # Catch indirect leakage through content K/V.
    config.layer_types = ["full_attention"] * 3
    baseline = Showo2UnifiedForCausalLM(config).eval()
    with torch.no_grad():
        torch.nn.init.normal_(baseline.image_flow_head.output_proj.weight, std=.05)
        for layer in baseline.image_flow_head.layers:
            torch.nn.init.normal_(layer.adaLN_modulation[-1].weight, std=.05)
    config = copy.deepcopy(config)
    config.dual_stream_attention_contract = "showo2_text_two_stream"
    model = Showo2UnifiedForCausalLM(config).eval()
    model.load_state_dict(baseline.state_dict(), strict=True)
    return baseline, model


def test_text_query_visibility_with_images_padding_segments_and_cfg():
    batch = image_batch()
    ids, types = batch["X0_input_ids"], batch["token_types"]
    content = omni_allowed_mask(ids, types, boi_token_id=11)
    query = text_query_allowed_mask(content, types)[0]
    assert query[1, 0] and query[2, :2].all()
    assert not query.triu().any()
    assert not query[3:7].any()  # No image query participates in prediction.
    assert query[7, :7].all()  # EOI query sees the entire preceding image.
    assert content[0, 2:7, 2:7].all()
    segments = torch.tensor([[0, 0, 1, 1, 1, 1, 1, -1]])
    packed = text_query_allowed_mask(omni_allowed_mask(ids, types, boi_token_id=11,
        segment_ids=segments), types)[0]
    assert not packed[2:, :2].any() and not packed[7].any()
    cfg = text_query_allowed_mask(omni_allowed_mask(ids, types, boi_token_id=11,
        image_uncond_rows=torch.tensor([True])), types)[0]
    assert not cfg[2].any()


def test_three_layer_text_queries_have_no_target_or_future_leakage():
    baseline, model = model_pair()
    ids = torch.tensor([[3, 4, 5, 6, 9]])
    original = model(X0_input_ids=ids).logits
    changed = ids.clone()
    changed[:, 2:] = torch.tensor([20, 21, 22])
    other = model(X0_input_ids=changed).logits
    torch.testing.assert_close(original[:, :3], other[:, :3], rtol=0, atol=0)
    assert not torch.equal(original[:, 3], other[:, 3])
    assert model.text_prediction_offset == 0 and baseline.text_prediction_offset == 1
    content = model.model(X0_input_ids=ids, calculate_likelihood=False).last_hidden_state
    assert not torch.equal(original[:, 1:], model.lm_head(content[:, :-1]))


@pytest.mark.parametrize("training", [False, True])
def test_flow_content_output_and_gradients_match_s2_single_exactly(training, monkeypatch):
    baseline, model = model_pair()
    baseline.train(training); model.train(training)
    batch = image_batch(2)
    batch["s2_image_uncond_rows"] = torch.tensor([False, True])
    state = dict(times=torch.tensor([[.1, .2], [.3, .4], [.5, .6], [.7, .8]]),
                 x_t=torch.randn(4, 2, 8, 4), velocity_target=torch.randn(4, 2, 8, 4))
    def reject_image_query(*args, **kwargs):
        raise AssertionError("Flow prediction must retain the S2 content-only backbone")
    monkeypatch.setattr(model.model, "_build_xt_inputs_embeds", reject_image_query)
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(parameter, dict(model.named_parameters())[name], rtol=0, atol=0)
    first = baseline(**batch, s2_flow_state=state)
    second = model(**batch, s2_flow_state=state)
    torch.testing.assert_close(first.loss, second.loss, rtol=0, atol=0)
    first.loss.backward(); second.loss.backward()
    for (name, p), (_, q) in zip(baseline.named_parameters(), model.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


def test_same_position_ce_keeps_s2_target_count_and_backpropagates_mask_embedding():
    _, model = model_pair()
    model.train()
    ids = torch.tensor([[3, 4, 5, 6, 9, 10, 0]])
    types = torch.tensor([[0, 0, 0, 0, 0, 0, 3]])
    segments = torch.tensor([[0, 0, 0, 1, 1, 1, -1]])
    labels = ids.clone(); labels[:, 5:] = -100
    batch = dict(X0_input_ids=ids, token_types=types, _text_segment_ids=segments)
    output = model(**batch, labels=labels)
    hidden = model.model(**batch, calculate_likelihood=True).last_hidden_state
    positions = torch.tensor([1, 2, 4])
    expected = F.cross_entropy(model.lm_head(hidden[:, positions]).flatten(0, 1).float(),
                               labels[:, positions].flatten())
    torch.testing.assert_close(output.loss, .05 * expected)
    assert output.per_modality_count["text_tokens"] == 3
    output.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.model.embed_tokens.weight.grad[model.config.mask_token_id].norm() > 0
    # An earlier packed document cannot influence later text queries.
    changed = ids.clone(); changed[:, :3] = torch.tensor([20, 21, 22])
    other = model.model(**{**batch, "X0_input_ids": changed}, calculate_likelihood=True).last_hidden_state
    torch.testing.assert_close(hidden[:, 3:], other[:, 3:], rtol=0, atol=0)


def test_caption_query_observes_full_image_but_no_target_text():
    _, model = model_pair()
    ids = torch.tensor([[11, 8, 8, 8, 8, 12, 4, 5, 6]])
    types = torch.tensor([[2, 1, 1, 1, 1, 2, 0, 0, 0]])
    latents = torch.randn(1, 9, 4)
    batch = dict(X0_input_ids=ids, token_types=types, image_latents=latents,
                 image_span_table=torch.tensor([[0, 0, 1, 5]]))
    original = model(**batch).logits
    changed = ids.clone(); changed[:, 7:] = torch.tensor([20, 21])
    other = model(**{**batch, "X0_input_ids": changed}).logits
    torch.testing.assert_close(original[:, 5:8], other[:, 5:8], rtol=0, atol=0)
    new_latents = latents.clone(); new_latents[:, 4] += 5
    image_changed = model(**{**batch, "image_latents": new_latents}).logits
    assert not torch.equal(original[:, 7], image_changed[:, 7])


def test_text_and_caption_generation_use_target_query_and_handle_padding():
    _, model = model_pair()
    ids = torch.tensor([[3, 4, 0], [5, 6, 9]])
    types = torch.tensor([[0, 0, 3], [0, 0, 0]])
    segments = torch.tensor([[0, 0, -1], [1, 1, 1]])
    generated, trace = model.generate_text(ids, token_types=types, segment_ids=segments,
                                           max_new_tokens=2, return_trace=True)
    assert trace["text_prediction_offset"] == 0 and trace["backbone_calls"] == 2
    for row, length in enumerate([2, 3]):
        single = model.generate_text(ids[row:row+1, :length], max_new_tokens=2)
        torch.testing.assert_close(generated[row, :length+2], single[0])
    batch = image_batch()
    prompt = batch["X0_input_ids"]
    full_ids = F.pad(prompt, (0, 1), value=7)
    full_types = F.pad(batch["token_types"], (0, 1), value=0)
    latents = F.pad(batch["image_latents"], (0, 0, 0, 1))
    expected = model(X0_input_ids=full_ids, token_types=full_types,
        image_latents=latents, image_span_table=batch["image_span_table"]).logits[:, -1].argmax(-1)
    generated = model.generate_text(prompt, token_types=batch["token_types"],
        image_latents=batch["image_latents"], max_new_tokens=1)
    torch.testing.assert_close(generated[:, -1], expected)


def test_multimodal_scorer_uses_same_position_queries():
    from utils.evaluation.multimodal_likelihood import CandidateRequest, score_candidate_requests
    _, model = model_pair()
    ids = torch.tensor([[11, 8, 8, 8, 8, 12, 4, 5, 6]])
    types = torch.tensor([[2, 1, 1, 1, 1, 2, 0, 0, 0]])
    latent = torch.randn(4, 4)
    request = CandidateRequest(0, 0, 7, tuple(ids[0].tolist()), tuple(types[0].tolist()),
                               tuple(range(9)), 7, 1, 0)
    score = score_candidate_requests(model, [request], SimpleNamespace(sample=lambda _: latent),
        batch_size=2, lm_head_chunk_tokens=1, attention_contract="showo2_text_two_stream",
        device=torch.device("cpu"))[0]
    dense = torch.zeros(1, 9, 4); dense[:, 1:5] = latent
    logits = model(X0_input_ids=ids, token_types=types, image_latents=dense,
        image_span_table=torch.tensor([[0, 0, 1, 5]])).logits
    expected = F.log_softmax(logits[:, 7:].float(), -1).gather(-1, ids[:, 7:, None]).sum()
    assert score.token_count == 2 and score.loglikelihood == pytest.approx(expected.item(), abs=2e-6)


def test_checkpoint_and_formal_contract_preserve_text_only_ablation(tmp_path):
    from omegaconf import OmegaConf
    from utils.showo2_unified_protocol import ROOT, config_path, validate_s2_config
    from utils.evaluation_model_source import _apply_checkpoint_model_contract
    from utils.evaluation.model_contracts import scoring_contract, validate_formal_image_order_scoring
    _, model = model_pair()
    model.save_pretrained(tmp_path)
    restored = Showo2UnifiedForCausalLM.from_pretrained(tmp_path).eval()
    assert restored.text_prediction_offset == 0
    ids = torch.tensor([[3, 4, 5]])
    torch.testing.assert_close(restored(X0_input_ids=ids).logits, model(X0_input_ids=ids).logits)
    cfg = OmegaConf.load(ROOT / config_path("single-text-two-stream"))
    baseline = OmegaConf.load(ROOT / config_path("single"))
    assert validate_s2_config(cfg)["variant"] == "single-text-two-stream"
    for section in ("dataset", "optimizer", "lr_scheduler", "training"):
        assert cfg[section] == baseline[section]
    left, right = dict(cfg.model), dict(baseline.model)
    left.pop("dual_stream_attention_contract"); right.pop("dual_stream_attention_contract")
    assert left == right
    target = OmegaConf.create({"model": {}})
    _apply_checkpoint_model_contract(target, model.config.to_dict(), label="test")
    assert target.model.dual_stream_attention_contract == "showo2_text_two_stream"
    assert target.model.flow_head_attention_contract == "showo2_omni_attention"
    attention = target.model.dual_stream_attention_contract
    exact = dict(dual_stream_attention_contract=attention, mc_samples=1,
                 scoring_contract=scoring_contract(attention, None),
                 contract=scoring_contract(attention, None), image_order_mc_contract="not_applicable_full_image_ar")
    assert validate_formal_image_order_scoring(exact, exact) == 1
    with pytest.raises(ValueError):
        validate_formal_image_order_scoring({**exact, "scoring_contract": "showo2_next_token_ar_target_aligned_v1"}, exact)
