"""Y's visibility, conditional independence, training and reveal contracts."""
import copy

import pytest
import torch
from transformers import AutoConfig

from models.modeling_model.modeling_y import (
    YConfig, YForCausalLM, YFlowLoss, cosine_reveal_counts, y_backbone_allowed,
)
from test_joint_dit import batch, tiny_model as z_model


def tiny_model():
    cfg = z_model().config.to_dict()
    cfg.pop("model_type", None)
    cfg.update(architecture_variant="selfless_y", flow_head_attention_contract="not_applicable",
               y_reveal_steps=2, y_empty_visible_prob=.1)
    model = YForCausalLM(YConfig(**cfg))
    with torch.no_grad():
        torch.nn.init.normal_(model.image_flow_head.net.final_layer.linear.weight, std=.1)
        torch.nn.init.normal_(model.image_flow_head.net.final_layer.adaLN_modulation[-1].weight, std=.1)
    return model


def visible():
    return torch.tensor([[False, False, True, False, True, False, False, False, False]])


def test_visibility_and_negative_rank_not_a_mask():
    b = batch()
    q, c = y_backbone_allowed(b["X0_input_ids"], b["token_types"], b["flow_sigma"],
                              visible(), b["image_loss_mask"], boi_token_id=11)
    assert q[0, 2:6][:, [2, 4]].all()
    assert not q[0, :, [3, 5]].any()
    assert c[0, [2, 4]][:, [2, 4]].all()
    assert not c[0, [0, 1, 2, 4, 6, 7, 8]][:, [3, 5]].any()
    assert not q[0, 2:6, 7:].any()
    segments = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, -1]])
    q, c = y_backbone_allowed(b["X0_input_ids"], b["token_types"], b["flow_sigma"],
        visible(), b["image_loss_mask"], segment_ids=segments, boi_token_id=11)
    assert not q[0, 2:, :2].any() and not c[0, 2:, :2].any()


def test_multilayer_unknown_isolation_and_visible_gradients():
    model = tiny_model().train()
    b = batch()
    b.update(y_visible_mask=visible(), compute_text_loss=False)
    observed = []
    hook = model.model.register_forward_hook(lambda m, a, out: observed.append(out.last_hidden_state))
    result = model(**b)
    assert result.per_modality_count["image_tokens"] == 2 * model.image_flow_batch_mul
    result.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.image_token_embedder.parameters())
    changed = copy.deepcopy(b)
    changed["image_latents"][:, [3, 5]] += 50
    changed["X0_input_ids"][:, 7:] = torch.tensor([21, 22])
    model(**changed)
    torch.testing.assert_close(observed[0][:, 2:6], observed[1][:, 2:6], rtol=0, atol=0)
    changed["image_latents"][:, 2] += 10
    model(**changed)
    assert not torch.allclose(observed[0][:, [3, 5]], observed[2][:, [3, 5]])
    hook.remove()


@pytest.mark.parametrize("v", [torch.zeros_like(visible()), visible()])
def test_cfg_cannot_relay_prompt_through_visible_content(v):
    model = tiny_model().eval()
    b = batch()
    b.update(y_visible_mask=v, y_image_uncond_rows=torch.tensor([True]))
    hidden = []
    hook = model.model.register_forward_hook(lambda m, a, out: hidden.append(out.last_hidden_state))
    first = model(**b)
    b["X0_input_ids"][:, 0] = 20
    second = model(**b)
    assert torch.isfinite(first.loss) and torch.isfinite(second.loss)
    torch.testing.assert_close(hidden[0][:, 2:6], hidden[1][:, 2:6], rtol=0, atol=0)
    hook.remove()


def test_loss_weights_images_equally(monkeypatch):
    head = YFlowLoss(target_channels=2, z_channels=4, depth=1, width=8, num_sampling_steps=10)
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.zeros_like(x))
    monkeypatch.setattr(head, "velocity", lambda x, t, z: torch.zeros_like(x))
    targets = torch.tensor([[[2., 2.]] * 3, [[1., 1.]] * 3])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    loss = head(targets, torch.zeros(2, 3, 4), mask)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    empty = head(targets, torch.zeros(2, 3, 4), torch.zeros_like(mask))
    assert empty == 0 and head.last_forward_stats == {}
    empty.backward()
    assert all(p.grad is not None for p in head.parameters())


def test_mlp_sampling_subset_matches_full_with_same_noise():
    head = tiny_model().image_flow_head.eval()
    z, noise = torch.randn(1, 4, 32), torch.randn(1, 4, 4)
    whole = head.sample(z, cfg=1, num_steps=2, initial_noise=noise)
    subset = head.sample(z[:, [1, 3]], cfg=1, num_steps=2, initial_noise=noise[:, [1, 3]])
    torch.testing.assert_close(whole[:, [1, 3]], subset)
    changed = noise.clone(); changed[:, 0] += 5
    after = head.sample(z, cfg=1, num_steps=2, initial_noise=changed)
    torch.testing.assert_close(whole[:, 1:], after[:, 1:])


@pytest.mark.parametrize("rounds", [1, 2, 4])
@pytest.mark.parametrize("cfg", [1., 3.5])
def test_generation_only_samples_new_tokens_and_preserves_old(monkeypatch, rounds, cfg):
    model = tiny_model().eval()
    b = batch()
    calls, samples = [], []
    h = model.model.register_forward_pre_hook(lambda m, a: calls.append(1))
    original = model.image_flow_head.sample
    def sample(*args, **kwargs):
        out = original(*args, **kwargs)
        samples.append(out.clone())
        return out
    monkeypatch.setattr(model.image_flow_head, "sample", sample)
    args = dict(input_ids=b["X0_input_ids"], token_types=b["token_types"], sigma=b["flow_sigma"],
        spans=[(0, 2, 6)], reveal_order=torch.arange(4)[None], initial_noise_bank=torch.randn(1, 4, 4),
        flow_cfg=cfg, flow_num_steps=2, reveal_steps=rounds, return_trace=True)
    out, trace = model.generate("t2i", **args)
    assert len(calls) == rounds and sum(t.shape[1] for t in samples) == 4
    torch.testing.assert_close(out.flatten(2).transpose(1, 2), torch.cat(samples, 1).to(out.dtype))
    assert trace["backbone_calls"] == rounds and trace["flow_head_calls"] == 4 * rounds
    assert not trace["backbone_kv_cache_enabled"] and torch.isfinite(out).all()
    replay, _ = model.generate("t2i", **args, initial_image_latents=b["image_latents"] + 100)
    torch.testing.assert_close(out, replay, rtol=0, atol=0)
    h.remove()


def test_i2t_backbone_matches_z_and_checkpoint_roundtrip(tmp_path):
    model, reference = tiny_model().eval(), z_model().eval()
    reference.model.load_state_dict(model.model.state_dict())
    b = batch()
    b.update(compute_image_loss=False, image_loss_mask=torch.zeros_like(b["image_loss_mask"]))
    seen = []
    hooks = [m.model.register_forward_hook(lambda m, a, out: seen.append(out.last_hidden_state)) for m in (model, reference)]
    model(**b); reference(**b)
    torch.testing.assert_close(seen[0], seen[1])
    for h in hooks:
        h.remove()
    model.save_pretrained(tmp_path)
    assert isinstance(AutoConfig.from_pretrained(tmp_path), YConfig)
    reloaded = YForCausalLM.from_pretrained(tmp_path).eval()
    assert reloaded.config.y_reveal_steps == 2
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)


def test_cosine_bounds_and_recipe():
    from omegaconf import OmegaConf
    from utils.y_protocol import CONFIG, validate_joint_dit_config
    from scripts.launch_short_ablation import launch_plan
    for rounds in [1, 4, 8, 16, 32, 64, 256]:
        counts = cosine_reveal_counts(256, rounds)
        assert len(counts) == rounds and sum(counts) == 256 and min(counts) >= 1
    assert validate_joint_dit_config(OmegaConf.load(CONFIG))["method"] == "Y"
    plan = launch_plan("y", environment={}, smoke=True, steps=2)
    assert plan["world_size"] == 16 and f"config={CONFIG}" in plan["command"]


def test_training_validation_uses_y_and_fixed_reveal_orders(monkeypatch, tmp_path):
    from test_training_image_generation import config, write_prompts, TinyVAE
    from test_training_unified_loss_validation import Tokenizer
    from utils import training_image_generation as generation
    write_prompts(tmp_path)
    model = tiny_model().train()
    monkeypatch.setattr(generation, "load_vae", lambda *args: TinyVAE())
    runner = generation.TrainingImageGenerator(config(tmp_path))
    reports = []
    for step in (2, 4):
        torch.manual_seed(step)
        report = runner.run(model, Tokenizer(), device=torch.device("cpu"), step=step, output_dir=tmp_path)
        assert report["complete"] and report["method"] == "Y" and model.training
        assert report["cache_mode"] == "backbone_condition_per_reveal"
        reports.append(report)
    for first, second in zip(reports[0]["images"], reports[1]["images"]):
        assert first["trace"]["reveal_order"] == second["trace"]["reveal_order"]
        assert first["trace"]["flow_head_calls"] == 40
        assert (tmp_path / "validation_generation/step-2" / first["image"]).read_bytes() == (
            tmp_path / "validation_generation/step-4" / second["image"]).read_bytes()
