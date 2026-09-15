from types import SimpleNamespace
import json

import pytest
import torch

from scripts.generate_unified_qualitative import (
    build_t2i_item, caption_sigmas, decode_suffix, fixed_posterior,
    noise_for, render, task_training, validate_generation_trace,
)
from utils.imagenet_flow_batching import collate_imagenet_flow_cache


class Tokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [4, 5, 6]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def model_stub():
    return SimpleNamespace(config=SimpleNamespace(
        image_tokens_per_img=4, image_latent_dim=4, boi_token_id=11,
        eoi_token_id=12, mask_token_id=7, image_mask_token_id=8,
    ))


@pytest.mark.parametrize("order", ["random", "sequential"])
def test_t2i_serialization_matches_training_and_eoi_precedes_image(order):
    item = build_t2i_item(Tokenizer(), model_stub(), "a red cube", 0, 42, order)
    assert item["input_ids"].tolist() == [4, 5, 6, 11, 7, 7, 7, 7, 12, 9]
    batch = collate_imagenet_flow_cache([item], pad_to_length=16)
    assert batch["token_types"][0].tolist() == [0, 0, 0, 2, 1, 1, 1, 1, 2, 2] + [3] * 6
    image_sigma = batch["sigma"][0, 4:8]
    assert sorted(image_sigma.tolist()) == [5, 6, 7, 8]
    assert batch["sigma"][0, 8] < image_sigma.min()
    assert not batch["image_latents"].any()
    if order == "sequential":
        assert image_sigma.tolist() == [5, 6, 7, 8]


def test_paired_noise_does_not_depend_on_batch_model_or_global_rng():
    original = noise_for(5, 42, 4, 4)
    torch.manual_seed(999)
    _ = torch.randn(100)
    assert torch.equal(original, noise_for(5, 42, 4, 4))
    assert not torch.equal(original, noise_for(5, 43, 4, 4))
    assert not torch.equal(original, noise_for(6, 42, 4, 4))


@pytest.mark.parametrize("order", ["random", "sequential"])
def test_caption_order_uses_native_contract_without_caption_leakage(order):
    model = model_stub()
    rows = [{"posterior_seed": 42, "reference": "must not enter prompt"}, {"posterior_seed": 73}]
    sigma = caption_sigmas(Tokenizer(), model, rows, order)
    assert sigma.shape == (2, 9)
    assert torch.equal(sigma[:, :4], torch.tensor([[0, 1, 2, 3]] * 2).float())
    assert sigma[:, -1].tolist() == [4, 4]
    for row in sigma[:, 4:8]:
        assert sorted(row.tolist()) == [5, 6, 7, 8]
    assert torch.equal(sigma, caption_sigmas(Tokenizer(), model, rows, order))


def test_scaled_mean_std_posterior_does_not_apply_scaling_twice():
    stats = torch.cat((torch.full((256, 16), 2.0), torch.zeros(256, 16)), dim=-1).half()
    assert torch.equal(fixed_posterior(stats, 42), torch.full((256, 16), 2.0).half())
    stats[0, 16] = -1
    with pytest.raises(ValueError, match="negative"):
        fixed_posterior(stats, 42)


def test_decode_records_eos_or_truncation_and_strips_prompt():
    assert decode_suffix(Tokenizer(), [4, 9, 5], [9])["token_ids"] == [4]
    assert decode_suffix(Tokenizer(), [4, 9], [9])["stop_reason"] == "eos"
    assert decode_suffix(Tokenizer(), [4, 5], [9])["stop_reason"] == "max_new_tokens"


def test_untrained_modalities_are_explicitly_marked():
    assert task_training("a-text-only")["t2i"] != "已训练"
    assert task_training("b-caption-only")["t2i"] != "已训练"
    assert task_training("unified-b-0p6b-caption-only-100bphys-s42-r1")["i2t"] == "已训练"
    assert task_training("unified-b-x0content-0p6b-100b-imagenet-split-s42-r1")["text"] == "已训练"
    assert task_training("unified-b")["text"] != "已训练"


def test_model_inventory_excludes_temporary_exports_before_loading_them(tmp_path, monkeypatch):
    from scripts import generate_unified_qualitative as generation

    names = ['unified-formal', 'unified-formal-smoke-check', 'unified-formal-debug', 'unified-formal-replay']
    for name in names:
        export = tmp_path / 'output' / name / 'hf_model-final-ema'
        export.mkdir(parents=True)
        (export / 'config.json').write_text(json.dumps({
            'architecture_variant': 'selfless_contextual',
            'dual_stream_attention_contract': 'selfless_strict',
        }))
    loaded = []
    def resolve(path):
        loaded.append(path.parent.name)
        assert path.parent.name == 'unified-formal'
        return SimpleNamespace(is_hf_final_ema=True, report=lambda: {})
    monkeypatch.setattr(generation, 'resolve_evaluation_model_source', resolve)
    models = generation.model_inventory(tmp_path)
    assert [model['run'] for model in models] == loaded == ['unified-formal']


def test_manifest_prompt_cardinality_and_uniqueness():
    from pathlib import Path
    data = json.loads(Path("configs/protocols/unified_qualitative_prompts_v1.json").read_text())
    for kind in ("t2i", "text"):
        assert len(data[kind]) == 32
        assert len({r["id"] for r in data[kind]}) == 32
        assert all(r["prompt"].strip() for r in data[kind])


def test_render_escapes_outputs_and_refuses_missing_records(tmp_path):
    spec = {"id": "b", "label": "B", "checkpoint": "/example/model", "source": {"global_step": 1},
            "architecture": "selfless_contextual", "backbone_attention": "xlnet_content_diagonal",
            "flow_attention": "xlnet_content_diagonal", "task_training": task_training("b")}
    samples = {"t2i": [{"id": "one", "group": "test", "prompt": "<script>bad</script>"}],
               "i2t": [], "text": []}
    (tmp_path / "manifest.json").write_text(json.dumps({"models": [spec], "samples": samples, "expected_per_model": {"t2i": 2, "i2t": 0, "text": 0}}))
    args = SimpleNamespace(output_dir=tmp_path, require_complete=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        render(args)
    page = (tmp_path / "index.html").read_text()
    assert "&lt;script&gt;bad&lt;/script&gt;" in page
    assert "<script>bad</script>" not in page
    assert not (tmp_path / "COMPLETED.json").exists()


def test_complete_gallery_validates_workers_and_produces_portable_zip(tmp_path):
    from PIL import Image
    import zipfile
    from scripts.generate_unified_qualitative import write_json

    spec = {"id": "b", "label": "B", "checkpoint": "/example/model", "source": {"global_step": 1},
            "architecture": "selfless_contextual", "backbone_attention": "xlnet_content_diagonal",
            "flow_attention": "xlnet_content_diagonal", "task_training": task_training("b")}
    samples = {"t2i": [{"id": "one", "group": "test", "prompt": "a bird"}], "i2t": [], "text": []}
    write_json(tmp_path / "manifest.json", {"models": [spec], "samples": samples, "expected_per_model": {"t2i": 2, "i2t": 0, "text": 0}})
    write_json(tmp_path / "preflight.json", {"complete": True})
    for rank in range(16):
        write_json(tmp_path / "progress" / f"rank-{rank:02d}.json", {"rank": rank, "world_size": 16,
                   "stage": "worker_complete", "complete": True, "limited_run": False})
        write_json(tmp_path / "models/b/load_reports" / f"rank-{rank:02d}.json", {"global_step": 1,
                   "full_checkpoint_value_check": {"complete": True}})
    for seed in (42, 43):
        relative = f"models/b/t2i/one-s{seed}.png"
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256)).save(tmp_path / relative)
        write_json((tmp_path / relative).with_suffix(".json"), {"sample_id": "one", "seed": seed,
                   "image": relative, "model": "b"})
    render(SimpleNamespace(output_dir=tmp_path, require_complete=True))
    assert json.loads((tmp_path / "COMPLETED.json").read_text())["complete"]
    with zipfile.ZipFile(tmp_path / "unified-generation-gallery.zip") as bundle:
        assert bundle.testzip() is None
        assert "index.html" in bundle.namelist()
        assert "models/b/t2i/one-s42.png" in bundle.namelist()
        assert len(bundle.read("results.jsonl").splitlines()) == 2


@pytest.mark.parametrize("variant", ["b", "c", "d", "e", "f", "s2"])
def test_actual_tiny_architectures_accept_all_three_generation_inputs(variant):
    from test_dynamic_xt_contract import tiny_config
    from test_positionwise_flow_on_b import _tiny_config as f_config
    from transformers import Qwen3Config
    from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
    from models.modeling_model.modeling_single_stream_text_ar import SingleStreamTextARConfig, SingleStreamTextARQwen3ForCausalLM
    from models.modeling_model.modeling_selfless_flow_dynamic_xt import DynamicXtQwen3ForCausalLM
    from models.modeling_model.modeling_selfless_flow_positionwise_on_b import PositionwiseFlowOnBQwen3ForCausalLM
    from pretrain.train_selfless_flow import _generate_i2t_caption_batch

    if variant == "s2":
        from test_showo2_unified import tiny_config as s2_config
        from models.modeling_model.modeling_showo2_unified import Showo2UnifiedForCausalLM
        config, cls = s2_config(), Showo2UnifiedForCausalLM
    elif variant == "d":
        config, cls = tiny_config(), DynamicXtQwen3ForCausalLM
    elif variant == "f":
        config, cls = f_config(), PositionwiseFlowOnBQwen3ForCausalLM
    else:
        config_class = SingleStreamTextARConfig if variant == "c" else Qwen3Config
        config = tiny_config(config_class)
        config.architecture_variant = "single_stream_text_ar" if variant == "c" else "selfless_contextual"
        cls = SingleStreamTextARQwen3ForCausalLM if variant == "c" else Qwen3ForCausalLM
    order = "sequential" if variant == "e" else "random"
    config.training_image_sigma_order = order
    model = cls(config).eval()
    tokenizer = Tokenizer()
    item = build_t2i_item(tokenizer, model, "a bird", 0, 42, order)
    batch = collate_imagenet_flow_cache([item], pad_to_length=16)
    with torch.no_grad():
        image, trace = model.generate("t2i", input_ids=batch["input_ids"], token_types=batch["token_types"],
            sigma=batch["sigma"], spans=[(0, item["image_start"], item["image_start"] + 4)],
            image_latent_dim=4, initial_noise_bank=noise_for(0, 42, 4, 4).unsqueeze(0),
            flow_temperature=1.0, flow_cfg=3.5, flow_cfg_schedule="constant", flow_solver="heun",
            flow_num_steps=2, parallel_rate=1, order_strategy="sequential" if variant == "e" else "spatial_halton",
            use_cache=True, return_trace=True)
        assert image.shape == (1, 4, 2, 2)
        assert torch.isfinite(image).all()
        validate_generation_trace(trace, use_cache=variant != "s2", task="t2i")
        captions, ids, reasons = _generate_i2t_caption_batch(model, tokenizer, torch.zeros(1, 4, 4),
            text_prefix="Describe this image in one detailed caption:", max_new_tokens=2, temperature=0,
            base_sigma_batch=caption_sigmas(tokenizer, model, [{"posterior_seed": 42}], order))
        assert len(captions) == len(ids) == len(reasons) == 1
        text, trace = model.generate("text", input_ids=torch.tensor([[4, 5, 6]]), max_new_tokens=2,
            temperature=0, eos_token_id=9, use_cache=True, return_trace=True)
        assert text.shape[1] >= 4
        validate_generation_trace(trace, use_cache=variant != "s2", task="text")


def test_s2_trace_requires_correct_generation_mode_and_cache():
    with pytest.raises(RuntimeError, match="cache differs"):
        validate_generation_trace({"backbone_kv_cache_enabled": True}, use_cache=False, task="t2i")
    with pytest.raises(RuntimeError, match="expected S2 generation mode"):
        validate_generation_trace({"backbone_kv_cache_enabled": False, "generation_mode": "showo2_text_ar"},
                                  use_cache=False, task="t2i")
