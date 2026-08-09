"""Single-NPU end-to-end smoke test for the NPU-only migration.

Uses the official Qwen3-0.6B-Base architecture with random initialization
(no pretrained weights needed) and a small fake ImageNet posterior cache.

Run:
    ASCEND_RT_VISIBLE_DEVICES=2 python tests/smoke_npu_e2e.py
"""

import torch
import torch_npu  # noqa: F401  (registers NPU)
from transformers import AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.utils import get_selfless_mask

MODEL_PATH = "/tmp/qwen3-0.6b-base"
FAKE_IMAGENET = "/tmp/fake_imagenet"


def build_tokenizer_and_config():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
    tok.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    tok.add_tokens(["<|boi|>", "<|eoi|>", "<|img_mask|>"], special_tokens=True)
    ids = {t: tok.convert_tokens_to_ids(t)
           for t in ("<|mdm_mask|>", "<|boi|>", "<|eoi|>", "<|img_mask|>")}

    cfg = Qwen3Config.from_pretrained(MODEL_PATH)
    cfg.mask_token_id = ids["<|mdm_mask|>"]
    cfg.boi_token_id = ids["<|boi|>"]
    cfg.eoi_token_id = ids["<|eoi|>"]
    cfg.image_mask_token_id = ids["<|img_mask|>"]
    cfg.use_flex_attention = True
    cfg.image_tokens_per_img = 256
    cfg.image_latent_dim = 16
    cfg.backbone_attention_output_gate = "none"
    cfg.image_flow_depth = 8
    cfg.image_flow_width = 1280
    cfg.image_flow_num_sampling_steps = "100"
    cfg.image_flow_batch_mul = 4
    cfg.image_flow_time_scale = 1000.0
    cfg.image_flow_time_sampling = "logit_normal"
    cfg.image_flow_logit_mean = 0.0
    cfg.image_flow_logit_std = 1.0
    cfg.image_flow_time_eps = 1.0e-5
    cfg.image_flow_time_uniform_mix = 0.1
    cfg.image_flow_solver = "heun"
    cfg.image_input_noise_strength = 1.0e-2
    cfg.image_uncond_prob = 0.1
    return tok, cfg, ids


def main():
    tok, cfg, ids = build_tokenizer_and_config()

    model = Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))
    model = model.to("npu")
    model.train()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n:.1f}M (Qwen3-0.6B-Base, random init, NPU)")

    ds = ImageNetFlowCacheDataset(
        cache_path=f"{FAKE_IMAGENET}/posterior_stats_all_fp16.pt",
        tokenizer=tok,
        boi_token_id=ids["<|boi|>"], eoi_token_id=ids["<|eoi|>"],
        mask_token_id=ids["<|mdm_mask|>"], eos_token_id=tok.eos_token_id,
        image_tokens_per_img=256, image_latent_dim=16,
        manifest_jsonl=f"{FAKE_IMAGENET}/manifest.jsonl",
        synset_mapping_path=f"{FAKE_IMAGENET}/LOC_synset_mapping.txt",
        conditioning_mode="class", max_seq_length=320,
    )
    print("dataset OK, len:", len(ds))

    batch = collate_imagenet_flow_cache(
        [ds[i] for i in range(4)], pad_to_length=320, pad_to_multiple_of=64
    )

    sigma = batch["sigma"].to("npu")
    B, S = sigma.shape
    attn_mask = get_selfless_mask(sigma.float(), S, "npu")
    print("attn_mask:", attn_mask.shape, attn_mask.dtype,
          "disallow_ratio=%.3f" % attn_mask.float().mean().item())

    out = model(
        X0_input_ids=batch["input_ids"].to("npu"),
        attention_mask=attn_mask,
        position_ids=batch["position_ids"].to("npu"),
        token_types=batch["token_types"].to("npu"),
        image_span_table=batch["image_span_table"].to("npu"),
        image_local_positions=batch["image_local_positions"].to("npu"),
        labels=batch["labels"].to("npu"),
        image_latents=batch["image_latents"].to("npu"),
        use_cache=False,
    )
    print("loss:", out.loss.item())
    assert torch.isfinite(out.loss), "loss not finite!"
    out.loss.backward()
    gn = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"backward OK, grad_norm={gn:.4f}")
    print("E2E PASS")


if __name__ == "__main__":
    main()
