"""Single-NPU throughput micro-benchmark for the Selfless-Flow NPU migration.

Measures fwd+bwd step time (s/optimizer step and samples/sec/GPU) using:
- official Qwen3-0.6B-Base architecture (random init, no pretrained weights)
- fake ImageNet posterior cache (N=115200, 256 tokens, fp16)
- class conditioning, B=32, seq 320, grad accum 2

This reproduces the compute workload that the training loop runs per step.
Run:
    ASCEND_RT_VISIBLE_DEVICES=2 python tests/bench_npu_step.py
"""

import time
import sys

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM
from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset
from utils.imagenet_flow_batching import collate_imagenet_flow_cache
from utils.utils import get_selfless_mask

MODEL_PATH = "/tmp/qwen3-0.6b-base"
FAKE = "/tmp/fake_imagenet_bench"

import os
GC = os.environ.get("GRAD_CKPT", "0") == "1"
import models.modeling_model.modeling_selfless_flow as _msf

# Match the 8xH100 config: B=32/GPU, total_batch 512 -> grad accum 2 (8 GPUs).
B = 16
GRAD_ACC = 4
SEQ = 320
WARMUP = 5
MEASURE = 10


def build():
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
    tok, cfg, ids = build()
    device = "npu"

    model = Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))
    model = model.to(device)
    if GC:
        model.gradient_checkpointing_enable()
    model.train()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n:.1f}M (Qwen3-0.6B-Base, random init, NPU)")

    ds = ImageNetFlowCacheDataset(
        cache_path=f"{FAKE}/posterior_stats_all_fp16.pt",
        tokenizer=tok,
        boi_token_id=ids["<|boi|>"], eoi_token_id=ids["<|eoi|>"],
        mask_token_id=ids["<|mdm_mask|>"], eos_token_id=tok.eos_token_id,
        image_tokens_per_img=256, image_latent_dim=16,
        manifest_jsonl=f"{FAKE}/manifest.jsonl",
        synset_mapping_path=f"{FAKE}/LOC_synset_mapping.txt",
        conditioning_mode="class", max_seq_length=SEQ,
    )
    print("dataset OK, len:", len(ds))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.01)

    def make_batch(i):
        idx = [(i * B + j) % len(ds) for j in range(B)]
        batch = collate_imagenet_flow_cache(
            [ds[k] for k in idx], pad_to_length=SEQ, pad_to_multiple_of=64
        )
        sigma = batch["sigma"].to(device)
        attn_mask = get_selfless_mask(sigma.float(), SEQ, device)
        return {
            "X0_input_ids": batch["input_ids"].to(device),
            "attention_mask": attn_mask,
            "position_ids": batch["position_ids"].to(device),
            "token_types": batch["token_types"].to(device),
            "image_span_table": batch["image_span_table"].to(device),
            "image_local_positions": batch["image_local_positions"].to(device),
            "labels": batch["labels"].to(device),
            "image_latents": batch["image_latents"].to(device),
            "use_cache": False,
        }

    # Pre-generate a few batches to exclude dataloader from timing.
    batches = [make_batch(i) for i in range(MEASURE + WARMUP)]

    def step(b):
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACC):
            out = model(**b)
            out.loss.backward()
        if GC and hasattr(_msf, "_checkpointed_selfless_forward"):
            _msf._checkpointed_selfless_forward.last_microbatch = False
        torch.npu.synchronize()
        return out.loss.item()

    def set_batch_mul(m):
        model.image_flow_batch_mul = int(m)

    # Warmup
    set_batch_mul(4)
    for i in range(WARMUP):
        step(batches[i])
    torch.npu.synchronize()

    def run_measure(m):
        set_batch_mul(m)
        for i in range(2):  # re-warm under this mul
            step(batches[i])
        torch.npu.synchronize()
        times = []
        for i in range(WARMUP, WARMUP + MEASURE):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            loss = step(batches[i])
            torch.npu.synchronize()
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        sps = (B * GRAD_ACC) / avg
        est = 17920 * avg / 3600
        print(f"batch_mul={m}: step {avg:.4f}s | {sps:.2f} samples/s/GPU "
              f"| est80ep(8x,no comm) {est:.2f}h")
        return avg

    print(f"\n=== Qwen3-0.6B-Base | B={B} ga={GRAD_ACC} seq={SEQ} gc={GC} | NPU ===")
    for m in (1, 2, 4):
        run_measure(m)


if __name__ == "__main__":
    main()
