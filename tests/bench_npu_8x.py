"""8-NPU DDP throughput benchmark for the Selfless-Flow NPU migration.

Validates that the fused-op implementation scales across all 8 Ascend 910B
devices (HCCL backend), matching the 8xH100 production configuration:
B=32/GPU x 8 ranks x grad_accum=2 = global batch 512.

Launch (torchrun, no accelerate config changes required):

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc_per_node=8 --master_port=29511 tests/bench_npu_8x.py

Environment knobs:
    GRAD_CKPT=1        enable gradient checkpointing (default off, matches
                       use_gradient_checkpointing=false in the class config)
    BENCH_STEPS=20     measured optimizer steps per run (default 20)
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402

from models.modeling_model.modeling_selfless_flow import Qwen3ForCausalLM  # noqa: E402
from utils.dataset_imagenet_flow_cache import ImageNetFlowCacheDataset  # noqa: E402
from utils.imagenet_flow_batching import collate_imagenet_flow_cache  # noqa: E402
from utils.utils import get_selfless_mask  # noqa: E402

MODEL_PATH = os.environ.get("QWEN3_BASE", "/tmp/qwen3-0.6b-base")
FAKE = os.environ.get("FAKE_IMAGENET", "/tmp/fake_imagenet_bench")

B = int(os.environ.get("BENCH_B", "32"))  # micro-batch per rank
GRAD_ACC = int(os.environ.get("BENCH_GA", "2"))  # grad-accum per rank
SEQ = 320
WARMUP = 3
MEASURE = int(os.environ.get("BENCH_STEPS", "20"))
GC = os.environ.get("GRAD_CKPT", "0") == "1"


class _SimpleTokenizer:
    """Bypass HF tokenizer download; selfless-flow class mode only needs
    ``encode`` (used by ``_text_ids``) and ``eos_token_id``."""

    eos_token_id = 151643

    def __len__(self):
        return 151936

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 + 100 for c in text]


def build_config():
    # Mirror utils/utils.py:load_model_tokenizer exactly: start from the
    # real Qwen3-0.6B-Base checkpoint config (151669 base tokens) and add
    # the selfless-flow extras. tie_word_embeddings stays True -> the
    # benchmark model has 760.9M params, matching production.
    cfg = Qwen3Config.from_pretrained(MODEL_PATH)
    cfg.mask_token_id = 151669  # <|mdm_mask|> added as first special token
    cfg.boi_token_id = 151670
    cfg.eoi_token_id = 151671
    cfg.image_mask_token_id = 151672
    cfg.use_flex_attention = True
    cfg.image_tokens_per_img = 256
    cfg.image_latent_dim = 16
    cfg.backbone_attention_output_gate = "none"
    cfg.image_flow_depth = 8
    cfg.image_flow_width = 1280
    cfg.image_flow_num_sampling_steps = "100"
    cfg.image_flow_batch_mul = int(os.environ.get("FLOW_MUL", "4"))
    cfg.image_flow_time_scale = 1000.0
    cfg.image_flow_time_sampling = "logit_normal"
    cfg.image_flow_logit_mean = 0.0
    cfg.image_flow_logit_std = 1.0
    cfg.image_flow_time_eps = 1.0e-5
    cfg.image_flow_time_uniform_mix = 0.1
    cfg.image_flow_solver = "heun"
    cfg.image_input_noise_strength = 1.0e-2
    cfg.image_uncond_prob = 0.1
    return cfg


def main():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    # ASCEND_RT_VISIBLE_DEVICES 已做物理→逻辑映射，set_device 用逻辑 id。
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)

    tok = _SimpleTokenizer()
    cfg = build_config()

    model = Qwen3ForCausalLM(cfg).to(dtype=torch.bfloat16)
    # Mirror production (utils/utils.py:221): resize to cover the 4 added
    # special tokens; 151669 -> 151673 rows in the embedding.
    model.resize_token_embeddings(151673)
    model = model.to(device)
    if GC:
        model.gradient_checkpointing_enable()
    # NOTE: no DDP wrapper — mirrors accelerate's default DDP config
    # (find_unused_parameters=False would fail on the flow-head stats
    # path; True breaks `last_hidden_state` kwarg). HCCL grad-sync cost is
    # measured separately below via an explicit all_reduce on the grads.
    model.train()
    model.image_flow_batch_mul = int(os.environ.get("FLOW_MUL", "4"))

    if rank == 0:
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"world={world} | model params {n:.1f}M | B={B} ga={GRAD_ACC} "
              f"gc={GC} | global_batch={B*GRAD_ACC*world}")

    ds = ImageNetFlowCacheDataset(
        cache_path=f"{FAKE}/posterior_stats_all_fp16.pt",
        tokenizer=tok,
        boi_token_id=cfg.boi_token_id, eoi_token_id=cfg.eoi_token_id,
        mask_token_id=cfg.mask_token_id, eos_token_id=tok.eos_token_id,
        image_tokens_per_img=256, image_latent_dim=16,
        manifest_jsonl=f"{FAKE}/manifest.jsonl",
        synset_mapping_path=f"{FAKE}/LOC_synset_mapping.txt",
        conditioning_mode="class", max_seq_length=SEQ,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.01)

    def make_batch(i):
        # Disjoint slices per rank (mimic DistributedSampler).
        base = (i * world + rank) * B
        idx = [(base + j) % len(ds) for j in range(B)]
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
            "record_flow_stats": False,
        }

    batches = [make_batch(i) for i in range(MEASURE + WARMUP)]

    def step(b):
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACC):
            out = model(**b)
            out.loss.backward()
        # Mirror DDP grad-sync: single flat all_reduce over all grads.
        # This is what HCCL does under DDP; isolated here so compute vs
        # comm cost is explicit (and avoids DDP wrapper incompatibilities
        # with the flow-head stats path).
        flat = [p.grad for p in model.parameters() if p.grad is not None]
        if flat:
            torch._foreach_mul_(flat, 1.0 / world)
            for g in flat:
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
        opt.step()
        torch.npu.synchronize()
        return out.loss.detach().float().item()

    # Warmup (includes HCCL kernel init + lazy op compilation).
    for i in range(WARMUP):
        loss = step(batches[i])
    torch.npu.synchronize()
    dist.barrier()

    times = []
    losses = []
    for i in range(WARMUP, WARMUP + MEASURE):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        losses.append(step(batches[i]))
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    t_min = min(times)
    t_max = max(times)
    sps_rank = (B * GRAD_ACC) / avg
    sps_global = sps_rank * world
    est_h = 17920 * avg / 3600
    peak_gib = torch.npu.max_memory_allocated() / 1024**3

    if rank == 0:
        print(f"\n=== 8x910B DDP | B={B} ga={GRAD_ACC} seq={SEQ} gc={GC} ===")
        print(f"loss first/last: {losses[0]:.4f} / {losses[-1]:.4f}")
        print(f"step avg {avg*1000:.0f}ms (min {t_min*1000:.0f} / "
              f"max {t_max*1000:.0f})")
        print(f"throughput: {sps_rank:.1f} img/s/rank | "
              f"{sps_global:.0f} img/s global")
        print(f"est 80ep (17920 steps, incl. comm): {est_h:.2f}h")
        print(f"peak HBM: {peak_gib:.1f} GiB/rank")
        acceptance = "PASS" if est_h < 6.0 * 1.10 else "CHECK"
        print(f"acceptance (8xH100=6h, +10% tolerance): {acceptance}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
