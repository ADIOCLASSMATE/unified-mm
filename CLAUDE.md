# Unified-MM Project Context

## Current scope

The default benchmark is the balanced ImageNet-100 architecture ablation. Keep
the two active paths comparable:

- Selfless-Flow: `models/modeling_model/modeling_selfless_flow.py`
- Qwen-Show-O: `models/modeling_model/modeling_qwen_showo.py`

Both use a Qwen3-0.6B backbone, the same 115K/10K split, global batch 256, and
35,920 optimizer steps. The authoritative protocol and results are in
`docs/IMAGENET100_ABLATION.md`.

## Non-negotiable evaluation defaults

- Selfless-Flow selected default: EMA `hf_model-final-ema`, BF16, CFG 3.5,
  constant schedule, 100-step Heun, `spatial_halton`, `PARALLEL_RATE=1`.
- Qwen-Show-O selected default: `hf_model-final`, official Show-O
  `guidance_scale s=11.75`, common CFG `w=12.75`, 12 steps, temperature 1.0.
- Show-O mapping: `w=1+s`; its official formula is
  `(1+s)*conditional - s*unconditional`.
- Do not run or document FP32 model-forward diagnostics. Flow's VAE decode and
  numerical integration may remain FP32 as required by the protocol.
- All formal scores use 10K samples, seed 42, eight GPUs, and the shared cached
  original-ImageNet Inception distribution.

## Data and training invariants

- Dataset root: `public/datasets/imagenet_ablation_100c_balanced`.
- Split membership comes from `split_seed42_val100.jsonl`; do not reconstruct it
  from a sorted manifest.
- The split has 100 classes, 1,150 train and 100 validation images per class.
- Flow uses 256 KL16 latent tokens of width 16.
- Show-O uses 256 official MAGVITv2 tokens and an 8,192-code image vocabulary.
- Keep the shared real-stat cache architecture-independent; VAE/VQ
  reconstruction distributions are not comparable substitutes.

## Active commands

```bash
bash script/ablation/build_imagenet_100c_balanced_cache.sh
bash script/ablation/prepare_qwen_showo_vq_100c.sh
bash script/ablation/cache_imagenet100_original_fid_stats.sh
bash script/ablation/pretraining_imagenet_flow_100c_80ep.sh
bash script/ablation/pretraining_qwen_showo_vq_100c_80ep.sh
bash script/ablation/evaluate_imagenet_flow_100c.sh
bash script/ablation/evaluate_qwen_showo_vq_100c.sh
```

The retained full-ImageNet and text-selfless launchers under `script/selfless/`
are optional paths, not default benchmark commands.

## Engineering rules

- Preserve same-position labels and the strict `sigma[kv] < sigma[q]` attention
  rule in Selfless-Flow.
- Preserve official cumulative Show-O ranking temperature updates.
- Sweep roots contain source/config/checkpoint hashes. After changing a bound
  file, use a new root; never rewrite an old contract to make it pass.
- Use the ablation wrappers for formal evaluation. Generic evaluator defaults
  are not the benchmark contract.
- Keep unrelated user changes in the dirty worktree intact.
