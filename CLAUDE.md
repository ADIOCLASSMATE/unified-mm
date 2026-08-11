# Unified-MM Project Context

## Architecture invariants

- Selfless-Flow image tokens are 256 KL16 tokens of width 16.
- Image attention uses row/column pure 2D RoPE in both the Qwen backbone and
  dynamic dual-stream contextual flow head.
- Do not add additive image positions or architecture-selection switches.
- `backbone_attention_output_gate` is the only retained architecture option:
  `none` by default, or `per_head_identity_sigmoid` for compatible research
  checkpoints.
- Preserve same-position labels and strict `sigma[kv] < sigma[q]` visibility.

## Data invariants

- `ImageNetFlowCacheDataset` supports exactly `class` and `caption`.
- Caption membership is strict, full captions are preserved, and training may
  use deterministic segment packing. Validation rows stay independent.
- Keep the latent cache contract at `[N, 256, 16]`.

## Evaluation defaults

- EMA checkpoint, BF16 model forward, CFG 3.5, constant schedule, 100-step
  Heun, `spatial_halton`, `parallel_rate=1`, seed 42.
- Formal ImageNet-1K evaluation uses 50K samples on 16×Ascend 910B with HCCL.
- Evaluator batch size is global before sharding: default 4096, or 256/rank.
- Use a real-stat cache matched to the evaluation prompt/data distribution.

## Runtime invariants

- The repository root `.venv` is the only supported environment and is managed
  by `uv`; do not recreate `.venv-npu` or install packages with ad-hoc pip calls.
- Keep Python 3.11, PyTorch 2.6.0+cpu, torch-npu 2.6.0.post5, torchvision
  0.21.0+cpu, and NumPy 1.26.4 aligned with the installed CANN runtime.
- Do not add CUDA, NCCL, Triton, `flash-attn`, or NVIDIA wheel dependencies.
- Preserve the CANN paths when setting `PYTHONPATH`; append the repository root
  instead of replacing `PYTHONPATH` with `.`.

The only retained research summary is `docs/ABLATION_CONCLUSIONS.md`.
