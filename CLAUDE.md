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
- Formal evaluation uses 10K samples on 8×H100.
- Evaluator batch size is global before sharding: default 4096, or 512/rank.
- Use a real-stat cache matched to the evaluation prompt/data distribution.

The only retained research summary is `docs/ABLATION_CONCLUSIONS.md`.
