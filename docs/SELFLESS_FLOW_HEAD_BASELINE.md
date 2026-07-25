# Selfless-Flow Flow Head Baseline

Status: **active**

Last updated: 2026-07-25

## Runtime interface

The active contextual flow head is the shared-attention/shared-MLP dynamic
dual-stream architecture `DF1`. New configs expose exactly two complete
position contracts:

| Baseline | Query/content position | Attention position | Role |
| --- | --- | --- | --- |
| `DF1-FH0` | additive 2D | none | default |
| `DF1-FH4` | none | row/column 2D RoPE | pure-RoPE alternative |

The only supported configuration surface is:

```yaml
model:
  image_flow_head_arch: "contextual"
  image_flow_head_variant: "DF1"
  image_flow_position_variant: "FH0"  # FH0 | FH4
```

`image_flow_position_variant` resolves the entire position contract. New
configs must not independently compose query position, context position, and
RoPE flags.

## Evidence

Under the matched ImageNet-100 seed-42 protocol:

| Baseline | FID ↓ | IS ↑ | generation samples/s |
| --- | ---: | ---: | ---: |
| `DF1-FH0` | 23.5699 | 64.7787 | 0.799 |
| `DF1-FH4` | 23.0230 | 64.6608 | 0.489 |

`FH0` is the default because it is the faster atomic position design; `FH4`
is retained as the clean relative-position design.

Sampling efficiency is reported but is not a pass/fail gate. The active pair is
explicitly an interface-convergence decision around the additive-only and
pure-RoPE endpoints, not a claim that the retained cells won every
single-seed metric. The amended quality-only selector outcome and every
non-retained cell appear only in the archived conclusion.

The complete matrices, rejected variants, historical raw selector output,
source snapshot, configs, checkpoints, metrics, and relocation map are
archived.
See
[`archive/SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL_HISTORICAL.md`](archive/SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL_HISTORICAL.md)
and `output/flow_head_ablation/relocation_manifest.json`.

The retained EMA checkpoints are:

```text
output/flow_head_ablation/dynamic_dual_stream_screen/runs/
├── selfless-flow-dual-df1-fh0-s42/hf_model-final-ema/
└── selfless-flow-dual-df1-fh4-s42/hf_model-final-ema/
```

Historical launchers, matrix builders, tests, and the exact shared runtime
source snapshot are under `scripts/archive/flow_head_ablation/`,
`script/ablation/archive/flow_head_ablation/`, and
`configs/ablation/archive/flow_head_ablation/`.
