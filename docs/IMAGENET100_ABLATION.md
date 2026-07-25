# ImageNet-100 Architecture Ablation: Protocol and Results

Status: final 10K evaluation; original architecture results completed
2026-07-19, flow-head position follow-up closed 2026-07-24.

This is the authoritative document for the balanced ImageNet-100 comparison
between Selfless-Flow, its flow-head variants, and Qwen-Show-O. It replaces the
former separate research, dataset, and Show-O pipeline notes.

## Decision summary

The repository defaults are selected by minimum FID. Maximum-IS points are
reported separately because increasing guidance trades diversity for class
confidence.

| Architecture / checkpoint | Selection | CFG | FID ↓ | IS ↑ |
| --- | --- | ---: | ---: | ---: |
| **Selfless-Flow EMA** `hf_model-final-ema` | minimum FID, default | **3.5** | **26.0110** | 59.5362 ± 1.1316 |
| Token-only MLP EMA `hf_model-final-ema` | architecture ablation; baseline-fixed inference | **3.5** | 27.9774 | 56.9485 ± 1.5827 |
| Token-only MLP ratio 4.5 EMA `hf_model-final-ema` | parameter-matched architecture ablation | **3.5** | 26.4404 | 58.3860 ± 1.1099 |
| Token-only MLP width 1936 EMA `hf_model-final-ema` | alternative parameter-matched scale | **3.5** | 26.9315 | 57.3898 ± 1.4622 |
| Selfless-Flow EMA `hf_model-final-ema` | maximum IS | 5.0 | 28.2501 | **61.6962 ± 1.0704** |
| Selfless-Flow non-EMA `hf_model-final` | EMA-selected diagnostic | 3.5 | 26.0782 | 60.0868 ± 1.4928 |
| **Qwen-Show-O** `hf_model-final` | minimum FID, default | common **w=12.75**, Show-O **s=11.75** | **31.1314** | 66.7629 ± 0.5617 |
| Qwen-Show-O `hf_model-final` | maximum IS | common w=11.0, Show-O s=10.0 | 32.0040 | **67.7769 ± 0.7228** |

At their FID-selected defaults, Selfless-Flow lowers FID by 5.1204 points
(16.45%) relative to Show-O, while Show-O has the higher IS. This is a
fidelity/diversity trade-off, not evidence that either metric alone captures
all qualitative differences. The comparison is controlled but single-seed;
small differences should not be presented as statistically significant.

## Dataset and training budget

- Source: official ImageNet `imagenet:v1`.
- Selection: 100 reproducibly distributed classes, each with 1,250 images.
- Training split: 1,150 images per class, 115,000 total.
- Validation split: 100 images per class, 10,000 total.
- Prompt: the first class name in `LOC_synset_mapping.txt`.
- Global training batch: 256 on eight GPUs.
- Budget: 449 optimizer steps per epoch and 35,920 steps for 80 epochs.
- Backbone scale: Qwen3-0.6B-Base for both architectures.

The authoritative membership file is:

```text
public/datasets/imagenet_ablation_100c_balanced/split_seed42_val100.jsonl
```

Do not reconstruct validation membership from a newly sorted manifest; the
explicit split file defines both membership and evaluation order.

## Architecture-specific training

### Selfless-Flow

- Continuous KL16 latents: 256 tokens × 16 channels.
- Two-stream Qwen with same-position targets and strict
  `sigma[kv] < sigma[q]` visibility.
- Contextual rectified-flow head trained with velocity MSE.
- Classifier-free training dropout: 0.1.
- EMA decay: 0.999 from step zero.
- Training config: `configs/ablation/imagenet_flow_100c_80ep.yaml`.
- Selected checkpoint: `hf_model-final-ema`.

### Qwen-Show-O

- Frozen official Show-O MAGVITv2 tokenizer.
- 8,192 image codes and 256 tokens per 256×256 image.
- Full unified-vocabulary CE only at masked image positions.
- Cosine random masking and classifier-free condition dropout 0.1.
- Training config: `configs/ablation/qwen_showo_vq_100c_80ep.yaml`.
- Selected checkpoint: `hf_model-final` (no EMA checkpoint).

The implementation follows Show-O's guidance convention:

```text
guided = (1 + s) * conditional - s * unconditional
common CFG weight w = 1 + s
```

Every Show-O result therefore records both common `w` and Show-O argument `s`.

## Shared formal metric protocol

Every formal point uses:

- all 10,000 validation prompts in authoritative order;
- seed 42 and eight H100 GPUs;
- original-ImageNet real statistics shared by both architectures;
- Inception feature dimension 2,048;
- 10 contiguous IS splits with population standard deviation;
- stable symmetric-eigendecomposition FID reduction;
- deterministic global sample indexing; evaluations that export images also
  validate exact filenames and counts.

Key provenance hashes:

| Artifact | SHA256 |
| --- | --- |
| Real-image manifest | `6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a` |
| Validation split manifest | `02e5c67c058f95bcca46c82f3c1fc81086f61dcec62ce25049843f09d930a5ba` |
| Inception weights | `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2` |
| Selfless-Flow EMA checkpoint | `81f86d1805d732f8c8e377a08cef6a6aad285eb533677405d4867bda90a86203` |
| Token-only MLP EMA checkpoint | `2f7af8c14b8f68a78eecae4312366c8660c9a13a4a71e94de8602e88438cd765` |
| Token-only MLP ratio 4.5 EMA checkpoint | `b678679c868be05d253bc11c21ba1b4e9304f79d3937ed94b85ed2bbe7f3499d` |
| Token-only MLP width 1936 EMA checkpoint | `bbad20da9c6dad7de27e489e17e224ea7203946092821e669198aa945739e1d4` |
| Selfless-Flow non-EMA checkpoint | `1af7302e4498a8bf4b50c8bd0d8fe3b008487ab2b82f1504eb34b9ac21b2dab1` |
| Qwen-Show-O checkpoint | `2eaf3c5958c36be4f2554ce88f67082cc6e40d67924df945c8b35a3efdec1806` |

Architecture-specific inference settings:

| Architecture | Model precision | Sampler |
| --- | --- | --- |
| Selfless-Flow | BF16 forward; FP32 VAE/numerical components | 100-step Heun, constant CFG, `spatial_halton`, `parallel_rate=1` |
| Qwen-Show-O | BF16 model; FP32 MAGVIT decode | 12-step MaskGIT, cosine remasking, temperature 1.0, official cumulative ranking-temperature update |

## Selected default commands

The single-point launchers now use the minimum-FID settings and distinct output
directories, so they do not overwrite historical pre-sweep outputs.

```bash
# EMA, CFG 3.5, BF16, parallel_rate=1
bash script/ablation/evaluate_imagenet_flow_100c.sh

# Token-only MLP architecture ablation; CFG is hard-fixed to 3.5
bash script/ablation/evaluate_imagenet_flow_token_mlp_100c.sh

# Token-only MLP, width 1280 / ratio 4.5; CFG is hard-fixed to 3.5
bash script/ablation/evaluate_imagenet_flow_token_mlp_param_matched_100c.sh

# Token-only MLP, width 1936 / ratio 1.0; CFG is hard-fixed to 3.5
bash script/ablation/evaluate_imagenet_flow_token_mlp_width1936_100c.sh

# Show-O s=11.75, equivalent to common w=12.75
bash script/ablation/evaluate_qwen_showo_vq_100c.sh
```

Environment overrides remain supported. For example:

```bash
CFG=5.0 OUTPUT_DIR=output/flow_cfg5 \
  bash script/ablation/evaluate_imagenet_flow_100c.sh

GUIDANCE_SCALE=10.0 OUTPUT_DIR=output/showo_w11 \
  bash script/ablation/evaluate_qwen_showo_vq_100c.sh
```

## Selfless-Flow EMA CFG sweep

All points below use `hf_model-final-ema`, BF16, 100-step Heun,
`spatial_halton`, and `parallel_rate=1`.

| CFG | FID ↓ | IS ↑ |
| ---: | ---: | ---: |
| 1.5 | 56.6840 | 34.6040 ± 1.3549 |
| 2.0 | 35.8983 | 47.2082 ± 0.8567 |
| 2.5 | 28.3411 | 54.2956 ± 1.2368 |
| 3.0 | 26.6047 | 57.7273 ± 0.8346 |
| **3.5** | **26.0110** | 59.5362 ± 1.1316 |
| 4.0 | 26.5438 | 60.2729 ± 0.7316 |
| 4.5 | 27.3226 | 60.7940 ± 1.0993 |
| 5.0 | 28.2501 | **61.6962 ± 1.0704** |

CFG 3.5 is an interior FID optimum on the tested grid. IS continues to rise
through CFG 5.0, with a corresponding FID regression.

## Flow-head architecture ablation

Architecture ablations reuse the baseline-selected inference protocol without
per-architecture tuning: CFG 3.5, BF16 model forward, FP32 VAE and flow
integration, 100-step Heun, `spatial_halton`, `parallel_rate=1`, seed 42, and
10,000 samples. This keeps the measured delta attributable to training-time
architecture choices rather than inference hyperparameter selection.

| Flow head | Head parameters | Cross-token mixing in head | FID ↓ | IS ↑ |
| --- | ---: | --- | ---: | ---: |
| Contextual baseline | 164.073M | clean-latent cross-attention | **26.0110** | **59.5362 ± 1.1316** |
| Token-only MLP, ratio 1.0 | 72.210M | none | 27.9774 | 56.9485 ± 1.5827 |
| Token-only MLP, ratio 4.5 | 163.996M | none | **26.4404** | **58.3860 ± 1.1099** |
| Token-only MLP, width 1936, ratio 1.0 | 163.828M | none | 26.9315 | 57.3898 ± 1.4622 |

The token-only head raises FID by 1.9664 (+7.56%) and lowers IS by 2.5877
(-4.35%). It is strictly pointwise: each prediction uses only its own noisy
latent token, timestep, and same-position backbone condition. Its checkpoint
contains no attention, QKV, positional-mixing, or clean-latent-context weights.

Both parameter-matched variants recover much of the loss from the 72.210M
head. The ratio-4.5 head is only 0.047% smaller than the contextual baseline;
it trails that baseline by 0.4294 FID (+1.65%) and 1.1502 IS (-1.93%), while
improving over the small token-only head by 1.5370 FID and 1.4375 IS. At nearly
the same parameter count, ratio 4.5 also beats width 1936 by 0.4911 FID and
0.9961 IS. Capacity is therefore better spent on the per-block expansion while
keeping the residual stream aligned to the 1280-wide backbone than on widening
the residual stream with ratio 1.0.

The formerly proposed “token-only ratio-4.5 + fixed query position” cell was
not run and is now closed rather than left as pending work. Subsequent
backbone experiments selected `E2-Q1`, and the dedicated contextual-head
position screen below found no position-only change that beat its baseline.
The remaining architecture question is therefore not whether to rescue a
pointwise MLP with one more position feature, but whether clean content should
be a dynamically updated stream inside the flow tower.

### Retained dynamic dual-stream flow-head baselines

The completed static-position and dynamic dual-stream screens have been
archived. New training uses the shared-attention/shared-MLP dynamic content
architecture `DF1` and exposes only two complete position contracts:

| Baseline | Position contract | FID ↓ | IS ↑ |
| --- | --- | ---: | ---: |
| **DF1-FH0** | additive query/content; no flow-head RoPE | **23.5699** | **64.7787** |
| **DF1-FH4** | no additive position; row/column 2D RoPE | **23.0230** | **64.6608** |

`DF1-FH0` is the default and `DF1-FH4` is the pure-RoPE alternative. All other
architecture/position cells are historical evidence rather than runtime
options.

The active contract is documented in
[`SELFLESS_FLOW_HEAD_BASELINE.md`](SELFLESS_FLOW_HEAD_BASELINE.md). Full
matrices and artifacts are preserved in
[`archive/SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL_HISTORICAL.md`](archive/SELFLESS_FLOW_DUAL_STREAM_FLOW_HEAD_ABLATION_PROPOSAL_HISTORICAL.md)
and `output/flow_head_ablation/relocation_manifest.json`.

### EMA versus non-EMA diagnostic

| Non-EMA CFG | FID ↓ | IS ↑ |
| ---: | ---: | ---: |
| 3.0 | 26.4090 | 57.6371 ± 1.2709 |
| **3.5** | **26.0782** | 60.0868 ± 1.4928 |
| 4.0 | 26.3739 | **60.7588 ± 0.9135** |

At CFG 3.5, EMA and non-EMA FID differ by only 0.0671 (0.258%). Treat them as
nominally tied under this single-seed 10K protocol. The non-EMA local sweep does
not improve on the EMA FID optimum.

## Qwen-Show-O CFG sweep

The sweep starts at common `w=1.0`, expands until both metric optima leave the
upper boundary, and adds 0.25 local points around the best regions.

| common w | Show-O s | FID ↓ | IS ↑ |
| ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 72.0998 | 22.3481 ± 0.9321 |
| 1.5 | 0.5 | 51.8948 | 33.8734 ± 1.4104 |
| 2.0 | 1.0 | 42.9546 | 44.9015 ± 1.0594 |
| 2.5 | 1.5 | 38.9403 | 51.9718 ± 0.8579 |
| 3.0 | 2.0 | 37.4271 | 57.1387 ± 0.6655 |
| 3.5 | 2.5 | 37.4617 | 59.6938 ± 1.2212 |
| 4.0 | 3.0 | 37.3406 | 62.0572 ± 0.9336 |
| 4.5 | 3.5 | 36.2308 | 63.2294 ± 0.9917 |
| 5.0 | 4.0 | 35.9663 | 64.3196 ± 1.2999 |
| 5.5 | 4.5 | 35.4717 | 65.2366 ± 0.5307 |
| 6.0 | 5.0 | 35.2622 | 65.9055 ± 1.0387 |
| 6.5 | 5.5 | 34.7138 | 66.3339 ± 0.6461 |
| 7.0 | 6.0 | 34.3908 | 66.9081 ± 0.9606 |
| 7.5 | 6.5 | 33.9322 | 67.1510 ± 1.1029 |
| 8.0 | 7.0 | 33.4324 | 67.0909 ± 0.9371 |
| 8.5 | 7.5 | 33.4562 | 67.3311 ± 0.7517 |
| 9.0 | 8.0 | 32.8049 | 67.1541 ± 0.7321 |
| 9.5 | 8.5 | 32.9703 | 67.2895 ± 1.0955 |
| 10.0 | 9.0 | 32.5182 | 67.6576 ± 0.7166 |
| 10.5 | 9.5 | 32.2986 | 67.4340 ± 0.6051 |
| 10.75 | 9.75 | 32.2112 | 67.2032 ± 0.6727 |
| **11.0** | **10.0** | 32.0040 | **67.7769 ± 0.7228** |
| 11.25 | 10.25 | 31.6710 | 67.0864 ± 1.0251 |
| 11.5 | 10.5 | 31.4362 | 67.3420 ± 1.0949 |
| 12.0 | 11.0 | 31.4913 | 67.0385 ± 0.6392 |
| 12.25 | 11.25 | 31.4416 | 67.1820 ± 0.5981 |
| 12.5 | 11.5 | 31.1535 | 67.3557 ± 0.9007 |
| **12.75** | **11.75** | **31.1314** | 66.7629 ± 0.5617 |
| 13.0 | 12.0 | 31.2012 | 67.1151 ± 1.1250 |

The FID-selected point is bracketed by worse values at `w=12.5` and `w=13.0`.
The IS-selected point is bracketed by lower IS at `w=10.75` and `w=11.25`.

## Result artifacts

Authoritative machine-readable summaries:

```text
output/selfless-flow-ablation-imagenet100-80ep/
  fid_is_cfg_sweep/summary.json
  fid_is_cfg_sweep/summary.csv
  fid_is_nonema_at_ema_selected_cfg/summary.json
  fid_is_nonema_cfg_local_bf16/cfg_3p0/metrics.json
  fid_is_nonema_cfg_local_bf16/cfg_4p0/metrics.json

output/selfless-flow-token-mlp-ablation-imagenet100-80ep/
  hf_model-final-ema/model.safetensors
  fid_is_selected_cfg3p5_ema/metrics.json

output/selfless-flow-token-mlp-param-matched-ablation-imagenet100-80ep/
  hf_model-final-ema/model.safetensors
  fid_is_selected_cfg3p5_ema/metrics.json

output/selfless-flow-token-mlp-width1936-ablation-imagenet100-80ep/
  hf_model-final-ema/model.safetensors
  fid_is_selected_cfg3p5_ema/metrics.json

output/qwen-showo-vq-ablation-imagenet100-80ep/
  fid_is_cfg_sweep_official/summary.json
  fid_is_cfg_sweep_official/summary.csv
```

The matched Flow qualitative sheet is:

```text
output/selfless-flow-ablation-imagenet100-80ep/
  fid_is_cfg_sweep/qualitative_cfg_comparison.png
```

## Exclusions and provenance

- The old Show-O `fid_is/metrics.json` used a superseded temperature schedule
  and lacks the current guidance-formula fields. It is not a formal result.
- FP32 model-forward diagnostics were stopped by design and are not reported.
  Production evaluation remains BF16.
- An exploratory token-only CFG sweep was stopped after the architecture
  protocol was clarified. Its CFG 2.0/2.5 diagnostics and incomplete work
  directories are excluded from architecture selection; only the independently
  rerun, baseline-fixed CFG 3.5 result above is formal.
- Each completed sweep root has an immutable `.sweep_contract.json` binding its
  checkpoint, config, evaluator, launcher, validator, real stats, tokenizer/VAE,
  and source hashes. Repository refactors do not invalidate the recorded result,
  but they intentionally prevent appending new points to an old root. Use a new
  `SWEEP_ROOT` for future evaluations.
- The full-ImageNet Selfless-Flow training entry remains available under
  `script/selfless/`, but it is not part of this 100C comparison and is not the
  repository default.
