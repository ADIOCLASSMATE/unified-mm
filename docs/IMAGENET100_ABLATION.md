# ImageNet-100 Architecture Ablation: Protocol and Results

Status: final 10K evaluation, 2026-07-18.

This is the authoritative document for the balanced ImageNet-100 comparison
between Selfless-Flow and Qwen-Show-O. It replaces the former separate research,
dataset, and Show-O pipeline notes.

## Decision summary

The repository defaults are selected by minimum FID. Maximum-IS points are
reported separately because increasing guidance trades diversity for class
confidence.

| Architecture / checkpoint | Selection | CFG | FID ↓ | IS ↑ |
| --- | --- | ---: | ---: | ---: |
| **Selfless-Flow EMA** `hf_model-final-ema` | minimum FID, default | **3.5** | **26.0110** | 59.5362 ± 1.1316 |
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
- saved generated images and exact filename/count validation.

Key provenance hashes:

| Artifact | SHA256 |
| --- | --- |
| Real-image manifest | `6c6fc84e6ec9cb8b92421659d44abbb24d1cc34558213d9fb9db7e4ec44f1c3a` |
| Validation split manifest | `02e5c67c058f95bcca46c82f3c1fc81086f61dcec62ce25049843f09d930a5ba` |
| Inception weights | `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2` |
| Selfless-Flow EMA checkpoint | `81f86d1805d732f8c8e377a08cef6a6aad285eb533677405d4867bda90a86203` |
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
- Each completed sweep root has an immutable `.sweep_contract.json` binding its
  checkpoint, config, evaluator, launcher, validator, real stats, tokenizer/VAE,
  and source hashes. Repository refactors do not invalidate the recorded result,
  but they intentionally prevent appending new points to an old root. Use a new
  `SWEEP_ROOT` for future evaluations.
- The full-ImageNet Selfless-Flow training entry remains available under
  `script/selfless/`, but it is not part of this 100C comparison and is not the
  repository default.
