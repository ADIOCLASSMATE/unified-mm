# Unified-MM: ImageNet-100 Architecture Ablations

This repository currently compares two Qwen3-0.6B image-generation
architectures under one controlled ImageNet-100 training and evaluation
protocol:

- **Selfless-Flow** uses a two-stream, same-position Qwen backbone and a
  contextual rectified-flow head over continuous KL16 image latents.
- **Qwen-Show-O** uses the same Qwen backbone scale with Show-O-style masked
  prediction over official MAGVITv2 image codes.

Both models train for 35,920 optimizer steps (80 epochs) on the same balanced
115K-image training split and are evaluated on the same 10K original-ImageNet
validation distribution. The authoritative protocol and all reported numbers
are in [docs/IMAGENET100_ABLATION.md](docs/IMAGENET100_ABLATION.md).

## Selected evaluation defaults

The single-point launchers default to the lowest-FID settings found by the
completed sweeps:

| Architecture | Checkpoint | Selected CFG | Other fixed settings |
| --- | --- | --- | --- |
| Selfless-Flow | `hf_model-final-ema` | `CFG=3.5` | BF16 model, 100-step Heun, `spatial_halton`, `PARALLEL_RATE=1` |
| Qwen-Show-O | `hf_model-final` | Show-O `s=11.75` (common `w=12.75`) | 12 MaskGIT steps, temperature 1.0 |

Show-O uses `(1+s)*conditional - s*unconditional`; therefore its command-line
`guidance_scale=s` is one lower than common CFG weight `w`.

## Current pipeline

Build the balanced subset from the existing full KL16 cache:

```bash
bash script/ablation/build_imagenet_100c_balanced_cache.sh
```

Prepare the official Show-O MAGVITv2 tokens and cache the shared original-image
FID distribution. Jobs that read raw ImageNet must explicitly attach dataset
`imagenet:v1`.

```bash
bash script/ablation/prepare_qwen_showo_vq_100c.sh
bash script/ablation/cache_imagenet100_original_fid_stats.sh
```

Train both 80-epoch ablations:

```bash
bash script/ablation/pretraining_imagenet_flow_100c_80ep.sh
bash script/ablation/pretraining_qwen_showo_vq_100c_80ep.sh
```

Evaluate the selected defaults on all 10K validation prompts:

```bash
bash script/ablation/evaluate_imagenet_flow_100c.sh
bash script/ablation/evaluate_qwen_showo_vq_100c.sh
```

The CFG sweep launchers remain available for explicit searches:

```bash
CFG_VALUES="1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0" \
  bash script/ablation/evaluate_imagenet_flow_cfg_sweep_100c.sh

GUIDANCE_SCALES="0.0 0.5 1.0" \
  bash script/ablation/evaluate_qwen_showo_vq_cfg_sweep_100c.sh
```

Use a new `SWEEP_ROOT` after changing any contract-bound source, config, or
protocol. Existing sweep roots are immutable experiment snapshots.

## Repository map

- `configs/ablation/`: the two active 100C training/evaluation configs.
- `models/modeling_model/modeling_selfless_flow.py`: Selfless-Flow backbone,
  flow objective integration, and sampler.
- `models/modeling_model/modeling_qwen_showo.py`: Qwen-Show-O model and official
  iterative masked sampler.
- `pretrain/`: the two active training loops plus retained text/full-dataset
  utilities.
- `script/ablation/`: data preparation, training, selected evaluation, and CFG
  sweep launchers.
- `scripts/`: evaluators, strict validators, immutable sweep contracts, and
  deterministic summary builders.
- `tests/`: architecture, dataset, metric-protocol, and default-regression
  tests.

## Preserved non-default paths

The full-ImageNet Selfless-Flow training path is intentionally retained, but it
is no longer the repository default:

```bash
bash script/selfless/encode_imagenet_full_kl16_vae.sh
bash script/selfless/pretraining_imagenet_flow_full_from_qwen3base.sh
```

The text-only selfless adaptation path is also retained:

```bash
bash script/selfless/pretraining_text_selfless_2048.sh
```

## Validation

```bash
PYTHONPATH=. .venv/bin/pytest -q tests
ruff check utils/dataset_utils.py tests/test_ablation_eval_defaults.py
```

For platform paths, resource policy, and the mandatory official ImageNet mount,
see [INSPIRE.md](INSPIRE.md).
