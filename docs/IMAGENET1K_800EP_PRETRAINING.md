# ImageNet-1K 800-epoch formal pretraining

## Fixed training contract

The formal run keeps the hyperparameters selected by the complete
ImageNet-100 sweep and changes only the controls that depend on dataset size or
training duration.

| Item | Formal value |
| --- | ---: |
| Ascend 910B NPUs | 64 (`4 x 16`) |
| Per-rank batch | 16 |
| Gradient accumulation | 1 |
| Global batch | 1024 |
| Backbone / Special-token LR | `30e-5` |
| Flow-head / Projector LR | `4e-5` |
| ImageNet train rows | 1,281,167 |
| Samples used per epoch | 1,281,024 |
| Randomly omitted rows per epoch | 143 |
| Optimizer steps per epoch | 1,251 |
| Epochs | 800 |
| Total optimizer steps | 1,000,800 |

The training population is the complete ImageNet-1K train set. The sample
budget is the largest multiple of 1024 below its size, so every epoch ends on a
complete optimizer step. The sampler is reshuffled deterministically each
epoch; the 143 omitted rows are not a permanent holdout.

The canonical files are:

- config:
  `configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml`;
- launcher:
  `script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep.sh`;
- preflight:
  `scripts/validate_ascend_imagenet1k_pretraining.py`.

## Duration-dependent adjustments

### EMA

Use FP32 rank-sharded EMA from step zero with decay `0.9999`.

The ImageNet-100 run used decay `0.999`, whose half-life was about 693 steps,
or 6.19 of its 112-step epochs. Keeping `0.999` on ImageNet-1K would reduce the
half-life to only 0.55 epoch. At `0.9999`, the half-life is about 6,931 steps,
or 5.54 ImageNet-1K epochs. This closely preserves the sweep's averaging scale
without introducing an irregular decimal. EMA stays sharded, so each rank owns
only its fraction of the FP32 shadow weights.

### WSD schedule

The WSD phases are exact epoch multiples:

- warmup: 5 epochs / 6,255 steps;
- stable: 595 epochs / 744,345 steps;
- decay: 200 epochs / 250,200 steps;
- final LR scale: 0.1.

Five warmup epochs preserve the successful short-run warmup scale. The final
quarter of the 800-epoch run is the WSD decay phase; copying the old 2,240-step
decay would make the formal decay less than two epochs.

### Recovery, validation, and logging

- save a complete resumable checkpoint every 10 epochs (12,510 steps);
- retain the latest 3 ordinary checkpoints, including their sharded EMA state;
- permanently retain every 100-epoch checkpoint (125,100 steps); these
  milestones do not participate in or consume slots from the rolling limit;
- export and permanently retain a complete BF16 EMA HF evaluation model every
  10 epochs as `hf_model-<step>-ema-eval`;
- do not save intermediate image-flow adapters; export only
  `image_flow_adapter-final.pt` at the end of training;
- run validation loss and validation-image probes every 10 epochs;
- log scalar training metrics every 50 steps;
- read and validate the DeepSpeed global gradient norm once per epoch;
- export both final raw and final EMA Hugging Face checkpoints.

The 50-per-class validation view is a deterministic diagnostic view whose rows
also remain in the training population. Formal FID does not use it as the real
distribution: it uses the official 50,000-image ImageNet validation set.

## Required data artifacts

Formal launch is intentionally blocked until these immutable artifacts pass
the preflight:

1. Full train posterior cache:
   `public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_train_fp16.pt`.
   Its posterior tensor is about 19.55 GiB before serialization overhead.
2. Local torch-fidelity Inception weights:
   `public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth`.
3. Locally computed moments over the official ImageNet validation set:
   `public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt`.

The canonical train manifest already contains all 1,281,167 rows and 1,000
classes. Any preprocessing Job that reads raw ImageNet must attach the official
`imagenet:v1` dataset at `/inspire/dataset/imagenet/v1`.

On the fixed 16-card Ascend development machine, build and validate the full
cache with one launcher. It uses 16 disjoint shards, batch 256 per NPU, four
data-loader workers per rank, and torch_npu's precompiled operator mode:

```bash
bash script/selfless/prepare_imagenet1k_cache_ascend16.sh
```

Compute the 50K validation moments locally from the official class directories,
using the fixed Inception weights and exactly the same resize, center-crop,
uint8 conversion, and torch-fidelity feature path as formal evaluation.
Root-level duplicate validation files are ignored:

```bash
bash script/selfless/prepare_imagenet1k_fid_stats_ascend16.sh
```

## Preflight, launch, resume, and evaluation

The CPU-only config contract can be checked before the large assets exist:

```bash
python scripts/validate_ascend_imagenet1k_pretraining.py --config_only
```

After preparing the assets, perform the one-time deep cache scan before
requesting 64 NPUs:

```bash
python scripts/validate_ascend_imagenet1k_pretraining.py --deep_cache_scan
```

The end-to-end 16-NPU development-machine smoke is:

```bash
bash script/selfless/smoke_imagenet1k_train_val_eval_ascend16.sh
```

It executes one real optimizer step, checkpoint and FP32 sharded-EMA save,
validation loss and image decoding, raw/EMA HF export, and an independent
fixed-Inception IS pipeline check. Smoke runs do not publish FID: comparable
FID results require exactly 50,000 fake samples, the frozen original
ImageNet-val moments, and `--require_formal_protocol`. The latest successful
smoke conclusion is kept
in
`public/datasets/imagenet_full/preparation/train_val_eval_smoke_report.json`;
large smoke checkpoints are discarded after verification.

The launcher runs the complete asset and per-node NPU preflight itself. Submit
one Inspire Job whose command is exactly this one launcher:

```bash
bash script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep.sh
```

To resume, point the same launcher at one complete retained checkpoint:

```bash
RESUME_FROM=output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/checkpoint-6255 \
  bash script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep.sh
```

The current reference recipe uses 50,000 generated samples, deterministic
canonical noise pairing, CFG 3.5, 10-step Heun, and the project ImageNet-val
moments. Sampling settings may change in later experiments and must be reported
with the result. This is a same-configuration project metric, not an ADM/DiT
leaderboard-comparable FID:

The permanently retained 10-epoch EMA evaluation exports are complete model
weights, not partial flow adapters. Evaluate one by passing its directory, for
example `hf_model-12510-ema-eval`, through `--model_path_override`. The rolling
DeepSpeed checkpoints remain the source for exact training recovery, while the
smaller BF16 EMA exports are the source for historical FID/IS curves.

```bash
torchrun --standalone --nproc_per_node=16 \
  scripts/evaluate_single_stream_fid_is.py \
  --config configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml \
  --model_path_override output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep/hf_model-final-ema \
  --output_dir output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-heun10-fid-is \
  --device npu --model_dtype bf16 \
  --samples 50000 --batch_size 4096 --vae_decode_batch_size 16 \
  --sampling_steps 10 --temperature 1.0 --cfg 3.5 \
  --cfg_schedule constant --flow_solver heun \
  --parallel_rate 1 --strategies spatial_halton \
  --inception_weights_path public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth \
  --real_stats_path public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt \
  --require_formal_protocol --canonical_pairing \
  --resume_progress --resume_checkpoint_interval_batches 1
```

After the initial status check, wait through one blocking CLI process. The
30-day timeout only bounds the local wait process and does not stop the Job:

```bash
inspire --json job wait <job-name> \
  --workspace 昇腾卡公共空间 \
  --interval 60 \
  --timeout 2592000
```

Do not repeatedly invoke `job wait`, status, events, logs, or utilization while
that blocking process remains active.

## Final EMA evaluation result

The final EMA HF export completed the project-formal ImageNet-val evaluation on
2026-08-19. All 50,000 requested samples were evaluated with deterministic
canonical pairing, CFG 3.5, 100-step Heun, `spatial_halton`, and the frozen
ImageNet-val moments. These historical values are valid only for comparisons
using exactly this project protocol; they are not ADM/DiT leaderboard values.

| Metric | Result |
| --- | ---: |
| FID | `18.996944032440638` |
| Inception Score | `450.06854248046875 ± 4.259687366514454` |
| Generation throughput | `4.228181407150334 samples/s` |

The retained machine-readable result is
`output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-fid-is/metrics.json`
(SHA256 `d6f066bffad8a4e3032ccc3aac4b9445e9589e21691c3519bdf5e2e722d506f8`).

### Position-wise flow-head ablation

The position-wise-head final EMA completed the same project-formal
evaluation on 2026-08-22. It used the same 50,000 canonically paired samples,
CFG 3.5, 100-step Heun solver, `spatial_halton` strategy, and frozen project
ImageNet-val moments as the baseline above.

| Metric | Position-wise | Baseline | Change |
| --- | ---: | ---: | ---: |
| FID | `18.2875253165069` | `18.996944032440638` | `-0.7094187159337366` (`-3.734%`) |
| Inception Score | `438.955712890625 ± 3.8873937344382083` | `450.06854248046875 ± 4.259687366514454` | `-11.112829589843727` (`-2.469%`) |
| Generation throughput | `18.023885168137856 samples/s` | `4.228181407150334 samples/s` | `4.263x` |

Thus the position-wise head improves FID and substantially reduces generation
cost, while its Inception Score is lower. The retained machine-readable result
is
`output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-positionwise-head-fid-is/metrics.json`
(SHA256 `a5850bc1072db7e5d5480ca02f6be33a849fb834682137e914bb1747fa405f83`).

### Sequential image-sigma ablation

The sequential-image-sigma final EMA completed its project-formal
evaluation on 2026-08-24. All 50,000 requested samples were evaluated with
canonical initial noise and sample pairing, CFG 3.5, a 100-step Heun solver,
the required `sequential` generation strategy, and the same frozen project
ImageNet-val moments.

| Metric | Seq-sigma (`sequential`) | Baseline (`spatial_halton`) | Descriptive change |
| --- | ---: | ---: | ---: |
| FID | `8.633703493770327` | `18.996944032440638` | `-10.363240538670311` (`-54.552%`) |
| Inception Score | `247.52949981689454 ± 3.36128814432888` | `450.06854248046875 ± 4.259687366514454` | `-202.5390426635742` (`-45.002%`) |
| Generation throughput | `4.224009451221147 samples/s` | `4.228181407150334 samples/s` | `0.999x` |

The canonical noise and ordered sample manifests match the baseline, but the
generation strategies do not. Consequently, these deltas describe the formal
end-to-end recipes and cannot isolate the effect of training sigma order: a
controlled attribution would also evaluate the baseline with `sequential`.
The retained machine-readable seq-sigma result is
`output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-seq-sigma-fid-is/metrics.json`
(SHA256 `443a67f83ee365065eac074de45eadfb9ac00ff4ef8ed1cd2c4dcfec397e6458`).

### Retired Dynamic-XT recipe

The ImageNet-only Dynamic-XT successor recipe described by older revisions of
this document is retired. The current Dynamic-XT implementation is unified
ablation D on baseline B and is specified by
`configs/protocols/unified_ablation_100b_ascend64.yaml`. It preserves
`image_flow_batch_mul: 4` and the B attention contract. Do not use an old
ImageNet-only Dynamic-XT checkpoint as its initialization.
