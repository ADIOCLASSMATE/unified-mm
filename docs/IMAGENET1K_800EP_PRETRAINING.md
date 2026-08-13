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
fixed-Inception FID/IS evaluation. Its 16-sample FID uses the evaluator's
explicit `--allow_nonofficial_fid` diagnostic opt-in and is only a pipeline
check; comparable formal results still require 50,000 fake samples and
`--require_official_protocol`. The latest successful smoke conclusion is kept
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

After training, formal FID/IS uses 50,000 generated samples, deterministic
canonical noise pairing, CFG 3.5, 100-step Heun, and the frozen official-val
moments:

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
  --output_dir output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-fid-is \
  --device npu --model_dtype bf16 \
  --samples 50000 --batch_size 4096 --vae_decode_batch_size 16 \
  --sampling_steps 100 --temperature 1.0 --cfg 3.5 \
  --cfg_schedule constant --flow_solver heun \
  --parallel_rate 1 --strategies spatial_halton \
  --inception_weights_path public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth \
  --real_stats_path public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt \
  --skip_target_decode --require_official_protocol --canonical_pairing \
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
