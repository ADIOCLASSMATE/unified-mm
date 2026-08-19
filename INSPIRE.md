# Inspire execution notes

## Shared paths

- Repository: `/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048/code/unified-mm`
- Shared user root: `/inspire/sj-ssd3/project/high-dimensionaldata/wanjiaxin-253108030048`
- Full ImageNet latent cache: `public/datasets/imagenet_full`
- ImageNet-100 distilled captions: `public/datasets/imagenet_distilled_captions/imagenet100`
- ImageNet-1k distilled captions: `public/datasets/imagenet_distilled_captions/imagenet1k`

## Official dataset mount

Any Notebook or Job that reads raw ImageNet must explicitly attach:

```text
Dataset ID: imagenet
Version ID: v1
Validated platform path: rclone-worker-1/imagenet/v1
Container path: /inspire/dataset/imagenet/v1
```

Verify that Job details contain non-empty `dataset_info`; shared storage does
not implicitly mount the official dataset.

## Resources

- Use `dev-wjx` for single-GPU micro-batch/smoke checks when it is running.
- Formal training and FID/IS evaluation use 8×H100.
- Evaluation uses 384 samples per H100, exposed by the evaluator as global
  batch 3072 on eight ranks. The previous 512-per-H100 setting OOMed during
  flow-cache batching on 80GB H100s.
- GPU Job priority is 4.
- Default project: `随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架`.
- The main project permits at most 16 concurrent GPUs, so submit at most two
  8-GPU formal jobs together.
- Secondary project: `多模态大模型新架构评测探索与scaling-law`. It may be used
  after a live quota/availability check, with at most 32 concurrent GPUs assigned
  to H100 work in `分布式训练空间`. Do not keep duplicate runnable jobs in
  both projects.
- Image: `docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1`.

### Ascend 910B training

- This Ascend workflow is isolated from Hopper / H100 / H200 workflows. Never
  fall back to, operate, or clean up those resources when following the Ascend
  contract below.
- Project: `多模态大模型新架构评测探索与scaling-law`
  (`high-dimensionaldata`).
- Dedicated Workspace: `昇腾卡公共空间`; use it only for Ascend workloads.
- Compute Group: `910B资源` (`ASCEND 910B (64GB)`).
- Ascend training allocation ceiling: 256 concurrent GPUs. This is a total
  project limit, not the per-instance `gpu,cpu,mem` quota triple.
- Preferred full-node Job row: `16,128,1024`; use `16,64,1024` only after a
  Live quota check shows it is the better valid row. At 16 GPUs per instance,
  256 GPUs corresponds to at most 16 instances.
- Job priority: 6. Verify the platform-assigned priority after submission.
- Permanent development Notebook: `dev-wjx-ascend`, with 16 Ascend GPUs in
  `昇腾卡公共空间`. Any smoke test that requires a GPU must start and run on
  this Notebook; do not create an alternative development Notebook or run the
  GPU smoke in the local Agent environment. Stop it after the smoke when no
  immediate follow-up debugging needs it, but do not delete this permanent
  Notebook.
- Base image:
  `docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.3`
  (platform image name `dev-wjx-ascend:v-1.3`).
- Before every submission, check Live Job quota, image status, active project
  Jobs, availability, and whole-node capacity; always dry-run first.

### Final ImageNet-100 training hyperparameters

- Canonical conclusion:
  `docs/IMAGENET100_HYPERPARAMETER_CONCLUSION.md`.
- Global batch size: `1024` on 64 Ascend NPUs (`4 x 16`), with per-rank batch
  `16` and gradient accumulation `1`.
- Final coupled learning rates: Backbone/Special-token `30e-5` and
  Flow-head/Projector `4e-5`.
- The selected 80-epoch checkpoint achieved FID `24.057695924272537` and IS
  `71.04673767089844` under the canonical 10,000-sample, 100-step evaluation.
- Sweep manifests、旧 checkpoint、评测产物、ImageNet-100 可执行配置和一次性
  launch 资产均已删除；仓库只保留结论。

### Formal ImageNet-1K 800-epoch pretraining

- Contract: `docs/IMAGENET1K_800EP_PRETRAINING.md`.
- Config:
  `configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml`.
- One-Job/one-launcher entry:
  `script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep.sh`.
- Use 64 Ascend 910B NPUs (`4 x 16`), per-rank batch 16, GA 1, and global
  batch 1024. Do not increase the card count without a new LR/batch contract.
- Fixed coupled LRs are Backbone/Special-token `30e-5` and
  Flow-head/Projector `4e-5`.
- The exact run is 1,251 optimizer steps per epoch and 1,000,800 steps total.
  WSD is 5 warmup + 595 stable + 200 decay epochs.
- EMA is FP32 rank-sharded, starts at step zero, and uses decay `0.9999` so its
  half-life remains close to the epoch-scale averaging selected on ImageNet-100.
- The launcher supports exact recovery through `RESUME_FROM=<checkpoint-dir>`.
- Full ImageNet-1K KL16 posterior cache, local Inception weights, and official
  ImageNet-val 50K real moments must exist before the launcher preflight passes.
- The completed full cache is
  `public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_train_fp16.pt`
  (`[1281167, 256, 32]`, FP16, SHA256
  `3fdb1341e682962bb3f10ff3b794d67e4cc8a46490a2c621b95d60e0d4b1fb82`).
  Rebuild it on `dev-wjx-ascend` with the single launcher
  `script/selfless/prepare_imagenet1k_cache_ascend16.sh`.
- The fixed torch-fidelity Inception weights are
  `public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth`
  (SHA256
  `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`).
- The locally computed official-val moments are
  `public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt`
  (50,000 images, 1,000 classes, 2,048 features, SHA256
  `7eb801931347be917b34077c5ab94c4c7c6b9c42bbe40b442f39947d9bb133`).
  They use the same fixed Inception weights and preprocessing as evaluation;
  rebuild them with
  `script/selfless/prepare_imagenet1k_fid_stats_ascend16.sh`.
- The complete 16-NPU train/validation/evaluation smoke passed on
  `dev-wjx-ascend`. Run it with the single launcher
  `script/selfless/smoke_imagenet1k_train_val_eval_ascend16.sh`; the retained
  conclusion is
  `public/datasets/imagenet_full/preparation/train_val_eval_smoke_report.json`.
  Its 16-sample FID/IS values are pipeline diagnostics, not paper metrics.
- W&B remains enabled. `WANDB_MODE` defaults to `offline` and may be set to
  `online` for a platform environment with working W&B credentials/network.
- The final EMA HF model completed the canonical official ImageNet-1K
  evaluation on 2026-08-19: 50,000 samples, deterministic canonical pairing,
  CFG 3.5, 100-step Heun, and the frozen official-val moments. The result is
  FID `18.996944032440638` and IS `450.06854248046875 ± 4.259687366514454`.
  The retained result is
  `output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-fid-is/metrics.json`
  (SHA256 `d6f066bffad8a4e3032ccc3aac4b9445e9589e21691c3519bdf5e2e722d506f8`).

### ImageNet-1K sequential image-sigma ablation

- This run keeps the formal 64-NPU, global-batch-1024, 800-epoch contract
  unchanged while setting `dataset.params.image_sigma_order: sequential`.
- Image sigma follows serialized latent-token order with strict
  `sigma[kv] < sigma[q]`: the diagonal is hidden, and EOI is ordered before
  every image token so image queries can still see EOI.
- Config:
  `configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_seq_sigma.yaml`.
- Launcher:
  `script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep_seq_sigma.sh`.
- Training-time validation and offline evaluation both use the fixed
  `sequential` generation strategy; the model asserts the exact 1..256 order
  at runtime for a complete sequential generation.
- The 16-NPU one-step train/validation/16-image evaluation smoke passed on
  `dev-wjx-ascend`. The retained report is
  `public/datasets/imagenet_full/preparation/seq_sigma_train_val_eval_smoke_report_v2.json`;
  run the retained one-off smoke launcher with
  `script/selfless/smoke_imagenet1k_seq_sigma_train_val_eval_ascend16.sh`.

### ImageNet-1K architecture ablations

- Both architecture controls preserve the formal 64-NPU, global-batch-1024,
  800-epoch optimization/data contract. A config-parity test permits only the
  architecture identity, run/output names, and architecture-required
  accounting fields to differ from the baseline.
- The position-wise-head control keeps the random-sigma selfless two-stream
  Qwen backbone and replaces only the contextual flow head with a vectorized
  MAR/NextStep-style AdaLN MLP. The head has no cross-token attention, content
  cache, previous-latent input, or image-position input.
- Position-wise-head config and launcher:
  `configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_positionwise_head.yaml`
  and
  `script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep_positionwise_head.sh`.
- The Dynamic-XT control keeps the same strict selfless X0/XT backbone and
  contextual dual-stream flow head. It replaces only predicted-image XT
  queries with `image_token_embedder(x_t) + backbone_flow_time_embedder(t)`;
  Heun predictor/corrector evaluations recompute XT while reading fixed X0
  K/V without committing XT to the cache.
- Dynamic-XT config and launcher:
  `configs/selfless/imagenet1k_class_dynamic_xt_800ep.yaml` and
  `script/selfless/pretraining_imagenet_class_dynamic_xt_800ep.sh`.
- Its 16-NPU official 50K FID/IS launcher is
  `script/selfless/evaluate_imagenet1k_dynamic_xt_ema_ascend16.sh`; it uses the
  dedicated Dynamic-XT evaluator entry while retaining the static protocol.
- The position-wise-head retained smoke report is
  `public/datasets/imagenet_full/preparation/positionwise_head_smoke_report.json`.
  Dynamic-XT passed the tiny train/backward plus Heun-CFG-cache NPU smoke and
  the exact formal per-rank shape (`B=16`, `L=320`, 256 image tokens,
  latent dim 16, flow batch multiplier 4) forward/backward smoke on
  `dev-wjx-ascend`. Dynamic-only backbone activation rematerialization keeps
  that exact batch contract within device memory; flow-head checkpointing and
  the underlying NPU attention/operator paths remain unchanged.

## Waiting

After one initial configuration/status check, wait with one blocking process:

```bash
inspire --json job wait <job-name> \
  --workspace 昇腾卡公共空间 \
  --interval 60 \
  --timeout 2592000
```

This 30-day timeout only bounds the local blocking process; it does not stop
the Job. Do not repeatedly invoke `job wait` or poll status, events, logs, or
GPU utilization while the blocking process is active.

## ImageNet-1K local-Qwen caption farm

This subsection is a workload-specific exception to the 8-GPU/priority-4
training guidance above. Caption-farm Workers are always preemptible
`priority=1` Jobs with exactly one H100; the Controller runs on the stable
zero-GPU `test-dev` Notebook in `CPU资源空间`.

Canonical configuration and outputs:

```text
Config: configs/caption_farm/imagenet1k_qwen36_35b_a3b_fp8.json
Run: public/datasets/imagenet_distilled_captions/imagenet1k/local_qwen36_35b_a3b_fp8_v1_run
Model: Qwen/Qwen3.6-35B-A3B-FP8 (local snapshot only)
Published JSONL: public/datasets/imagenet_distilled_captions/imagenet1k/local_qwen36_35b_a3b_fp8_v1.jsonl
```

The queue key is `(image identity, model fingerprint, caption slot)`. Claims
are atomic shared-filesystem leases with heartbeats; expired leases are
reclaimed, and results become visible with a no-replace atomic link. A Worker
loads the complete local vLLM model and passes `/health` plus a real image
request before claiming work. The selected H100 tuning is recorded in
`worker_tuning.json` and currently uses request concurrency 16, max sequences
32, and claim batches of 32.

Inspire CLI 6.2.0 does not expose the Web UI's official-dataset field. The farm
therefore dry-runs every Job through the CLI, then uses the narrow
`caption_farm.inspire_submit` adapter to add exactly this audited payload:

```json
{"dataset_info":[{"dataset_id":"imagenet","version_id":"v1","path":"rclone-worker-1/imagenet/v1"}]}
```

After creation it reads Job status back and stops the Job immediately unless
the Job is LOW priority, has one H100, uses the fixed
`docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1` image, and has the
expected non-empty `dataset_info`.

Useful commands:

```bash
RUN_DIR=public/datasets/imagenet_distilled_captions/imagenet1k/local_qwen36_35b_a3b_fp8_v1_run
PYTHONPATH=. .venv/bin/python scripts/imagenet_qwen_caption_farm.py queue status --run-dir "$RUN_DIR"
PYTHONPATH=. .venv/bin/python scripts/imagenet_qwen_caption_farm.py controller status --run-dir "$RUN_DIR"
PYTHONPATH=. .venv/bin/python scripts/imagenet_qwen_caption_farm.py controller pause --run-dir "$RUN_DIR"
PYTHONPATH=. .venv/bin/python scripts/imagenet_qwen_caption_farm.py controller resume --run-dir "$RUN_DIR"
PYTHONPATH=. .venv/bin/python scripts/imagenet_qwen_caption_farm.py controller stop --run-dir "$RUN_DIR"
```

There must be one Controller only. It discovers the live project/group/quota
whitelist and targets at most 16 one-card LOW-priority Workers. Either project
may carry all 16 during a peer circuit break; when both are healthy, weighted
round-robin uses the configured 2:1 project weights. Submissions remain a burst
of one with at least 30 seconds between creates. It refills lost/preempted Jobs,
uses exponential backoff for API/quota rejection, and opens a circuit after
repeated rejection. It diagnoses each terminal
failure once and records `NEEDS_ATTENTION.json` for unrecoverable task failure,
missing official mount, a running Worker with no business progress, or a
Controller exception. Pure low-priority queueing is not considered a stalled
Worker.

Run the single Controller in the operator's foreground supervision call. This
one command is silent while healthy and returns a compact JSON payload only on
completion, explicit stop, or `NEEDS_ATTENTION`; do not split it into a detached
Controller plus a second wait process, and do not manually poll Jobs while it
is active:

```bash
PYTHONPATH=. .venv/bin/python scripts/imagenet_qwen_caption_farm.py controller supervise \
  --run-dir "$RUN_DIR"
```

Normal completion is automatic: all 3,843,501 caption slots must be COMPLETE,
with zero PENDING/LEASED/FAILED; all farm Jobs must drain; the audit must verify
the exact canonical key set, ImageNet mapping, and ImageNet-100 compatibility;
then the Controller atomically publishes 1,281,167 rows containing the original
caption plus three local-Qwen captions and writes `COMPLETED.json`. Do not treat
a merely running Controller or partially filled staging tree as completion.
