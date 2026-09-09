# Inspire execution notes

## Shared paths

- Repository: `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/unified-mm`
- Shared user root: `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048`
- Repository `public` link target: `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048`
- Full ImageNet latent cache: `public/datasets/imagenet_full`
- ImageNet-100 distilled captions: `public/datasets/imagenet_distilled_captions/imagenet100`
- ImageNet-1K synthetic caption/T2I dataset:
  `public/datasets/imagenet1k_synthetic_v1`
- Seekable joint-training text index:
  `public/datasets/imagenet1k_synthetic_v1/indexed/train/manifest.json`

## Repository-wide generation default

- All Selfless-Flow model variants and all training, validation, evaluation,
  generalization, smoke, and benchmark configurations default to 10-step Heun
  generation. Production YAMLs set both
  `model.image_flow_num_sampling_steps=10` and
  `evaluation.sampling_steps=10`; model-class fallbacks and CLI/launcher
  defaults are also 10.
- Explicit non-default step counts are experimental overrides and must use a
  distinct output directory. Historical 100-step reports and artifacts retain
  their original labels and metrics as completed-experiment evidence.

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

Current production training and project-formal evaluation use Ascend 910B.
Current formal models are A/B (formerly A_x0/B_x0), with C–F as ablations on B;
older A/B and retired branches are historical. Names and method definitions live in
[EXPERIMENTS.md](docs/EXPERIMENTS.md). The resource and project contracts below
also retain the placement needed to reproduce those historical experiments;
[historical H100 recipes](docs/archive/IMAGENET_EXECUTION_NOTES.md) are retained
only as provenance for earlier experiments.

### Ascend 910B training

- This Ascend workflow is isolated from Hopper / H100 / H200 workflows. Never
  fall back to, operate, or clean up those resources when following the Ascend
  contract below.
- Project: `多模态大模型新架构评测探索与scaling-law`
  (`high-dimensionaldata`).
- Project override for the unified ClimbMix + ImageNet work: every 1B LR-sweep
  Job and every formal 100B baseline-b or C-on-B Job must use
  `随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架`. Never submit those
  Jobs to `high-dimensionaldata`. The explicit exception is the D/E/F-on-B
  study: submit those three formal Jobs to
  `多模态大模型新架构评测探索与scaling-law` (`high-dimensionaldata`), with
  no duplicate D/E/F Jobs in the random-order project.
- Unified training sets `training.runtime_hashing_enabled: false`. Its data
  loading, checkpoint/resume checks, EMA layout checks, sweep selection, and
  formal continuation must use readable fields and must not calculate hashes.
  Unified launchers also set `WANDB_MODE=disabled`; metrics are retained in
  local logs and readable JSON instead of initializing a third-party tracker.
- The frozen 1B grid is
  `configs/protocols/unified_baseline_lr_sweep_1b_ascend64.yaml`; launch one
  arm with
  `script/selfless/pretraining_unified_baseline_lr_sweep_arm_ascend64.sh` and
  select only after all nine complete with
  `scripts/select_unified_lr_sweep.py`.
- Formal A/B and B-based C–F share the
  [100B training contract](configs/protocols/unified_ablation_100b_ascend64.yaml)
  and `script/selfless/pretraining_unified_ablation_100b_ascend64.sh` launcher.
  That contract owns architecture differences, optimizer settings and
  architecture-specific startup/checkpointing controls. Every fresh arm starts
  from Qwen3-0.6B-Base at optimizer step zero; do not resume LR-sweep weights or
  substitute an older A/B/C/D checkpoint. Historical run paths remain stable.
- Dedicated Workspace: `昇腾卡公共空间`; use it only for Ascend workloads.
- Compute Group: `910B资源` (`ASCEND 910B (64GB)`).
- The `high-dimensionaldata` Ascend training allocation ceiling is 256
  concurrent GPUs. This is a total project limit, not the per-instance
  `gpu,cpu,mem` quota triple; do not assume the same ceiling for the unified
  project override without a Live platform check.
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
- The permanent development Notebook currently uses image
  `dev-wjx-ascend:v-1.4` and belongs to `公共科研项目`. This Notebook-specific
  placement does not change the training Job project/image contracts above.
- Notebook access must reach both `qz.sii.edu.cn` and
  `notebook-inspire-sj.sii.edu.cn` directly. If an Agent shell exports a local
  proxy, clear it for the Inspire process and its browser/tunnel children:
  `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy inspire ...`.
  This is scoped to the command and does not change global proxy settings.
- Base image:
  `docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.3`
  (platform image name `dev-wjx-ascend:v-1.3`).
- Before every submission, check Live Job quota, image status, active project
  Jobs, availability, and whole-node capacity; always dry-run first.

### B flow-head depth scaling

- The two depth-scaling arms use
  `configs/selfless/unified_b_x0_flow_depth{16,30}_100b_ascend64.yaml` and
  `script/selfless/pretraining_unified_flow_head_scaling_ascend64.sh --depth 16|30`.
  Both are fresh, complete 100B runs with 64 Ascend NPUs in
  `随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架`.
- Width stays 1280; head depth 16/30 gives 321,543,696/597,117,456 parameters.
  Keep the B learning rates, source schedule, local batches, 4-step gradient
  accumulation, and four RF samples per image. Initialization is Qwen3-0.6B-Base
  with a random flow head, never the trained baseline checkpoint.
- Both arms require `image_flow_grad_checkpointing=true` and
  `image_flow_share_content=true`. Following D's infrastructure pattern, only
  the T2I flow branch checkpoints its blocks. X0 content and per-layer K/V
  remain batch B; query-only RF expansion has logical batch 4B. Pure-text and
  I2T minibatches keep their existing backbone path. This is activation
  checkpointing; gradient accumulation remains four steps.
- Run the 16-NPU smoke with the formal per-rank shapes before submission.
  Validate NPU BF16 loss/backward parity, finite training, peak memory, final
  checkpoint/EMA exports, and cached generation after reloading each EMA.
  Freeze the complete source for formal jobs and keep reports under
  `output/experiments/unified-b-x0-flow-head-scaling/`.

### Unified 0.6B ImageNet-native full evaluation

- The reusable 16-NPU protocol is
  `configs/protocols/unified_full_evaluation_ascend16.yaml`. Text benchmark
  assets live at `public/benchmarks/selfless_text_v1`; its readable manifest
  records source URLs, file sizes, and row counts only. Runtime hashing and
  contamination/decontamination hashing remain disabled.
- The canonical complete-suite entry is
  `script/selfless/evaluate_unified_native_full_checkpoint_ascend16.sh`. It
  combines project-formal ImageNet-val 50K T2I FID/IS, the eight-task text
  suite, complete ImageNet-val 50K generative zero-shot classification,
  MSCOCO Karpathy 5K test retrieval, Flickr30K Karpathy test retrieval,
  SugarCrepe, and ARO. All image-text scores use fixed-alpha language-prior
  correction. MMBench and SEED are internal ablation-trend diagnostics only;
  custom ImageNet 1K/5K retrieval, ReaL, and the old generated-caption CLIP
  score have been removed. The text-only entry is
  `script/selfless/evaluate_selfless_text_ascend16.sh`.
- The completed step-95415 artifact previously named baseline b predates the
  flow-head content diagonal. It is retained only as the no-diagonal control at
  `output/unified-b-flow-head-no-diagonal-0p6b-100b-imagenet-split-s42-r1/hf_model-final-ema`.
  The flow-diagonal-only B path
  `output/unified-b-0p6b-100b-imagenet-split-s42-r1` retains the historical
  shared-query AdaLN condition and remains an intermediate control. The new
  canonical B path
  `output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1` is reserved for
  a fresh retraining run and must not reuse either historical checkpoint. New B
  explicitly sets `flow_head_attention_contract: xlnet_content_diagonal`:
  its flow query remains strict (`sigma_kv < sigma_q`) while the content stream
  consumes the same latent token as the backbone, uses `sigma_kv <= sigma_q`,
  and conditions its content AdaLN with backbone X0 hidden while query AdaLN
  stays conditioned by XT hidden. Legacy checkpoints with no attention field
  are interpreted as `selfless_strict`; checkpoints with no
  `flow_condition_contract` retain the shared-query condition. Rank-sharded
  checkpoint directories remain legacy inputs only for historical trends.
- The reusable T2I-only entry is
  `script/selfless/evaluate_unified_t2i_fid_is_ascend16.sh`. Project-formal IS uses
  ten deterministic `stratified_by_synset` splits. Each split must contain
  all 1,000 ImageNet classes with exactly five samples per class; the formal
  gate also requires the source dataset itself to declare `split=val`. This
  val-reference FID is for same-protocol comparisons and is not ADM/DiT
  leaderboard-comparable.
- Text scoring follows this model's same-position Selfless query-stream
  likelihood contract rather than a stock next-token lm-eval adapter. MMLU is
  5-shot; the evaluation-only maximum context is 4096 and does not change the
  2048-token training contract.
- Text protocol v3 (full evaluation configuration v10) normalizes choice
  likelihoods by original-choice Unicode character counts, not token counts.
  WinoGrande scores only the shared suffix given each prefix-plus-option
  context. Legacy v2 text scores cannot satisfy the current formal gate;
  migrate retained likelihoods with `scripts/repair_text_benchmark_results.py`
  and rerun WinoGrande before publishing an eight-task macro. Migration keeps
  a `legacy-before-p1/` backup inside the affected text result directory.
- Historical checkpoint evaluation output is under
  `output/evaluation/unified-a-0p6b/checkpoints/step-{56000,58000,60000}`;
  the compact trend is `output/evaluation/unified-a-0p6b/trend/trend.md`.
  Per-sample benchmark/text results and qualitative image/caption artifacts are
  retained, while rank shards, resume state, duplicate logs, smoke output, and
  results from removed protocols are excluded.
- The step-56000/58000/60000 archive predates language-prior correction and is
  historical only. Its former `paper_protocol_complete` flag does not satisfy
  the current v8 protocol, and its ImageNet/retrieval/benchmark metrics must
  not be reused as new paper results.
- New Flickr30K and MSCOCO runs must use retrieval schema v3. Independent COCO
  query partitions may still be used, but every partition must score all
  25,010 captions and `scripts/merge_cross_dataset_retrieval_partitions.py`
  must calibrate the complete, duplicate-free 5,000-row matrix before R@K.
- The current multimodal asset manifest is schema v2 with 61,036 images,
  including three fixed language-prior null images. Its cache target is
  `public/benchmarks/selfless_multimodal_likelihood_v1/vae_posterior_mar_kl16_v2`.
  Build that 16-shard cache before evaluation. The former 64,973-image cache
  lacks the null images and is explicitly not a valid evaluation input.
- `output/evaluation-checkpoints` remains outside the result archive because it
  contains 77GB of legacy checkpoint-trend inputs, not evaluation output.
  Moving it would invalidate recorded historical paths; it is not the
  canonical input for a new final evaluation.
- Evaluation launchers use the repository `.venv/bin/python` explicitly;
  do not rely on a bare `python` being present in non-interactive Ascend Job
  images. Completed core results can be reused through
  `REUSE_CORE_EVAL_ROOT` without repeating 50K FID generation; retained
  external predictions can be reused through `REUSE_BENCHMARK_EVAL_ROOT`.
  Standard retrieval results can be reused through
  `REUSE_COCO_RETRIEVAL_ROOT` and `REUSE_FLICKR30K_RETRIEVAL_ROOT`.

### Canonical evaluation output

- All evaluation results live under `output/evaluation/`; the single human-facing
  entry is `output/evaluation/index.html`. Build/update it on CPU with
  `python3 scripts/build_evaluation_report.py`. Current source selection is
  versioned in `configs/protocols/evaluation_report.json`; missing or invalidated
  results must never be substituted from historical runs.
- Keep training checkpoints, raw/final EMA exports and optimizer/data/RNG state
  in their training run. Training validation goes to
  `output/evaluation/training-validation/<run>/`, qualitative output to
  `output/evaluation/qualitative/<run>/`, and representation studies to
  `output/evaluation/research/`. Standalone evaluators retain their explicitly
  selected result directory under `output/evaluation/`.
- Historical path relocation and original metadata are recorded under
  `output/evaluation/migrations/`. Frozen execution logs/source snapshots retain
  the original recorded paths; use the relocation manifest to resolve them.
- `output/evaluation-checkpoints/` contains historical input weights, not result
  files. Those checkpoint identities remain outside the result directory.

### B sampling sweeps

- The CFG/Heun sweep uses independent one-node, 16-NPU Jobs in the
  user-selected project
  `随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架`.
  The sampling plan and explicit platform fields are retained in the sweep's
  `protocol.json`; this evaluation placement does not change training rules.
- The sweep scheduler is `scripts/sweep_unified_t2i_sampling.py`, the Job
  submitter is `scripts/submit_unified_t2i_sweep.py`, and the CPU report watcher
  is `scripts/report_sampling_sweep.py`. Results live under
  `output/evaluation/unified-b-x0content-0p6b/sweeps/`.
- Evaluate each CFG in 0.5 increments at 10-step Heun on ImageNet-val 50K;
  select minimum FID and maximum IS separately, then evaluate both selected
  CFGs at 5/10/20/50/100 steps. Keep the paired sample/noise contract and ten
  class-stratified IS splits. Any guard-boundary optimum remains unresolved
  until the range is extended.
- The evaluator's `--save_image_count` retains an evenly spaced subset of
  the full evaluation ordering, plus the actual prompts, image IDs, and
  canonical noise seeds. This export does not change generation or metrics.
  The current sweep keeps 64 paired images per combination.
- Add selected sweep roots to `sampling_sweeps` in
  `configs/protocols/evaluation_report.json`. The homepage includes the
  sweep conclusions, curves, and paired-image comparison while preserving
  the common fixed-parameter model comparison.
- Before a reveal-order study, audit CFG ±0.5 and ±1.0 at the selected
  minimum-FID Heun step. Reuse exact completed 50K points when present;
  a boundary winner still requires extension. Preparation is handled by
  `scripts/prepare_unified_order_sweep.py` and preserves the evidence in
  `cfg-refinement.json`.
- Reveal-order studies use the same independent 16-NPU Job placement and
  paired metric/image protocol. Register their roots in `order_sweeps` in
  the report selection. The `confidence_*` experimental policies rank each
  next 16 Halton positions from generated context; they require cached
  generation, constant CFG != 1, and canonical initial noise. They do not replay training sigma
  or read target latents. Preserve saved `order_trace` records as well as PNGs.
  See `docs/B_ORDER_SWEEP.md` for the score definitions and controls.
- The full nine-model CFG=2.0 / Heun=10 matrix is prepared with
  `scripts/prepare_unified_matrix_sweep.py --include-random`. Evaluate Halton,
  `confidence_stability`, and `random` for each model, plus E's native sequential control.
  Random preserves the fixed evaluator seed/batch partition and its native
  uniform permutation; verify matching saved random orders across models.
  Each final EMA keeps its own attention and flow-condition contracts; D
  refreshes the dynamic XT condition for both probes, and F retains no flow
  content cache. Follow `docs/UNIFIED_MATRIX_CFG2_ORDER.md` for the smoke and
  pairing gates. Register the root in `matrix_sweeps` in the report selection.
  Use independent 16-NPU Jobs in the same user-selected random-order project,
  with up to 12 concurrent Jobs and rolling admission after per-model smoke.
- Job launchers must include existing mounted Ascend driver library
  directories, as in the qualitative launcher. Throttle submissions and
  retry rejected rate-limited reads. A controller/resource-query error must
  preserve healthy independent Jobs and their progress.

### Unified qualitative generation

- The reusable all-final-EMA generation entry is
  `script/selfless/generate_unified_qualitative_ascend16.sh`, backed by
  `scripts/generate_unified_qualitative.py` and the fixed custom prompts in
  `configs/protocols/unified_qualitative_prompts_v1.json`.
- Prepare frozen inputs on CPU before submission. One final EMA per completed
  Unified-MM condition is selected; unfinished runs and intermediate exports
  are not implicitly substituted. Every checkpoint owns its architecture,
  backbone/flow attention, flow condition, and image-order contract. D must
  pass the post-load time-embedding value guard.
- Default output per model is 128 T2I images (64 prompts, two paired spatial
  noise seeds), 64 I2T captions (ImageNet-val and COCO/Flickr Karpathy test),
  and 64 pure-text continuations (32 prefixes, greedy and temperature 0.8).
  Retain all outputs. Label untrained tasks for single-source controls;
  captions used for human comparison must never enter I2T model inputs.
- Use 10-step Heun, CFG 3.5, BF16 model, FP32 KL16 VAE, fixed input posterior
  samples, and the model-native image order. Text is base-model continuation,
  not chat-template instruction following. Qualitative artifacts are not
  formal benchmark scores.
- Outputs live under `output/evaluation/qualitative/<run>/`. The final renderer requires
  exact sample coverage, all 16 worker/load reports, and valid images before
  publishing a portable `index.html` gallery and ZIP. Keep smoke outputs in
  a separate directory. Runtime hashing and W&B remain disabled.
- Before submitting, inspect the effective CLI `me` path as well as the job
  command: account-level path aliases can point at another project's fileset
  or an unavailable storage pool. For isolated submissions, use a temporary
  CLI context with only a `me` override to the validated global shared output
  path; leave global account settings and other projects unchanged. Continue
  to pass scheduling fields explicitly, without a Workload Profile.

### Historical ImageNet recipes

Earlier ImageNet-100, class-conditioned, caption-joint, 80/400-epoch T2I,
sequential-sigma and position-wise recipes are retained in
[the historical execution notes](docs/archive/IMAGENET_EXECUTION_NOTES.md).
They describe their original runs and do not override the Unified defaults.

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
