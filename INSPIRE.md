# Inspire execution notes

## Shared paths

- Repository: `/inspire/hdd/global_user/wanjiaxin-253108030048/code/unified-mm`
- Shared user root: `/inspire/hdd/global_user/wanjiaxin-253108030048`
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
  to this work. Do not keep duplicate runnable jobs in both projects.
- Image: `docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1`.

## Waiting

After one initial configuration/status check, wait with one blocking process:

```bash
inspire --json job wait <job-name> \
  --workspace 分布式训练空间 \
  --interval 30 \
  --timeout 14400
```

Do not repeatedly poll status, events, logs, or GPU utilization while a formal
job is running.

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
