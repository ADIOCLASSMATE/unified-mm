# ImageNet multi-caption distillation

## Teacher selection and API policy

Qwen API generation is disabled. The generator rejects every `qwen*` model
before creating an API client. Existing Qwen responses remain untouched; any
future Qwen captions are generated locally on a separate H100 task.

The active API teacher is `MiniMax-M3`. Legacy `kimi-k2.6` responses remain
readable, but Kimi is excluded from new generation and publication. A CPU-side
ImageNet benchmark on 2026-08-03 used independent processes, independent
concurrency pools, non-overlapping images, one request per image/model, and
three requested captions per response:

| model/config | success | completed req/s | mean / P50 / P90 / P95 / max latency (s) | decision |
| --- | ---: | ---: | ---: | --- |
| `kimi-k2.6`, concurrency 10 | 64/64 | 0.87 | 10.15 / 9.10 / 13.82 / 15.66 / 18.88 | Recommended throughput point. |
| `kimi-k2.6`, concurrency 12 | 64/64 | 0.86 | 12.57 / 9.09 / 24.95 / 34.44 / 43.03 | Stable, but no throughput gain. |
| `kimi-k2.6`, concurrency 16 | 64/64 | 0.73 | 15.12 / 7.27 / 42.96 / 50.92 / 64.86 | Stable success rate but slower; do not use formally. |
| `kimi-k2.6`, concurrency 32 | 74/96 | 0.63 | timeout-dominated | Unstable. |
| `kimi-k2.6`, concurrency 64 | 37/64 | 0.53 | timeout-dominated | Unstable. |
| `MiniMax-M3`, concurrency 96 | 94/96 | 4.22 | 16.92 / 17.19 / 19.88 / 20.92 / 22.73 | Good. |
| `MiniMax-M3`, concurrency 128 | 125/128 | 4.71 | 18.15 / 18.46 / 21.43 / 23.28 / 27.14 | Short probe only; the full run later overloaded badly. |

`MiniMax-M2.5`, `MiniMax-M2.7`, `deepseek-v4*`, and `glm-5.2` are text-only and
are rejected. `qwen-image-2.0*`, `doubao-seedream*`, and `doubao-seedance*`
generate media rather than caption text, while `qwen3-vl-embedding` and
`doubao-embedding-vision*` return embeddings; these are rejected too.

`doubao-seed-2-1-pro-260628` timed out after 120 seconds, `mimo-v2.5` consumed
the entire 500-token probe budget without returning text, and `mimo-v2.5-pro`
returned a proxy 404. They remain unverified and are intentionally not enabled
for formal generation.

The generator defaults to MiniMax only. `kimi-k2.6` remains a supported legacy
alias but must be selected explicitly. MiniMax-M3 omits the thinking field,
leaving thinking off by provider default. `--temperature` and `--thinking`
remain explicit overrides but normally stay unset.

Caption length uses a 32--60 word prompt target only. Generation requires
exactly the requested number of non-empty captions; partial or oversized arrays
are retried and are not considered resumable completion. It does not reject
based on length or lexical similarity. Word counts remain metadata only.

Generation retries each missing group up to two times with a 45-second request
timeout by default. Per-model request-rate limiting prevents fast empty
responses from flooding the API, and a rolling failure circuit stops scheduling
when 50% of 20 health-relevant completed groups fail. A repeated, already-known
image-level `empty_response` or `malformed_response` does not count as a new
provider outage; fresh content failures, timeouts, disconnects, 429s, and 529
overload responses still do. Formal synthesis runs on CPU/API workers only: no
CLIP or VLM judge is part of the bulk path. Rerunning the same generation
command preserves exact three-caption records and requests only missing or
malformed groups.

## Data contract

The published file has exactly one JSONL row per canonical ImageNet manifest
row. Identity is preserved with all of the following fields:

- `manifest_index`, `img_id`, `id`, `path`, and `synset`;
- the SHA-256 of the exact manifest used for generation;
- the SHA-256 of the image bytes observed by the teacher;
- the original caption as caption index zero;
- multiple API captions with model, prompt version, word count, and text hash.

The merge is strict by default. It refuses unknown/missing image IDs, path or
synset mismatches, duplicate identities, mixed manifest hashes, incomplete
image/model groups, empty captions, and any group whose caption count differs
from the configured count. Publishing uses an atomic rename.

`ImageNetFlowCacheDataset` remains compatible with the old one-caption JSONL.
For the new format it reads the nested `captions` list. Training deterministically
cycles through every caption using `(seed, sample index, epoch)`; validation
always uses caption index zero (the original caption), unless explicitly
configured otherwise.

## Commands

Run with the repository environment:

```bash
PYTHONPATH=. uv run --no-sync python scripts/distill_imagenet_captions.py capabilities
```

Always inspect the planned request count before spending API quota:

```bash
PYTHONPATH=. uv run --no-sync python scripts/distill_imagenet_captions.py generate \
  --dataset imagenet100 \
  --models MiniMax-M3 \
  --model-concurrency MiniMax-M3=48 \
  --model-rps MiniMax-M3=4 \
  --dry-run
```

A limited MiniMax smoke is fully resumable:

```bash
PYTHONPATH=. uv run --no-sync python scripts/distill_imagenet_captions.py generate \
  --dataset imagenet100 \
  --models MiniMax-M3 \
  --limit 1000 \
  --model-concurrency MiniMax-M3=48 \
  --model-rps MiniMax-M3=4 \
  --timeout 45 \
  --max-retries 2 \
  --circuit-window 20
```

For formal generation, run MiniMax alone. Each exact three-caption response is
immediately reusable:

```bash
PYTHONPATH=. uv run --no-sync python scripts/distill_imagenet_captions.py generate \
  --dataset imagenet100 \
  --models MiniMax-M3 \
  --captions-per-model 3 \
  --model-concurrency MiniMax-M3=48 \
  --model-rps MiniMax-M3=4 \
  --timeout 45 \
  --max-retries 2 \
  --circuit-window 20
```

Use the same command with `--dataset imagenet1k` for the full 1,281,167-row
manifest. Do not add Qwen or Kimi to the API generation command.

Large runs can be partitioned into disjoint shards. Each worker must use a
different `--shard-id` with the same `--num-shards` and generation settings:

```bash
PYTHONPATH=. uv run --no-sync python scripts/distill_imagenet_captions.py generate \
  --dataset imagenet1k \
  --models MiniMax-M3 \
  --model-concurrency MiniMax-M3=48 \
  --model-rps MiniMax-M3=4 \
  --num-shards 32 \
  --shard-id 0
```

Do not merge until coverage is complete. A one-shot pass may leave timeout or
connection gaps; rerun only the matching single-model command to fill them,
then consolidate again. Merge and validate:

```bash
PYTHONPATH=. uv run --no-sync python scripts/distill_imagenet_captions.py coverage \
  --dataset imagenet1k \
  --response-dir public/datasets/imagenet_distilled_captions/imagenet1k/responses/REQUEST_FINGERPRINT

PYTHONPATH=. uv run --no-sync python scripts/distill_imagenet_captions.py merge \
  --dataset imagenet1k \
  --response-dir public/datasets/imagenet_distilled_captions/imagenet1k/responses/REQUEST_FINGERPRINT

PYTHONPATH=. uv run --no-sync python scripts/distill_imagenet_captions.py validate \
  --dataset imagenet1k \
  --verify-images
```

The default published paths are:

- `public/datasets/imagenet_distilled_captions/imagenet100/captions/imagenet100_multicap_v1.jsonl`
- `public/datasets/imagenet_distilled_captions/imagenet1k/captions/imagenet1k_multicap_v1.jsonl`

The response fingerprint is printed by both dry-run and generation. Staging
databases live below `responses/<fingerprint>/` and are safe to resume.

### Progress and runtime logs

Generation displays a request progress bar and appends immediately flushed
JSONL events to `<response-dir>/generation.runtime.jsonl`. Every request records
the model, `img_id`, manifest index, concurrency, start/end timestamps, latency,
success/failure, error class, and saved caption count. Each independent model
pool emits `model_scheduler_completed` as soon as it finishes; it does not wait
for another model process. Per-run aggregate JSON files are stored under
`<response-dir>/benchmark_runs/` with success rate, requests/s, latency P50/P90/
P95/max, errors by class, and actual per-minute completions.

Every stage of `evaluate_imagenet_caption_smoke.py` displays an image/record
progress bar. It also appends structured JSONL runtime events and flushes every
event immediately, so the last completed image and failure message remain
visible after an interruption. `prepare` and `clip` write
`<output>.runtime.jsonl`; `summarize` writes
`<output-dir>/summarize.runtime.jsonl`.

Use `--log-file` to select another path and `--log-every N` to reduce logging
frequency on a larger smoke. `--no-progress` disables terminal rendering for a
batch job but deliberately leaves runtime logging active.

## Training configuration

Point either the ImageNet-100 or ImageNet-1k config at its matching merged
file. Keep the canonical cache and manifest for that dataset unchanged:

```yaml
dataset:
  params:
    conditioning_mode: caption
    caption_jsonl: public/datasets/imagenet_distilled_captions/imagenet100/captions/imagenet100_multicap_v1.jsonl
    caption_text_key: recaption_short
    caption_list_key: captions
    caption_list_text_key: text
    caption_path_key: path
    caption_id_key: id
    caption_validation_index: 0
    caption_manifest_sha256: SHA256_FROM_THE_MERGE_METADATA
```

Use the analogous `imagenet1k` path with the full cache/manifest. The merged
metadata file beside the JSONL contains the exact output SHA-256 to place in
`caption_manifest_sha256`.

## Operational notes

- Raw ImageNet must be mounted at `/inspire/dataset/imagenet/v1`; the manifest
  is not a substitute for the official dataset attachment.
- The generator reads `SII_API_KEY` from the environment and never writes it.
- A full 1k run makes 1,281,167 MiniMax requests before retries and asks for three
  captions per request. Confirm
  provider quota, cost, rate limits, and permission to transmit ImageNet bytes
  before starting it.
- `--allow-incomplete` exists only for smoke/debug merges. Do not use a partial
  merge as a formal training dataset.
