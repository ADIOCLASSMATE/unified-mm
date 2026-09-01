# Selected image-understanding evaluation

The frozen protocol is
`configs/protocols/pretraining_native_understanding_evaluation_ascend16.yaml`.
Every new final metric loads the FP32 `hf_model-final-ema` Hugging Face export
directly with `from_pretrained`, without a rank-sharded EMA overlay. Its step
comes from `ema_export_metadata.json.source_global_step`. Rank-sharded EMA
directories remain accepted only for retained historical checkpoint trends.
No evaluation uses downstream fine-tuning, instruction tuning, or a learned
linear probe. Runtime content hashing is disabled.

## ImageNet-val custom retrieval

`scripts/evaluate_imagenet_pretraining_native.py` reads only the official
50,000-image ImageNet validation split and its val-only KL16 posterior cache.
ImageNet train is never used by an evaluation loader.

- Retrieval uses deterministic class-balanced 1K and 5K subsets drawn from
  ImageNet val. Both I2T and T2I exact-instance R@1, R@5, and R@10 are primary
  outputs. Same-class relevance recall and median rank remain secondary
  diagnostics because different images in one ImageNet class can have
  semantically similar synthetic descriptions.
- Only the uncalibrated normalized likelihood is evaluated and written.
  Visual calibration has been removed from the evaluator and launcher.

The large candidate sets use a shared-prefix KV cache. A
repeated-full-sequence backend remains available as a correctness reference.

ImageNet class-name Top-1/Top-5 and ReaL are not part of the protocol. Their
generative class-caption likelihood did not match the training objective well
enough to support a classification claim in the paper main table.

## Standard image-text retrieval

The paper main table uses the standard Karpathy test splits in addition to the
custom ImageNet-val diagnostic:

- MSCOCO 5K test: 5,000 images and 25,000 captions;
- Flickr30K test: 1,000 images and 5,000 captions.

Both report I2T and T2I R@1, R@5, and R@10. COCO is evaluated once on the full
5K test set; it is not the legacy five-fold 1K average. Every image has five
positive captions, and every caption has one positive image.

Normalize an authorized Karpathy split with
`scripts/prepare_cross_dataset_retrieval_assets.py`, then build its no-hash
KL16 cache with
`script/selfless/prepare_cross_dataset_retrieval_cache_ascend16.sh`. Formal
scoring uses `script/selfless/evaluate_cross_dataset_retrieval_ascend16.sh`.
COCO images can be obtained from the official 2014 release; Flickr30K images
require the dataset owner's access flow.

## Hard negatives and compositionality

The custom ImageNet random-class, same-class, and WordNet-near caption
negative protocols have been removed. Correct-caption versus hard-negative
evaluation now uses the official benchmark examples from:

- SugarCrepe, including all seven perturbation categories;
- ARO Visual Genome Relation;
- ARO Visual Genome Attribution.

All three are scored by positive-versus-negative mean token log-likelihood.
Their pairwise random baseline is 50%.

## Internal ablation diagnostics

MMBench Dev-EN circular and the single-image portion of SEED-Bench remain only
for internal ablation trends because the three-checkpoint audit found clear
margins over their random baselines. They are excluded from paper main tables:
both use our semantic candidate-likelihood scorer rather than the leaderboards'
free-form answer extraction.

POPE was removed because accuracy stayed at chance while the model predicted
“yes” for only about 3% of examples. COCO Caption PPL was removed from the
image-understanding table because it has neither a random baseline nor a
matched reference checkpoint that makes the absolute value interpretable.
Winoground, SVO-Probes, and What’sUp are not part of the selected protocol
because their complete official assets were unavailable in the completed
runs.

The numerical selection audit is recorded in
`docs/IMAGE_UNDERSTANDING_BENCHMARK_SELECTION.md`.

## Launchers

Run the canonical complete evaluation:

```bash
RUN_ROOT=output/unified-a-0p6b-100b-imagenet-split-s42-r1
OUTPUT_ROOT=/path/to/unified-a-0p6b-final-ema-native-full
EVAL_PROFILE=formal \
  script/selfless/evaluate_unified_native_full_checkpoint_ascend16.sh \
  "${RUN_ROOT}/hf_model-final-ema" "${OUTPUT_ROOT}"
```

This launcher uses one 16-NPU Job and runs every component serially.

The model-source directory must contain `config.json`, `model.safetensors`,
`tokenizer.json`, and `ema_export_metadata.json`. The ablation-a export above
records `source_global_step=95415`. The launcher's historical `checkpoint`
naming does not change the load contract: this path is loaded as the final HF
model, and no `ema_manifest.json` is read.

Set `REUSE_CORE_EVAL_ROOT` to a completed core evaluation to avoid repeating
generation, validation, and text evaluation. Set
`REUSE_BENCHMARK_EVAL_ROOT` to a completed likelihood benchmark root for the
same model source to reuse previously computed MMBench, SEED, SugarCrepe, and
ARO predictions. Both reuse paths validate model-source identity, completion,
and the no-hash contract.

Set `REUSE_COCO_RETRIEVAL_ROOT` and `REUSE_FLICKR30K_RETRIEVAL_ROOT` to reuse
completed standard-retrieval results for the same model source.

Run only the selected understanding suite:

```bash
RUN_ROOT=output/unified-a-0p6b-100b-imagenet-split-s42-r1
EVAL_PROFILE=formal \
  script/selfless/evaluate_pretraining_native_understanding_ascend16.sh \
  "${RUN_ROOT}/hf_model-final-ema" /path/to/output
```
