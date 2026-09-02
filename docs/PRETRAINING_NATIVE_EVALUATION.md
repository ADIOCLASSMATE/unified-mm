# Selected image-understanding evaluation

The frozen protocol is
`configs/protocols/pretraining_native_understanding_evaluation_ascend16.yaml`.
New final runs load the FP32 `hf_model-final-ema` export directly, without
fine-tuning, instruction tuning, a linear probe, or runtime content hashing.
Rank-sharded EMA inputs are accepted only for retained historical trends.

## ImageNet-1K zero-shot classification

`scripts/evaluate_imagenet_pretraining_native.py` evaluates all 50,000
official ImageNet validation images against all 1,000 class texts. It does not
subsample a custom 1K or 5K retrieval set.

- Class order and names come from OpenAI CLIP's ImageNet notebook. The two
  duplicate labels are disambiguated by WordNet synset identity (`projectile`
  versus `missile`, and `sunglass` versus `sunglasses`).
- The single frozen class text is `a photo of a {class_name}.`; no template is
  selected by looking at ImageNet-val labels.
- Selfless, not an external CLIP encoder, supplies the mean token
  log-likelihood `s(i,c)=log P(t_c|i)`.
- The text prior is estimated from all 50,000 evaluation images:
  `b(c)=logmeanexp_i s(i,c)`. Ranking uses only `s(i,c)-b(c)` with fixed
  `alpha=1`.
- Only debiased Top-1 and Top-5 accuracy are reportable. The raw score matrix
  is not exposed as an alternative metric.

This is a fixed *generative zero-shot classification* protocol. It uses CLIP's
class vocabulary convention, but it must not be described as OpenAI CLIP
cosine-similarity evaluation.

## Standard bidirectional retrieval

The paper-facing retrieval suite uses the Karpathy test split and the complete
candidate pool:

- MSCOCO 5K test: 5,000 images and the 25,010 captions present in the source
  split (4,990 images have five captions and ten have six);
- Flickr30K 1K test: 1,000 images and 5,000 captions.

Both datasets report I2T and T2I R@1/R@5/R@10. COCO is one full 5K run, not
the legacy five-fold 1K average. Every I2T query accepts all reference
captions of its image; every T2I caption accepts its paired image.

For the complete image-by-text score matrix `S`, the only retained score is
`S'[:,c] = S[:,c] - logmeanexp_i S[i,c]` (`alpha=1`). The estimator uses every
candidate image and no relevance labels. This correction can substantially
change I2T because one query ranks different texts. It cannot change T2I
ranks: all images for a fixed caption receive the same subtracted constant.
That invariance is a mathematical consequence of the requested calibration,
not evidence that T2I was left uncalibrated.

Assets are normalized with `scripts/prepare_cross_dataset_retrieval_assets.py`,
cached with
`script/selfless/prepare_cross_dataset_retrieval_cache_ascend16.sh`, and scored
with `script/selfless/evaluate_cross_dataset_retrieval_ascend16.sh`.

## Compositional and hard-negative matching

The paper-facing tasks are SugarCrepe (all seven perturbation categories), ARO
VG Relation, and ARO VG Attribution. Each example compares the positive and
official negative caption. Candidate scores use fixed `alpha=1` and a prior
estimated with the log-mean-exp score over exactly three fixed, label-free,
Gaussian null images in the model's normalized `[-1,1]` VAE input space
(`mean=0`, `std=0.25`). Only strict debiased pairwise win rate is reportable;
ties are not counted as wins.

The number and distribution of null images are frozen project choices. They
are compatible with the VisualGPTScore calibration family but must be stated
when comparing results, because the paper reports model/dataset-specific null
image choices rather than one universal protocol.

After preparing asset schema v2, build the corresponding cache once with
`script/selfless/prepare_multimodal_likelihood_cache_ascend16.sh`. Its default
target is `vae_posterior_mar_kl16_v2`; the former 64,973-row cache is rejected
because it predates the three null images.

## Internal diagnostics and removed protocols

MMBench Dev-EN circular and single-image SEED-Bench remain internal ablation
diagnostics. Their semantic candidate-likelihood adapter is not comparable to
the official free-form leaderboards.

The following are rejected by the current schemas and are not reusable:
custom ImageNet-val 1K/5K retrieval, ImageNet ReaL, generated-caption CLIP
score, raw/uncalibrated likelihood, custom ImageNet caption negatives, POPE,
COCO caption perplexity, and incomplete Winoground/SVO-Probes/What'sUp runs.

## Canonical launch

```bash
RUN_ROOT=output/unified-b-0p6b-100b-imagenet-split-s42-r1
EVAL_PROFILE=formal \
  script/selfless/evaluate_unified_native_full_checkpoint_ascend16.sh \
  "${RUN_ROOT}/hf_model-final-ema" /path/to/evaluation-output
```

The launcher validates completeness, model-source identity, schema versions,
fixed calibration, and the no-hash contract before reusing any result.
