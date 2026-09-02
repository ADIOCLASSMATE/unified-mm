# Image-understanding benchmark selection audit

This file records the protocol decision, not model results. Historical
56K/58K/60K likelihood numbers were produced before language-prior correction
and are deliberately not copied into the current report. Their schemas are
rejected by the launchers and summarizers; every publishable value must be
rerun with the frozen protocol.

| evaluation | current role | reportable primary metric | comparison boundary |
| --- | --- | --- | --- |
| ImageNet-1K val 50K | paper-facing generative zero-shot classification | debiased Top-1/Top-5 | same fixed Selfless likelihood protocol |
| MSCOCO Karpathy 5K | paper-facing standard retrieval | debiased I2T/T2I R@1/5/10 | same split, candidates, model score, and calibration |
| Flickr30K Karpathy 1K | paper-facing standard retrieval | debiased I2T/T2I R@1/5/10 | same split, candidates, model score, and calibration |
| SugarCrepe | paper-facing compositional matching | strict debiased pairwise win rate plus seven categories | same null-image calibration |
| ARO VG Relation/Attribution | paper-facing compositional matching | strict debiased pairwise win rate | same null-image calibration |
| MMBench Dev-EN circular | internal ablation only | debiased circular accuracy | not leaderboard-comparable |
| SEED-Bench image | internal ablation only | debiased candidate accuracy | not leaderboard-comparable |

ImageNet's former class-balanced 1K/5K image-caption retrieval was removed.
ImageNet validation is now used once, in full, for 1,000-way zero-shot
classification. Standard retrieval claims are restricted to COCO and
Flickr30K because they supply recognized image-caption retrieval splits and
multiple ground-truth captions per image.

SugarCrepe must retain all seven categories. ARO and SugarCrepe ties are
reported separately and never converted to index-order wins. MMBench and SEED
remain internal because this project scores semantic text candidates directly,
whereas their normal leaderboards rely on answer-generation and extraction
conventions.

Removed metrics are custom ImageNet retrieval, ImageNet ReaL, generated-caption
CLIP score, uncalibrated likelihood, custom caption negatives, POPE, COCO
caption perplexity, and tasks whose complete authorized assets were not
available. Missing tasks are never represented as zero scores.
