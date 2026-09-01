# Image-understanding benchmark selection audit

This audit uses the completed 0.6B EMA checkpoints at steps 56,000, 58,000,
and 60,000. The table below shows the step-60,000 normalized-likelihood score.
The 95% intervals are Wilson intervals over benchmark examples. MMBench uses
the conservative option-count-weighted random baseline for one semantic
choice per original question rather than the much smaller independent-row
circular baseline.

| benchmark | examples | score | random baseline | margin | 95% interval | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MMBench Dev-EN circular | 1,164 questions | 38.06% | 27.98% | +10.08 pp | 35.31–40.88% | internal ablation trend only |
| SEED-Bench image | 14,233 | 37.92% | 25.00% | +12.92 pp | 37.12–38.71% | internal ablation trend only |
| SugarCrepe | 7,511 | 62.87% | 50.00% | +12.87 pp | 61.77–63.95% | retain |
| ARO VG Relation | 23,937 | 71.08% | 50.00% | +21.08 pp | 70.50–71.65% | retain |
| ARO VG Attribution | 28,748 | 87.54% | 50.00% | +37.54 pp | 87.15–87.91% | retain |
| POPE COCO | 9,000 | 50.08% | 50.00% | +0.08 pp | 49.04–51.11% | remove |

POPE also has a 5.43% positive-class F1 and a 2.79% predicted-yes ratio at
step 60,000, confirming a degenerate almost-always-negative solution rather
than useful hallucination discrimination.

Across steps 56,000 to 60,000, the absolute changes were only 0.02–0.26
percentage points for every benchmark above. Paired exact McNemar tests found
no reliable checkpoint difference at the 5% level; ARO Relation was borderline
at p=0.0501. The retained tasks therefore demonstrate non-chance capability,
not a claimed late-training improvement.

SugarCrepe must retain its seven-category breakdown. At step 60,000 the
`add_obj` category is below chance (35.50%) even though the overall score is
62.87%; reporting only the aggregate would conceal that failure mode.

ImageNet generative Top-1/Top-5 and ReaL have been removed because class-name
likelihood does not match the pretraining objective closely enough for a paper
main-table classification claim. ImageNet-val 1K/5K I2T/T2I R@1/R@5/R@10
remain as explicitly labelled custom in-domain retrieval protocols.

The paper-facing standard retrieval suite adds MSCOCO Karpathy 5K test and
Flickr30K Karpathy 1K test. Both report bidirectional R@1/R@5/R@10 over all
five reference captions per image. MMBench and SEED are excluded from the
paper main table and retained only for comparing internal ablation trends.

COCO Caption PPL is removed from the selected image-understanding protocol:
its absolute PPL has no random-accuracy analogue and no matched external
reference in these runs. Winoground, SVO-Probes, and What’sUp are also omitted
because complete official assets were not evaluated; unavailable tasks are not
represented as zero scores.
