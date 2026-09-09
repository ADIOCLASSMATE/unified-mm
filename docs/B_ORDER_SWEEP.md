# B sigma / reveal-order comparison

B is the formal model defined in [EXPERIMENTS.md](EXPERIMENTS.md).

This study fixes the minimum-FID choice from the completed CFG/Heun sweep:
CFG 2.0, Heun 10. `cfg-refinement.json` revalidates the exact 50K results at
CFG 1.0, 1.5, 2.0, 2.5, 3.0. The winner is interior, so the requested final-step
±0.5 / ±1.0 neighborhood needs no duplicate computation or boundary extension.

The model is the step-95415 final EMA of
`unified-b-x0content-0p6b-100b-imagenet-split-s42-r1`. Each arm uses an independent
16-NPU Job in the user-selected random-order language-modeling project, global
batch 4096, ImageNet-val 50K, seed 42, canonical per-image/per-position noise,
BF16 model, FP32 VAE and ODE integration, constant CFG, temperature 1.0,
serialized reveal, and the backbone cache. IS has ten synset-stratified splits.
These are validation-set comparisons, not independent holdout estimates.

## Strategies

Existing policies are `spatial_halton`, `sequential`, `spatial_uniform`
(the existing center-out/checker ordering, despite its name), and `random`.
Random permutations use the evaluator's frozen batch/rank seed rule; the
canonical initial noise remains paired with the other arms. Changing rank or
batch layout would change random order and requires a separate protocol.

The four additional policies retain the same ordered blocks of 16 Halton
positions. At the start of each block, they query all its candidates using only
the caption and already generated image content. The candidates cannot attend
to one another. This is adaptive ranking within blocks, not global re-ranking
after every token. All candidates are eventually decoded sequentially with
fresh conditions and their original canonical noise.

Let `vc`, `vu` be conditional and unconditional flow velocities at the actual
per-position initial noise `x`, with `t=0`. Let `v = vu + cfg*(vc-vu)`.
Mean is over latent channels and all scoring arithmetic uses FP32.

| Policy | Score / ordering |
|---|---|
| `confidence_cfg` | Ascending `mean((vc-vu)^2) / (0.5*mean(vc^2+vu^2)+1e-8)` |
| `confidence_cfg_reverse` | The same score, descending; tests the opposite direction of guidance sensitivity |
| `confidence_stability` | Ascending `mean((v_next-v)^2)/(mean(v^2)+1e-8)`, where `x_next=x+0.1*v`, `t_next=0.1`, and `v_next` uses the same generated context |
| `confidence_halton` | Both velocity probes are computed, but Halton order is retained; numerical/cache and cost control |

Ties preserve original Halton order. These proxies are not calibrated confidence
probabilities or latent log likelihoods. Agreement may prefer background or
weakly caption-dependent positions. Stability measures a local ODE property,
which may not predict perceptual correctness. The comparison tests these
hypotheses without using true images to choose orders.

The general high-confidence-first idea is informed by
[MaskGIT](https://arxiv.org/html/2202.04200v1), which uses discrete token prediction
probabilities. Our continuous-latent score definitions above are experimental
adaptations, not that paper's method. Keeping spatially spread candidate blocks
is motivated by the coverage argument in
[Halton Scheduler](https://arxiv.org/abs/2503.17076); that paper also gives reasons
not to assume confidence-based selection always improves FID.

## Cache and reproducibility checks

A probe commits only the previously completed content token, exactly once,
with its X0 backbone condition. Probe candidate queries and Euler proposals
never enter either content cache. The chosen candidates are re-evaluated when
actually generated, so conditions include all preceding committed positions.
Probe scoring does not consume RNG draws. CPU tests compare adaptive generation
against an independent probe-free decode forced to use its recorded order,
cross two block boundaries, and verify independence from training sigma order.

Run NPU smoke only on permanent `dev-wjx-ascend`; require its saved audit before
submission. Pass an explicit canonical shared `--cwd` to `notebook exec` because
the account's `me` alias can point at a fileset absent from this Notebook.
The full study re-runs Halton to reproduce the previous 50K result,
and measures the numerical and timing effect of the probe control explicitly.

Each arm saves 64 identical selected image indices, prompts, canonical noise
seeds, PNGs, and `order_trace` JSON containing actual ranks and proxy scores.
The report audits these permutations, score sorting, image identities and
decoding, checkpoint identity, precision, split coverage, and immutable inputs
using readable file fields. No runtime hashes or third-party trackers are used.

Prepare with `scripts/prepare_unified_order_sweep.py --previous-sweep <completed>
--output-dir <new>`, run its bounded `launch/smoke.sh` on the development
Notebook, validate smoke, then submit `launch/submit_unified_t2i_sweep.py` with
`--output-dir <new> --name-prefix <unique>`. The standard report watcher
`scripts/report_sampling_sweep.py --output-dir <new> --watch` updates
`output/evaluation/index.html`, PNG/PDF plots, CSV, and final audits.

Potential follow-ups, separate from this frozen comparison: enlarge the
candidate pool with equal compute controls; re-score more often; evaluate
agreement across independent probe noises; calibrate a confidence predictor
on a separate training subset. None should be claimed better before measured.
