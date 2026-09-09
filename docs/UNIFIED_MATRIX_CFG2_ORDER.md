# Unified final-EMA matrix at CFG 2.0 / Heun 10

The study evaluates the nine models selected for formal metrics in
`configs/protocols/evaluation_report.json`: B-X0, B-flowdiag, B-no-flowdiag,
A-X0, legacy A, C-on-B, D-on-B r4, E-on-B, and F-on-B. Each model uses its own
step-95415 final EMA. The five extra qualitative-only models are outside this
formal matrix.

Every model receives independent `spatial_halton`, `confidence_stability`, and `random`
arms. E receives an additional `sequential` arm, because that is its native
training/evaluation order. The matrix consists of 28 full 50K evaluations. The new
Halton matrix holds inference settings fixed across all nine models; the extra
E arm isolates the CFG change from its change of reveal order.

## Frozen evaluation

- ImageNet validation 50,000, one synthetic T2I prompt per image, seed 42.
- CFG 2.0, constant schedule, ten Heun steps, temperature 1, serialized reveals.
- Canonical CPU FP32 noise indexed by global image index and spatial position;
  identical prompts/noise across every model and order.
- Global batch 4096 on 16 NPUs, 256 samples per rank, padded length 512.
- BF16 model, FP32 VAE and flow integrator, KL16 scaling factor 0.2325.
- FID uses the existing ImageNet-val 50K reference and torch-fidelity Inception.
  IS uses ten fixed class-stratified splits, five images per class per split.
- Retain 64 evenly spaced PNGs, prompt/image/noise identity JSON and actual
  reveal-order/score traces per arm: 1,792 paired images in total.
- Runtime hashing and W&B remain disabled. Input identities use readable
  metadata, file size/mtime and exact post-load tensor-value checks.

CFG 2.0 was chosen on B-X0. This study is a common-setting comparison; it does
not claim the independently optimal CFG for every other checkpoint. These
validation-reference scores are project comparisons, not public leaderboard
scores. A fixed seed and IS split standard deviations do not establish
statistical significance of small differences.

## Confidence stability across architectures

The policy is unchanged from `docs/B_X0_ORDER_SWEEP.md`: score the next 16
Halton positions at t=0 and after an Euler proposal of dt=0.1, using

`mean((v_next - v)^2) / (mean(v^2) + 1e-8)`, with `v = vu + 2*(vc - vu)`.

Reveal lower scores first within each block; stable ties preserve Halton.
Proposals are discarded, and actual reveals start from the original noise.
Only captions and completed generated content are visible. Unrevealed target
latents and the original training sigma permutation do not enter the policy.

The implementation preserves each architecture's semantics:

| Model family | Probe and cache behavior |
|---|---|
| B-X0, A-X0, C, E | Static XT query condition; pending flow content uses the previous backbone X0 hidden. A retains strict content attention. |
| Legacy A and both legacy B controls | Static XT query condition; pending flow content retains the previous static query condition, as trained. |
| D Dynamic-XT | Conditional/unconditional caches remain separate. Both velocity probes refresh the XT hidden from the current x and t. Pending content uses the completed X0 hidden exactly once. |
| F position-wise | Probe velocities use the independent per-position MLP. The backbone remains cached; there is no flow content stream/cache. |

Non-confidence generation paths and repository sampling defaults are retained.
The new paths require a backbone cache, constant CFG != 1, and canonical
initial noise.

## Correctness and numerical controls

CPU tests exercise 36 positions (three candidate blocks), distinct prompts,
all eight implementation variants, both stability and an unchanged-order probe
control. A fresh decode in the observed order checks generated latents; tests
also check RNG preservation, training-sigma independence, D velocity refresh
counts and content commits, and F's absent flow cache.

The permanent 16-NPU development Notebook smoke assigns every model to one or
two ranks, checks every loaded checkpoint tensor after BF16 casting, runs all
256 reveals for Halton/stability/probe control, and tests a second candidate
block at the full per-rank batch. E also runs sequential. Per-model gates are
published only after that model's assigned workers, PNGs and traces pass.

BF16 changes to fused query matrix shapes can cause different outputs even
under the same forced order. The smoke records this difference and additionally
replays the observed order with identical probe shapes to isolate cache
semantics. B-X0's three formal 50K arms are checked against the previous study's
metrics and all 64 saved images per arm.

Random uses the existing native uniform `torch.randperm(256)` order, with no
confidence probes. The evaluator resets the order RNG to
`42 + batch_idx * 1009 + rank * 1000003` for every single-strategy arm. Matching
world size, batch partition and prompts therefore gives matching random orders
across models; saved order traces are compared explicitly. Canonical spatial
noise remains independent of this order RNG.

## Execution and artifacts

Preparation: `scripts/prepare_unified_matrix_sweep.py --include-random`. Each arm uses an
independent one-node, 16-NPU Inspire Job in the user-selected random-order
language modeling project. The controller keeps up to 12 concurrent Jobs
(192 NPUs), starting long D arms as soon as their smoke passes. It uses one
blocking waiter per Job, admits only validated models, and fills freed slots.
A controller or monitor timeout does not cancel or duplicate healthy Jobs.

The current root is
`output/evaluation/ablation-matrix/cfg2-heun10-order-20260909-r1/`.
Raw scores, launch evidence, input audit, smoke controls, CSV, standalone PNG/PDF
plots and per-arm paired samples stay there. The CPU watcher
`scripts/report_sampling_sweep.py` updates the homepage's dedicated matrix view.
Final publication requires all 28 full evaluations and the complete metric,
input, PNG, trace, pairing and B-X0 reproduction audits.

The user added Random while the initial 19 arms were running. The actual study
therefore preserves its pre-extension protocol/state in
`launch/extensions/random/`, then appends nine new tasks under the state lock.
All original code, weights, per-model configs, fixed settings and arm identities
remain unchanged. The existing controller admits the added tasks without
restarting the active evaluations.
