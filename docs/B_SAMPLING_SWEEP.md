# B CFG / Heun sweep

B is the formal model defined in [EXPERIMENTS.md](EXPERIMENTS.md).

This study evaluates the final step-95415 EMA from
`output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/hf_model-final-ema`.
The retained run is
`output/evaluation/unified-b-x0content-0p6b/sweeps/cfg-heun-20260908-r1`.

The first phase evaluates CFG 1.0–6.0 in 0.5 increments at 10-step Heun.
A winning endpoint extends the range with four additional 0.5-spaced points
in parallel, clipped at zero and the upper guard. The upper guard is 12.0; an optimum on that guard is reported as
unresolved and requires extension. Minimum FID and maximum IS are selected
separately. The second phase evaluates both selected CFGs at Heun steps
5, 10, 20, 50, and 100, reusing their complete 10-step results. This is a
sequential search, not an exhaustive two-dimensional grid.

Every combination evaluates all 50,000 ImageNet validation examples with
the same captions and canonical initial noise, seed 42, spatial Halton order,
BF16 model, FP32 VAE, global batch 4096, and 16 ranks. IS uses ten
synset-stratified splits, each with five examples from every class. Its
reported standard deviation is the population deviation across those splits.
FID uses the existing ImageNet-val reference and is comparable within that
protocol. Selection and scoring use the same validation set.

Each combination retains 64 images covering the full evaluation ordering.
The fixed indices are `floor(i * 49999 / 63)` for `i=0..63`. Each PNG has a
JSON record with its global index, ImageNet image ID, actual prompt, and
canonical noise seed. Export happens after metric accumulation and consumes
no random numbers. The final audit verifies image decoding, exact coverage,
and identical sample/prompt/noise records across all combinations.

Each parameter combination owns an independent 16-NPU Job in
`随机序语言建模-统一自回归与掩码扩散的随机顺序生成框架`, using the
`昇腾卡公共空间` workspace and `910B资源` group. The initial grid contains
11 Jobs. The controller records a dry-run and submitted resources for each
Job, spaces submissions by at least 30 seconds, and waits on each Job with
one blocking CLI wait. Failed arms can resume independently; resource/API
errors preserve other running Jobs. Model code is frozen under `launch/source`.
Runtime hashing and W&B remain disabled.

If a terminal instance has neither NPU devices nor a host driver library,
the controller retains its environment and scheduling events, excludes its
actual assigned node, and retries the affected arm. These infrastructure
failures do not consume the three evaluation-attempt limit. Exclusion is
bounded to eight nodes; API or controller failures preserve healthy Jobs.
The exact exclusions are recorded in the retained protocol and job manifest.

To prepare another run, pass a new output directory to
`scripts/sweep_unified_t2i_sampling.py prepare` together with `--model-source`,
`--config`, `--source-repo`, `--baseline-metrics`, and `--platform-json`.
The retained run's `launch/source`, baseline arm's `metrics.json`, and
`launch/platform.json` supply those last three inputs. Preparation creates
the frozen source, submission script, CLI path context, and input size/mtime
audit. It defaults to 64 saved images and re-evaluates the baseline to retain
the same image coverage. It submits no GPU work.

Launch the prepared study with
`python3 scripts/submit_unified_t2i_sweep.py --output-dir <new-directory> --name-prefix <unique-prefix>`.
Live resource checks still run before every submission. Register the selected
study directory in the report configuration before running
`scripts/report_sampling_sweep.py --output-dir <new-directory> --watch` in
a CPU environment with Pillow and Matplotlib. The current study's isolated
plotting dependencies are under `launch/report-deps`; the main `.venv` was
not changed to install them.

The result files are `protocol.json`, `state.json`, `results.csv`,
`summary.json`, `cfg-sweep.{png,pdf}`, `heun-sweep.{png,pdf}`, and
`audit.json`. Per-arm metrics, images, and prompt records live under `arms/`.
Platform submission and recovery evidence lives under `launch/`.

The selected run is registered in
`configs/protocols/evaluation_report.json`. Running
`python3 scripts/build_evaluation_report.py` rebuilds
`output/evaluation/index.html` from validated raw results. Its sampling tab
shows the table, curves, independently selected FID/IS optima, and an image
comparison with synchronized sample selection. Incomplete sweeps have no
final optimum. The model table defaults to CFG=2.0 / Heun=10 and retains
CFG=3.5 as an explicit historical protocol selection.
