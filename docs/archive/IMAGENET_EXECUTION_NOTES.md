# 早期 ImageNet 执行记录

这些记录保留既有实验的硬件与配方，不作为当前 Unified-MM 的默认值。
当前资源约定见 [INSPIRE.md](../../INSPIRE.md)，训练入口见 [README](../../README.md)。

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
  to H100 work in `分布式训练空间`. Do not keep duplicate runnable jobs in
  both projects.
- Image: `docker.sii.shaipower.online/inspire-studio/dev-wjx:v-2.1`.

### Final ImageNet-100 training hyperparameters

- Canonical conclusion:
  `docs/IMAGENET100_HYPERPARAMETER_CONCLUSION.md`.
- Global batch size: `1024` on 64 Ascend NPUs (`4 x 16`), with per-rank batch
  `16` and gradient accumulation `1`.
- Final coupled learning rates: Backbone/Special-token `30e-5` and
  Flow-head/Projector `4e-5`.
- The selected 80-epoch checkpoint achieved FID `24.057695924272537` and IS
  `71.04673767089844` under the canonical 10,000-sample, 100-step evaluation.
- Sweep manifests、旧 checkpoint、评测产物、ImageNet-100 可执行配置和一次性
  launch 资产均已删除；仓库只保留结论。

### Final ImageNet-1K synthetic Caption/T2I joint training

- The selected default is
  `configs/selfless/imagenet1k_caption_joint_10ep_ascend16_b1024.yaml`, launched
  by `script/selfless/pretraining_imagenet1k_caption_joint_ascend16.sh`.
- It starts from the completed class-conditioned EMA, uses `2e-5` for the
  backbone, special-token rows, image projector, and flow head, with
  `lambda_text=0.05` and `lambda_image=1.0`.
- The fixed contract remains 16 NPUs, per-rank batch 16, GA 4, global batch
  1024, 10-epoch WSD, 12,020 optimizer steps, fixed seeds/splits/data order,
  six synthetic captions per train image, and twelve aligned T2I prompts.
- The sweep evidence, generation metrics, T2I-regression caveat, and source
  report hashes are retained in `docs/IMAGENET1K_CAPTION_JOINT_CONCLUSION.md`.
  Sweep/probe/control checkpoints and one-use orchestration assets are deleted.

### ImageNet-1K T2I-only 80-epoch training

- The three formal configurations and launchers are paired as follows:
  baseline uses
  `configs/selfless/imagenet1k_t2i_baseline_80ep_ascend_64npu_bs1024.yaml`
  with
  `script/selfless/pretraining_imagenet1k_t2i_baseline_ascend_64npu_bs1024_80ep.sh`;
  position-wise flow head uses
  `configs/selfless/imagenet1k_t2i_positionwise_head_80ep_ascend_64npu_bs1024.yaml`
  with
  `script/selfless/pretraining_imagenet1k_t2i_positionwise_head_ascend_64npu_bs1024_80ep.sh`;
  sequential sigma uses
  `configs/selfless/imagenet1k_t2i_seq_sigma_80ep_ascend_64npu_bs1024.yaml`
  with
  `script/selfless/pretraining_imagenet1k_t2i_seq_sigma_ascend_64npu_bs1024_80ep.sh`.
- Each run continues from its matching completed 800-epoch class-conditioned
  EMA. Position-wise preserves `architecture_variant: positionwise_selfless`
  with random image-sigma order and `spatial_halton` generation; sequential
  sigma preserves the contextual flow head with sequential image-sigma and
  generation order. Never cross-load another variant's EMA.
- All three train only T2I image flow: `caption_sequence_modes: ["t2i"]`,
  `lambda_text=0`, and `lambda_image=1`. None uses the joint-caption
  checkpoint.
- The fixed contract is 64 Ascend NPUs (`4 x 16`), per-rank batch 16, GA 1,
  global batch 1024, 1,202 optimizer steps per epoch, 80 epochs, and 96,160
  total optimizer steps. WSD uses 8 warmup + 48 stable + 24 decay epochs; all
  trainable parameter groups use learning rate `2e-5`.
- Training uses the aligned seek index at
  `public/datasets/imagenet1k_synthetic_v1/indexed/train/manifest.json` and its
  twelve synthetic T2I prompts per image. Every image receives a deterministic
  random starting offset and rotates without replacement through all twelve
  prompts before reuse; consecutive epochs never reuse its prompt. Epoch state
  is shared with DataLoader workers and restored by the exact-resume cursor.
- Canonical output roots are the matching
  `output/selfless-flow-imagenet1k-t2i-{baseline,positionwise-head,seq-sigma}-ascend64-b1024-80ep`
  directories.
- The canonical T2I generation launchers are
  `script/selfless/evaluate_imagenet1k_t2i_baseline_ascend16.sh`,
  `script/selfless/evaluate_imagenet1k_t2i_positionwise_head_ascend16.sh`, and
  `script/selfless/evaluate_imagenet1k_t2i_seq_sigma_ascend16.sh`. Each uses 16
  Ascend NPUs, the matching final EMA HF export, 50,000 validation-image
  synthetic T2I prompts, 10-step Heun, CFG 3.5, canonical paired noise, and
  the frozen project ImageNet validation moments for FID/IS. Baseline and
  position-wise use `spatial_halton`; sequential sigma uses `sequential`. Each
  result is retained below its run root at
  `generation-evaluation/heun10/t2i-fid-is/metrics.json`. The former 100-step
  comparison result remains at `generation-evaluation/t2i-fid-is/metrics.json`.
- Formal T2I comparisons use only independent ImageNet-val prompts, 50,000
  generated samples, frozen original ImageNet-val moments, and ten
  synset-stratified IS partitions. Results made with training-image prompts or
  row-contiguous IS partitions are not part of the repository protocol.

### ImageNet-1K T2I-only 400-epoch training

- The 400-epoch experiments are independent full training runs initialized
  from the same three matching 800-epoch class-conditioned EMA exports as the
  80-epoch experiments. They do not resume the already-decayed 80-epoch
  optimizer/scheduler state and never overwrite the retained 80-epoch runs.
- Their configs are the matching
  `configs/selfless/imagenet1k_t2i_{baseline,positionwise_head,seq_sigma}_400ep_ascend_64npu_bs1024.yaml`
  files; their launchers are the matching
  `script/selfless/pretraining_imagenet1k_t2i_{baseline,positionwise_head,seq_sigma}_ascend_64npu_bs1024_400ep.sh`
  files.
- The shared contract remains 64 Ascend NPUs (`4 x 16`), per-rank batch 16,
  GA 1, global batch 1024, learning rate `2e-5`, T2I-only image flow, and the
  deterministic twelve-prompt rotation. Each run uses 1,202 optimizer steps
  per epoch and 480,800 total steps. WSD is extended proportionally to 40
  warmup + 240 stable + 120 decay epochs.
- Canonical output roots are the matching
  `output/selfless-flow-imagenet1k-t2i-{baseline,positionwise-head,seq-sigma}-ascend64-b1024-400ep`
  directories.

### Formal ImageNet-1K 800-epoch pretraining

- Contract: `docs/IMAGENET1K_800EP_PRETRAINING.md`.
- Config:
  `configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024.yaml`.
- One-Job/one-launcher entry:
  `script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep.sh`.
- Use 64 Ascend 910B NPUs (`4 x 16`), per-rank batch 16, GA 1, and global
  batch 1024. Do not increase the card count without a new LR/batch contract.
- Fixed coupled LRs are Backbone/Special-token `30e-5` and
  Flow-head/Projector `4e-5`.
- The exact run is 1,251 optimizer steps per epoch and 1,000,800 steps total.
  WSD is 5 warmup + 595 stable + 200 decay epochs.
- EMA is FP32 rank-sharded, starts at step zero, and uses decay `0.9999` so its
  half-life remains close to the epoch-scale averaging selected on ImageNet-100.
- The launcher supports exact recovery through `RESUME_FROM=<checkpoint-dir>`.
- Full ImageNet-1K KL16 posterior cache, local Inception weights, and official
  ImageNet-val 50K real moments must exist before the launcher preflight passes.
- The completed full cache is
  `public/datasets/imagenet_full/vae_posterior_mar_kl16/posterior_stats_imagenet1k_train_fp16.pt`
  (`[1281167, 256, 32]`, FP16, SHA256
  `3fdb1341e682962bb3f10ff3b794d67e4cc8a46490a2c621b95d60e0d4b1fb82`).
  Rebuild it on `dev-wjx-ascend` with the single launcher
  `script/selfless/prepare_imagenet1k_cache_ascend16.sh`.
- The fixed torch-fidelity Inception weights are
  `public/models/torch-fidelity/weights-inception-2015-12-05-6726825d.pth`
  (SHA256
  `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`).
- The locally computed ImageNet-val moments are
  `public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt`
  (50,000 images, 1,000 classes, 2,048 features, SHA256
  `7eb801931347be917b34077c5ab94c4c7c6b9c42bbe40b442f39947d9bb133`).
  They use the same fixed Inception weights and preprocessing as evaluation;
  rebuild them with
  `script/selfless/prepare_imagenet1k_fid_stats_ascend16.sh`.
- The complete 16-NPU train/validation/evaluation smoke passed on
  `dev-wjx-ascend`. Run it with the single launcher
  `script/selfless/smoke_imagenet1k_train_val_eval_ascend16.sh`; the retained
  conclusion is
  `public/datasets/imagenet_full/preparation/train_val_eval_smoke_report.json`.
  Its 16-sample FID/IS values are pipeline diagnostics, not paper metrics.
- W&B remains enabled. `WANDB_MODE` defaults to `offline` and may be set to
  `online` for a platform environment with working W&B credentials/network.
- The final EMA HF model completed the project-formal ImageNet-val
  evaluation on 2026-08-19: 50,000 samples, deterministic canonical pairing,
  CFG 3.5, 100-step Heun, and the frozen val moments. This historical result
  is same-protocol-only, not ADM/DiT leaderboard-comparable. The result is
  FID `18.996944032440638` and IS `450.06854248046875 ± 4.259687366514454`.
  The retained result is
  `output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-fid-is/metrics.json`
  (SHA256 `d6f066bffad8a4e3032ccc3aac4b9445e9589e21691c3519bdf5e2e722d506f8`).

### ImageNet-1K sequential image-sigma ablation

- This run keeps the formal 64-NPU, global-batch-1024, 800-epoch contract
  unchanged while setting `dataset.params.image_sigma_order: sequential`.
- Image sigma follows serialized latent-token order with strict
  `sigma[kv] < sigma[q]`: the diagonal is hidden, and EOI is ordered before
  every image token so image queries can still see EOI.
- Config:
  `configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_seq_sigma.yaml`.
- Launcher:
  `script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep_seq_sigma.sh`.
- Training-time validation and offline evaluation both use the fixed
  `sequential` generation strategy; the model asserts the exact 1..256 order
  at runtime for a complete sequential generation.
- The 16-NPU one-step train/validation/16-image evaluation smoke passed on
  `dev-wjx-ascend`. The retained report is
  `public/datasets/imagenet_full/preparation/seq_sigma_train_val_eval_smoke_report_v2.json`;
  run the retained one-off smoke launcher with
  `script/selfless/smoke_imagenet1k_seq_sigma_train_val_eval_ascend16.sh`.
- The sequential-image-sigma final EMA completed its project-formal 50K
  evaluation on 2026-08-24 with deterministic canonical pairing, CFG 3.5,
  100-step Heun, the required `sequential` generation strategy, and the frozen
  ImageNet-val moments. Its FID is `8.633703493770327`, IS is
  `247.52949981689454 ± 3.36128814432888`, and generation throughput is
  `4.224009451221147 samples/s` on 16 NPUs. The retained result is
  `output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-seq-sigma-fid-is/metrics.json`
  (SHA256 `443a67f83ee365065eac074de45eadfb9ac00ff4ef8ed1cd2c4dcfec397e6458`).
  Baseline and position-wise formal results use `spatial_halton`, so their
  metric deltas against this result also include the inference-order change;
  do not interpret those deltas as an isolated training-sigma ablation.

### ImageNet-1K architecture variants

- The position-wise control preserves the historical formal 64-NPU,
  global-batch-1024, 800-epoch optimization/data contract.
- The position-wise-head control keeps the random-sigma selfless two-stream
  Qwen backbone and replaces only the contextual flow head with a vectorized
  MAR/NextStep-style AdaLN MLP. The head has no cross-token attention, content
  cache, previous-latent input, or image-position input.
- Position-wise-head config and launcher:
  `configs/selfless/imagenet1k_class_pretrain_800ep_ascend_64npu_bs1024_positionwise_head.yaml`
  and
  `script/selfless/pretraining_imagenet1k_class_ascend_64npu_bs1024_800ep_positionwise_head.sh`.
- Ablation D is the new Dynamic-XT implementation on unified baseline B. It
  keeps B's `xlnet_content_diagonal` attention contract, contextual flow head,
  data/schedule/LR contract, and `image_flow_batch_mul: 4`. One X0 content
  stream is computed while four independent RF states form a `4B` XT query
  stream; predicted-image queries use
  `image_token_embedder(x_t) + backbone_flow_time_embedder(t)`.
- D's model and generation behavior live in the dedicated files
  `models/modeling_model/modeling_selfless_flow_dynamic_xt.py` and
  `models/modeling_model/modeling_selfless_flow_dynamic_xt_generation.py`.
  During Heun generation every predictor/corrector evaluation rebuilds the XT
  query while reading fixed X0 K/V without committing XT to the cache.
  The flow query AdaLN condition is the refreshed backbone XT hidden. The flow
  content AdaLN condition is instead the backbone X0 hidden: training computes
  it once at batch B and repeats it for the four RF states, while generation
  obtains the previous token's X0 hidden from the fused cache-commit/current-
  query forward. The retired shared/static condition behavior has no switch.
- The formal D launcher is
  `script/selfless/pretraining_unified_ablation_d_on_b_0p6b_formal_ascend64.sh`;
  the independent evaluation entry is
  `scripts/evaluate_dynamic_xt_single_stream_fid_is.py`.
- The minimal 64-NPU startup guard was validated for 10 optimizer steps in
  `umm-d-on-b-startupguard10-v13-64-s42-r1`: the first-boundary guard passed,
  disabled itself immediately, and step 10 remained finite at loss `0.6762`
  and `4.1813 s/step`. Fully masked attention rows are a normal shared input
  pattern across arms and are not treated as the D failure cause.
- The position-wise-head retained smoke report is
  `public/datasets/imagenet_full/preparation/positionwise_head_smoke_report.json`.
- The position-wise-head final EMA completed the project-formal 50K
  evaluation on 2026-08-22 with deterministic canonical pairing, CFG 3.5,
  100-step Heun, `spatial_halton`, and the frozen val moments. Its
  FID is `18.2875253165069`, IS is
  `438.955712890625 ± 3.8873937344382083`, and generation throughput is
  `18.023885168137856 samples/s` on 16 NPUs. Relative to the same-protocol
  baseline final EMA, FID improves by `0.7094187159337366` (`3.734%`), IS
  decreases by `11.112829589843727` (`2.469%`), and throughput is `4.263x`.
  The retained result is
  `output/selfless-flow-imagenet1k-class-ascend64-b1024-800ep-positionwise-head-fid-is/metrics.json`
  (SHA256 `a5850bc1072db7e5d5480ca02f6be33a849fb834682137e914bb1747fa405f83`).
- The retired ImageNet-only, A-based Dynamic-XT recipe must not be resumed or
  used for D. D starts fresh from the same Qwen3-0.6B weights as B.
