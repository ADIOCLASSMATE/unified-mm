# 评测与产物

[评测总览](../output/evaluation/index.html)展示指标、采样实验、训练曲线和 T2I / I2T / 文本样例。结果选择由 [evaluation_report.json](../configs/protocols/evaluation_report.json)维护。

## 目录

```text
output/evaluation/
├── index.html / summary.json / selection.json
├── unified-*/                         # 模型及 checkpoint 评测
├── qualitative/<run>/                 # 逐样本输出、报告、ZIP
├── training-validation/<run>/         # 训练验证
├── training-loss/                     # 原始曲线 JSON/CSV、PNG/SVG
├── data-provenance/                   # 数据来源与模板
├── research/                          # 表征研究
├── ablation-matrix/                   # 跨模型采样矩阵
├── comparisons/                       # 指标对照
└── diagnostics/ / audits/ / migrations/
```

训练权重、优化器、随机状态和训练日志位于 `output/<run>/`；`output/evaluation-checkpoints/` 存放历史评测输入权重。

## 完整评测

```bash
EVAL_PROFILE=formal \
  script/selfless/evaluate_unified_native_full_checkpoint_ascend16.sh \
  /path/to/hf_model-final-ema output/evaluation/<model>/<evaluation-name>
```

协议为 [unified_full_evaluation_ascend16.yaml](../configs/protocols/unified_full_evaluation_ascend16.yaml)。当前 final 输入为完整 FP32 EMA 导出，评分按架构选择 Selfless 同位置或 S2 / 文本 AR 对齐方式。

S2 的验证样例按选中的图像裁出 batch，并在每次 ODE 计算中刷新整图，不使用 backbone KV cache。候选评分使用单次确定性 AR 计算；正式汇总按 S2 契约接受 `mc_samples=1`，Selfless 保持 MC64。

| 项目 | 文档 |
| --- | --- |
| ImageNet 分类、COCO/Flickr 检索、SugarCrepe/ARO | [图像理解](PRETRAINING_NATIVE_EVALUATION.md) |
| 文本、图文评分公式与样本 | [评分协议](EVALUATION_PROTOCOL_AUDIT.md) |
| ImageNet 50K FID/IS、跨模型换序 | [固定参数矩阵](UNIFIED_MATRIX_CFG2_ORDER.md) |
| GenEval、DPG-Bench、MJHQ-30K | [官方生成评测](OFFICIAL_T2I_BENCHMARKS.md) |

主比较矩阵采用 CFG=2、Heun=10：E 使用 Sequential，其余使用 Halton。网页另保留 CFG=3.5 的完整评测结果。各结果的采样参数保存在对应 manifest。

## 训练验证

```text
training-validation/<run>/
├── validation_unified_loss_metrics_step_<N>.json
├── validation_summary_step_<N>.json
└── downstream_validation/step-<N>/
    ├── subset.json
    └── summary.json
```

统一 loss 使用当前训练权重，下游指标使用 EMA。历史图片、caption 和 `validation_metrics_step_*.json` 保存在同一 run 目录。训练阶段的样本与时点见 [下游验证](TRAINING_DOWNSTREAM_VALIDATION.md)。

`training-loss/curves.csv` 保存训练点，`validation.csv` 保存验证点，`curves.json` 保存两者及协议。图中平滑只作用于训练显示；不同验证聚合协议分段绘制。

## 更新网页

```bash
python3 scripts/build_evaluation_report.py
python3 scripts/build_evaluation_report.py --plots
python3 scripts/build_evaluation_report.py --unified-plots
```

第一条更新网页和原始 JSON/CSV，第二条同时导出全部图表；第三条仅额外更新 B / T2I-only 的 CFG 折线图和 FID / IS 权衡图（PNG、PDF、SVG）。报告依赖位于 [evaluation_report_requirements.txt](../scripts/evaluation_report_requirements.txt)。构建读取已经完成的结果与训练日志，保留未完成项的空值。

网页入口为研究总览、模型评测、Unified 训练消融、采样与解码、训练与验证、定性样例、数据与协议。模型筛选按正式 A/B、C–F、历史实验分组。

S2-single 的完整 final FP32 EMA 评测选取 `unified-s2-single-0p6b/final-ema-20260914-r4`，其中复用同一 checkpoint 的 r1 文本结果。研究总览和“模型评测 → B / S2 对照”（`#matrix/showo2`）展示 CFG=3.5、Heun=10 的生成结果及全部理解、文本指标；S2 的候选评分为整图可见的确定性 next-token AR（MC1）。尚无 CFG=2.0 结果的模型在该协议下保留空值。

S2-single 的 CFG sweep 位于 `unified-s2-single-0p6b/sweeps/cfg-1-6-20260915-r1`：CFG=1.0–6.0、间隔 0.5，固定 Heun=10 / seed 42 / ImageNet-val 50K，复用 CFG=3.5 完整评测。两个 16 卡 Job 分别评测 1.0–3.0 和 4.0–6.0，沿用 r4 的冻结源码及配置。B / S2 对照页每分钟读取 `comparisons/s2-single-cfg-sweep/comparison.json`；FID / IS 图及 CSV 位于同一目录，未完成档位断开曲线。S2 全局 batch 2048，B 为 4096，IS 阴影表示十个类别分层 split 的标准差。使用包含 Matplotlib 的 CPU Python 执行 `scripts/report_s2_cfg_sweep.py --watch` 可持续核验并更新图表；任务全部完成或结束后自动退出。

对于定性样例清单中尚未包含的模型，在 `evaluation_report.json` 的 `models` 条目中填写已登记的 `run`、`checkpoint_step` 和评测 `root`。构建从完整评测汇总与 checkpoint 元数据读取模型身份，并核对最终 EMA 和步数；固定输入定性样例仍按实际已完成的清单展示。

### B / S2-single 配对定性与速度（2026-09-15）

`qualitative/b-s2-single-final-ema-20260915-r1/` 复用 9 月 8 日画廊的全部固定输入与噪声：每模型 128 张 T2I、64 个 I2T、64 条文本续写，final FP32 EMA step 95415。B CFG=2.0，S2-single CFG=1.5，Heun=10；S2 使用整图 flow 和无 KV cache 的 AR 文本路径。全部图片与逐条输出保留。此为定性样例，不包含 GenEval 评分。

同一 16 卡 910B Job，每卡 batch=8，BF16 模型。图像生成每卡预热一批、正式测量三次，调用前后同步设备；平均秒/图为全部 48 次批次耗时之和除以 384。B 为 2.1847 秒/图，S2-single 为 0.2680 秒/图；各卡中位吞吐的中位数分别为 0.4583、3.7564 图/秒，S2/B 为 8.20 倍。计时包含输入搬运，排除模型加载、VAE 解码和写盘，不是单张请求延迟或整机实测吞吐。VAE/PNG 保存另记一次耗时，包含首次调用开销，不用于预热后的速度比较。

配对目录提供 `speed.html`、`speed.json`、`speed.csv`（96 条原始计时）及含图片和计时的 ZIP。复现汇总：`python3 scripts/report_qualitative_speed.py output/evaluation/qualitative/b-s2-single-final-ema-20260915-r1`。总览选择 `qualitative/unified-all-final-ema-with-s2-20260915-r1`：保留既有模型结果和 B CFG=3.5，追加 S2 CFG=1.5；“定性样例 → B / S2”可筛选，速度链接另提供 B CFG=2.0 与 S2 CFG=1.5 的配对页面。合并清单的 `reused_model_sources` 记录每个模型的原始画廊。

Unified 训练消融（`#unified-training`）比较正式 B 与 I2T-only、T2I-only、text-only 的对应任务结果。由 `evaluation_report.json` 的 `unified_training_ablation` 选择原始 final EMA 评测，构建时核对 checkpoint、完成状态、样本覆盖和评分协议。展示核心指标、文本八任务、双向检索明细与内部诊断，差值统一为 B − only；可下载 `comparisons/unified-training-ablation/comparison.json`。I2T/T2I 匹配对应任务曝光量，text-only 沿用已有约 100B 文本对照；生成按 CFG=1.0–6.0、间隔 0.5 分别比较，固定 Heun=10。T2I-only 复用已完成的 1.0 / 2.0 / 3.5，其余八档单独评测。

生成板块展示 FID、IS 折线图、FID / IS 权衡曲线、逐 CFG 表格，以及每个模型独立选择最低 FID 和最高 IS 的结果；两个最优值不视为同一工作点。图表及 `cfg-sweep.csv` 位于 `comparisons/unified-training-ablation/`。IS 的阴影为十个类别分层 split 的标准差，不是置信区间。未完成档位保留空值且折线断开；普通构建如检测到新数据，先隐藏旧图，执行 `--unified-plots` 后生成与当前数据一致的图表。

“采样与解码消融 → Flow head scale 消融”（`#sampling/flow-head-scale`）比较 B contextual head 与 F position-wise AdaLN MLP 的 depth 8 / 16 / 30，分别列出 CFG=1.0、2.0、3.5 的 Halton / Heun-10 结果及 head / 全模型参数量。来源在选择文件的 `flow_head_scale` 中显式登记，构建时核对 final EMA step、head 配置、参数量与评分协议；训练未完成或未评测的单元格保留空白。后续收齐 F 结果时更新对应 `training_status` 和 `metrics` 路径后重新构建。

### ARO 文本先验口径（2026-09-15）

ARO Relation / Attribution 的主指标统一为 `conditional_pairwise.win_rate`，使用含文本先验的平均 token 条件似然。运行 `python scripts/update_aro_conditional_results.py --apply` 从保存的候选分数恢复并同步各级汇总，无需模型重跑；随后运行 `python scripts/build_evaluation_report.py` 更新网页。迁移审计、旧汇总备份和更新副本位于 `output/evaluation/migrations/20260915-aro-conditional/`，原始预测不变。
