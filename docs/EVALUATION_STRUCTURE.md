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
```

第一条更新网页和原始 JSON/CSV，第二条同时导出 PNG/SVG。报告依赖位于 [evaluation_report_requirements.txt](../scripts/evaluation_report_requirements.txt)。构建读取已经完成的结果与训练日志，保留未完成项的空值。

网页入口为研究总览、模型评测、采样与解码、训练与验证、定性样例、数据与协议。模型筛选按正式 A/B、C–F、历史实验分组。
