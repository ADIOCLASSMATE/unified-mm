# 训练中下游验证

`training_downstream_v1` 的 100B 配方默认每 10,000 optimizer steps 执行；当前 31,800-step 消融每 3,180 步执行，EMA decay 按预算缩短为 0.9997。所有 rank 参与，使用固定全局样本和当前 EMA，结果保存在 `output/evaluation/training-validation/<run>/`。

| 任务 | 样本 | 指标 |
| --- | --- | --- |
| 八项文本 benchmark | 全部 34,507 题 | 各任务主指标与算术均值 |
| ImageNet 分类 | 每类 2 张，共 2,000 图，全部 1,000 类候选 | 校准 Top-1 / Top-5 |
| ARO Relation | 分层抽样 512 条 | 严格胜率、平局率、分差 |
| SugarCrepe | 分层抽样 512 条 | 同上 |

文本使用 [评分协议](EVALUATION_PROTOCOL_AUDIT.md)。ImageNet 从选中的 2,000 图估计语言先验；ARO / SugarCrepe 使用三张固定 null image、alpha=1、MC16，顺序模型 MC1。正式完整图文评测使用 MC64。

## 执行

固定清单按 `indices[rank::world_size]` 分片，posterior 与 order seed 由样本身份决定。首次准备的 CPU 数据与 token 跨轮缓存。分片 FP32 EMA 临时复制入现有模型，完成后恢复训练权重、buffers、模式和 RNG；设备评分采用训练 dtype。

工作预算 540 秒，另留 60 秒用于汇总与恢复。结果分别记录 `complete`、`within_time_budget`、已处理样本数和耗时。

同一验证点先用当前训练权重执行 [统一 loss](UNIFIED_LOSS_VALIDATION.md)，短预算配置随后用 raw 权重生成 16 张固定 prompt/种子的图像，最后用 EMA 执行下游评分。生成统一采用 CFG 3.5、Heun10，并单独记录耗时；S2 使用完整模型刷新，Z 复用固定 backbone 条件，B+S2 使用 KV cache。统一 loss 有独立计时；T2I/I2T loss 与分类共用 2,000 张图像清单。

## 文件

```text
training-validation/<run>/
├── validation_summary_step_<N>.json
├── validation_unified_loss_metrics_step_<N>.json
├── downstream_validation/step-<N>/
│   ├── subset.json
│   └── summary.json
└── validation_generation/step-<N>/
    ├── overview.png / index.html
    ├── <prompt-images>.png
    └── summary.json
```

Tracker 使用 `val/downstream/<task>`、`val/downstream/text_mean` 和完成／耗时字段。FID/IS 与完整检索使用独立 [完整评测入口](EVALUATION_STRUCTURE.md)。

## 计时

```bash
bash script/selfless/benchmark_training_validation_ascend16.sh \
  <HF-EMA-or-sharded-EMA-checkpoint> <timing-output-directory>
```

16×910B 实测：F final EMA 的下游评分为 419.51 秒；depth16 在两轮训练验证中为 388.43 / 382.63 秒，depth30 为 387.26 / 389.71 秒。三组均完成全部 11 项任务。

F 的计时记录位于 `output/evaluation/diagnostics/validation-timing-16npu-20260906-f-r3/`；深度实验的整轮计时见 [统一 loss](UNIFIED_LOSS_VALIDATION.md)。
