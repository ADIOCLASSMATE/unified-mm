# B：CFG / Heun 扫描

模型为 B step 95415 final EMA。保留结果位于 `output/evaluation/unified-b-x0content-0p6b/sweeps/cfg-heun-20260908-r1/`；当前主矩阵采用其中的最低 FID 配方 CFG 2 / Heun 10。

## 搜索范围

| 阶段 | 设置 | 选择规则 |
| --- | --- | --- |
| CFG | Heun 10；CFG 1.0–6.0，间隔 0.5 | 分别选最低 FID、最高 IS |
| 边界扩展 | 最优在边界时，向外增加 4 个间隔 0.5 的点；范围 0–12 | 到 12 仍为边界最优时记录待扩展 |
| Heun | 对两个入选 CFG 测 5/10/20/50/100 步，复用已完成的 10 步 | 分别报告质量与耗时 |

这是分阶段搜索。调参与计分共用 ImageNet-val 50K。

## 固定条件

每组使用相同 50000 条 prompt 和 canonical 初始噪声，seed42、Halton、BF16 模型、FP32 VAE、16 ranks、全局 batch4096。FID 使用项目 ImageNet-val reference；IS 为十个 synset 分层 split，每类每 split 五张，标准差为 split 间总体标准差。

每组保留 64 个固定样本，索引为 `floor(i * 49999 / 63)`，i=0..63。PNG 配套 JSON 记录图像 ID、prompt、索引和噪声 seed。

## 执行

准备：`scripts/sweep_unified_t2i_sampling.py prepare`，传入 `--model-source`、`--config`、`--source-repo`、`--baseline-metrics`、`--platform-json`、`--output-dir`。已有结果的 `launch/source`、基线 `metrics.json` 和 `launch/platform.json` 可作为输入。

```bash
python3 scripts/submit_unified_t2i_sweep.py   --output-dir <new-directory> --name-prefix <unique-prefix>
python3 scripts/report_sampling_sweep.py --output-dir <new-directory> --watch
```

每组独立使用随机序项目的 16-NPU Job；准备阶段冻结源码，每次提交检查资源。平台约定见 [INSPIRE](../INSPIRE.md)。本次 sweep 关闭 runtime hashing 和 W&B。

## 产物

`protocol.json`、`state.json`、`results.csv`、`summary.json`、`audit.json`；曲线为 `cfg-sweep.{png,pdf}`、`heun-sweep.{png,pdf}`。各组指标和图片位于 `arms/`，运行入口位于 `launch/`。

在 `configs/protocols/evaluation_report.json` 登记结果，运行 `scripts/build_evaluation_report.py` 更新 [评测首页](../output/evaluation/index.html)。页面提供表格、曲线和同步样本对比。
