# V5 产物与检查记录

记录日期：2026-09-07。七组模型、14 个路径/提示设置、41 条读出曲线的冻结前向与分析已完成。范围见 [协议](CROSS_MODEL_GEOMETRY_PROTOCOL_V5.md)，结论见 [结果](CROSS_MODEL_GEOMETRY_V5_RESULTS_20260907.md)，重算见 [复现入口](CROSS_MODEL_GEOMETRY_V5_REPRODUCE.md)。

## 覆盖数量

| 设置 | 层对×读出行数 | 固定／dev端点记录数 |
| --- | --- | --- |
| B native / bare / neutral | 各90 | 各120 |
| F native / bare / neutral | 各90 | 各120 |
| DINOv2＋Qwen / MAE＋Qwen | 各90 | 各120 |
| SigLIP 内容读出 | 58 | 80 |
| SigLIP 原生 pooler | 1 | 40 |
| JanusFlow 理解 / 生成 | 78 / 104 | 120 / 160 |
| Show-o2 理解 / 生成 | 90 / 120 | 120 / 160 |
| 合计 | 1171 | 1640 |


共 37472 条映射记录、328 个源 dev 选择、984 个层搜索置换检验、1260 组配对对照、1680 条扰动结果及 48 条事后残差分解。保存 25 对正式 PNG/PDF；V3/V4/V5 合计 34 项单元测试通过。

## 检查入口

| 内容 | 覆盖 | 记录 |
| --- | --- | --- |
| 来源 | 模型类、权重逐参数加载、revision、B/F 训练配置 | [模型来源](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/model-sources-and-training.json)、[资产](../output/evaluation/research/cross-model-geometry-v5-20260907/asset-manifest.json) |
| 数据 | 32000 ImageNet 图、11776 COCO 场景、868 ARO 原图及全部文本；96 图像素检查 | [样本与输入](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/samples-and-preprocessing.json) |
| 层与读出 | 1171 个层对 × 读出组合 | [比较契约](../output/evaluation/research/cross-model-geometry-v5-20260907/comparison-contract.json) |
| B 复用 | 270 行、1080 几何 bundle、63936 个映射 R²、1152 个 ARO 正确率；R² 最大差 0 | [V4 一致性](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/b-v4-parity.json) |
| 汇总 | 全部 CSV 数值、dev 选择、层搜索统计 | [汇总检查](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/summary-tables.json) |
| 配对 | 1260 组及所有 2000-bootstrap 区间重算 | [配对检查](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/paired-model-differences.json) |
| 扰动解释 | 48 条原始误差重现、平移/变形分解 | [残差分解](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/perturbation-error-decomposition.json) |
| 图表 | 25 对 PNG/PDF；PNG 解码与目视、PDF 解析 | [索引](../output/evaluation/research/cross-model-geometry-v5-20260907/figures/index.json)、[目视记录](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/visual-review.json) |
| 资源 | 17 个提取阶段结束，开发机停止 | [停止记录](../output/evaluation/research/cross-model-geometry-v5-20260907/resource-cleanup.json) |
| 整体覆盖 | 正式文件、报告、图表和检查记录 | [产物检查](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/completion-artifacts.json) |

特征分片：B/F 各 480，Qwen/DINOv2/MAE 各 48，SigLIP 96，JanusFlow/Show-o2 各 352。完整性检查覆盖样本身份、契约、形状和有限值。数值配置与统计适用条件集中在 [方法](CROSS_MODEL_GEOMETRY_V5_METHODS_20260907.md)。
