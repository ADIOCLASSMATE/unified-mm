# 跨模型关系几何 V5 协议

在冻结权重下比较 B、F、JanusFlow-1.3B、Show-o2-1.5B、SigLIP-so400m-patch14-384、DINOv2 ViT-B/14＋Qwen3-0.6B-Base、MAE ViT-B/16＋同一 Qwen。结果见 [V5 报告](CROSS_MODEL_GEOMETRY_V5_RESULTS_20260907.md)。

## 数据与主对照

完整复用 [V4 数据](B_GEOMETRY_PROTOCOL_V4.md)：ImageNet 32000 图 / 12000 模板、COCO 11776 场景 / 58909 caption、身份互斥的 ARO 868 图 / 1736 候选文本。划分、顺序和种子 20260909 保持一致。

主对照固定最终 norm、`content_mean`、欧氏模式、32 维。扩展包括全部实际层、原生任务读出、单位球、128/512 维及源 dev 选择；原始宽度相同的模型另报 full。

## 模型契约

| 比较组 | 加载与路径 |
| --- | --- |
| B / F | 各自 final EMA；F 使用 `PositionwiseFlowOnBQwen3ForCausalLM`；native / bare / neutral |
| SigLIP | 两塔独立前向；共同内容均值与训练过的 pooler 分列 |
| DINOv2 / MAE＋Qwen | 原始视觉编码器配原始 Qwen Base；MAE 全 patch 可见、固定顺序 |
| JanusFlow / Show-o2 | 理解、生成路径分列；保存视觉编码/融合输入和原生生成槽位 |

每侧保存输入、每个 block、最终 norm；不同深度按预先固定的相对深度网格配对，记录真实层号。图像共享 Resize256 / CenterCrop256 内容视图，再适配原生分辨率和归一化。

生成探针使用两次独立前向：图像侧为 clean VAE posterior mean、t=1、空文本；文本侧为文本、t=0、全样本共享的固定高斯噪声。记录内容和生成槽位，另测两个噪声 seed 与图像 t=0.5。B/F 另测 sigma 与后验均值控制。

## 分析和统计

沿用 V4 的几何、fit-only 正交映射和全参数冻结跨域协议；共同维度 32/128/512。报告 kNN@5/10/20、RSA、CKA、R²、PCA 方差覆盖、秩、跨协方差条件数、打乱配对及打乱图像控制。

固定最终层与源 dev 选择分列。199 次身份置换含逐曲线层搜索零分布；2000 次身份级 bootstrap 用于测试区间和模型间配对差值。区间条件于现有权重、fit 和 dev 选择。

完整算法、模型语义来源和数值配置集中在 [方法文档](CROSS_MODEL_GEOMETRY_V5_METHODS_20260907.md)。

## 产物

根目录：`output/evaluation/research/cross-model-geometry-v5-20260907/`。模型和官方源码位于 `public/models/`；精确 revision 见 `asset-manifest.json`。

保存冻结适配器/比较/统计契约、全部特征、逐层与映射结果、配对区间、图表和来源检查。运行命令见 [复现入口](CROSS_MODEL_GEOMETRY_V5_REPRODUCE.md)，覆盖数量和检查记录见 [产物清单](CROSS_MODEL_GEOMETRY_V5_COMPLETION_AUDIT_20260907.md)。
