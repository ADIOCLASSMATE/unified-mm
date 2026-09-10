# 跨模型关系几何 V5：方法

数据与模型范围见 [V5 协议](CROSS_MODEL_GEOMETRY_PROTOCOL_V5.md)，数值见 [结果](CROSS_MODEL_GEOMETRY_V5_RESULTS_20260907.md)。本实验测量固定权重的关系结构、正交子空间泛化和困难负例排序。

## 输入和语义来源

图文独立前向。外部图像先共享 Resize256 / CenterCrop256 内容视图，再适配各模型原生输入；各模型使用自己的 VAE。MAE 全 patch 可见。ImageNet 类别留出针对映射拟合；COCO 是相对 B/F 本次 ImageNet 图像训练的域外数据。ARO 的 868 张图按 VG→COCO 身份与全部 COCO 划分互斥。

| 比较组 | 预训练与本次权重来源 |
| --- | --- |
| B / F | Qwen3 Base、MAR VAE；ClimbMix＋配对 ImageNet 的 CE / flow 训练；两者独立训练 |
| JanusFlow | 预训练语言模型、SigLIP 理解编码器、SDXL VAE；AR / flow / REPA / SFT |
| Show-o2 | Qwen2.5-Instruct、Wan VAE；SigLIP 蒸馏语义路径，再进行两阶段 AR / flow 训练；使用基础 1.5B 版本 |
| SigLIP | 成对图文 sigmoid 对齐；最终 pooler 为训练组件 |
| DINOv2＋Qwen | 独立视觉自监督与语言预训练，DINOv2 小模型包含教师蒸馏 |
| MAE＋Qwen | 视觉 masked-pixel reconstruction 与独立语言预训练 |

模型规模、训练数据和教师不同；外部模型用于比较已有表征，B/F 是本项目内的架构对照。各模型实际参数量和 revision 见[来源记录](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/model-sources-and-training.json)。Show-o2 去除 embedding / lm_head 共享存储的重复计数后为 2,830,777,056 个元素，另载 Wan VAE。

## 层和几何

保存输入、每个 block、最终 norm；不同深度按固定相对深度网格配对。共同内容均值、末 token、原生任务位置和 SigLIP pooler 分列。

先按语义单位平均原始特征。欧氏模式减模态校准均值，单位球模式再逐单位 L2 归一化。kNN 排除自身，RSA 去对角线，线性 CKA 中心化 Gram；这些指标衡量关系对应。

映射在源 fit 上估计两侧均值、独立 PCA、整体 RMS 单位和正交旋转/反射 Q，不逐轴白化。共同维度为 32/128/512；full 用于两侧原始宽度相同的模型。跨数据集复用全部源参数。

`R² = 1 − Σ||XQ − Y||² / Σ||Y||²`，Y 以源 fit 均值居中。0 对应预测 fit 均值，负值照常报告。另报打乱 fit 配对、单个整体尺度补充、测试 PCA 方差覆盖、两侧样本秩及跨协方差谱。V5 的 `rotation_identified` 还要求跨协方差在相对阈值 1e-6 下满秩。

## 生成路径与扰动

图像侧使用 clean posterior mean、t=1、空文本；文本侧使用正文、t=0、全样本共享的固定高斯噪声。探针读取 backbone，完整 ODE 生成质量由正式评测衡量。

在固定 32 个 ImageNet test 类和 512 个 COCO test 场景上，增加两个噪声 seed 及图像 t=0.5，文本侧复用主表示。B/F 的 sigma 控制同时改变 Content 次序与首图像 query 槽位；后验均值另列。所有扰动使用冻结的均值、PCA、RMS 和 Q。

事后残差分解覆盖 48 个 COCO 扰动条件，将平方误差拆成 `n × ||平均残差||² + Σ||中心化残差||²`，并测量原始表示变化中的平移能量。分解描述已有误差，不更新预测或映射。入口为 [explain_geometry_v5_perturbations.py](../scripts/explain_geometry_v5_perturbations.py)。

## 统计

199 次语义身份置换沿层共享，另存每条曲线的层搜索零分布。dev 在预定读出、模式和 fit 规模内选择层与维度；固定最终层保留。

测试、ARO 和模型配对差值使用 2000 次身份级 bootstrap，caption 随图归组；kNN 固定候选图、重采样 query。区间条件于当前权重、fit/PCA 和 dev 选择，逐项报告。不同模型 R² 使用各自表示空间的分母。

## 数值配置

下列精度由 cal 的单样本 / 正式 batch 全层一致性检查确定。

| 模型 | 正式计算 / 特征存储 | cal 最低逐层余弦 |
| --- | --- | ---: |
| SigLIP | FP32 / BF16 | 0.99999994（原 BF16 为 0.94839） |
| JanusFlow | FP32 / BF16 | 0.99999952（原 BF16 为 0.97835） |
| Show-o2 | FP32 / FP32，输入对所选 token 直接求均值 | 0.99998772（原 BF16 为 0.935019） |
| B / F | BF16 / BF16 | 主 native 内容均值稳定；neutral 末 token 最低约 0.99177 / 0.95020 |

Show-o2 批量误差相对中心化信号的 RMS，图像最高约 0.00205、COCO 文本约 0.00373；ImageNet 首 block 单点生成槽位为 0.03768，该弱信号读出保留数值标记。旧 Show-o2 特征和契约归档于 `calibration-archive/showo2-fp32-cal-revision-2/`。

JanusFlow 旧契约的 `generation_noise` 描述遗留“cast BF16”；实际实现转换到 `self.dtype`，本轮为 FP32。基础噪声在 CPU 以 FP32 生成。

## 参考

[关系几何](https://proceedings.mlr.press/v235/huh24a.html)、[CKA](https://proceedings.mlr.press/v97/kornblith19a.html)、[ARO](https://arxiv.org/abs/2210.01936)、[JanusFlow](https://arxiv.org/html/2411.07975v1)、[Show-o2](https://arxiv.org/html/2506.15564v2)、[SigLIP](https://arxiv.org/abs/2303.15343)、[DINOv2](https://arxiv.org/abs/2304.07193)、[MAE](https://arxiv.org/abs/2111.06377)、[Qwen3 Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)。
