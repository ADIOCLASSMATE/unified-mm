# 跨模型关系几何 V5：结果

B 存在跨模态关系结构和可泛化的部分子空间。独立的 DINOv2＋Qwen 也有明显对应；B 相对 F 的优势随读出、层和迁移方向变化。实验比较冻结权重中的表征，head 的因果分工不在本轮测量范围内。

七组模型、14 个路径/提示设置、41 条读出曲线和 1171 个层对 × 读出组合全部保留。[完整数值](../output/evaluation/research/cross-model-geometry-v5-20260907/RESULTS_ZH.md)、[方法](CROSS_MODEL_GEOMETRY_V5_METHODS_20260907.md)、[复现](CROSS_MODEL_GEOMETRY_V5_REPRODUCE.md)。

## 比较口径

使用 V4 的 ImageNet 32000 图、COCO 11776 场景及 868 张身份互斥 ARO 图。几何候选集为 200 个 ImageNet test 类、固定 512 个 COCO test 场景；COCO 映射误差用全部 2048 test 场景。

主对照固定最终 norm、共同 `content_mean`、欧氏 32 维；源 fit 决定均值、PCA、RMS 和正交 Q，跨域全部冻结。源 dev 选层、原生任务读出和其他维度另报。外部模型具有不同预训练数据、教师与规模，参数来源如下：

| 比较组 | 实际载入参数范围（约） | 语义来源和关键差别 |
| --- | --- | --- |
| B／F | 各 0.761B，不含 MAR VAE | Qwen Base、MAR VAE 初始化；ClimbMix＋配对 ImageNet 条件生成；文本 CE＋图像 flow |
| DINOv2＋Qwen | 0.087B＋0.596B | 独立单模态；DINOv2 小模型含自监督教师蒸馏 |
| MAE＋Qwen | 编码器 0.086B＋0.596B | 未分类微调的重建式视觉编码器；全部 patch 可见，无新连接器 |
| SigLIP | 两塔合计 0.878B | 显式成对图文 sigmoid 对齐；原生 pooler 训练过 |
| JanusFlow | 2.046B，另载 VAE 0.084B | SigLIP 理解编码器＋REPA；AR／rectified flow／SFT |
| Show-o2 | 去重复共享存储后 2.831B，另载 VAE 0.127B | Qwen2.5-Instruct、SigLIP 蒸馏语义路径、Wan VAE、AR／flow 与指令数据 |

## 无拟合关系

固定最终层共同内容均值，B 的 ImageNet CKA / RSA / kNN@10 为 0.597 / 0.337 / 0.2475，COCO 为 0.373 / 0.164 / 0.182。DINOv2＋原始 Qwen 的 COCO 对应值为 0.632 / 0.371 / 0.310。

MAE＋Qwen 的 COCO 最终层 CKA 为 0.086。JanusFlow 理解路径的 COCO 输入 CKA 已约 0.628，最终约 0.705，输入包含 SigLIP 语义。B 的输入→最终 CKA 在 ImageNet 为 0.238→0.597，COCO 为 0.189→0.373。

## 留出映射与跨域

固定最终层、共同内容均值、欧氏 32 维的 R²。每个模型使用自身表示空间的分母，0 对应预测源 fit 的目标均值。

| 模型／路径 | IN→IN | IN→COCO | COCO→COCO | COCO→IN |
| --- | --- | --- | --- | --- |
| B | 0.128 | -0.913 | -0.005 | 0.193 |
| F | -0.003 | -1.059 | -0.176 | -0.006 |
| DINOv2＋Qwen | 0.171 | 0.035 | 0.290 | -0.040 |
| MAE＋Qwen | -0.566 | -1.068 | -0.569 | -0.094 |
| SigLIP 内容均值 | 0.330 | -0.386 | 0.327 | 0.020 |
| JanusFlow 理解 | 0.307 | 0.099 | 0.461 | 0.070 |
| JanusFlow 生成 | -0.209 | -1.219 | -0.317 | -0.171 |
| Show-o2 理解 | 0.204 | -0.077 | 0.307 | 0.066 |
| Show-o2 生成 | 0.201 | -0.121 | 0.317 | 0.059 |
| SigLIP 原生 pooler（另列参考） | 0.526 | -0.277 | 0.357 | 0.016 |

B 的 COCO 留出区间为 [−0.033, 0.025]。ImageNet32 保留图像/文本方差约 0.807 / 0.393，COCO32 为 0.691 / 0.752。DINOv2＋Qwen 的 COCO 留出 R² 为 0.290，MAE＋Qwen 为 −0.569，后者经源 dev 选层后为 −0.182。

Show-o2 理解路径经 COCO dev 选到 block26/d32 后，同域 R² 从 0.307 升到 0.396，迁移到 ImageNet 从 0.066 变为 −0.289。源域最优层的跨域表现可能下降。

在上述主口径的配对区间中，B 四个方向均高于 F、MAE＋Qwen、JanusFlow 生成内容均值；COCO 留出低于 DINOv2、SigLIP、JanusFlow 理解和 Show-o2 两条路径。COCO→ImageNet 高于八个共同均值参照；该目标域也是 B/F 的多模态训练域。

## B 与 F

两者 dataset、optimizer、scheduler、training 配置逐字段相同，final EMA 均为 step 95415，flow head 参数量相差约 0.149%。F 使用逐位置 MLP head 和自己的模型类及权重。

主对照中的 B−F 配对差值：

| 来源→目标 | ΔR² | 条件 95% CI |
| --- | --- | --- |
| ImageNet→ImageNet | 0.131 | [0.071, 0.193] |
| ImageNet→COCO | 0.146 | [0.116, 0.175] |
| COCO→COCO | 0.171 | [0.155, 0.187] |
| COCO→ImageNet | 0.199 | [0.171, 0.226] |

改用各自 COCO dev 选择的原生 query：B 为 block14/d32，F 为 block22/d32。同域 R² 为 0.28739 / 0.28772，差值 −0.00033，95% 区间 [−0.03700, 0.03602]；迁移到 ImageNet 为 0.500 / 0.252，差值区间 [0.225, 0.272]。

## Flow 路径与扰动

生成探针使用图像 posterior mean/t=1 和文本/固定噪声/t=0 两次独立前向，读取 backbone 内容或生成槽位。

| 路径 / 读出 | COCO dev 选择 | COCO 留出 R² | COCO→ImageNet R² |
| --- | --- | ---: | ---: |
| JanusFlow 生成槽位均值 | block6/d32 | 0.377 | 0.465 |
| Show-o2 生成内容均值 | block26/d32 | 0.399 | −0.106 |
| Show-o2 生成槽位均值 | block6/d32 | 0.116 | 0.056 |

JanusFlow 的 block6 与原论文施加 REPA 的层一致。其前部内容受因果掩码保护而看不到后部噪声；生成槽位的两个额外 seed 同分母误差变化约 −0.0004 / +0.0088，图像 t=0.5 为 −0.077，区间 [−0.095, −0.060]。

Show-o2 生成槽位在额外 seed1/seed2 下的同分母误差变化为 −95.317 / −111.800，CKA 却从约 0.449 变为 0.463 / 0.463，kNN 基本保留。事后分解发现原始变化能量的 99.99996% / 99.99973% 来自整体平移。图像 t=0.5 的 block26 内容均值则包含明显变形：CKA 从约 0.664 降至 0.082，中心化变形 RMS 约为原语义变化的 2.48 倍。

B 的 dev 选层原生 query 在 sigma2 下变化为 −0.337，区间 [−0.370, −0.306]；Content mean 约 +0.004，区间跨 0。sigma 同时移动 query 槽位，后验均值控制影响较小。[48 条残差分解](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/perturbation-error-decomposition.json)保留原始预测和冻结映射。

## ARO 困难负例

固定最终层、共同内容均值、欧氏 32 维、COCO8192 映射。B 的属性/关系余弦正确率为 54.5% / 56.1%，相对同任务打乱图像的增益区间均跨 0。距离读出的关系正确率为 48.9%，增益约 7.4 个百分点，区间 [1.9, 12.9]。

SigLIP 原生 pooler 的属性余弦正确率约 63.4%，相对打乱图像的增益区间约 [6.0, 19.5] 个百分点。全部读出及余弦/距离结果分列，ARO 不参与设置选择。

## 数值与产物

SigLIP、JanusFlow 正式计算使用 FP32，Show-o2 使用 FP32 计算与存储；cal 最差余弦分别为 0.99999994、0.99999952、0.99998772。B/F neutral 末 token 及 Show-o2 早期弱槽位保留数值标记，详见方法。

保存 37472 条映射、328 个 dev 选择、984 个层搜索检验、1680 条扰动、1260 组配对对照及 25 对 PNG/PDF。34 项相关测试通过。区间为当前权重、fit 和 dev 选择下的身份级条件区间。

[汇总检查](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/summary-tables.json)、[配对检查](../output/evaluation/research/cross-model-geometry-v5-20260907/audits/paired-model-differences.json)、[图表索引](../output/evaluation/research/cross-model-geometry-v5-20260907/figures/index.json)、[产物清单](CROSS_MODEL_GEOMETRY_V5_COMPLETION_AUDIT_20260907.md)。
