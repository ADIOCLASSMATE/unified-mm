# B 跨模态几何实验 V3：结果

日期：2026-09-07。指定 B run，step 95415，主模型为 `hf_model-final-ema`。本轮复用 [V2](B_SEMANTIC_EMERGENCE_V2_RESULTS_20260907.md) 已经核验的逐层特征，仅在 CPU 上分析；没有重新启动开发机、修改模型、训练或新增 NPU 前向。

已完成全部 **660 条逐层记录**及完整性审计，包含 EMA、raw、三个初始化流程参考、30 个位置和两种读出。结论是：**B 存在可泛化的跨模态语义关系和部分共享子空间，但尚未证实整个表示空间只差一个全局旋转。**

快速查看：[欧氏逐层几何][geometry-euclidean]、[单位球逐层几何][geometry-sphere]、[欧氏旋转泛化][rotation-test]、[跨域旋转迁移][rotation-transfer]；[完整逐层 CSV][csv]。

## 1. 我们实际检验的命题

你的“空间一致”可以拆成三个层次：

- **邻居和关系共享**：猫在图像空间的邻居，与猫在文本空间的邻居是不是相同概念？不用学习图文映射。
- **旋转可对应**：只在一部分概念上求一个全局正交变换，是否还能对应新的概念？允许平移、旋转/反射和整体单位换算，不允许逐轴任意拉伸。
- **跨数据集可复用**：ImageNet 上学到的变换，不重新校准就能否迁移到 COCO，反向是否成立？

完整 [实验协议](B_GEOMETRY_PROTOCOL_V3.md) 已固定。采用形状比较和核/邻居关系比较的视角，分别参考 [Generalized Shape Metrics](https://proceedings.neurips.cc/paper/2021/hash/252a3dbaeb32e7690242ad3b556e626b-Abstract.html)、[CKA](https://proceedings.mlr.press/v97/kornblith19a.html) 和 [Platonic Representation Hypothesis](https://phillipi.github.io/prh/)。这是本项目的探索性实现，不是对某篇论文的完整复现。

## 2. 数据和可解释范围

| 数据 | 拟合 / 开发 / 测试单位 | 几何评估 | 重复参考 |
| --- | --- | --- | --- |
| ImageNet | 600 / 200 / 200 个互斥类别；每类 3 图均值，对 2 个类名模板均值 | 200 测试类 | 同 200 类的另一组 3 图 / 2 模板 |
| COCO | 500 / 500 / 1000 个互斥场景；1 图，对其全部 caption 的均值 | 固定 500 测试场景子集；映射误差用全部 1000 场景 | 同场景前 2 条 caption，对其余 caption |

先对原始特征求语义单位均值，再做中心化。ImageNet 校准只用 600 个拟合类别的独立 cal 样本，不用测试类别；COCO 使用独立 500 场景校准。所有 PCA、拟合中心、整体尺度和旋转只使用拟合集。

“未见类别”指映射拟合未见，不是 B 训练未见；COCO 是相对本次 ImageNet 图像训练的域外自然场景，不承诺预训练去污染。V2 测试集已查看，本轮是后续探索，不是独立确认。每类仅 3 图会使图像类别原型比较嘈杂。

## 3. 固定最终 RMSNorm：近邻关系确实共享

下表是 **EMA、仅减模态均值、不逐向量 L2 归一化** 的 kNN@10 身份重合率。它不是图文检索正确率：例如 30% 表示同一语义单位两侧各自 10 个邻居中，平均有 3 个身份相同。

| 条件 / 读出 | ImageNet，200 类 | COCO，500 场景 |
| --- | ---: | ---: |
| 机会水平 | 5.03% | 2.00% |
| 无提示，Content | 7.90% | 7.74% |
| 原生提示，Content | 17.40% | 19.30% |
| 原生提示，异构 query | 27.85% | 30.26% |

这表明，哪怕不要求图文直接坐标接近，也有可对应的局部关系。无提示 Content 的关系信号较弱，但不等于完全没有；这与 V2 中无提示直接检索接近随机不是矛盾。

原生 query 的输入 mask 在每侧是常量，所以输入 0 的 query 几何记为无效；不是“零层已经完美统一”。Content 不读取目标 query，因此其共享关系不能由两边共享末尾 mask 的解释单独覆盖。

### 原始几何和单位球几何不能混为一谈

| final RMSNorm，native | ImageNet RSA / CKA / kNN@10 | COCO RSA / CKA / kNN@10 |
| --- | --- | --- |
| Content，欧氏 | 0.154 / 0.485 / 17.40% | 0.176 / 0.363 / 19.30% |
| Content，单位球 | 0.389 / 0.537 / 24.90% | 0.290 / 0.372 / 22.84% |
| query，欧氏 | 0.371 / 0.411 / 27.85% | 0.273 / 0.464 / 30.26% |
| query，单位球 | 0.325 / 0.449 / 30.45% | 0.423 / 0.495 / 33.56% |

球面处理有时改善邻居一致性，但不是所有距离排序都改善。尤其无提示 ImageNet Content：欧氏 CKA 为 0.227，而 RSA 是 −0.110；单看 CKA 会漏掉这个冲突。整体核统计、全部距离的排序和局部邻居身份是不同问题。

仅平移中心不会改变各自内部欧氏距离；再做逐行 L2 才会改变几何。不能把单位球上的成功表述成原始点云只差一个旋转。

## 4. 一个旋转能解释多少？局部关系强于全局等距

以下是在独立拟合集求 Q、测试集评估的 **正交映射 R²**，固定 final RMSNorm。R²=1 才是精确对应，R²=0 表示与预测拟合均值一样；负数保留，不等于“没有任何语义”。两侧仅做拟合集的全局 RMS 单位换算，没有逐轴白化。

| native / 欧氏模式 | 完整 1024 维 | PCA 32 维 | PCA 128 维 |
| --- | ---: | ---: | ---: |
| ImageNet Content | −0.425 | −0.208 | −0.366 |
| ImageNet query | −0.258 | −0.075 | −0.119 |
| COCO Content | −0.247 | −0.065 | −0.106 |
| COCO query | −0.021 | **0.225** | 0.167 |

COCO query 的 32 维结果，按 1000 个场景 bootstrap 的条件 95% 区间为 **[0.201, 0.248]**；同容量、打乱拟合配对的 R² 为 **−1.034**。该 PCA 在测试图像/文本上分别保留 **58.2% / 89.8%** 方差，因而是部分子空间证据，不是用几乎全部信息得到了完整等距。

允许再从正确配对上估计一个全局收缩尺度时，同一 COCO query 32 维的 R² 为 **0.349**；仍远未接近 1。这个补充对应更一般的 similarity Procrustes，完整数据都保留了此结果，不隐藏整体缩放的作用。

**不能据完整 1024 维的误差断言全空间不存在好旋转**：600 类/500 场景居中后秩最多 599/499，拟合数据不足以识别一个完整 1024 维旋转，SVD 只能给出其中一个解。当前更稳妥的陈述是：本轮没有证实全空间近似等距；已证实的正向信号主要来自关系与部分子空间。

## 5. 开发集选层后，能推广到新类别和另一个数据集吗？

下表只展示 native query；全部 Content / bare / neutral / 初始化组合也保留在机器结果中。联合在开发集按正交 NRMSE 选层和维度，同分优先较早层、再较低维，**不是挑测试峰值**。四项都选到 32 维。

| 拟合数据 / 模式 | 开发集选层 | 同域新单位 R² | 另一数据集 R² |
| --- | ---: | ---: | ---: |
| ImageNet，欧氏 | 13 | 0.022 | COCO：−0.269 |
| ImageNet，单位球 | 13 | **0.210** | COCO：**0.239** |
| COCO，欧氏 | 23 | **0.267** | ImageNet：**0.152** |
| COCO，单位球 | 13 | **0.319** | ImageNet：−0.028 |

跨域时不重估目标域均值、PCA、尺度或 Q；表内映射方向都为 image→text，“双向”是两个数据集之间的迁移方向。单位球的 ImageNet 13 层子空间在同域测试图像/文本保留 **77.6% / 69.1%** 方差。

这比“只在拟合的几个类别上能旋转对齐”更有说服力，但效果明显依赖预处理和迁移方向。不支持一个域无关、层无关、读出无关的统一等距空间。欧氏 ImageNet query 的开发 R² 为 0.257，测试只有 0.022，也提醒我们不能拿拟合/开发集效果替代未参与拟合类别的泛化表现。

## 6. 初始化和 raw：训练后的增量不是 EMA 独有

固定 final RMSNorm、native、欧氏 kNN@10：

| 状态 / 读出 | ImageNet | COCO |
| --- | ---: | ---: |
| 初始化参考 Content，3 seeds 范围 | 7.80–8.70% | 5.42–6.02% |
| final raw Content | 16.10% | 19.60% |
| final EMA Content | 17.40% | 19.30% |
| 初始化参考 query，3 seeds 范围 | 6.85–8.45% | 4.60–5.02% |
| final raw query | 28.35% | 29.78% |
| final EMA query | 27.85% | 30.26% |

各状态独立使用开发集选层/维度后，native query 的同域旋转 R²：

| 拟合数据 / 模式 | EMA | raw | 初始化 42 / 43 / 44 |
| --- | ---: | ---: | --- |
| ImageNet，欧氏 | 0.022 | 0.009 | −1.029 / −1.158 / −0.592 |
| ImageNet，单位球 | 0.210 | 0.191 | −0.515 / −0.787 / −0.643 |
| COCO，欧氏 | 0.267 | 0.270 | −0.597 / −0.570 / −0.585 |
| COCO，单位球 | 0.319 | 0.318 | −0.593 / −0.554 / −0.559 |

raw 与 EMA 选择到相同的上述层和维度，主要趋势一致。初始化也可能包含高于机会的邻居结构，不能用“p 显著”定义训练涌现；效应幅度和泛化误差的增量更有信息。

初始化参考保留预训练 Qwen/VAE，只重新初始化项目增加的部分；不是原 run 的可信历史 step 0，也不是三个完整训练重复。结果支持训练后共享结构增加，但缺少匹配的冻结 backbone、打乱训练配对等训练臂，不能区分 projector 接入已有语言结构、backbone 更新、配对监督和生成目标各自的贡献。

## 7. 与“同模态自己有多稳定”相比

原生最终 query，欧氏模式：

| 参考关系 | RSA | kNN@10 |
| --- | ---: | ---: |
| ImageNet 图像原型，3 张图对另 3 张图 | 0.663 | 32.40% |
| ImageNet 文本原型，2 模板对另 2 模板 | 0.932 | 70.05% |
| ImageNet 图文跨模态 | 0.371 | 27.85% |
| COCO 同场景两组 caption | 0.765 | 37.30% |
| COCO 图文跨模态 | 0.273 | 30.26% |

每类 3 张图的图像原型本身不是非常稳定，因此不能要求跨模态几何必然接近 1。跨模态邻居重合已与这些有限视图参考处于同一量级，但全局距离排序仍弱得多。参考不是严格上界，不对它做简单比值后声称“语义完成度达到多少”。COCO 主缓存没有独立图像重复视图，所以本轮没有给出其图像侧稳定性上界。

## 8. 沿层变化与可读的邻居示例

下面是固定深度间隔的 EMA native query，欧氏模式。完整 30 个位置同时报告，不以测试峰值选择结论。

| 深度 | ImageNet RSA / kNN@10 | COCO RSA / kNN@10 |
| --- | --- | --- |
| 输入 0 | 无效：query 为常量 | 无效：query 为常量 |
| block 4 | 0.120 / 11.80% | 0.223 / 10.82% |
| block 8 | 0.316 / 23.05% | 0.407 / 19.60% |
| block 12 | 0.534 / 34.00% | 0.452 / 25.72% |
| block 16 | 0.456 / 31.60% | 0.296 / 25.80% |
| block 20 | 0.353 / 29.85% | 0.253 / 25.32% |
| block 24 | 0.385 / 28.95% | 0.292 / 30.52% |
| block 28，Norm 前 | 0.256 / 26.20% | 0.217 / 27.10% |
| 最终 RMSNorm | 0.371 / 27.85% | 0.273 / 30.26% |

关系在 backbone 中逐步变得可读，但中间有回落，不是严格单调。最终 RMSNorm 与最后一个 block 必须分开：模型自身的归一化和逐通道缩放会改变池化后的几何。本表是深度轴，不是训练过程的时间轴。

为方便理解，额外导出 native 的全部 **200 类**在第 13 层和 final RMSNorm 的邻居，涵盖两种读出、两种几何，共 1600 行。以下是事后选的说明性正反例，不能代替总体结果：

| 类别 | final query，欧氏前 10 邻居的交集 | 重合数 |
| --- | --- | ---: |
| newt | fire salamander、smooth newt、box turtle、water snake、slug、mink | 6 |
| great grey owl | rooster、bulbul、quail、coucal | 4 |
| airliner | projectile | 1 |
| box turtle | 无 | 0 |

例如 newt 的图像与文本邻居都含近缘两栖动物，但也有跨类混杂；不能把重合邻居全都自动认定为合理语义关系。全部邻居及不重合项保存在 `imagenet-neighbor-examples-v3.json`。

## 9. 对假设的回答，以及最值得补的实验

**本轮支持“不同模态存在可泛化的共同语义关系”，尤其是原生 query 的邻域和低维子空间；没有证实“全部表示只差一个全局旋转”。** 这两句话并不矛盾。直接余弦弱，也不再能作为“没有统一语义结构”的充分反证。

建议下一轮优先补两个缺口，而不是继续堆相似性指标：

1. **更稳定的 ImageNet 原型**：增加每类独立图像数，画原型视图数的稳定性曲线。当前 3 图对 3 图的同模态近邻一致率仅约 32%，限制了图文原型比较的解释。这解决测量噪声，但不会增加类别数，不能解决 1024 维全空间旋转的样本不足。
2. **足够多且独立的场景对**：换未参与 V1/V2 的图文场景，在固定 dev/test 下逐步增加拟合例数，例如 512 / 2048 / 8192，检查完整旋转、PCA 子空间与跨域结果是否稳定。拟合点数超过 1024 只是必要起点，还要检查覆盖、数值秩、有效维度和误差是否随样本数收敛；重复同一类名模板不能冒充新的独立语义方向。

新域可继续选 COCO 未使用场景或另一个有真实描述的数据集。ImageNet 适合类别几何，复杂场景用于检验类别之外的迁移；如果要进一步称为“完整语义”，仍需属性、数量、关系和绑定的受控对照。

最后，实际 B 使用预训练 LM/VAE、配对条件、文本 CE 与图像 flow matching。没有显式对比/表征对齐损失，不等于从零初始化、无配对监督或字面的单一重建损失。本轮也没有对 flow head / LM head 做因果干预，**不能据此推导 head 不承担语义计算**。

## 10. 统计、核验与复现产物

- 660/660 条逐层记录完整，生成 2640 项几何比较、7920 项映射设置；输入层常量 query 对应的 44 项几何和 132 项映射明确记为无效，不是缺失任务。
- 每个有效几何设置有 199 次语义身份置换。EMA native 两种读出、两个数据集的欧氏 kNN@10 曲线，经同置换的层间最大值校准后，经验 p 均为 0.005；这是置换分辨率下限，不声明整个实验家族均已校正。
- 保存 88 个开发集选层/维度结果、528 条指标曲线的层搜索零分布校准，以及固定 final RMSNorm 的条件 bootstrap 区间。bootstrap 不包括重新拟合 PCA/Q、原型重采样和训练重复的不确定性。
- [完整性审计][audit] 通过：来源组合、样本数量、有限值、秩与维数、方差保留边界、R²/NRMSE 恒等式和 bootstrap 字段均符合协议。
- 1600 条邻居明细重新聚合后，与对应 8 项正式 kNN@10 结果逐项一致。
- 新旧诊断相关单元测试 **21 passed**；Ruff 和差异空白检查通过。六组曲线同时保存 PNG / PDF。
- 初次小规模自测发现 FP64 并列秩赋值与稳定排序 API 调用问题，均在正式分析前修正。批量运行发现容器实际配额为 16 CPU 核，已中断本次分析进程、保留完成记录并以较低并行度恢复；没有改变数据、指标或权重，也没有删除文件。

代码：[分析器](../scripts/analyze_unified_geometry_v3.py)、[绘图器](../scripts/plot_unified_geometry_v3.py)、[测试](../tests/test_geometry_v3.py)。完整运行方法见 [协议](B_GEOMETRY_PROTOCOL_V3.md)。

原始结果：[JSON][results]、[CSV][csv]、[开发集选择][selected]、[层搜索置换校准][null]、[全部邻居明细][neighbors]。这些文件均在指定 run 的 `representation-diagnostic/geometry-v3-20260907` 下，旧 V2 结果未覆盖。

[geometry-euclidean]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/geometry-layers-centered_euclidean.png
[geometry-sphere]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/geometry-layers-centered_unit_sphere.png
[rotation-test]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/procrustes-test-centered_euclidean.png
[rotation-transfer]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/procrustes-transfer_test-centered_unit_sphere.png
[csv]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/geometry-v3.csv
[results]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/results-geometry-v3.json
[selected]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/dev-selected-geometry-v3.json
[null]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/layer-search-null-v3.json
[neighbors]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/imagenet-neighbor-examples-v3.json
[audit]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v3-20260907/audit-geometry-v3.json
