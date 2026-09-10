# B 关系几何 V3：结果

2026-09-07，复用 B 的 V2 冻结特征。图文邻居关系高于机会水平；可泛化的旋转对应集中在部分低维子空间，跨域迁移依赖方向和几何处理。协议与复现命令见 [V3](B_GEOMETRY_PROTOCOL_V3.md)。

## 样本

| 数据 | 拟合 / 开发 / 测试单位 | 几何评估 | 重复参考 |
| --- | --- | --- | --- |
| ImageNet | 600 / 200 / 200 个互斥类别；每类 3 图均值，对 2 个类名模板均值 | 200 测试类 | 同 200 类的另一组 3 图 / 2 模板 |
| COCO | 500 / 500 / 1000 个互斥场景；1 图，对其全部 caption 的均值 | 固定 500 测试场景子集；映射误差用全部 1000 场景 | 同场景前 2 条 caption，对其余 caption |

## 无拟合关系

固定 EMA / final RMSNorm，中心化欧氏模式的 kNN@10 身份重合：

| 条件 / 读出 | ImageNet，200 类 | COCO，500 场景 |
| --- | ---: | ---: |
| 机会水平 | 5.03% | 2.00% |
| 无提示，Content | 7.90% | 7.74% |
| 原生提示，Content | 17.40% | 19.30% |
| 原生提示，异构 query | 27.85% | 30.26% |

欧氏与单位球分别报告：

| final RMSNorm，native | ImageNet RSA / CKA / kNN@10 | COCO RSA / CKA / kNN@10 |
| --- | --- | --- |
| Content，欧氏 | 0.154 / 0.485 / 17.40% | 0.176 / 0.363 / 19.30% |
| Content，单位球 | 0.389 / 0.537 / 24.90% | 0.290 / 0.372 / 22.84% |
| query，欧氏 | 0.371 / 0.411 / 27.85% | 0.273 / 0.464 / 30.26% |
| query，单位球 | 0.325 / 0.449 / 30.45% | 0.423 / 0.495 / 33.56% |

## 正交映射

固定最终层的留出 R²：

| native / 欧氏模式 | 完整 1024 维 | PCA 32 维 | PCA 128 维 |
| --- | ---: | ---: | ---: |
| ImageNet Content | −0.425 | −0.208 | −0.366 |
| ImageNet query | −0.258 | −0.075 | −0.119 |
| COCO Content | −0.247 | −0.065 | −0.106 |
| COCO query | −0.021 | **0.225** | 0.167 |

COCO query 的 32 维 R²=0.225，条件 95% 区间 [0.201, 0.248]；打乱 fit 为 −1.034，保留图像/文本方差 58.2% / 89.8%。加入单个整体尺度后的 R² 为 0.349。full 1024 维的样本秩上限在 ImageNet / COCO 为 599 / 499。

只用源 dev 选层与维度，原生 query 四项均选到 32 维；跨域冻结全部分析参数：

| 拟合数据 / 模式 | 开发集选层 | 同域新单位 R² | 另一数据集 R² |
| --- | ---: | ---: | ---: |
| ImageNet，欧氏 | 13 | 0.022 | COCO：−0.269 |
| ImageNet，单位球 | 13 | **0.210** | COCO：**0.239** |
| COCO，欧氏 | 23 | **0.267** | ImageNet：**0.152** |
| COCO，单位球 | 13 | **0.319** | ImageNet：−0.028 |

## 训练状态参考

最终层、native、欧氏模式的 kNN@10：

| 状态 / 读出 | ImageNet | COCO |
| --- | ---: | ---: |
| 初始化参考 Content，3 seeds 范围 | 7.80–8.70% | 5.42–6.02% |
| final raw Content | 16.10% | 19.60% |
| final EMA Content | 17.40% | 19.30% |
| 初始化参考 query，3 seeds 范围 | 6.85–8.45% | 4.60–5.02% |
| final raw query | 28.35% | 29.78% |
| final EMA query | 27.85% | 30.26% |

源 dev 选择后的 query 留出 R²：

| 拟合数据 / 模式 | EMA | raw | 初始化 42 / 43 / 44 |
| --- | ---: | ---: | --- |
| ImageNet，欧氏 | 0.022 | 0.009 | −1.029 / −1.158 / −0.592 |
| ImageNet，单位球 | 0.210 | 0.191 | −0.515 / −0.787 / −0.643 |
| COCO，欧氏 | 0.267 | 0.270 | −0.597 / −0.570 / −0.585 |
| COCO，单位球 | 0.319 | 0.318 | −0.593 / −0.554 / −0.559 |

raw 与 EMA 接近；三个初始化参考对应新增组件 seed 42/43/44，保留预训练 Qwen/VAE。

## 重复视图与逐层变化

同模态重复参考和跨模态关系：

| 参考关系 | RSA | kNN@10 |
| --- | ---: | ---: |
| ImageNet 图像原型，3 张图对另 3 张图 | 0.663 | 32.40% |
| ImageNet 文本原型，2 模板对另 2 模板 | 0.932 | 70.05% |
| ImageNet 图文跨模态 | 0.371 | 27.85% |
| COCO 同场景两组 caption | 0.765 | 37.30% |
| COCO 图文跨模态 | 0.273 | 30.26% |

原生 query 的完整深度节选：

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

固定最终 query 的 ImageNet 邻居示例：

| 类别 | final query，欧氏前 10 邻居的交集 | 重合数 |
| --- | --- | ---: |
| newt | fire salamander、smooth newt、box turtle、water snake、slug、mink | 6 |
| great grey owl | rooster、bulbul、quail、coucal | 4 |
| airliner | projectile | 1 |
| box turtle | 无 | 0 |

## 产物

660 条基础记录、2640 组几何、7920 组映射；其中 44 组几何和 132 组映射为常量或秩不足。另有 88 个 dev 选择、528 个层搜索检验、1600 条邻居示例；21 项相关测试通过。

统计采用 199 次身份置换和 2000 次测试 bootstrap，区间条件于当前权重与拟合。算法细节见协议。

[完整结果][results]、[CSV][csv]、[dev 选择][selected]、[层搜索零分布][null]、[邻居示例][neighbors]、[检查记录][audit]；图表：[欧氏几何][geometry-euclidean]、[球面几何][geometry-sphere]、[同域映射][rotation-test]、[跨域映射][rotation-transfer]。

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
