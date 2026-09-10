# B 关系几何 V4：结果

2026-09-07，B step 95415，主权重为 final EMA。扩大样本后，backbone 的跨模态关系与部分共同子空间更清楚；全空间误差、跨域不对称和 query 位置依赖仍显著。全部前向冻结，协议见 [V4](B_GEOMETRY_PROTOCOL_V4.md)。

## 数据与采集

| 数据 | 实际语义单位与划分 | 本次用途 |
| --- | --- | --- |
| ImageNet | 1000 类 × 32 张新图；排除 V1/V2 的 18000 张；每类 2 cal、15 a、15 b。映射类别 600 fit / 200 dev / 200 test；每类 12 个模板 | 更稳定的类别原型、独立图像视图、1/3/5/10/15 图原型曲线、模板变化 |
| COCO | 11776 个新场景、58909 条 caption；512 cal / 8192 fit / 1024 dev / 2048 test | 相对 B 的 ImageNet 图像训练的域外检验；512→2048→8192 拟合规模曲线 |
| ARO / VG | 1000 关系 + 1000 属性原图，内部不重复；身份审计后另报 417 关系 + 451 属性的明确 COCO-pool-disjoint 子集 | 配对正确 caption 与控制负例，不在 ARO 拟合映射 |

EMA 测 native / bare / neutral，raw 与 init42/43/44 测 native；每组 30 层、三种读出。图文各自独立前向。ImageNet 的类别留出针对映射拟合；ARO 主解释采用明确互斥的 868 图。

## 关系和原型稳定性

固定最终 RMSNorm、EMA、欧氏模式。kNN@10 为两侧邻居身份重合：

| 状态 / 读出 | ImageNet：200 测试类 | COCO：固定 512 测试场景 |
| --- | ---: | ---: |
| 机会水平 | 5.03% | 1.96% |
| EMA bare Content mean | 14.90% | 8.16% |
| EMA native Content mean | 24.75% | 18.22% |
| EMA native Content last-sigma | 23.70% | 6.35% |
| EMA native query | **31.35%** | **30.21%** |
| raw native query | 30.75% | 29.92% |
| 三个初始化参考 native query | 10.55–12.15% | 3.75–4.08% |

原生 query 的 ImageNet RSA / CKA 为 0.422 / 0.463，COCO 为 0.329 / 0.533。raw 与 EMA 接近；初始化参考保留预训练 Qwen/VAE。

每类图像数的影响，固定 native query / final RMSNorm / 欧氏模式：

| 每类图像数 | ImageNet 图文近邻重合 | 两组独立图像原型近邻重合 |
| --- | ---: | ---: |
| 1 | 17.85% | 16.45% |
| 3 | 25.95% | 31.10% |
| 5 | 29.15% | 39.55% |
| 10 | 31.05% | 53.35% |
| 15 | 31.35% | 61.40% |

15 图原型的图像重复 RSA 为 0.922；独立文本模板 RSA 为 0.982、kNN@10 为 85.15%。原型先平均原始特征，再中心化或球面化。

## 拟合规模和子空间

COCO native query / final RMSNorm / 欧氏模式，测试始终为 2048 场景：

| 拟合场景数 | 完整 1024 维测试 R² | PCA 32 维测试 R² |
| --- | ---: | ---: |
| 512 | −0.028 | 0.222 |
| 2048 | 0.060 | 0.217 |
| 8192 | **0.101** | **0.222** |

8192-fit 的 full / 32-D 条件 95% 区间为 [0.085, 0.117] / [0.205, 0.238]，32-D 保留图像/文本方差 61.0% / 85.6%。raw 为 0.096 / 0.219，初始化参考为 −0.739～−0.711 / −0.679～−0.652。

两个 fit-bootstrap 参考的 full R² 为 0.092 / 0.085，32-D 为 0.228 / 0.220。允许单个整体尺度时，full / 32-D 为 0.306 / 0.376。

8192-fit 两侧样本秩为 1024；跨协方差在相对阈值 1e−6 下秩为 1017，条件数约 5.7×10⁹，32-D 约 1.86×10³。ImageNet 的 600 fit 类使 full 1024 维存在样本秩不足。

## dev 选择和跨域

源 dev 选择原生 query 的层和维度，四项均为 32 维。括号为固定拟合/选择后的 2000-bootstrap 条件 95% 区间：

| 源域 / 几何 | 选层 | 同域新语义单位 R² | 全部参数冻结后的跨数据集 R² |
| --- | ---: | ---: | ---: |
| ImageNet / 欧氏 | 14 | **0.443** [0.385, 0.495] | COCO：−0.868 [−0.988, −0.749] |
| ImageNet / 单位球 | 13 | **0.475** [0.425, 0.523] | COCO：**0.266** [0.250, 0.282] |
| COCO / 欧氏，8192 fit | 14 | **0.287** [0.247, 0.327] | ImageNet：**0.500** [0.470, 0.530] |
| COCO / 单位球，8192 fit | 14 | **0.374** [0.357, 0.389] | ImageNet：**0.208** [0.144, 0.268] |

四项同域的打乱 fit R² 依次为 −1.300、−1.182、−0.939、−0.901。映射均为 image→text；表中的迁移方向表示源数据集互换。

## ARO 困难负例

固定 final RMSNorm / native query / COCO 8192-fit / 32-D 欧氏映射，以余弦排序。868 图按 VG→COCO 身份与全部 COCO 池互斥：

| 严格身份隔离子集 | 正确率及条件 95% 区间 | 同任务打乱图像 | 正确率增量的 95% 区间 |
| --- | ---: | ---: | ---: |
| 属性，451 图 | **53.66%** [49.22, 58.76] | 45.90% | [1.33, 14.41] 个百分点 |
| 关系，417 图 | **50.84%** [46.28, 55.16] | 49.64% | [−5.28, 7.43] 个百分点 |

负欧氏距离排序的属性/关系正确率为 46.56% / 52.04%。只排除 76 个已知重合的 1924 图版本，query 余弦正确率为 55.43% / 52.56%，打乱图像为 52.53% / 52.25%。属性有局部图像增益，关系增益区间跨 0。

## caption、模板和位置

COCO 同场景平均 1/3/5 条 caption，kNN@10 为 21.50% / 28.40% / 30.20%，冻结 32-D 映射的 R² 为 0.172 / 0.211 / 0.222。同场景两组 caption 的重复 kNN@10 为 43.26%、RSA 为 0.790。

ImageNet 的 a→b 模板切换使 32-D 欧氏 R² 从 0.144 变为 0.126，单位球从 0.194 变为 0.177。

COCO 固定 512 test 场景、最终 query、32-D 欧氏映射：

| 条件 | 首图像 query 的 0-based 空间位置 | R² |
| --- | --- | ---: |
| native | (7,8) | 0.239 |
| sigma 1 | (7,7) | 0.242 |
| sigma 2 | (14,13) | −0.104 |
| VAE 后验均值 | (7,8) | 0.239 |

sigma2 同时移动 query 槽位，跨模态 kNN@10 仍为约 30.10%（native 30.21%）。只用 512 cal 减去位置均值偏移后 R² 为 −0.175；两槽位间单独拟合的 32-D 正交映射测试 R² 为 0.646。结果显示关系保留与固定坐标稳定是不同性质，后验均值替换影响较小。

## 产物

完成 630 条逐层记录、20160 个映射设置、216 条扰动记录；1512 个常量或秩不足子空间标记无效。864 个正式特征分片、867466 行通过检查，29 项测试通过，保存 16 组 PNG/PDF。

[CSV][csv]、[完整结果][results]、[dev 选择][selection]、[选择后区间][ci]、[扰动][robust]、[ARO 身份隔离][arodata]、[位置诊断][position]；图表：[逐层关系][geometry]、[映射泛化][rotation]、[拟合规模][learning]、[原型大小][prototype]、[ARO][arofig]；检查：[特征][fa]、[样本][sa]、[分析][aa]。

[csv]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/geometry-v4.csv
[results]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/results-geometry-v4.json
[selection]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/dev-selected-geometry-v4.json
[ci]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/selected-uncertainty-v4.json
[geometry]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/figures/layers-knn-centered_euclidean.png
[rotation]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/figures/rotation-test-centered_euclidean.png
[learning]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/figures/coco-fit-size-final-norm.png
[prototype]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/figures/imagenet-prototype-size-centered_euclidean.png
[arofig]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/figures/aro-controls-centered_euclidean.png
[fa]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/feature-audit-v4.json
[sa]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/sample-audit-v4.json
[aa]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/analysis-audit-v4.json
[cleanup]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/notebook-cleanup.json
[robust]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/robustness-geometry-v4.json
[arodata]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/aro-disjoint-v4.json
[position]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/position-exploration-v4.json
