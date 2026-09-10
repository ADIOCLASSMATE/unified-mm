# B 关系几何 V4 协议

扩大 [V3](B_GEOMETRY_PROTOCOL_V3.md) 的样本与拟合规模，新增原型稳定性、困难负例、顺序和后验控制。冻结 B step 95415，结果见 [V4 报告](B_GEOMETRY_V4_RESULTS_20260907.md)。

## 数据

| 数据 | 规模与划分 | 控制 |
| --- | --- | --- |
| ImageNet | 1000 类 × 32 图；每类 2 cal / 15 a / 15 b；12 模板按 4/4/4 分组；类别 600 fit / 200 dev / 200 test | 排除 V1/V2 的 18000 图；a 原型嵌套取 1/3/5/10/15 图，b 图和模板作独立参考 |
| COCO | 11776 场景、58909 caption；512 cal / 8192 fit / 1024 dev / 2048 test | 排除旧检索 test 和 SugarCrepe 图像；fit 曲线为 512/2048/8192 |
| ARO / VG | 1000 关系图 + 1000 属性图，原图身份不重复 | 正负 caption 排序，不拟合映射 |

后续 VG→COCO 身份核验发现 76 个已知重合、1056 个未核对身份。补充报告明确互斥的 868 图（451 属性 / 417 关系），以及只排除已知重合的 1924 图。V5 主表采用 868 图版本。

## 特征

EMA 测 native / bare / neutral，raw 和 init42/43/44 测 native；7 个组合 × 30 层 × 3 读出，共 630 条记录。读出定义沿用 [V2](B_SEMANTIC_EMERGENCE_PROTOCOL_V2.md)。

图文各自独立前向，首批执行隐藏目标替换检查。模型前向使用固定 16 张 910B；保存源权重、样本、契约与分片身份。

## 几何与选择

指标、fit-only 中心/PCA/RMS/Q、dev 选择和跨域冻结沿用 V3。维度扩为 32 / 128 / 512 / full 1024；主几何使用 200 个 ImageNet 测试类和固定 512 个 COCO 测试场景，COCO 映射误差使用全部 2048 个测试场景。

保留测试方差覆盖、样本秩、跨协方差谱与条件数。V4 的 `rotation_identified` 表示两侧样本张成空间足够；跨协方差的秩和条件数另读。

## 扰动

sigma 1、sigma 2 和 VAE 后验均值控制，使用 COCO 全部 cal + 固定 512 test、ImageNet 固定 32 test 类。主分析参数冻结，汇总第 13、23 层及最终 RMSNorm；原始特征保存全部 30 层。

| 条件 | 首图像 query 的空间位置（从 0 计） |
| --- | --- |
| native | (7,8) |
| sigma 1 | (7,7) |
| sigma 2 | (14,13) |

sigma 同时改变上下文次序和 query 槽位。事后位置诊断用独立 512 cal 估计平移或位置间映射，结果单独保存。

## 执行入口

- [准备样本](../scripts/prepare_unified_geometry_v4.py)
- [采集监督](../scripts/run_unified_geometry_v4.py)
- [CPU 分析](../scripts/analyze_unified_geometry_v4.py)
- [绘图](../scripts/plot_unified_geometry_v4.py)

产物根目录：`output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/geometry-v4-20260907/`。
