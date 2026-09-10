# B 表征诊断 V2 协议

在冻结 B 权重下，测量图文检索、类别迁移、关系几何和困难负例。结果见 [V2 报告](B_SEMANTIC_EMERGENCE_V2_RESULTS_20260907.md)。

## 模型与读出

模型为 `unified-b-x0content-0p6b-100b-imagenet-split-s42-r1`，step 95415 的 EMA / raw，以及按项目初始化流程生成的 seed 42/43/44 参考。初始化参考保留预训练 Qwen/VAE；三个 seed 对应新增组件初始化。

EMA 测 bare、native、neutral；raw 和初始化参考测 bare、native。每侧独立前向，采集输入、28 个 block 和最终 RMSNorm，共 30 个位置。读出为：

| 读出 | 定义 |
| --- | --- |
| `content_mean` | 正文图像或文本 token 均值，排除提示、目标和 padding |
| `content_last_sigma` | 可见完整正文的最后一个 Content token |
| `query_native` | 首目标 query；native 使用 I2T 首文本位置、T2I 首图像位置，bare/neutral 使用共同文本 query |

提示和读出定义在采集前固定。每个分片首批替换隐藏目标，检查所有层读出不变。

文本 query mask 为 151669，图像 query mask 为 151672；初始化时两行相同，训练后各自更新。mask-only 控制只换 XT mask embedding，保持位置、token type、X0 和可见性一致。

## 数据

种子为 20260907，按图像身份划分；同图描述归入同一集合。

| 数据 | 规模与划分 |
| --- | --- |
| COCO | 2500 图；500 cal / 500 fit / 500 dev / 1000 test；test 共 5003 条描述 |
| ImageNet | 1000 类 × 8 图；每类 2 cal / 3 fit / 3 test；6 个模板按 2/2/2 划分 |
| SugarCrepe | 1050 图；7 类负例各 25 cal / 25 dev / 100 test，合计 700 test |

校准均值来自 cal；映射和分类器来自 fit；参数选择来自 dev。ImageNet 分类探针按图像留出；类别外推映射另用 600 fit / 200 test 类，cal 覆盖全部 1000 类。

## 分析

- 检索：原始余弦、减模态均值后 L2 归一化，多正例 R@1/5/10。
- 映射：固定 ridge 与正交 Procrustes，含同容量打乱配对控制。
- 类别：图像域内线性分类、文本分类器迁移到图像、图像分类器迁移到文本。
- 几何：线性 CKA、去对角相似度秩相关及语义身份置换。
- SugarCrepe：正确与错误描述排序，附同类负例内打乱图像控制。
- 稳定性：提示、raw/EMA、初始化、mask-only 和三个 image sigma。

区间按图像身份 bootstrap，同图描述一起重采样。完整层曲线与固定最终层分别报告。

## 产物

根目录：`output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/`。

保存样本清单、逐层特征、完整指标、拟合参数及图表。此版完成冻结表征分析；训练目标与 head 干预未执行。类别隔离和更严格的旋转泛化检验见 [V3](B_GEOMETRY_PROTOCOL_V3.md)。
