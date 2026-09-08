# B 跨模态语义几何 V4：扩展实验

日期：2026-09-07。模型为指定 B run 的 step 95415，主模型 `hf_model-final-ema`。这次实际启动了固定 `dev-wjx-ascend`，在 16 张 910B 上新增提取特征；不是仅复用 V3 结果。模型始终冻结，没有新增训练。开发机已停止，Notebook 对象与共享盘产物均保留。

已完成全部 **630 条逐层记录、20160 个映射设置、216 条冻结坐标扰动记录**及身份隔离 ARO／位置补充；特征、数据身份、分析完整性均通过审计。保存 **16 组 PNG/PDF 图表**，相关 **29 项测试通过**。1512 个常量或不足秩的请求子空间按无效保留，没有用任意数值补成成功结果。

## 1. 对你的假设，当前结果支持到哪里

**支持较弱但实质的版本：B 的 backbone 表示中存在可泛化的跨模态关系结构和部分共同子空间。不支持把它直接概括为“全部表示只差一个全局旋转”，更没有证明 head 只做采样。**

这里的“有关联”不是图像向量与文本向量直接余弦高，而是：同一批语义单位在两个空间中的邻居、距离排序是否一致；只用拟合集求出的正交映射是否还能对应未参与拟合的类别/场景；源数据集的全部映射参数冻结后能否迁移。

你的直觉与 [Platonic Representation Hypothesis](https://proceedings.mlr.press/v235/huh24a.html) 的关系几何视角很接近。我们采用不拟合跨模态变换的 kNN 身份重合、RSA、[线性 CKA](https://proceedings.mlr.press/v97/kornblith19a.html)，再用 held-out 正交 Procrustes 检验更强的形状对应。它们不是可互换的“语义分数”：例如 CKA 高不自动意味着属性绑定正确，也不证明整体近似等距。

## 2. 本次测量范围

完整约束见 [V4 协议](B_GEOMETRY_PROTOCOL_V4.md)。采集清单在新特征得分之前固定；后续位置解释及 VG/COCO 身份修正单独标明，未改写原始清单或覆盖主结果。

| 数据 | 实际语义单位与划分 | 本次用途 |
| --- | --- | --- |
| ImageNet | 1000 类 × 32 张新图；排除 V1/V2 的 18000 张；每类 2 cal、15 a、15 b。映射类别 600 fit / 200 dev / 200 test；每类 12 个模板 | 更稳定的类别原型、独立图像视图、1/3/5/10/15 图原型曲线、模板变化 |
| COCO | 11776 个新场景、58909 条 caption；512 cal / 8192 fit / 1024 dev / 2048 test | 相对 B 的 ImageNet 图像训练的域外检验；512→2048→8192 拟合规模曲线 |
| ARO / VG | 1000 关系 + 1000 属性原图，内部不重复；身份审计后另报 417 关系 + 451 属性的明确 COCO-pool-disjoint 子集 | 配对正确 caption 与控制负例，不在 ARO 拟合映射 |

ImageNet 的“未见类别”仅对映射拟合而言，B 训练见过这些类别；COCO/VG 不代表已排除 Qwen/VAE 全部预训练污染。COCO 的 Karpathy train 是数据来源划分，不是说本次 B 训练过这些 COCO 图。

EMA 测 native / bare / neutral，final raw 与 init42/43/44 测 native。每种组合记录输入 0、28 个 block、最终 RMSNorm，共 30 个位置；每处读取 Content mean、Content last-sigma、首目标 query。只有 native 的两侧是异构原生 query，bare/neutral 都使用文本 query。三种读出不能混称为一个 embedding。

图像和文本分别独立前向，不向图像分支提供配对 caption，也不向文本分支提供真实目标图像。每个分片首批执行隐藏目标替换不变性检查。正式特征 **864 个分片、867466 行**通过数值、身份、形状及来源核验；EMA/raw 分别逐一核验源文件中 486/487 个存储张量的 BF16 加载值。

## 3. 不需要拟合旋转，就已经能看到关系对应

以下固定 final RMSNorm、EMA、仅减模态均值的欧氏模式。kNN@10 指两侧各自 10 个邻居的身份重合比例，**不是跨模态检索准确率**。

| 状态 / 读出 | ImageNet：200 测试类 | COCO：固定 512 测试场景 |
| --- | ---: | ---: |
| 机会水平 | 5.03% | 1.96% |
| EMA bare Content mean | 14.90% | 8.16% |
| EMA native Content mean | 24.75% | 18.22% |
| EMA native Content last-sigma | 23.70% | 6.35% |
| EMA native query | **31.35%** | **30.21%** |
| raw native query | 30.75% | 29.92% |
| 三个初始化参考 native query | 10.55–12.15% | 3.75–4.08% |

原生 query 的 RSA / CKA 分别为 ImageNet **0.422 / 0.463**、COCO **0.329 / 0.533**。Content mean 也有关系信号，所以不能把全部关联解释为末尾 query 的身份效应；但读出方式及提示条件确实重要。

初始化参考按项目初始化流程保留预训练 Qwen/VAE、初始化新增组件，不是该 run 的可信历史 step 0，也不是三个训练重复。初始化也有部分高于机会的结构；训练后增量的幅度比单独一个显著性 p 值更有解释力。

文本 query mask 是 ID **151669**，图像 query mask 是 ID **151672**，并非同一个 token 或绑定参数。初始化时两行复制为相同值，EMA 中余弦已为 **0.24327**。输入 0 的每侧 query 是常量，几何记为无效；不能用它来声称输入层语义已经统一。

## 4. 原型噪声会显著改变你的实验结论

固定 native query / final RMSNorm / 欧氏模式：

| 每类图像数 | ImageNet 图文近邻重合 | 两组独立图像原型近邻重合 |
| --- | ---: | ---: |
| 1 | 17.85% | 16.45% |
| 3 | 25.95% | 31.10% |
| 5 | 29.15% | 39.55% |
| 10 | 31.05% | 53.35% |
| 15 | 31.35% | 61.40% |

15 图原型的两组图像 RSA 为 **0.922**；两组文本模板 RSA 为 **0.982**、kNN@10 为 **85.15%**。同模态原型已经相当稳定，而跨模态仍有明显差距。不能把这些重复参考直接当作严格噪声上界，或取比值叫“语义完成度”。

这也解释为什么单图、少图类别原型容易低估关系。应先平均原始特征，再分别做中心化或球面处理；不能把归一化后平均与平均后归一化混为同一种估计。

## 5. 一个旋转能泛化多少：大样本与子空间

设图像、文本的 fit-only 表示分别为 X、Y，求 `min ||XQ−Y||², QᵀQ=I`。这里只允许平移、旋转/反射和两侧全局 RMS 单位换算；不逐轴白化。32/128/512 维使用两侧独立、仅在拟合集估计的 PCA。

报告 `R² = 1 − ||X_test Q−Y_test||² / ||Y_test||²`；Y 已用拟合中心居中，0 对应预测拟合均值，1 才是精确对应，负数不截断。单位球模式在这些步骤前还做了逐语义单位 L2 归一化，因此不是原始欧氏点云的同一命题。

固定 COCO native query / final RMSNorm / 欧氏模式，测试始终为 2048 个场景：

| 拟合场景数 | 完整 1024 维测试 R² | PCA 32 维测试 R² |
| --- | ---: | ---: |
| 512 | −0.028 | 0.222 |
| 2048 | 0.060 | 0.217 |
| 8192 | **0.101** | **0.222** |

大样本使完整空间从弱负值变成正值，但仍远不接近 1；低维结果没有随着增加样本突然变成完整等距。final raw 对应 8192-fit 的 full / 32-D R² 为 **0.096 / 0.219**。三个初始化参考对应为 **−0.739～−0.711 / −0.679～−0.652**。

8192-fit 的 full / 32-D 条件 95% 区间分别为 **[0.085, 0.117] / [0.205, 0.238]**；32-D 保留测试图像 **61.0%**、文本 **85.6%** 方差。两个额外 fit-bootstrap 参考的 full R² 为 **0.092 / 0.085**、32-D 为 **0.228 / 0.220**，不是新的训练重复或置信区间。允许额外从 fit 估计一个整体收缩尺度后，两者 R² 为 **0.306 / 0.376**；这个较宽松的 similarity-Procrustes 结果也不接近精确对应。

**样本数充足不等于全空间旋转稳定可识别。** 8192-fit 的两侧样本数值秩达到 1024，但最终 query 的跨协方差在相对奇异值 1e−6 阈值下秩为 **1017**，条件数约 **5.7×10⁹**；32-D 的对应条件数约 **1.86×10³**。主要机器结果中的 `rotation_identified` 只表示两侧样本张成空间足够，不能当作跨协方差条件良好的保证。ImageNet 只有 600 拟合类，其 full 1024-D 始终存在显然的样本不可识别问题。

## 6. 只用开发集选层后，迁移是否成立

下表为 native query；在各源域开发集按正交 R² 选层和维度，不看测试集挑峰值。四项均选到 32 维。括号是固定拟合/开发选择后按独立测试语义单位做 2000 次 bootstrap 的条件 95% 区间，不含新训练或拟合/开发抽样的不确定性。

| 源域 / 几何 | 选层 | 同域新语义单位 R² | 全部参数冻结后的跨数据集 R² |
| --- | ---: | ---: | ---: |
| ImageNet / 欧氏 | 14 | **0.443** [0.385, 0.495] | COCO：−0.868 [−0.988, −0.749] |
| ImageNet / 单位球 | 13 | **0.475** [0.425, 0.523] | COCO：**0.266** [0.250, 0.282] |
| COCO / 欧氏，8192 fit | 14 | **0.287** [0.247, 0.327] | ImageNet：**0.500** [0.470, 0.530] |
| COCO / 单位球，8192 fit | 14 | **0.374** [0.357, 0.389] | ImageNet：**0.208** [0.144, 0.268] |

同域四项打乱 fit 配对的 R² 分别为 −1.300、−1.182、−0.939、−0.901。ImageNet 欧氏/球面选择在测试图像/文本分别保留 **85.3% / 65.3%**、**86.0% / 68.8%** 方差；COCO 分别保留 **76.6% / 80.4%**、**73.8% / 79.9%**。因此是明显的共同子空间证据，不是保留了几乎全部信息后证明全空间同构。

跨数据集方向明显不对称，而且依赖是否球面化。这里所有映射方向均为 image→text，“两方向迁移”指源数据集互换，不是分别拟合两个图文方向。共享语义并不要求所有模态私有细节也等距；本实验不能把“完整空间未等距”进一步推出“backbone 没有统一抽象语义”。

## 7. 困难负例：不能把几何关联称作完整组合语义

[ARO](https://arxiv.org/abs/2210.01936) 特意检验属性、关系及顺序等组合信息。我们的 ARO 只使用已有 VG 属性/关系子任务与缓存裁剪，不是对整个 benchmark 的标准生成/似然评估；比较的是冻结表征经过 COCO 映射后对正确/错误 caption 的排序。

原始 2000 张 ARO 图内互不重复，但与 COCO 池并不天然互斥。依据作者维护的 [VG 官方身份字段](https://homes.cs.washington.edu/~ranjay/visualgenome/api_readme.html)，发现 76 个已知重合、1056 个无法核对的 COCO 链接；严格子集仅保留明确不在全部 V4 COCO 池中的 **451 属性 / 417 关系**图。

固定 final RMSNorm、native query、COCO 8192-fit、32-D 欧氏映射，以余弦排序：

| 严格身份隔离子集 | 正确率及条件 95% 区间 | 同任务打乱图像 | 正确率增量的 95% 区间 |
| --- | ---: | ---: | ---: |
| 属性，451 图 | **53.66%** [49.22, 58.76] | 45.90% | [1.33, 14.41] 个百分点 |
| 关系，417 图 | **50.84%** [46.28, 55.16] | 49.64% | [−5.28, 7.43] 个百分点 |

属性存在一些图像依赖信号，但绝对正确率区间包含 50%；关系的余弦证据很弱。以负欧氏距离排序，属性/关系正确率为 **46.56% / 52.04%**，虽部分相对打乱图像的差值区间为正，绝对成绩仍不能叫作良好的组合理解。这里的多个区间未经整个实验指标家族校正。

保留身份未能核对者、只排除 76 个已知重合的较大子集（967 属性 / 957 关系），对应 query 余弦正确率为 **55.43% / 52.56%**，打乱图像为 **52.53% / 52.25%**。结论对“相对哪个零模型”很敏感；不能只看超过 50% 就认定模型使用了正确图像关系。

原始 2000 图的完整层结果继续存档，严格子集的固定最终层覆盖全部七个状态/提示组合、三个读出、两种几何、四个维度。不能把挑选其中某个较高的维度/读出分数再当成独立确认。

## 8. caption、模板、后验与 query 位置

固定最终 native query、COCO、欧氏模式：同场景平均 1/3/5 条 caption 时，kNN@10 为 **21.50% / 28.40% / 30.20%**；冻结 all-caption 拟合的 32-D 映射，测试 R² 为 **0.172 / 0.211 / 0.222**。同场景前两条对其余 caption 的重复参考 kNN@10 为 **43.26%**、RSA 为 **0.790**。这里每张图仍只算一个独立语义单位。

ImageNet 最终 query 的 32-D 映射只在 a 模板拟合，切换到 b 模板后，欧氏 R² 从 **0.144 降至 0.126**，单位球从 **0.194 降至 0.177**。这比重新拟合模板后的成绩更能反映原映射稳定性，但本轮模板是通用类名句式，不覆盖所有自然语言措辞变化。

顺序/后验扰动沿用冻结 native 的全部分析参数。COCO 固定 512 test 场景上，最终 query 的 32-D 欧氏 R²：

扰动统计固定在第 13、23 层及最终 RMSNorm 三处；原始采集仍保存全部 30 个位置。以下仅列固定最终层。

| 条件 | 首图像 query 的 0-based 空间位置 | R² |
| --- | --- | ---: |
| native | (7,8) | 0.239 |
| sigma 1 | (7,7) | 0.242 |
| sigma 2 | (14,13) | −0.104 |
| VAE 后验均值 | (7,8) | 0.239 |

sigma 并不是固定位置的纯顺序干预：图像上下文的 Content 可见次序改变，文本条件首图像 query 的空间槽位也改变。远处槽位的文本 query 对 native 的同样本余弦约 **0.494**，但图文跨模态 kNN@10 仍约 **30.10%**，接近 native 的 **30.21%**。这正说明直接坐标稳定、关系稳定及固定图文映射稳定是不同问题。

事后只用独立 512 cal 场景估计位置均值偏移，再冻结全部已有图文映射测试：去掉偏移没有恢复 sigma2（R² 为 **−0.175**）。同一文本两槽位各自校准后的余弦为 **0.658**、内部几何 RSA 为 **0.690**、近邻重合 **58.79%**；用 512 cal 拟合它们之间的 32-D 正交映射，独立测试 R² 为 **0.646**。它既不是简单常量平移，也不是语义完全消失，而是位置相关的读出发生了实质变化。

因此，**不能把任意图像位置的生成 query 都当成同一个固定坐标系下的全局语义摘要**。Content mean 没有这个目标槽位选择问题，但观察图像侧仍依赖 sigma 上下文。后验均值替换本身影响很小，不能据此认为所有 query 扰动都稳定。

## 9. “纯重建涌现”与“head 只采样”需要怎样表述

本次 B 的 [实际训练配置](../output/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/config.yaml) 使用预训练 Qwen3-0.6B-Base、MAR KL16 VAE、ClimbMix 文本，以及 ImageNet 合成 caption 的 I2T/T2I 配对条件；文本交叉熵与图像 flow loss 的权重为 0.05 / 1.0。它没有在本次表示实验中增加对比损失，但也不是随机初始化、无跨模态配对信息的“只有压缩”。

需要分开四个命题：

- **Backbone 内存在跨模态可复用关系：有支持。** 读出在 head 之前；原生 query 的 held-out 形状对应、部分跨域迁移及 Content 的关系信号构成证据。
- **全部模态共享一个固定、全局等距坐标系：没有证实。** 全维误差、跨域不对称、提示和位置依赖都与这个强版本有距离；弱方向的数值病态又限制了简单否定。
- **这种增量只由重建目标造成：没有因果证明。** 缺少匹配的冻结 backbone、打乱训练图文配对、去除预训练等训练臂。三个初始化参考不能分离这些贡献。
- **Head 不做语义计算、仅负责采样：本实验不能判断。** B 的 flow head 本身有 8 层、宽度 1280，并接收 backbone query/content 条件；“backbone 中已经有语义”不推出“head 中不存在进一步语义处理”。LM 的 unembedding 与 flow 的条件向量场也不能与一个无参数采样器混同。

更一般地，重建/预测损失不唯一规定内部坐标：对表示做可逆重参数化并相应改变解码器，可以保留输入输出功能。因此“只差旋转”是一个额外、强而可检验的结构假设，不是“能压缩／能重建”的数学必然结果。BPE tokenizer 与已学习的 VAE encoder 也不能都视为完全没有数据结构的同类组件。

目前最准确的概括是：**在预训练与配对生成训练的条件下，B 学到了跨模态的关系对应；这种对应以特定中间层、读出和部分子空间最清楚。复杂绑定、位置无关性和 head 的因果分工仍然是开放问题。**

## 10. 统计与可复现产物

主几何使用 199 次整语义身份置换，kNN 排除自身，RSA 不用对角线；同一曲线沿层使用相同置换，另存 max-over-layer 零分布。它只校准给定状态/提示/读出/数据/指标的一条层曲线，不是整个探索实验的统一多重比较控制。

固定最终层有按语义组的 2000 次 bootstrap；预先指定第 13、23 层及最终层、Content mean / query、full / 32-D 的 native 条件另有两次 fit 重采样。Caption 始终随所属图像分组。所有方差保留、秩、条件数、负 R²、同容量打乱拟合、同模态重复参考、措辞/后验/顺序结果均保留，不只展示正向分数。

主要入口：

- [逐层 CSV][csv]、[完整机器结果][results]、[开发集选择结果][selection]、[四个主要选择的测试区间][ci]。
- [全层邻居关系][geometry]、[各层旋转泛化][rotation]、[COCO 拟合规模][learning]、[ImageNet 原型大小][prototype]、[严格身份隔离 ARO 对照图][arofig]。图目录还包含另一种几何、RSA、跨域迁移、caption 与扰动曲线。
- [特征审计][fa]、[数据身份审计][sa]、[分析完整性审计][aa]、[开发机停止记录][cleanup]。
- [冻结坐标扰动结果][robust]、[ARO 身份隔离结果][arodata]、[事后 query 位置诊断][position]。

采集入口是 [run_unified_geometry_v4.py](../scripts/run_unified_geometry_v4.py)；主 CPU 入口是 [analyze_unified_geometry_v4.py](../scripts/analyze_unified_geometry_v4.py)，绘图入口是 [plot_unified_geometry_v4.py](../scripts/plot_unified_geometry_v4.py)。CPU 分析使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，单线程 BLAS/Torch 配合多进程；不在 Agent 环境触发 NPU。绘图依赖隔离临时目录中的 Matplotlib 3.10.9 / NumPy 1.26.4，没有修改项目的运行依赖。

本轮使用 Inspire / Ascend 技能限定到固定的 16 卡开发机，执行真实 NPU 算术与提取检查，并在特征审计通过后停止该 Notebook；没有删除基础设施或旧结果，没有改训练权重，也没有计算数据/权重哈希。

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
