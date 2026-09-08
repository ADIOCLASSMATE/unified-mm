# B 跨模态语义实验 V2：执行结果

日期：2026-09-07。模型：`unified-b-x0content-0p6b-100b-imagenet-split-s42-r1`，最终 step **95415**。主结果使用 `hf_model-final-ema`，另测 final raw 与三个初始化流程参考。

已完成 [V2 设计](B_SEMANTIC_EMERGENCE_PROTOCOL_V2.md) 的冻结模型阶段：Content 提示控制、原生首目标 query、mask-only 干预、逐层几何/检索/探针、初始化参考和稳定性检查。**没有执行新增训练或原生 head 的生成因果干预。** 所有 NPU 前向在 `dev-wjx-ascend` 的 16 张 910B 上完成，结束后已停止该 Notebook 并读回确认 `STOPPED`；没有删除 Notebook、修改模型权重或训练配置。

## 1. 结论先行

结果支持一个有限但实质性的命题：**在预训练 Qwen/VAE 的基础上，B 的配对生成训练之后，backbone 中存在初始化参考没有呈现的、可跨模态复用的语义结构；它明显依赖上下文和读出方式，并不是一个处处直接对齐的统一向量空间。**

- 不依赖共有末尾文本 query 的 **X0 Content mean**，在原生任务前缀下已有直接图文检索、跨模态类别探针和部分困难负例信号，因此 V1 的关联不能完全归因于“用了同一个文本 mask”。
- 去掉前缀后，直接检索接近随机量级；仅加中性 `Input:` 已恢复相当一部分信号。不能把这一差异单独归因为任务词的语义，因为前缀长度、位置、首 token 和注意力上下文也一起变化。
- **原生异构 query** 的直接距离不如 Content，但关系结构较强，独立拟合集学习的正交映射可以明显提高检索。共享结构不等于逐维坐标一致。
- 类别和部分属性较清楚，关系、属性/对象绑定的证据弱。不能把本实验概括为“完整语义已经统一”。
- 不能据此证明“随机初始化 + 无配对监督的纯重建必然涌现语义”，也不能证明“head 只做采样，没有语义计算”。本次没有训练因果对照和 head 干预。

## 2. 实际测了什么

### 数据与隔离

协议与样本清单在读取 V2 分数前固定，种子为 `20260907`。图像、文本始终独立前向：图像输入不含配对 caption，文本输入不含真实目标图像。

| 数据 | 校准 / 拟合 / 开发 / 测试 | 用途 |
| --- | --- | --- |
| COCO Karpathy 缓存 | 500 / 500 / 500 / 1000 张图；测试有 5003 条 caption | 场景检索、几何、配对映射 |
| ImageNet 1000 类 | 每类 2 / 3 / 0 / 3 张图，共 8000 张；6 个文本模板按 2 / 2 / 0 / 2 分组 | 域内与跨模态探针、类别外推映射 |
| SugarCrepe 7 类困难负例 | 每类 25 / 0 / 25 / 100 个独立图像，共 1050 图、2100 候选文本 | 属性、对象、关系与绑定诊断 |

COCO 替换 V1 的 Flickr 数据；ImageNet 图像 ID 与 V1 使用的 10000 张不重叠。SugarCrepe 按真实 COCO 图像身份排除了本次选中的全部 COCO 图像，不同困难负例类别之间也不复用图像。一个图像的全部 caption 保持同组。

这是固定子集的表征诊断，**不是正式 COCO 5K 检索或 ImageNet 零样本基准**。新样本指未参与 V1 分析，不意味着已经排除模型全部预训练/训练数据污染；类名及部分模板也不是此前从未接触的词面。

### 三种上下文与三个主要读出

| 条件 | 图像上下文 | 文本上下文 | query |
| --- | --- | --- | --- |
| `bare` | BOI + 256 latent + EOI，无文本前缀 | caption 正文，无前缀 | 两侧均为末尾文本 query |
| `neutral` | `Input:` + 图像 | `Input:` + caption | 两侧均为末尾文本 query |
| `native` | `Describe this image in one detailed caption:` + 图像 | `Generate an image matching this description:` + caption | I2T 首目标文本 query 对 T2I sigma 首目标图像 query |

V2 均不使用 V1 的共同语义后缀。原生图像首目标的空间位置在一个 sigma 复本内对所有样本固定，不由类名或配对图像 ID 决定。

- `content_mean`：只对已观察正文 / latent 的 X0 求均值，排除提示、边界、padding 和隐藏目标。
- `content_last_sigma`：数据 token 中 sigma 最大的 X0，能够看到全部先前数据和自身；仍有末 token / 空间位置偏置。
- `query_native`：读取该条件所定义的目标 query。注意这个字段在 `bare` / `neutral` 中表示共有文本 query，**只有 `native` 是原生异构 query 对**。

保持 B 原有 X0 对角可见、XT 严格 sigma 因果规则。采集 0 号输入、28 个 block 输出及最终 RMSNorm，共 **30 个位置**。最终 Norm 是模型自身的变换，不是分析者学习的对齐层。

主指标逐层、逐条件、逐读出，分别用独立校准集估计两个模态均值，然后先减均值、再 L2 归一化。原始余弦结果也保留。均值不使用测试集；不同条件之间不混用均值。

## 3. 固定最终层的主结果

以下均为 final RMSNorm 后、模态中心化后的 COCO 测试 R@1（百分比）。1000 图像候选、5003 caption；两个方向的随机期望均约 **0.1%**。初始化范围是 seed 42/43/44 三个参考的最小—最大值，不是置信区间。

| 模型 / 上下文 / 读出 | 图→文 R@1 | 文→图 R@1 |
| --- | ---: | ---: |
| EMA，bare，Content mean | 0.20 | 0.10 |
| EMA，neutral，Content mean | 4.70 | 4.78 |
| EMA，native，Content mean | **16.80** | **11.35** |
| EMA，native，原生异构 query | 6.00 | 3.46 |
| EMA，neutral，共有文本 query | 12.80 | 6.30 |
| final raw，native，Content mean | 15.70 | 10.27 |
| final raw，native，原生异构 query | 4.90 | 2.74 |
| 初始化参考，native，Content mean | 0.10–0.40 | 0.04–0.12 |
| 初始化参考，native，原生异构 query | 0.00–0.20 | 0.08–0.18 |

EMA native Content 的 R@1/5/10 分别为图→文 **16.80 / 39.00 / 50.20**，文→图 **11.35 / 27.32 / 38.40**。其不减均值的 R@1 是 **8.60 / 4.22**；原生 query 不减均值只有 **0.50 / 0.38**。中心化有帮助，但不会把所有读出都变成同一种表示。

按图像连同全部 caption 做 2000 次 bootstrap，固定候选池与校准均值：

| EMA 条件 | 图→文 R@1 的 95% 区间 | 文→图 R@1 的 95% 区间 |
| --- | --- | --- |
| bare Content | [0.00, 0.50] | [0.02, 0.20] |
| native Content | [14.50, 19.10] | [9.97, 12.72] |
| native query | [4.60, 7.50] | [2.70, 4.28] |
| neutral query | [10.80, 14.90] | [5.44, 7.20] |

补充的 999 次整图身份置换保留 caption 分组，native Content 的图→文 / 文→图随机均值为 **0.098 / 0.100%**，两方向描述性置换 p 均为 0.001；EMA bare Content 对应 p 为 0.251 / 0.545。p 未做整个指标家族的多重比较校正；不把某个接近随机量级但偶然较小的 p 当作“语义涌现”结论。零命中时经验 bootstrap 可能退化为零宽区间，不代表总体成功概率严格为零。

## 4. 每一层：深度曲线不是训练时间轴

完整 30 个位置、2610 行分析见 [逐层 CSV][csv]、[完整 JSON][results]；[曲线 PNG][plot] / [PDF][pdf]。下面取固定的深度间隔帮助阅读，不用于选择最优测试层。

| 位置 | native Content 图→文 / 文→图 R@1 | native query 图→文 / 文→图 R@1 | Content / query CKA |
| --- | --- | --- | --- |
| 输入 0 | 0.10 / 0.12 | 0.10 / 0.10 | 0.179 / 0.000 |
| block 4 | 0.80 / 0.64 | 3.00 / 1.34 | 0.235 / 0.387 |
| block 8 | 2.10 / 1.26 | 4.20 / 2.46 | 0.260 / 0.543 |
| block 12 | 8.40 / 7.38 | 8.80 / 5.98 | 0.332 / 0.607 |
| block 16 | 13.40 / 7.14 | 7.40 / 4.30 | 0.338 / 0.483 |
| block 20 | 16.00 / 8.57 | 3.60 / 2.30 | 0.326 / 0.409 |
| block 24 | 15.00 / 9.47 | 5.70 / 2.90 | 0.330 / 0.558 |
| block 28，Norm 前 | 8.20 / 4.58 | 2.20 / 2.04 | 0.219 / 0.364 |
| final RMSNorm 后 | 16.80 / 11.35 | 6.00 / 3.46 | 0.364 / 0.485 |

Content 的直接对齐总体在中后层更明显；原生 query 的曲线不是单调增长。最后一个 block 与最终 RMSNorm 后不可混称“最后一层”：学到的逐通道缩放、逐 token 归一化及后续池化会改变几何，即使之后再次 L2 归一化也不等价。

只用独立开发集图→文 R@1 选层，固定同分时选较早层：native Content 选 final Norm，native query 选 block 12，其测试结果是 **8.80 / 5.98**；neutral 共有文本 query 选 block 27，其测试结果是 **18.10 / 13.25**。这是开发集选层结果，不是事后挑测试峰值。详见 [选层结果][selected]。

这里只能描述最终模型沿深度的表征变化，不能说“训练到某一步发生了涌现”。

## 5. 原生 query 与 mask 身份效应

原生文本 mask 是 `<|mdm_mask|>`，ID **151669**；图像 mask 是 `<|img_mask|>`，ID **151672**。同一张表的不同行，不绑定参数。EMA 的两行余弦为 **0.24327**；三个初始化参考中两行逐元素相等，因为初始化时复制。

对固定的原生上下文和槽位，仅替换选定位置的初始 XT 向量；X0、token types、RoPE / 位置和可见集合不变：

| 图像上下文 query 的 mask / 文本上下文 query 的 mask | 图→文 R@1 | 文→图 R@1 |
| --- | ---: | ---: |
| 文本 / 图像：原生 | 6.00 | 3.46 |
| 文本 / 文本：只改文本上下文的图像槽位 | 9.50 | 5.54 |
| 图像 / 图像：只改图像上下文的文本槽位 | 3.70 | 1.96 |
| 图像 / 文本：两侧交换 | 4.40 | 2.88 |

同为文本 mask 有帮助，但同为图像 mask 没有相同效果。**这不是只要 mask ID 相同就一定对齐。** 非原生注入可能分布外，不称为原生图像 query 评测。

在 native 上下文中，更换 mask 后，分别减掉各自校准均值，同一样本 query 的平均余弦为图像侧 **0.235**、文本侧 **0.784**；中心化变化 RMS / 原生样本离散 RMS 为 **1.033 / 0.754**。图像侧变化与跨样本离散已是同一量级，说明 mask 会改变读取上下文的结果，而不仅是加一个可去掉的常量偏移。

实现审计确认，mask-only 干预下每层完整 X0 张量逐元素不变。因此 native Content 的 16.8% 不可能由本次末尾 XT mask 替换直接造成。

## 6. 关系结构共享，不等于坐标已经相同

按场景将同图 caption 聚合后比较两侧内部关系；99 次整图对应置换得到几何零分布。

| final Norm 条件 | CKA | 置换 CKA 均值 | 去对角相似度秩相关 |
| --- | ---: | ---: | ---: |
| EMA bare Content | 0.236 | 0.026 | 0.190 |
| EMA native Content | 0.364 | 0.023 | 0.299 |
| EMA native query | 0.485 | 0.019 | 0.422 |
| EMA neutral query | 0.477 | 0.031 | 0.364 |
| 初始化 42 native Content | 0.134 | 0.017 | 0.104 |
| 初始化 42 native query | 0.111 | 0.022 | 0.081 |

bare Content 虽几乎不能直接检索，CKA 仍大于随机，初始化也有非零结构。共同的低层属性、场景统计或其他混杂都可能贡献 CKA，不能用 CKA 单独定义语义成功。

在独立 500 张 COCO 拟合图上，固定正则 ridge 或带拟合均值的正交 Procrustes 从图像映射到文本坐标，再在 1000 张测试图上评测。图→文 / 文→图始终是同一个 image→text 映射产生的分数矩阵及其转置；不是另外拟合一个逆映射。

| EMA final Norm | 不学映射 | 配对 ridge | 配对正交映射 | 打乱配对正交映射 |
| --- | --- | --- | --- | --- |
| bare Content | 0.20 / 0.10 | 0.90 / 1.66 | 2.20 / 1.84 | 0.10 / 0.08 |
| native Content | 16.80 / 11.35 | 10.50 / 10.37 | 13.90 / 10.47 | 0.00 / 0.02 |
| native query | 6.00 / 3.46 | 12.80 / 10.25 | **19.90 / 9.81** | 0.00 / 0.06 |

对于 Content，已有的直接坐标反而胜过这个小拟合集学习的映射；对于原生 query，映射明显提高可对应性。映射成功是“低容量后处理可建立对应”的证据，不应写成“原生 head 接口已直接对齐”。

另在 ImageNet 600 个拟合类别上学习映射，在不参加映射拟合的 200 个类别上测 600 张图、400 条类名文本，随机同类命中为 0.5%：

| EMA final Norm | 不学映射 | 配对正交映射 | 打乱配对正交映射 |
| --- | --- | --- | --- |
| native Content | 18.33 / 27.25 | 9.83 / 20.00 | 0.83 / 1.25 |
| native query | 12.17 / 10.00 | 17.00 / 22.75 | 0.83 / 1.00 |

这个类别隔离针对**配对映射的拟合**。无标签校准集覆盖全部 1000 类，包括测试类别，所以不是整条分析管线严格不接触测试类别分布的零样本设置。

## 7. 语义内容：类别较强，关系与绑定有限

### ImageNet 探针

固定正则 ridge；源模态每类 3 张图或 2 条文本拟合分类器，目标模态使用另一组图像 / 文本模板测试。表中均为 1000 类 top-1，随机为 0.1%。

| final Norm | 图像域内分类 | 文本分类器→图像 | 图像分类器→文本 |
| --- | ---: | ---: | ---: |
| EMA bare Content | 3.03 | 0.10 | 0.10 |
| EMA neutral Content | 20.27 | 1.43 | 9.65 |
| EMA native Content | 24.30 | 4.57 | 12.50 |
| EMA native query | 31.63 | 1.70 | 0.75 |
| EMA neutral 共有文本 query | 30.43 | 10.17 | 7.20 |
| 初始化参考 native Content | 1.50–2.03 | 0.00–0.07 | 0.00–0.05 |
| 初始化参考 native query | 1.67–2.03 | 0.07–0.13 | 0.00–0.10 |

文本域内接近 100% 主要说明类名在模板中可读，不能据此声称通用语言语义能力。跨模态分类器共享 1000 个标签词表，测试图像与拟合图像分离，但不属于未见类别分类。

同数据、同采样 / BF16 输入舍入 / 校准与固定正则的 VAE-only 图像域内参考：空间均值 16 维 **1.87%**，直接展平 4096 维 **0.73%**。这是两个简单线性读出，不是 VAE 语义能力上限。native backbone 特征更易读出类别，bare 的提升则小得多。见 [VAE 参考][vae]。

### SugarCrepe

700 个独立测试图，每类 100 个；正负候选独立编码，余弦两选一，平分时给半分。按类别分层、按图像 bootstrap 2000 次；打乱图像的对照在同一困难负例类别内进行。

| EMA final Norm | 正确率 | 95% 区间 | 类内打乱图像后 | 相对打乱的差值 95% 区间（百分点） |
| --- | ---: | --- | ---: | --- |
| bare Content | 47.29 | [43.57, 51.00] | 44.29 | [-2.00, 8.00] |
| native Content | **58.86** | [55.28, 62.43] | 52.71 | [0.71, 11.14] |
| native query | 55.29 | [51.43, 59.00] | 53.57 | [-3.57, 6.86] |
| neutral Content | 54.71 | [51.14, 58.43] | 48.71 | [0.71, 11.29] |

native Content 比打乱图像更好，而 native query 相对这一对照的优势区间包含零。不能只拿高于 50% 就排除候选文本偏好。

| 困难负例类型 | native Content | native query |
| --- | ---: | ---: |
| add_att | 62 | 66 |
| add_obj | 50 | 52 |
| replace_att | 76 | 51 |
| replace_obj | 69 | 62 |
| replace_rel | 54 | 54 |
| swap_att | 58 | 55 |
| swap_obj | 43 | 47 |

这些是描述性子类分数，未做七类及层间的多重显著性声明。困难负例沿用缓存中的 SugarCrepe 标签，没有逐张人工重审；没有另建干净的数量 / 颜色最小对照集。不能将对象/属性替换上的优势外推成稳健的关系和组合语义。

## 8. 初始化与训练归因

参考模型使用该 run 的配置及项目 `load_model_tokenizer` 完整 B 初始化路径：Qwen3-0.6B-Base、特殊 token 扩展、图像 projector 与 flow head 初始化；同样使用 MAR KL16 缓存 latent。seed 42/43/44 只改变新初始化部分。抽查预训练第一层 Q 投影、最后一层 MLP 权重与 Qwen 源权重逐元素相等，并确认初始两个 mask 行相等。

没有可信的历史 step-0 权重，所以必须称“初始化流程参考”，不能称这个 run 的真实 step 0。三个初始化种子也不是三个训练重复。raw 与 EMA 结果分别列出，不把权重平均方式视为新增训练阶段。

native Content 从初始化约 0.1–0.4% 的直接图→文命中到 final raw 15.7%、EMA 16.8%，配合类别探针和困难负例，支持训练后存在很大的表征增量。不过没有冻结 backbone 或打乱训练配对的匹配训练臂，无法隔离：

1. 新 projector 如何把图像接入已有语言结构；
2. backbone 更新建立了哪些新共享结构；
3. 预训练 LM / VAE、配对数据和生成目标各自贡献多少。

实际目标是文本 CE 与图像 flow matching，在图文配对条件和 `climbmix, t2i, climbmix, i2t` 混合下训练。未加入显式对比 / 分类 / 表征对齐损失，**不等于没有监督、所有模块从零初始化或只有一个字面重建损失**。

现存顶层检查点为 92000、94000、95415；它们不能提供早期时间轴。本次没有把 loss 日志或层深曲线当作早期表征权重的替代。

## 9. 稳定性与实现核验

### 固定 100 图候选池的 sigma 检查

使用预先选定的相同 100 张 COCO 测试图及 500 caption，沿用相同 500 图校准集；这里随机 R@1 为 **1%**，不能与 1000 图主结果的绝对分数直接比较。

| 条件 | sigma 0 | sigma 1 | sigma 2 |
| --- | --- | --- | --- |
| native Content，图→文 / 文→图 | 48.0 / 33.2 | 47.0 / 33.4 | 47.0 / 32.6 |
| native query，图→文 / 文→图 | 34.0 / 21.0 | 38.0 / 24.8 | 32.0 / 21.8 |
| bare Content，图→文 / 文→图 | 2.0 / 2.2 | 3.0 / 3.0 | 1.0 / 2.2 |
| bare query，图→文 / 文→图 | 2.0 / 1.2 | 1.0 / 1.8 | 1.0 / 2.0 |

native Content 结论在这三个 sigma 复本下稳定；query 有更明显位置 / 顺序波动。bare 使用后验均值时 Content 为 **2.0 / 2.2**、query 为 **2.0 / 1.4**，未出现足以解释主差异的恢复。后验均值控制仅对 bare 子集执行，不能据此声称所有 native 结果都对后验选项不敏感。

### 正确性检查

- 最终 EMA 的 **486 个存储权重张量**逐一与实际加载模型的 BF16 对应张量比较，全部一致；state dict 有 487 键来自 tied embedding。
- 在原始 V1 的 rank-0 图像 / 文本各 16 个样本上复现原协议：30 位置、两种旧读出逐元素一致，最大绝对误差 0。
- 同一可见 I2T 上下文、缓存与 sigma 下，V2 native 图像 Content 与 V1 Content 逐元素一致。因此 bare 与 native 的差异不是新提取器回归。
- 每个提取数据分片首批执行隐藏目标替换不变性检查；对实际 mask 交换的首批还检查每层完整 X0 不变。目标内容替换不会影响定义的读出，未读取配对目标真值。
- **1216 个正式特征分片**通过有限值、完整样本 / rank 覆盖、无重复、形状 / dtype 与模型来源一致性检查。并行 stdout 的行级事件计数可能受交织影响，持久化分片完整性与断言才是验收依据。
- 2610 行层级分析全部完成；新增相关测试与 V1 测试共 **12 passed**，诊断代码 Ruff、shell 语法和差异空白检查通过。

开发机 SSH 初始化曾因内部 apt 源不可达失败，随后使用 Inspire 自身认证的 Jupyter terminal 通道运行同一 Notebook，没有修改 CLI 全局配置或系统软件源。CPU 分析首轮遇到多线程父进程 fork 死锁，已停止该次分析进程，改为单线程父进程、worker 内启用小线程池后完成；没有改变已冻结样本、模型或指标。

## 10. 如何重新表述假设，下一步是什么

更符合本次证据的假设是：

> 配对生成目标能够在已有预训练基础上，使统一 backbone 形成可跨模态复用的、上下文依赖的语义结构；原生 query 在该结构上执行任务 / 槽位相关的读取，因此不必直接共用同一坐标几何。

“head 专注采样”仍是独立的计算分工假设。当前 B 的 flow head 有 8 层、宽度 1280，同时接收 XT 与 X0 条件，不能仅凭 backbone 检索好就排除 head 中进一步的语义处理。

下一轮最有区分力的冻结模型控制是等长度无语义前缀、边界 / 首 token 与位置控制，区分任务词含义和上下文支撑作用；本次 `Input:` 的恢复值得优先追查。随后才是固定 head 的匹配范数语义干预、XT / X0 条件路径分别控制，以及需要单独限定训练预算的配对 / 打乱 / 冻结 backbone 对照。

尚未完成原设计中的全部强化项：原生 head 的输出因果验证、可信早期训练时间轴、匹配训练臂、人工复核最小语义对、以及更细的类内关系几何零分布。当前只有整体配对几何置换、类别内困难负例图像置换和打乱配对映射等相应对照。不要把冻结阶段完成写成这些因果结论也已验证。

## 11. 工件与复现

输出目录约 **84 GiB**，保留特征以便后续 CPU 复核，不需重新启动开发机：

`output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907`

- [冻结协议][protocol]、[样本清单][samples]、[执行完成记录][execution]。
- [全量指标 JSON][results]、[逐层 CSV][csv]、[开发集选层][selected]、[曲线 PNG][plot] / [PDF][pdf]。
- [权重与旧实验回放核验][replay]、[特征完整性审计][audit]。
- [检索身份置换][null]、[困难负例区间][hard]、[VAE 参考][vae]。
- [提取与核验脚本](../scripts/probe_unified_semantics_v2.py)、[CPU 分析](../scripts/analyze_unified_semantics_v2.py)、[补充审计](../scripts/audit_unified_semantics_v2.py)、[有界提取调度](../scripts/run_unified_semantics_v2.py)、[16 卡入口](../script/selfless/probe_unified_semantics_v2_dev_ascend16.sh)、[测试](../tests/test_semantics_v2.py)。

仅重新合并 CPU 结果，不重新前向：

```bash
repr_v2_root=output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python scripts/analyze_unified_semantics_v2.py \
  --output-dir "$repr_v2_root" --summarize-only
```

CPU 重算例：`--state final_ema --profiles bare,native,neutral --workers 24 --threads 2 --skip-summary`；其他 state 分别为 `final_raw, init42, init43, init44`，其 profiles 为 `bare,native`。已有层结果会复用，不静默覆盖；如需更改分析定义，应使用独立版本结果路径，不把新协议混入本次冻结工件。

本次画图依赖放在临时目录 `/tmp/unified-mm-repr-plot-UM3fnO`，未改项目依赖；有 matplotlib 时可在合并命令加 `--plot`。所有权重与大工件检查均未计算 hash。

[protocol]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/protocol.json
[samples]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/samples.json
[results]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/results-v2.json
[csv]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/layer-metrics-v2.csv
[selected]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/dev-selected-layers.json
[plot]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/layer-curves-v2.png
[pdf]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/layer-curves-v2.pdf
[execution]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/extraction-complete.json
[replay]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/legacy-replay-and-weight-verification.json
[audit]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/integrity-audit.json
[null]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/retrieval-permutation-controls.json
[hard]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/hard-negative-intervals.json
[vae]: ../output/evaluation/research/unified-b-x0content-0p6b-100b-imagenet-split-s42-r1/representation-diagnostic/semantic-v2-20260907/vae-baseline.json
