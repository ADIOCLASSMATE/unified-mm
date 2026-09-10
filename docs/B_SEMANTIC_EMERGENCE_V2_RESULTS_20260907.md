# B 表征诊断 V2：结果

2026-09-07，B step 95415 的冻结表征。原生提示下的 Content 检索和跨模态关系明显强于无提示及初始化参考；属性和关系绑定表现较弱。协议见 [V2](B_SEMANTIC_EMERGENCE_PROTOCOL_V2.md)。

## 数据和提示

| 数据 | 校准 / 拟合 / 开发 / 测试 | 用途 |
| --- | --- | --- |
| COCO Karpathy 缓存 | 500 / 500 / 500 / 1000 张图；测试有 5003 条 caption | 场景检索、几何、配对映射 |
| ImageNet 1000 类 | 每类 2 / 3 / 0 / 3 张图，共 8000 张；6 个文本模板按 2 / 2 / 0 / 2 分组 | 域内与跨模态探针、类别外推映射 |
| SugarCrepe 7 类困难负例 | 每类 25 / 0 / 25 / 100 个独立图像，共 1050 图、2100 候选文本 | 属性、对象、关系与绑定诊断 |

| 条件 | 图像上下文 | 文本上下文 | query |
| --- | --- | --- | --- |
| `bare` | BOI + 256 latent + EOI，无文本前缀 | caption 正文，无前缀 | 两侧均为末尾文本 query |
| `neutral` | `Input:` + 图像 | `Input:` + caption | 两侧均为末尾文本 query |
| `native` | `Describe this image in one detailed caption:` + 图像 | `Generate an image matching this description:` + caption | I2T 首目标文本 query 对 T2I sigma 首目标图像 query |

图文独立前向，所有 caption 按图归组。记录输入、28 个 block、最终 RMSNorm；Content mean 排除提示与目标。初始化 seed 42/43/44 保留预训练 Qwen/VAE，属于新增组件的初始化参考。

## COCO 检索

固定最终 RMSNorm，先用独立 cal 均值中心化，再 L2 归一化；1000 图、5003 caption，R@1 单位为 %。

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

对应身份级 bootstrap 95% 区间：

| EMA 条件 | 图→文 R@1 的 95% 区间 | 文→图 R@1 的 95% 区间 |
| --- | --- | --- |
| bare Content | [0.00, 0.50] | [0.02, 0.20] |
| native Content | [14.50, 19.10] | [9.97, 12.72] |
| native query | [4.60, 7.50] | [2.70, 4.28] |
| neutral query | [10.80, 14.90] | [5.44, 7.20] |

原生 Content 的图→文 R@1/5/10 为 16.80 / 39.00 / 50.20，文→图为 11.35 / 27.32 / 38.40。原始余弦 R@1 为 8.60 / 4.22；原生 query 为 0.50 / 0.38。

## 层与 mask

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

输入 query 为常量，表中输入层的 query CKA=0 是记录占位。文本/图像 mask ID 分别为 151669 / 151672，EMA 两行余弦为 0.24327。只替换 XT mask、保持位置和可见性后的最终 query：

| 图像上下文 query 的 mask / 文本上下文 query 的 mask | 图→文 R@1 | 文→图 R@1 |
| --- | ---: | ---: |
| 文本 / 图像：原生 | 6.00 | 3.46 |
| 文本 / 文本：只改文本上下文的图像槽位 | 9.50 | 5.54 |
| 图像 / 图像：只改图像上下文的文本槽位 | 3.70 | 1.96 |
| 图像 / 文本：两侧交换 | 4.40 | 2.88 |

## 关系几何

COCO 固定最终层；几何关系和直接余弦检索分别计量。

| final Norm 条件 | CKA | 置换 CKA 均值 | 去对角相似度秩相关 |
| --- | ---: | ---: | ---: |
| EMA bare Content | 0.236 | 0.026 | 0.190 |
| EMA native Content | 0.364 | 0.023 | 0.299 |
| EMA native query | 0.485 | 0.019 | 0.422 |
| EMA neutral query | 0.477 | 0.031 | 0.364 |
| 初始化 42 native Content | 0.134 | 0.017 | 0.104 |
| 初始化 42 native query | 0.111 | 0.022 | 0.081 |

## 映射和类别

COCO 只在 500 fit 图上拟合，测试仍用 1000 图；表内为图→文 / 文→图 R@1（%）。

| EMA final Norm | 不学映射 | 配对 ridge | 配对正交映射 | 打乱配对正交映射 |
| --- | --- | --- | --- | --- |
| bare Content | 0.20 / 0.10 | 0.90 / 1.66 | 2.20 / 1.84 | 0.10 / 0.08 |
| native Content | 16.80 / 11.35 | 10.50 / 10.37 | 13.90 / 10.47 | 0.00 / 0.02 |
| native query | 6.00 / 3.46 | 12.80 / 10.25 | **19.90 / 9.81** | 0.00 / 0.06 |

ImageNet 另在 600 fit 类学习映射，测试 200 类的 600 图 / 400 文本，随机同类命中为 0.5%；本版无标签 cal 覆盖全部 1000 类。

| EMA final Norm | 不学映射 | 配对正交映射 | 打乱配对正交映射 |
| --- | --- | --- | --- |
| native Content | 18.33 / 27.25 | 9.83 / 20.00 | 0.83 / 1.25 |
| native query | 12.17 / 10.00 | 17.00 / 22.75 | 0.83 / 1.00 |

ImageNet 最终层线性分类，单位为 %：

| final Norm | 图像域内分类 | 文本分类器→图像 | 图像分类器→文本 |
| --- | ---: | ---: | ---: |
| EMA bare Content | 3.03 | 0.10 | 0.10 |
| EMA neutral Content | 20.27 | 1.43 | 9.65 |
| EMA native Content | 24.30 | 4.57 | 12.50 |
| EMA native query | 31.63 | 1.70 | 0.75 |
| EMA neutral 共有文本 query | 30.43 | 10.17 | 7.20 |
| 初始化参考 native Content | 1.50–2.03 | 0.00–0.07 | 0.00–0.05 |
| 初始化参考 native query | 1.67–2.03 | 0.07–0.13 | 0.00–0.10 |

## SugarCrepe

700 test 图，先比较总体正确率和同类型内打乱图像的对照：

| EMA final Norm | 正确率 | 95% 区间 | 类内打乱图像后 | 相对打乱的差值 95% 区间（百分点） |
| --- | ---: | --- | ---: | --- |
| bare Content | 47.29 | [43.57, 51.00] | 44.29 | [-2.00, 8.00] |
| native Content | **58.86** | [55.28, 62.43] | 52.71 | [0.71, 11.14] |
| native query | 55.29 | [51.43, 59.00] | 53.57 | [-3.57, 6.86] |
| neutral Content | 54.71 | [51.14, 58.43] | 48.71 | [0.71, 11.29] |

各类型正确率（%）：

| 困难负例类型 | native Content | native query |
| --- | ---: | ---: |
| add_att | 62 | 66 |
| add_obj | 50 | 52 |
| replace_att | 76 | 51 |
| replace_obj | 69 | 62 |
| replace_rel | 54 | 54 |
| swap_att | 58 | 55 |
| swap_obj | 43 | 47 |

原生 Content 的总体图像增益区间为正；原生 query 的增益区间跨 0。关系与对象交换是较弱项。

## sigma 稳定性

固定 100 图 / 500 caption 子集，R@1（%）：

| 条件 | sigma 0 | sigma 1 | sigma 2 |
| --- | --- | --- | --- |
| native Content，图→文 / 文→图 | 48.0 / 33.2 | 47.0 / 33.4 | 47.0 / 32.6 |
| native query，图→文 / 文→图 | 34.0 / 21.0 | 38.0 / 24.8 | 32.0 / 21.8 |
| bare Content，图→文 / 文→图 | 2.0 / 2.2 | 3.0 / 3.0 | 1.0 / 2.2 |
| bare query，图→文 / 文→图 | 2.0 / 1.2 | 1.0 / 1.8 | 1.0 / 2.0 |

## 产物

完成 2610 条逐层分析、1216 个特征分片及 12 项相关测试。

[完整结果][results]、[逐层 CSV][csv]、[dev 选择][selected]、[曲线 PNG][plot] / [PDF][pdf]、[困难负例区间][hard]、[置换对照][null]、[样本][samples]、[完整性检查][audit]。本轮测量固定 backbone 读出；head 的因果分工属于独立实验。

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
