# 历史 ImageNet 架构消融

本页记录早期 ImageNet 研究；当前 A/B 和 C–F 见 [实验定义](EXPERIMENTS.md)。

## 位置与 head

该轮选择 backbone 与 dynamic contextual flow head 全部使用 row/column pure 2D RoPE，移除 additive image position。backbone 三 seed 的 1D→2D 对照：FID 从 26.3528 ± 0.5440 降至 25.2463 ± 0.3010，IS 从 59.1364 ± 0.6996 升至 61.5805 ± 0.4910。

联合位置消融：

| Backbone position | Flow position | FID ↓ | IS ↑ |
| --- | --- | ---: | ---: |
| query additive + 2D RoPE | additive-only | 23.695 | 63.167 ± 1.149 |
| query additive + 2D RoPE | pure 2D RoPE | **22.949** | 64.184 ± 1.195 |
| **pure 2D RoPE** | additive-only | 23.372 | 63.933 ± 1.360 |
| **pure 2D RoPE** | **pure 2D RoPE** | **23.014** | **64.974 ± 0.967** |
| observed additive + 2D RoPE | additive-only | 23.677 | 64.148 ± 1.286 |
| observed additive + 2D RoPE | pure 2D RoPE | **23.017** | **64.819 ± 1.304** |

全链路 pure 2D RoPE 距最低 FID 0.065，IS 最高。additive-only / no-RoPE 控制为 FID23.908、IS63.301 ± 1.313。

参数匹配 pointwise MLP 的 FID / IS 为 26.4404 / 58.3860 ± 1.1099，contextual baseline 为 26.0110 / 59.5362 ± 1.1316。后续更长训练的 position-wise 结果见 [800 epoch 研究](IMAGENET1K_800EP_PRETRAINING.md)。

## Attention output gate

接口为 `none | per_head_identity_sigmoid`，gate 使用 `2 * sigmoid(W_g h)`、W_g=0。10K 配对评测：

| Suite | 无 gate FID / IS | 有 gate FID / IS | 结论 |
| --- | ---: | ---: | --- |
| class | 165.5937 / 7.2954 | **154.1903 / 8.9043** | gate 更好 |
| caption | **34.0955 / 46.3868** | 34.7437 / 46.2953 | gate 更差 |

gate 的最终验证 loss 增加1.10%，吞吐下降8.09%，单卡峰值 allocated 增加2.62GiB。默认选 `none`。

## 当时配方

EMA、BF16、CFG3.5 constant、100-step Heun、Halton、parallel_rate1；正式评测10000样本、seed42、8×H100。evaluator batch 为分片前全局 batch，该轮使用4096，每 rank512。

数据支持 class 与 caption：class 直接使用类名；caption 使用完整文本、固定 T2I 前缀、一一对应 membership 和确定性 segment packing。Caption × initialization 矩阵未完成。
