# Selfless-Flow 消融结论

本文件是仓库中唯一保留的消融记录。历史实验矩阵、proposal、临时协议、
兼容分支和运行快照均已移除；表中的名称只用于说明实验来源，不再是可配置接口。

## 最终架构

- Image backbone：observed image token 与 mask query 都不使用 additive image
  position；attention 固定使用 row/column **pure 2D RoPE**。
- Flow head：固定使用逐层更新 clean-content stream 的 dynamic dual-stream
  contextual head；content/query attention 同样只使用 row/column **pure 2D
  RoPE**。
- Backbone attention output gate：保留
  `none | per_head_identity_sigmoid` 接口，默认 `none`。启用时采用
  `2 * sigmoid(W_g h)`，`W_g=0`，因此 step 0 严格等价于无 gate。
- 数据条件：只支持 `class` 与 `caption` 两种模式。

## 位置编码与 flow head

Backbone 三 seed 结果表明，将 flatten 1D RoPE 换成 row/column 2D RoPE 后，
FID 从 `26.3528 ± 0.5440` 降至 `25.2463 ± 0.3010`，IS 从
`59.1364 ± 0.6996` 升至 `61.5805 ± 0.4910`。2D RoPE 对三个 seed 均同时
改善 FID 与 IS。

Flow head 单独消融表明，主要收益来自 dynamic dual-stream，而不是静态读取
clean context。最终联合消融如下：

| Backbone position | Flow position | FID ↓ | IS ↑ |
| --- | --- | ---: | ---: |
| query additive + 2D RoPE | additive-only | 23.695 | 63.167 ± 1.149 |
| query additive + 2D RoPE | pure 2D RoPE | **22.949** | 64.184 ± 1.195 |
| **pure 2D RoPE** | additive-only | 23.372 | 63.933 ± 1.360 |
| **pure 2D RoPE** | **pure 2D RoPE** | **23.014** | **64.974 ± 0.967** |
| observed additive + 2D RoPE | additive-only | 23.677 | 64.148 ± 1.286 |
| observed additive + 2D RoPE | pure 2D RoPE | **23.017** | **64.819 ± 1.304** |

Pure 2D RoPE flow head 对三个 backbone 都同时改善 FID 与 IS。最终选择的全链路
pure 2D RoPE 距全表最低 FID 仅 `0.065`，同时取得最高 IS，并完全移除 image
additive position。额外的全链路 additive-only/no-RoPE control 得到
`FID 23.908 / IS 63.301 ± 1.313`，也支持这一选择。

Pointwise token-only MLP 即使做到参数量匹配，仍落后 contextual head：
参数匹配版本为 `FID 26.4404 / IS 58.3860 ± 1.1099`，contextual baseline 为
`FID 26.0110 / IS 59.5362 ± 1.1316`。因此仓库只保留 contextual dynamic
dual-stream flow head。

## Attention output gate

Gate 的完整 10K paired evaluation 是 mixed result：

| Suite | 无 gate FID / IS | 有 gate FID / IS | 结论 |
| --- | ---: | ---: | --- |
| class | 165.5937 / 7.2954 | **154.1903 / 8.9043** | gate 更好 |
| caption | **34.0955 / 46.3868** | 34.7437 / 46.2953 | gate 更差 |

有 gate 的 final validation loss 相对退化 `1.10%`，训练吞吐下降 `8.09%`，
单卡峰值 allocated 显存增加 `2.62 GiB`。Gate 确实学到了非平凡的分层抑制，
但没有在 caption-trained 模型上带来一致收益。因此接口保留用于研究和已有
checkpoint，工程默认固定为 `none`。

## 推理默认值

- Selfless-Flow：EMA checkpoint、BF16 model forward、CFG `3.5`、constant
  schedule、100-step Heun、`spatial_halton`、`parallel_rate=1`。
- 正式评测：10,000 samples、seed 42、8×H100。
- Evaluator 的 `--batch_size` 是 **shard 前的全局 batch**。单 H100 实测
  batch `512` 时，class/caption 分别使用约 `67.44 GiB` allocated、
  `72.45/72.59 GiB` reserved；因此 8×H100 默认使用全局 batch `4096`
  （每 rank `512`），不再使用全局 `512`。rank sharding 在 dataset
  collation 前完成，避免每个进程重复构造全局 CPU batch。

## 数据结论

`class` 模式直接使用 ImageNet class name；`caption` 模式使用完整 caption、
固定 T2I prefix、严格一一对应的 caption membership，并对变长样本做确定性
segment packing。Caption × initialization 的正式矩阵未完成，仓库不保留或
宣称该实验的质量结论；caption 仅作为与 class 并列的受支持数据加载方式。
