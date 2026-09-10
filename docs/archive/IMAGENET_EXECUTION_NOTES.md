# 历史 ImageNet 运行索引

当前资源配置见 [INSPIRE](../../INSPIRE.md)，当前 Unified 训练见 [TRAINING](../TRAINING.md)。本页记录早期实验的独立入口。

## 早期硬件

早期生成评测使用8×H100，后来 flow-cache batching 使用每卡384 / 全局3072；曾用镜像 `dev-wjx:v-2.1`，优先级4。ImageNet-100 及后续正式 ImageNet-1K训练迁至64×910B。

## 已归档结论

| 实验 | 文档 |
| --- | --- |
| 位置、contextual head、gate | [架构消融](../ABLATION_CONCLUSIONS.md) |
| ImageNet-100 LR | [扫描结果](../IMAGENET100_HYPERPARAMETER_CONCLUSION.md) |
| ImageNet-1K class 800 epoch | [配方与最终结果](../IMAGENET1K_800EP_PRETRAINING.md) |
| Caption / T2I 联合10 epoch | [配方与选择](../IMAGENET1K_CAPTION_JOINT_CONCLUSION.md) |

## ImageNet-1K T2I-only：80 / 400 epoch

三个变体为 baseline、positionwise_head、seq_sigma，各从自己对应的800-epoch class EMA初始化。400-epoch组独立启动，保持前序80-epoch产物。

| 配置 | 80 epoch | 400 epoch |
| --- | ---: | ---: |
| NPU / 每rank batch / GA / 全局batch | 64 / 16 / 1 / 1024 | 相同 |
| 每epoch optimizer steps | 1202 | 1202 |
| 总steps | 96160 | 480800 |
| WSD warmup / stable / decay epochs | 8 / 48 / 24 | 40 / 240 / 120 |
| 所有可训练参数 LR | 2e-5 | 2e-5 |
| 文本 / 图像 loss 权重 | 0 / 1 | 0 / 1 |

每图12条 T2I prompt 以确定性随机起点无放回轮换，epoch 和数据游标支持恢复。数据索引为 `public/datasets/imagenet1k_synthetic_v1/indexed/train/manifest.json`。

配置模式：`configs/selfless/imagenet1k_t2i_{baseline,positionwise_head,seq_sigma}_{80,400}ep_ascend_64npu_bs1024.yaml`。launcher 前缀为 `script/selfless/pretraining_imagenet1k_t2i_`；80-epoch 后缀为 `_ascend_64npu_bs1024_80ep.sh`，400-epoch为 `_ascend_64npu_bs1024_400ep.sh`。

输出模式：`output/selfless-flow-imagenet1k-t2i-{baseline,positionwise-head,seq-sigma}-ascend64-b1024-{80,400}ep/`。

评测使用各自 final EMA、16×910B、ImageNet-val50000 prompt、canonical 噪声、CFG3.5、Heun10和项目 val moments。baseline/positionwise 用 Halton，seq_sigma 用 sequential；十个 IS split 按 synset 分层。结果位于各 run 的 `generation-evaluation/heun10/t2i-fid-is/metrics.json`，旧100步结果位于 `generation-evaluation/t2i-fid-is/metrics.json`。

这批历史 T2I-only 从 class EMA继续训练；当前匹配 B 任务曝光的 only 定义见 [EXPERIMENTS](../EXPERIMENTS.md)。
