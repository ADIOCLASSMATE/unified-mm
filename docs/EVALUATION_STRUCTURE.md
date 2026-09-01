# 评测结构

当前标准 final 协议直接加载训练结束导出的 FP32 EMA Hugging Face 目录
`hf_model-final-ema`。仍在训练中的消融实验可直接加载一个完整的
rank-sharded EMA checkpoint；这种来源只加载一次，不会再叠加其他权重。
评测 step 来自模型来源自己的 metadata。协议不做下游微调、指令微调或
线性探针，运行时不计算哈希。

## 统一评测入口

所有统一模型评测只接收一个 `--model_source`。模型来源 metadata 是
`architecture_variant` 和 `dual_stream_attention_contract` 的唯一依据，因此
同一份中性的评测 YAML 可以正确加载 A、B、C，不需要在 evaluator 或 shell
脚本里复制消融分支。

生成式任务统一调用模型自身的 `model.generate(task, ...)`：T2I 使用 `task="t2i"`，
I2T 使用 `task="i2t"`，纯文本续写使用 `task="text"`。正式生成默认并明确启用
KV cache；attention mask、CFG 和 token 更新逻辑全部由模型实现，evaluator 只负责
数据、分布式调度、解码和指标。无 cache 单流仅用于正确性测试，双流仅保留在测试中
验证 hidden 严格一致。likelihood 与 retrieval 本质是候选打分，继续直接调用模型
forward，不伪装成生成任务。

文本 PPL 按目标 token 的总 NLL 加权计算。每个 padded batch 只执行一次双流
backbone forward；随后只选取有效 target hidden，并分块执行 `lm_head`，避免创建
完整 `[batch, sequence, vocabulary]` logits。该路径不使用 KV cache，因为一次整段
forward 比写入再读取 cache 更快。

## 论文主结果

- 图像生成：ImageNet val 50K prompt，报告 FID 和按类别分层的 10-split IS。
- 标准图文检索：
  - MSCOCO Karpathy 5K test：5,000 张图、25,010 条官方 caption（4,990 张图各 5 条，10 张图各 6 条）；
  - Flickr30K Karpathy test：1,000 张图、5,000 条 caption；
  - 两者均报告 I2T/T2I R@1、R@5、R@10。COCO 不使用旧式 5 次 1K 平均。
- 自定义域内检索：官方 ImageNet val 上的 class-balanced 1K/5K，报告 I2T/T2I R@1、R@5、R@10，并明确标注为自定义协议。
- 组合与 hard-negative：SugarCrepe、ARO VG Relation、ARO VG Attribution。
- 纯文本：ARC-Easy、ARC-Challenge、HellaSwag、PIQA、WinoGrande、BoolQ、OpenBookQA、MMLU 5-shot。

[Karpathy 原始页面](https://cs.stanford.edu/people/karpathy/deepimagesent/)提供 COCO/Flickr30K split JSON；[微软 BEiT-3 官方说明](https://github.com/microsoft/unilm/blob/master/beit3/get_started/get_started_for_retrieval.md)也按 `dataset_coco.json` 和 `dataset_flickr30k.json` 构建 retrieval test split。Flickr30K 图片需要按其授权流程取得。

## 仅内部消融

MMBench Dev-EN circular 和 SEED-Bench image 只用于比较消融趋势。当前使用的是本项目的 semantic candidate likelihood，不能当作官方 free-form leaderboard 分数，也不进入论文主表。

ImageNet held-out loss、生成图片和生成 caption 放在 `diagnostics/`，用于训练健康检查与定性展示，不作为论文理解主指标。

## 明确删除

ImageNet Top-1/Top-5、ReaL、I2T-CLIP、Visual calibration、自定义 ImageNet caption negatives、POPE、COCO Caption PPL、Winoground、SVO-Probes、What’sUp 均不在当前协议。归档也不保留 rank shards、FID resume state、重复日志或 smoke 输出。

## 历史 checkpoint 输出目录

```text
output/evaluation/
├── README.md
└── unified-a-0p6b/
    ├── manifest.json
    ├── checkpoints/
    │   ├── step-56000/
    │   ├── step-58000/
    │   └── step-60000/
    │       ├── paper-generation/
    │       ├── paper-understanding/
    │       ├── paper-text/
    │       ├── internal-ablation/
    │       ├── diagnostics/
    │       └── summary.json
    └── trend/
        ├── trend.json
        └── trend.md
```

`output/evaluation-checkpoints/` 是历史趋势的 legacy 输入而非评测输出，因此保持原位，避免破坏既有记录路径；它不是新 final 评测的标准输入。

历史 checkpoint 协议只对 step 60000 执行了完整重评；step 56000/58000 仅保留既有趋势结果，不重复跑标准跨数据集检索。Flickr30K 直接由一个 16 卡 Job 完成；MSCOCO 为缩短墙钟时间，将 5,000 个 image query 按索引奇偶拆到两个独立 16 卡 Job，每个分片仍对完整 25,010 条 caption 候选打分，随后做无重复、无缺行的严格合并。分片只改变执行调度，不改变 Karpathy 5K test 协议或指标定义。

历史归档 `output/evaluation/unified-a-0p6b` 已完成上述 checkpoint 协议，`paper_protocol_complete=true` 且没有 pending task；这不替代对 `hf_model-final-ema` 的新标准 final 评测。
