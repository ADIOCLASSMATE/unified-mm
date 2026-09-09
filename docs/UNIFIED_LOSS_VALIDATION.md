# 训练与验证统一 loss

`UnifiedMixedDataset` 在 `experiment.val_every` 执行三个活跃任务的 loss 验证。
新协议名为 `unified_schedule_microbatch_mean_v1`；训练优化目标没有更改。
训练与验证共用前向参数准备函数和源任务统计函数，验证使用当前训练权重。

## 汇总公式

设调度长度（梯度累积次数）为 G，任务 s 在调度中出现 n_s 次。
每个任务分别保存两种统计：

- 原始 loss：`sum(任务原始 loss × 有效目标数) / sum(有效目标数)`。
- 对总 loss 的贡献：`(n_s / G) × mean(该任务各 microbatch 的 model.loss)`。

`model.loss` 已含 `lambda_image` 或 `lambda_text`，因此不能再次乘权重。
验证总 loss 为全部活跃任务贡献之和。源 microbatch 均值和目标 token 加权均值
分别计算；在不同 batch 目标数不同时，不能用“原始 loss × 系数”替代前者。

默认调度 `ClimbMix → T2I → ClimbMix → I2T`、G=4、图像权重 1、文本权重 0.05：

```text
L = 0.25 × mean(T2I microbatch MSE)
  + 0.0125 × mean(I2T microbatch CE)
  + 0.025 × mean(ClimbMix microbatch CE)
```

这里的 mean 指 microbatch 均值，不是网页“原始 loss”列的 token 加权均值。
单任务调度即使重复八次，也有 n_s/G=1，不会额外缩小八倍。
缺失、无目标或未完成的活跃任务不能生成新协议总 loss。

## 前向与样本协议

验证使用 `train()` + `no_grad()`：保留训练的图像输入噪声、CFG 条件丢弃、
flow 时间/噪声采样、图像顺序、任务开关和注意力规则，不进行反向传播或优化。
这与用于下游准确率的 `eval()` 模式区分。每次验证前后恢复原模块模式以及
Python、NumPy、Torch 和当前设备的随机数状态；不会消费训练 DataLoader 的游标。

ImageNet 直接复用下游轻量评测的有序清单：默认从独立 val split 每类固定随机
抽取 2 张，共 1,000 类、2,000 张。保留原轻量评测的抽样算法和样本顺序；
T2I / I2T / ImageNet 分类共享同一个清单对象，不再单独随机抽取 loss 样本。
按官方 `image_id`、内部 `img_id` 和 synset 校验并映射各自的 cache / Subset 行号，
不假设缓存行号等于图像 ID；缺失或身份不符直接报错，不用别的图像补齐。
T2I / I2T 各自沿用训练的序列化设置和独立复制的训练 collator（含 packing / pad
schedule）。训练和验证 caption 的数据来源仍不同，详见网页“数据与协议”。
纯文本使用现有 400 条 ClimbMix 固定源记录协议；文档窗口与独立性说明见
[纯文本验证](CLIMBMIX_VALIDATION.md)。纯文本序列长度与 batch 大小必须与训练源一致。

先按训练源的 micro_batch_size 固定全局 batch，再按 `batch_ids[rank::world_size]`
分配。各 batch 的随机种子只依赖验证 seed、源任务和全局 batch ID，不依赖
rank 或训练步数。尾 batch 保留真实样本数；不补齐、不重复数据，空 rank 仍参与
归约。目标均值与 microbatch 均值都在全局归约后计算，改变卡数不会重新分组。

## 配置与执行

默认启用，沿用 `val_every` 的间隔。`TrainingValidator` 在训练进程中只创建一次，
依次执行当前权重 loss 与下游 EMA 评测。下游继续使用自己的 540/600 秒预算；
另外记录 loss、各源任务和整轮验证耗时，600 秒不代表整轮验证的总上限。
CPU / Gloo 回归及固定 16 卡开发机的真实训练内验证均已通过；NPU 计时见下文。

```yaml
experiment:
  val_every: 10000
  loss_validation:
    enabled: true
    seed: 424242  # 前向随机损失的全局 batch seed
  downstream_validation:
    seed: 424242  # T2I / I2T / 分类共用的样本抽样 seed
    imagenet_per_class: 2
```

图像抽样规模只由 `downstream_validation.imagenet_per_class` 控制；旧的
`loss_validation.image_samples` 配置会明确报错，避免两套清单悄悄分离。
`loss_validation.enabled=false` 可关闭整组 loss 验证；`val_every=0` 关闭周期验证。
包含纯文本的联合协议要求 ClimbMix 启用，禁止只停用该项却继续发布“三任务总 loss”。
外部纯文本 JSONL、manifest 和训练排除规则沿用 `climbmix_validation` 配置。
正在运行的进程须在正常重启/恢复时加载新代码；不会自动修改已有作业或历史记录。

## 验证基础设施

- 全局样本计划只准备一次，分片不引入额外 DataLoader worker 或补齐样本。
- loss 仅缓存本 rank 使用的固定图文样本，含验证 caption / prompt token 和
  posterior latent；完整 posterior 继续使用 mmap。collator 每次生成新 batch，
  模型的 CFG、flow 等随机前向仍按固定全局 batch seed 重新计算。
- 下游保留 CPU 样本、分词和 posterior 缓存；模型分数、权重相关前缀不跨验证轮缓存。
- loss 的非有限值 / 无目标检查在设备端累积，每个源任务同步一次，减少逐 batch
  的设备等待。任一源失败均不发布不完整总 loss。
- rank 0 的原子写入也纳入分布式失败同步，读数据、前向或写盘失败不会让其他
  rank 继续进入不匹配的 collective；模型模式与随机状态按原协议恢复。

2026-09-09 校验：与已有三份 16 卡轻量评测清单逐项比较，2,000 个官方 ID
及其顺序完全一致，1,000 类各 2 张；实际 val posterior cache 的身份映射通过。
真实 CPU 下游数据准备首次约 2.34 秒，缓存命中小于 1 毫秒；此计时不包含
模型前向、EMA 操作或 NPU 工作，不能换算为整轮验证加速比。
审计见 [`audit.json`](../output/evaluation/diagnostics/shared-validation-infra-20260909/audit.json)。
相关 CPU / Gloo 回归共 112 项通过，覆盖采样与行号映射、冷/热缓存 loss 一致、
训练前向一致性、任务调度、空 rank、非有限值与 rank 0 写盘异常同步。

同日固定 `dev-wjx-ascend` 的 16 张 Ascend 910B 实测，两档 flow head 均在真实
训练 step 2 / step 4 执行完整验证，保持正式的每卡 batch 形状：

| 配置 | 首轮 loss | 缓存命中后的 loss | 首轮完整验证 | 第二轮完整验证 |
| --- | ---: | ---: | ---: | ---: |
| depth16 | 7.14 秒 | 4.10 秒 | 395.59 秒 | 386.75 秒 |
| depth30 | 8.76 秒 | 5.08 秒 | 396.05 秒 | 394.81 秒 |

每轮 T2I / I2T 各完成 2,000 张，纯文本完成 400 条源记录；ImageNet 与下游的
有序清单完全一致，1,000 类各 2 张。三任务加权贡献之和等于总 loss，下游 11 项
全部完成并各轮均在 600 秒预算内。第二轮确认缓存命中，但只有两轮观测，不能
将完整验证耗时差全部解释为缓存收益。此计时仅代表 16 卡开发机。
原始摘要见
[`restart-validation-20260909/smoke`](../output/experiments/unified-b-x0-flow-head-scaling/restart-validation-20260909/smoke/)。

## 产物与历史记录

新文件原子写入：

```text
output/evaluation/training-validation/<run>/validation_unified_loss_metrics_step_<N>.json
```

schema 为 `selfless_unified_loss_validation_metrics_v1`。Tracker 和 JSON 包含
`val/loss`、三个 `val/loss_<source>`、`val/weighted_contribution_<source>`、
`val/mean_microbatch_loss_<source>`、真实目标数与 batch 数；没有模糊的 `val/loss_text`。
同时保存调度、权重、累积次数、batch 大小、样本索引、随机种子及纯文本独立性。
`imagenet_subset` 包含完整官方图像 ID、内部 ID、类别索引和抽样配置，与同一步
`downstream_validation/step-<N>/subset.json` 中的清单一致。
`validation_summary_step_<N>.json` 汇总整轮状态、耗时和两个详细结果的路径；
Tracker 包含 `val/validation_seconds`、`val/unified_loss_seconds`、
`val/unified_loss/<source>_seconds`、`val/downstream_prepare_seconds` 和缓存命中标记。
训练日志增加 `train/loss`（`step_loss` 的同值别名）和同一 `loss_protocol` 元数据。

网页识别新文件，在同一步优先使用完整新记录，不拼入旧权重/旧协议的任务值。
历史 ImageNet 总 loss 仍为 `lambda_image × T2I + lambda_text × I2T`；旧独立纯文本
文件也保留历史定义。页面、CSV 与导出图保留协议信息，在协议变化处断开曲线，
网页可筛选“仅与训练同口径”或“仅历史协议”。历史缺失的纯文本验证不进行反推。

运行 `python3 scripts/build_evaluation_report.py` 刷新网页；加 `--plots` 刷新 PNG/SVG。
