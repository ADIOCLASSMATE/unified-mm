# 统一 loss 验证

协议为 `unified_schedule_microbatch_mean_v1`。每个 `val_every` 使用当前训练权重，在 `train()` + `no_grad()` 下执行相同的输入噪声、CFG dropout、flow 采样、任务与 attention 路径。

## 聚合

设调度长度为 G，任务 s 出现 n_s 次：

```text
原始任务 loss = sum(raw_loss × 有效目标数) / sum(有效目标数)
任务贡献       = n_s / G × mean(model.loss of source microbatches)
总 loss        = sum(任务贡献)
```

`model.loss` 已含任务权重。默认 `climbmix → t2i → climbmix → i2t` 的目标是：

```text
L = 0.025 × mean(ClimbMix microbatch CE)
  + 0.25  × mean(T2I microbatch MSE)
  + 0.0125 × mean(I2T microbatch CE)
```

单任务的调度比例为 1。原始 loss 按目标数加权，任务贡献按 microbatch 平均，两种统计分别保存。

## 样本与执行

| 来源 | 固定样本 |
| --- | --- |
| T2I / I2T | ImageNet-val 每类 2 张，共 2,000 张，与下游分类共用有序清单 |
| ClimbMix | 400 条固定源记录，窗口设置跟随训练源 |

验证 batch 按训练源大小构造，再按 `batch_ids[rank::world_size]` 分片。随机 seed 由任务和全局 batch ID 确定；每轮恢复模型模式和 RNG。图像身份映射、posterior 与 token 缓存跨轮复用，模型输出每轮计算。

```yaml
experiment:
  val_every: 10000
  loss_validation:
    enabled: true
    seed: 424242
  downstream_validation:
    seed: 424242
    imagenet_per_class: 2
```

`loss_validation.enabled=false` 关闭 loss 验证，`val_every=0` 关闭周期验证。所有活跃任务完成后汇总总 loss；ClimbMix 来源设置见 [纯文本验证](CLIMBMIX_VALIDATION.md)。

## 结果与耗时

```text
output/evaluation/training-validation/<run>/
  validation_unified_loss_metrics_step_<N>.json
  validation_summary_step_<N>.json
```

保存各源 loss、加权贡献、microbatch 均值、目标数、样本 ID、随机 seed、权重来源和耗时。Tracker 使用 `val/loss_<source>`、`val/weighted_contribution_<source>`、`val/unified_loss_seconds` 等字段。

16×910B、正式每卡 batch、训练 step 2 / 4 的实测：

| Head | 首轮 loss 秒 | 缓存命中后 loss 秒 | 首轮整轮验证秒 | 第二轮整轮验证秒 |
| --- | ---: | ---: | ---: | ---: |
| depth16 | 7.14 | 4.10 | 395.59 | 386.75 |
| depth30 | 8.76 | 5.08 | 396.05 | 394.81 |

整轮含 [下游验证](TRAINING_DOWNSTREAM_VALIDATION.md)。历史 `val/loss_text` 表示 I2T caption CE，历史图像总 loss 仅含 T2I/I2T；网页按协议分段展示。结果统一通过 `python3 scripts/build_evaluation_report.py` 更新。
