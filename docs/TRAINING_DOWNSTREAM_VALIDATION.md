# 训练中下游验证

`UnifiedMixedDataset` 的训练验证使用 `training_downstream_v1`，默认每
10,000 个 optimizer step 执行一次。所有训练 rank 参与；16、64、256 卡使用
相同的全局样本、类别候选、随机种子和评分规则。`val_every=0` 禁用定期验证。

| 项目 | 固定全局规模 | 分数 |
| --- | --- | --- |
| ARC-E/C、HellaSwag、PIQA、WinoGrande、BoolQ、OpenBookQA、MMLU | 全部 34,507 题 | 原文本协议的各任务主指标，另报 8 项算术平均 |
| ImageNet 分类 | 每类随机 2 张，共 2,000 张；仍比较全部 1,000 类 | 校准后的 Top-1、Top-5 |
| ARO VG relation | 按类别分层抽取 512 条 | 校准后的严格配对胜率、平局率、平均分差 |
| SugarCrepe | 按类别分层抽取 512 条 | 同上 |

文本评分复用正式评测的 `selfless_text_benchmark_v3`：字符长度归一化、
WinoGrande 共享后缀评分、MMLU 5-shot 与学科宏平均均保持原定义。
评分采用训练模型的精度；本次计时与 DeepSpeed 一致，浮点缓冲也为 BF16。
`summary.json` 记录 `floating_buffer_dtypes` 供比较。旧纯文本离线程序可能
保留 FP32 RoPE 缓冲，因此不能要求与该程序逐题、逐位完全相同。
ImageNet 用选中的全部 2,000 张图估计无标签语言先验，alpha=1。
两个图文任务使用 MC16、3 张固定空图和 alpha=1 的先验校准；顺序图像模型
只有一个确定顺序，使用 MC1，记录实际 MC 数。MC16 快评与正式 MC64 结果分开解释。

每次都按相同 seed 从整个数据池分层抽样，先固定全局清单，再用
`indices[rank::world_size]` 分片。分片不补齐，不重复样本，支持空分片。
图像 posterior 噪声与顺序 MC 由样本身份确定，与 rank 无关。
计数、正确数和分类分数矩阵全局汇总，类别宏平均在汇总后计算。
不同卡数或 batch 形状可能产生细微 BF16 舍入差异。

验证使用当前模型对象。当训练启用 EMA 时，将各 rank 持有的 FP32 EMA 分块
广播并临时复制进模型，验证完恢复 CPU 备份中的训练权重。不会建立第二份完整
NPU 模型，也不会重置优化器；模型模式、共享参数、buffers 和 Python/NumPy/
Torch 随机数状态在正常结束及异常退出时恢复。该方式沿用训练已有的
ZeRO-2 完整参数复制前提，不能直接用于 ZeRO-3 参数分片。

训练验证不再调用旧的前几个 batch loss/生成图片/生成 caption 流程。
FID/IS、检索与其余图文任务留在独立的完整评测流程中。快评用于观察文本和图像
理解趋势，不能由此推断生成 FID 的排名。独立完整评测仍可调用原有全量 loss
验证函数；历史 LR sweep 的 loss 选择脚本也只适用于原有历史产物。

## 时间与结果

工作预算为 540 秒，预留 60 秒用于汇总、EMA 恢复和日志。
时间包含首次评分模块初始化、数据准备、EMA 切换和恢复。批次之间检查截止时间；
未完成的任务记录真实已处理条数与 `time_budget_exhausted`，不产生准确率。
`complete` 表示样本是否全部完成，`within_time_budget` 单独表示是否在 600 秒内。
该截止时间是协作式的，不会在执行中的设备算子内强制打断训练进程。

每个验证点写入：

- `output/evaluation/training-validation/<run>/downstream_validation/step-<N>/subset.json`：全局样本 ID 与配置。
- 同目录 `summary.json`：模型契约、EMA 来源、各任务完成情况、得分与时间。
- Tracker 的 `val/downstream/<task>`、`val/downstream/text_mean` 与耗时/完成标记。

计时验收只使用固定的 16 卡开发机，调用同一个训练验证函数。计时脚本额外包含
checkpoint 加载和分片 EMA 初始化，并检查验证后训练权重、训练模式恢复。
`launcher.status` 另记进程启动、HCCL 建组和退出在内的总时间。

```bash
bash script/selfless/benchmark_training_validation_ascend16.sh \
  <HF-EMA-or-sharded-EMA-checkpoint> <new-timing-output-directory>
```

2026-09-06，16 × Ascend 910B，Qwen3-0.6B：F 最终 EMA 的完整快评
**419.51 秒（约 7 分钟）**；包含进程启动和退出的总时间为
**443 秒（7 分 23 秒）**。11 项全部完成，EMA 恢复检查通过。
其中全文本 83.64 秒，ImageNet 257.02 秒，
ARO 18.95 秒，SugarCrepe 19.92 秒。
其余为模型加载、EMA 操作、数据准备与汇总。最大 NPU 已分配显存约
2.95 GiB（独立计时程序，不含训练优化器）。
ImageNet Top-1 为 45.15%，与从已有全量分数矩阵重算同一 2,000 张子集的结果一致。
产物：`output/evaluation/diagnostics/validation-timing-16npu-20260906-f-r3/`。

这次计时证明当前 0.6B 配置在 16 卡上的预算；没有对 64/256 卡做额外计时。
1.7B 复用同一实现与截止时间，尚无其 16 卡完整通过的时间结论。
已经运行中的训练进程需要在下一次正常启动/恢复时才会加载新版代码。
