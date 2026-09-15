# 训练

模型定义见 [实验](EXPERIMENTS.md)，数据资产见 [数据](DATA.md)，提交位置见 [Inspire](../INSPIRE.md)。

## Unified 配方

| 项目 | A/B、C–F、depth16/30 |
| --- | --- |
| 初始化 | Qwen3-0.6B-Base，optimizer step 0 |
| 设备 | 64×Ascend 910B，HCCL，DeepSpeed ZeRO-2 |
| 任务顺序 | `climbmix → t2i → climbmix → i2t`，GA4 |
| 每卡 batch | text 4；T2I / I2T 各 16 |
| 序列长度 | text 2,048；图文 512 |
| 预算 | 95,415 updates，约 100B 名义文本目标 |
| 图像曝光 | 每次更新 T2I / I2T 各 1,024 图，各累计 97,704,960 图 |
| Flow 采样 | 每图 4 份独立噪声／时间，logit-normal(0,1) + 10% uniform |
| Loss 权重 | image 1.0；text 0.05 |
| AdamW LR | backbone / tied embedding 3e-4；projector / flow 5e-5 |
| WSD | warmup 596；decay 23,854；最低 LR 比例 0.1 |
| 数值 | BF16；FP32 梯度累积与分片 EMA，EMA decay 0.9999 |
| 随机性 | seed 42；conditioning dropout 0.1；输入微噪声 0.01 |
| 记录 | 本地 JSON/log；runtime hashing 与 W&B 关闭 |

完整字段在 [100B 协议](../configs/protocols/unified_ablation_100b_ascend64.yaml)。训练 loss 按任务 microbatch 汇总，公式见 [统一 loss](UNIFIED_LOSS_VALIDATION.md)。

## 启动入口

```bash
bash script/selfless/pretraining_unified_ablation_a_0p6b_formal_ascend64.sh
bash script/selfless/pretraining_unified_ablation_b_0p6b_formal_ascend64.sh
```

C–F 使用 `script/selfless/pretraining_unified_ablation_{c,d,e,f}_on_b_0p6b_formal_ascend64.sh`。
B 深度扩展配置为 `configs/selfless/unified_b_x0_flow_depth{16,30}_100b_ascend64.yaml`。
F 参数匹配扩展配置为 `configs/selfless/unified_f_on_b_flow_depth{16,30}_100b_ascend64.yaml`，在随机序语言建模项目各提交一个 64 卡任务：

```bash
bash script/selfless/pretraining_unified_positionwise_flow_head_scaling_ascend64.sh --depth 16
bash script/selfless/pretraining_unified_positionwise_flow_head_scaling_ascend64.sh --depth 30
```

两档 F 均从基座 step 0 开始，完整执行同一 95,415-step 配方。固定 16 卡开发机验收入口为 `script/selfless/smoke_unified_positionwise_flow_head_scaling_ascend16.sh --output-dir <report-dir> --label <unique-label>`，保持正式每卡 batch，先训练 12 步，再从完整 checkpoint 恢复到 14 步，检查 raw/EMA 重载与完整 256-latent 生成。

单任务入口：

```bash
bash script/selfless/pretraining_unified_b_i2t_only_0p6b_ascend16.sh
bash script/selfless/pretraining_unified_b_t2i_only_0p6b_ascend16.sh
```

两项 image-only 均为 16 卡、每卡 batch 32、GA2、95,415 updates，分别对齐 B 对应任务的数据与曝光。T2I 保留 RF4，共享 X0 content/KV，并对 flow block 做 activation checkpointing。开发机实测 I2T 约 1.77 秒/update、T2I 约 2.5 秒/update；记录位于 [only launch](../output/experiments/unified-b-image-only-matched/launch-20260910-r1/)。

S2 与 B+SigLIP 的入口及当前设置见 [S2](SHOWO2_UNIFIED_ABLATION_DESIGN.md)和 [B+SigLIP](B_SIGLIP_UNIFIED_ABLATION.md)。

无 ClimbMix 的 T2I + I2T 对照在随机序语言建模项目使用 2 节点 × 16 卡，配置为 [joint 32 卡](../configs/selfless/unified_b_t2i_i2t_matched_ascend32.yaml)：

```bash
bash script/selfless/pretraining_unified_b_t2i_i2t_ascend32.sh
```

该入口读取平台 PET 多节点环境，使用 ZeRO-2、每卡 batch 32 和 GA2。两个图文流共享只读图像缓存，独立保存和恢复数据游标。固定 16 卡开发机验收使用同一入口加 `--smoke-suite --output-dir <report-dir> --label <unique-label>`，训练 12 步、恢复到 14 步，并验证 raw/EMA 重载及完整生成。正式与 smoke 输出目录独立。

## 权重与续训

```text
output/<run>/
├── config.yaml / experiment_identity.json
├── training_metrics.jsonl
├── checkpoint-<step>/             # optimizer、scheduler、RNG、数据游标、EMA
├── hf_model-<step>-eval/          # 周期 raw BF16 导出
├── hf_model-<step>-ema-eval/      # 周期 EMA BF16 导出
├── hf_model-<step>-eval-pair.json
├── hf_model-final/
└── hf_model-final-ema/
```

| 保存项 | 默认策略 |
| --- | --- |
| 普通 checkpoint | `save_every` 控制，滚动保留最近 3 个 |
| 完整里程碑 | 每 100 图像 epoch，永久保留 |
| Raw + EMA 评测导出 | 每 20 图像 epoch，完整 BF16 模型与 tokenizer |
| Final | 完整 raw / EMA；当前正式 final EMA 为 FP32 导出 |

Unified / class 配方每参考图像 epoch 为 1,251 steps，对应里程碑 125,100、周期导出 25,020；早期 T2I / caption-joint 为 1,202，对应 120,200 / 24,040。纯文本使用 unified 参考周期，实际值保存在各 run 配置。

保存先写 `.partial` 目录，完成全部 rank 的文件后发布完成标记，再执行保留策略。续训加载完整 checkpoint，恢复原 world size、优化器、调度器、数据游标和随机状态。相关实现为 `utils/training_checkpoint.py` 与 `utils/checkpoint_transaction.py`。

## 训练验证

默认每 10,000 steps 验证一次，产物统一位于：

```text
output/evaluation/training-validation/<run>/
```

当前权重计算统一 loss，EMA 计算下游指标。具体样本、耗时和文件名见 [统一 loss](UNIFIED_LOSS_VALIDATION.md)与 [下游验证](TRAINING_DOWNSTREAM_VALIDATION.md)。历史训练保存的图片、caption 和 `validation_metrics_step_*.json` 也位于该目录。

## 开发验证

```bash
bash script/check_repo.sh
```

保存／恢复、RF4 执行和缓存改动使用固定 16 卡开发机验证数值、完整 checkpoint、raw/EMA 重载与生成。[维护与验收](REPO_REFACTOR_20260909.md)列出通用入口；各实验的 `output/experiments/` 保存具体配置与日志。
