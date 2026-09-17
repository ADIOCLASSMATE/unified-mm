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

## 当前四项短预算消融

2026-09-16 起，当前启动的是 S2-single、S2-single 文本双流、Z、Z+B head，使用各自的 `*_33b_ascend64.yaml`。B+S2 调制的短预算配置已准备，按要求停止，暂不启动。四项均在随机序语言建模项目从 Qwen 基座 step 0 开始，64 卡、GA4，`max_train_steps = stop_after_steps = 31800`，约 33.328B 名义文本目标。WSD warmup 199、decay 7950，从 step 23850 开始衰减，终点 LR 为峰值的 0.1。

每 3180 步验证一次，共 10 轮，包含 step 31800；下游 bench 的样本、seed 和评分协议保持原设置，EMA decay 为 **0.9997**。按 `N × (1 − d)` 近似不变从原 95,415-step / 0.9999 配方缩放，EMA 时间尺度与 WSD 衰减区间的比例同步保持约 0.419。当前输出使用 `33b-…-r2`，从基座 step 0 重启 EMA。按相同 bench 和相同权重口径比较排名；短预算排名仍需完整预算对照确认，不能直接视为最终排名。

短预算配置由 [short_ablation_protocol.py](../utils/short_ablation_protocol.py) 校验，统一设置预算、EMA、验证频率、固定 prompt 出图与运行身份；模型、数据和优化器沿用各自 100B 配方。新输出名使用 `33b`，不续接旧 run。旧配置和 `--smoke-suite` 保留用于原架构验收。31,800 步是优化步预算，并非 5 小时的时限。

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

S2-single 的 31,800-step 入口为 `bash script/selfless/pretraining_s2_single_ascend64.sh`，配置为 [S2-single 33B](../configs/selfless/unified_s2_single_33b_ascend64.yaml)。与同预算文本双流版本相比，模型仅有文本 attention contract 的差异。

[S2-single 文本双流](S2_TEXT_TWO_STREAM.md) 使用 `bash script/selfless/pretraining_s2_text_two_stream_ascend64.sh`，只改变 backbone 的文本预测位置，保留完整 S2 flow head；当前为上述 31,800-step 预算。

[Z](Z_EXPERIMENT.md) 使用 `bash script/selfless/pretraining_z_ascend64.sh`，保留 B 双流 backbone，同一图像共用 sigma，单流 DiT 联合去噪；当前为 64 卡、31,800 updates。

[Z + B 单流 head](Z_B_HEAD_EXPERIMENT.md) 使用 `bash script/selfless/pretraining_z_b_head_ascend64.sh`，复用 Z 的训练与整图生成流程，head 使用 B 的全部层和初始化，`proj(h) + time_embed(t)` 进入 AdaLN。

[B + S2-single 调制](B_S2_MODULATION.md) 使用 `bash script/selfless/pretraining_b_s2_modulation_ascend64.sh`。Backbone 和 head 均保留 B 双流，只将 backbone 条件移到 head 输入，AdaLN 仅接收时间；当前为 64 卡、31,800 updates，每轮验证使用 raw 权重和 KV cache 生成 16 张图像。

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

100B 配方默认每 10,000 steps 验证一次；当前四项 33B 消融每 3,180 steps 验证一次。产物统一位于：

```text
output/evaluation/training-validation/<run>/
```

当前权重计算统一 loss，EMA 计算下游指标。具体样本、耗时和文件名见 [统一 loss](UNIFIED_LOSS_VALIDATION.md)与 [下游验证](TRAINING_DOWNSTREAM_VALIDATION.md)。历史训练保存的图片、caption 和 `validation_metrics_step_*.json` 也位于该目录。

所有五份短预算配置（包括暂不启动的 B+S2）每轮验证都使用当前 raw 权重生成 16 张固定 prompt、固定种子的图像（CFG 3.5、Heun10）。S2-single 和文本双流使用整图 flow，每次 ODE/CFG 计算刷新整个 backbone/head，无 KV cache，Heun10 共 40 次调用；Z 和 Z+B 复用一次 backbone 前向产生的固定条件；B+S2 开启 backbone/head content KV cache。单图、`overview.png`、`index.html` 和生成参数写入 `validation_generation/step-<step>/`；`experiment.validation_generation.weights: raw` 显式记录权重选择。EMA 仍用于下游评分和原有权重导出，见 [Z 验证图像](Z_EXPERIMENT.md#每轮验证的图像)。

固定 16 卡开发机的验证出图验收入口：`bash script/selfless/pretraining_short_ablation_ascend64.sh --generation-smoke --output-dir <report-dir>`。两种 S2 各连续验证两次，检查完整 256-latent 生成、VAE 解码、固定图像复现、参数及 RNG/训练模式恢复。

## 开发验证

```bash
bash script/check_repo.sh
```

保存／恢复、RF4 执行和缓存改动使用固定 16 卡开发机验证数值、完整 checkpoint、raw/EMA 重载与生成。[维护与验收](REPO_REFACTOR_20260909.md)列出通用入口；各实验的 `output/experiments/` 保存具体配置与日志。
