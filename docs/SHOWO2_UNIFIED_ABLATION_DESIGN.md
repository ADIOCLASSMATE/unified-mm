# S2 / SigLIP 对照

检验语义／细节双路径在本项目架构和训练范式下是否必要。直接比较 B 与 B+SigLIP，并以 S2-single / dual 为参照。

## 模型

```text
VAE latent ┬→ latent projector ────────────────┐
           └→ input projection → semantic ViT ┴→ concat → RMSNorm → MLP → backbone
```

| 组 | 视觉前端 | 文本／图像建模 |
| --- | --- | --- |
| B | projector | Selfless 同位置预测，随机序逐 latent flow |
| B+SigLIP | projector + semantic + fusion | B 路径，语义层遵守 sigma 因果可见性 |
| S2-single | projector | 文本 next-token AR，图像块 full attention，整图 flow |
| S2-dual-siglip | projector + semantic + fusion | 与 S2-single 相同 |

S2 每次 ODE 调用用当前 noisy latent 刷新视觉前端、backbone 和 head。每幅图共享一个时间 t，保留四份 Monte Carlo 样本；B 按 token 采样 t。

S2 flow head 为 8 层、宽 1280、20 query heads / 5 KV heads、head dim 64、MLP intermediate 1472，共 163,295,760 参数；B head 为 164,072,976。两个 S2 组的共有模块初始化逐位相同。

## SigLIP 与训练阶段

语义初始化来自 `google/siglip-so400m-patch14-384`：前 26 层、宽 1152，加载 417 个张量。KL16 输入采用新 Linear，位置表由 27×27 插值到 16×16。

[Show-o2 原流程](https://arxiv.org/html/2506.15564v2)：

| 阶段 | 语义分支与主模型 |
| --- | --- |
| 预蒸馏 | 用 RGB SigLIP 特征训练接收 clean/noisy VAE latent 的语义分支 |
| Stage-1 | 冻结语义分支和 LLM，训练 projector、fusion、flow head |
| Stage-2 | 联合训练主模型，VAE 冻结；语义 LR=2e-6 |

当前 S2 / B+SigLIP 变体直接加载 SigLIP Transformer，省略预蒸馏和 Stage-1，从 step 0 全参数训练。它们使用 Qwen3、MAR-KL16 和 B 的 [unified 配方](TRAINING.md)。按 Show-o2 原流程建立的 KL16 语义预蒸馏与分阶段对照尚未实现。

原生实现见 [Stage-1](../../Show-o/show-o2/train_stage_one.py)、[Stage-2](../../Show-o/show-o2/train_stage_two.py)。

## 配置与运行

| 组 | 配置 | 训练目录 |
| --- | --- | --- |
| S2-single | [single](../configs/selfless/unified_s2_single_100b_ascend64.yaml) | `output/unified-s2-single-0p6b-100b-imagenet-split-s42-r1` |
| S2-dual-siglip | [dual](../configs/selfless/unified_s2_dual_siglip_100b_ascend64.yaml) | `output/unified-s2-dual-siglip-0p6b-100b-imagenet-split-s42-r1` |

```bash
bash script/selfless/pretraining_showo2_unified_ascend64.sh --variant single
bash script/selfless/pretraining_showo2_unified_ascend64.sh --variant dual-siglip
```

两组均在 `high-dimensionaldata` 使用 64 卡、95,415 updates。B+SigLIP 的入口见 [分支实现](B_SIGLIP_UNIFIED_ABLATION.md)。数值合同为 [showo2_unified_100b_ascend64.yaml](../configs/protocols/showo2_unified_100b_ascend64.yaml)，执行配置见 [S2 infra](S2_INFRA_20260910.md)。

## 评测与验证

主表报告理解、生成、文本保留及训练／推理成本。B+SigLIP 相对 B 的变化是直接比较；S2-dual 相对 single 的变化用于参照。

S2 评分使用 AR shift，图像候选整图可见，order MC=1。生成使用 CFG2 / Heun10；每图 20 次速度估计，条件与无条件合计 40 次 backbone/head 前向。

16×910B 开发机已完成两个 S2 的真实 batch 训练、四份 flow 采样、完整 checkpoint、raw/EMA 重载和图文生成。结果在 [launch-20260909](../output/experiments/showo2-unified/launch-20260909/)；infra 续训验证在 [infra-20260910](../output/experiments/showo2-unified/infra-20260910/)。
