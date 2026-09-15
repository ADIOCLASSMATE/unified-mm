# 实验定义

正式 A/B 对应持久 ID `a_x0` / `b_x0`，B 为基线。展示名称与分组由 [experiment_registry.json](../configs/protocols/experiment_registry.json)维护；训练目录和 checkpoint 保留原名称。

## 架构

双流设计统一目标位置上的预测，允许文本与图像采用各自的生成次序，并将 backbone 的跨模态上下文建模与 flow head 的图像时间步迭代分开，动机见[双流架构的设计动机](DUAL_STREAM_DESIGN_MOTIVATION.md)。

A/B 从 Qwen3-0.6B-Base 初始化，使用共享参数的 X0/content 与 XT/query 两流，以及 8 层、宽 1280 的 contextual flow head。Backbone 的 XT 输入为 learned mask，flow head 接收 noisy latent；flow query/content 分别由 backbone XT/X0 hidden 调制。

| 设置 | A | B |
| --- | --- | --- |
| Backbone / head 的 query 可见性 | `sigma_kv < sigma_q` | `sigma_kv < sigma_q` |
| Backbone / head 的 content 可见性 | `sigma_kv < sigma_q` | `sigma_kv <= sigma_q` |
| Attention contract | `selfless_strict` | `xlnet_content_diagonal` |
| 图像训练顺序 | random | random |

文本与 caption 均按从左到右训练。图像为 256 个 16 维 KL16 latent token；位置编码使用 row/column 2D RoPE，backbone 实际基数为 10,000。Attention output gate 默认 `none`。

| B 上的消融 | 改动 |
| --- | --- |
| C · 文本 AR | 文本使用单流 next-token 预测，图像路径沿用 B |
| D · Dynamic-XT | Backbone query 接收 `x_t,t`，每次 ODE 速度计算刷新条件 |
| E · sequential | 图像训练与原生生成均用顺序排列 |
| F · position-wise head | 参数量匹配的逐位置 AdaLN MLP 替代 contextual head |
| depth16 / depth30 | Flow head 深度由 8 增至 16 / 30 |

A/B 与 C–F 使用 [统一训练协议](../configs/protocols/unified_ablation_100b_ascend64.yaml)，深度扩展使用 [B scaling 协议](../configs/protocols/unified_b_x0_flow_head_scaling_100b_ascend64.yaml)与 [F scaling 协议](../configs/protocols/unified_f_on_b_flow_head_scaling_100b_ascend64.yaml)。新训练均从基座 step 0 开始；数据、预算、学习率和产物见 [训练](TRAINING.md)。

F 的逐位置 head 为共享 AdaLN 残差 MLP，不含跨 token attention。两档扩展沿用 B 的完整训练配方，使用 flow activation checkpointing；参数预算逐档匹配 B，容差为 0.5%。

| 档位 | F 深度 × 宽度 | F head 参数 | B head 参数 | 相对差异 |
| --- | --- | --- | --- | --- |
| 基础档 | 8 × 1936 | 163,828,208 | 164,072,976 | −0.149% |
| depth16 | 16 × 1960 | 321,655,616 | 321,543,696 | +0.035% |
| depth30 | 30 × 1968 | 595,579,792 | 597,117,456 | −0.258% |

## 训练数据消融

| 实验 | 数据与预算 | 配置 |
| --- | --- | --- |
| I2T-only | B 的 I2T 数据与曝光；95,415 updates | [I2T](../configs/selfless/unified_b_i2t_only_matched_ascend16.yaml) |
| T2I-only | B 的 T2I 数据与曝光；95,415 updates，RF4 | [T2I](../configs/selfless/unified_b_t2i_only_matched_ascend16.yaml) |
| text-only | ClimbMix，100B 物理文本位置；95,368 updates | [text](../configs/selfless/unified_single_text_0p6b_100b_ascend16.yaml) |
| T2I + I2T | 无 ClimbMix；两项图文任务各对齐 B 曝光，95,415 updates | [joint](../configs/selfless/unified_b_t2i_i2t_matched_ascend32.yaml) |

两项 image-only 使用 16 卡、每卡 batch 32、GA2、全局 1,024 图；每任务累计 50,024,939,520 个物理位置。协议为 [image-only matched](../configs/protocols/unified_b_image_only_matched_ascend16.yaml)。

T2I + I2T 使用 32 卡、每卡 batch 32、GA2，交替执行两个任务，完全不加载 ClimbMix；从 Qwen3-0.6B-Base step 0 初始化。每次更新每个任务各 1,024 图，合计 100,049,879,040 个物理位置。沿用 only 的有效任务均值约定，GA2 下每个任务的 loss 占比为 B 的 GA4 下的两倍；此对照匹配各任务数据曝光与优化步数，不匹配 B 的总计算量或任务梯度占比。详见 [双任务协议](../configs/protocols/unified_b_image_joint_matched_ascend32.yaml)。

## Z：整图联合去噪

[Z + B 单流 head](Z_B_HEAD_EXPERIMENT.md) 使用 B 的完整 head 模块，单流双向去噪 256 个 latent，backbone 条件和共享时间共同进入 AdaLN；其余配置沿用 Z。

相关独立对照：[B + S2-single 调制](B_S2_MODULATION.md)，保留 B 的双流和随机序，只改变 backbone 条件注入的位置，参数量与 B 完全一致。

[Z](Z_EXPERIMENT.md) 保留 B 的双流 backbone，将同一图像的 sigma 设为相同值；一次 backbone 生成固定条件，单流双向 DiT 在共享 t 下联合去噪全部 256 latent，默认 Heun10。数据和训练预算对齐 B，每轮验证额外生成 16 张固定 prompt 和种子的 EMA 图像，checkpoint 使用独立身份。

## S2 与 SigLIP

目标是检验额外语义／细节双路径在本项目架构和训练范式下是否必要。

| 组 | 视觉前端 | 图像建模 |
| --- | --- | --- |
| B | KL16 projector | 随机序逐 latent flow |
| B+SigLIP | SigLIP 语义分支 + projector + fusion | B 路径，语义层使用 sigma 因果可见性 |
| S2-single | KL16 projector | 图像块 full attention，整图 flow |
| S2-dual-siglip | SigLIP 语义分支 + projector + fusion | 与 S2-single 相同 |

当前 S2 与 B+SigLIP 配置采用 B 的 unified 数据和 95,415-step 预算，从 step 0 全参数训练，语义分支 LR=2e-6。Show-o2 原流程包含语义预蒸馏、Stage-1 冻结和 Stage-2 解冻；当前变体省略前两步。

详细配置和原流程见 [S2](SHOWO2_UNIFIED_ABLATION_DESIGN.md)、[B+SigLIP](B_SIGLIP_UNIFIED_ABLATION.md)，性能设置见 [S2 infra](S2_INFRA_20260910.md)。

## 历史身份

| ID | 方法 |
| --- | --- |
| `a_legacy` | 严格 attention，flow query/content 共用 XT 条件 |
| `b_flowdiag` | Backbone/head content 含对角线，flow 共用 XT 条件 |
| `b_no_flowdiag` | Backbone content 含对角线、head content 严格，flow 共用 XT 条件 |
| `c_on_a_legacy` | 历史 A 加文本 AR |
| `a_caption_only` | 历史 A 的 I2T-only |
| `a_text_only` / `b_text_only` | 已完成的 ClimbMix 单任务对照 |

早期 ImageNet 配方见 [历史实验](ABLATION_CONCLUSIONS.md)。结果选择由 [evaluation_report.json](../configs/protocols/evaluation_report.json)维护，首页使用各模型对应的 final EMA；D 使用训练 r4、修正后的完整评测 r2。
