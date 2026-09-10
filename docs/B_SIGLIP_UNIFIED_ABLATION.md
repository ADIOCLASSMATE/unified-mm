# B + SigLIP

在 B 的 X0 图像输入加入语义分支，比较其对理解、生成、文本保留及计算成本的影响。实验关系与 Show-o2 训练阶段见 [S2 / SigLIP 对照](SHOWO2_UNIFIED_ABLATION_DESIGN.md)。

| 项目 | 当前实现 |
| --- | --- |
| 初始化 | Qwen3-0.6B-Base，step 0 |
| 语义分支 | SigLIP 前 26 层，宽 1152、16 heads、MLP 4304 |
| 输入 | 同一 KL16 latent 分别进入 B projector 与新语义投影 |
| 融合 | concat → RMSNorm → MLP，写入 X0/content 图像位置 |
| 可见性 | 语义层只读取同图且 `sigma_k <= sigma_j` 的位置 |
| XT / head | B learned-mask query 与 8 层 contextual flow head |
| 额外参数 | 400,367,520；预训练语义参数 397,066,912 |
| 学习率 | semantic 2e-6；新输入／fusion 5e-5 |
| 当前训练阶段 | step0 全参数训练，VAE 冻结，尚无预蒸馏 |

共有 B 参数在新增模块前初始化；语义模块使用独立初始化 seed。语义 KV cache 保存已生成内容，生成、完整重算和候选似然共用可见性规则。

## 运行

```bash
bash script/selfless/pretraining_unified_b_siglip_ascend64.sh
```

[配置](../configs/selfless/unified_b_siglip_100b_ascend64.yaml)采用 [B 训练配方](TRAINING.md)：64 卡、GA4、每卡 batch 4/16/4/16、RF4、95,415 updates。项目归属为随机序语言建模，输出为 `output/unified-b-siglip-0p6b-100b-imagenet-split-s42-r1`。

## Infra

16×910B、每段 8 updates、去除前 2 次，以最慢 rank 的中位时间计：

| 语义 checkpoint | MC content | 秒/update | 峰值 GiB |
| --- | --- | ---: | ---: |
| 开 | 重复四份 | 2.881 | 42.85 |
| 关 | 重复四份 | 2.866 | 46.49 |
| 开 | 共享一份 | 2.786 | 37.05 |
| 关 | 共享一份 | 2.728；复测 2.736 | 40.69 |

当前选择关闭语义 checkpoint、共享 content；backbone / semantic 每 update 分别调用 4 / 2 次，四份 query 在 head 中合批一次。NPU BF16 对照 loss 相对误差 9.67e-6，最坏单参数梯度相对 L2 误差 1.88%。完整 6→8 step 续训、data/RNG/EMA 恢复及 raw/EMA 图文生成通过。

配置、profiler、提交和验证记录：[launch-20260910](../output/experiments/unified-b-siglip/launch-20260910/)。
