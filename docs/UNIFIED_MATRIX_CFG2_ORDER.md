# Unified：CFG2 / Heun10 生成矩阵

9 个模型：正式 A/B、C–F、历史 shared-condition A、历史 B 含/不含 flow 对角线。各自使用 step95415 final EMA，D 使用训练 r4。身份见 [实验定义](EXPERIMENTS.md)，结果选择见 `configs/protocols/evaluation_report.json`。

每个模型测 Halton、confidence_stability、random；E 另测原生 sequential，共 28 组完整 50K 评测。主表中 E 使用 sequential。

## 固定条件

| 项目 | 设置 |
| --- | --- |
| 数据 | ImageNet-val 50000，每图一条合成 T2I prompt |
| 采样 | CFG2，constant，Heun10，温度1，逐位置 reveal |
| 噪声 | seed42，CPU FP32，按全局图像索引和空间位置配对 |
| 运行 | 16 NPUs，全局 batch4096，每 rank256，padding512 |
| 精度 | BF16 模型、FP32 VAE / ODE；KL16 scale0.2325 |
| FID / IS | 项目 ImageNet-val reference；十个 synset 分层 IS split |
| 图片 | 每组固定64张，共1792张；保存身份与顺序记录 |

CFG2 来自 B 的验证集选择，其他模型共享该配方。IS split 标准差描述样本分组波动。

## 顺序和架构

confidence_stability 沿用 [B 顺序实验](B_ORDER_SWEEP.md)：每16个 Halton 候选，在 t=0 和 Euler dt=0.1 计算 `mean((v_next-v)^2)/(mean(v^2)+1e-8)`，低分优先；`v=vu+2*(vc-vu)`。实际生成从原始噪声开始。

| Model family | Probe and cache behavior |
|---|---|
| A/B and B-based C/E | Static XT query condition; pending flow content uses the previous backbone X0 hidden. A retains strict content attention. |
| Legacy A and both legacy B controls | Static XT query condition; pending flow content retains the previous static query condition, as trained. |
| D Dynamic-XT | Conditional/unconditional caches remain separate. Both velocity probes refresh the XT hidden from the current x and t. Pending content uses the completed X0 hidden exactly once. |
| F position-wise | Probe velocities use the independent per-position MLP. The backbone remains cached; there is no flow content stream/cache. |

random 使用 `42 + batch_idx * 1009 + rank * 1000003`；各模型保持相同 batch/rank 划分。三个策略都保存真实生成顺序。

## 运行和检查

准备入口为 `scripts/prepare_unified_matrix_sweep.py --include-random`，每组独立 16-NPU Job，平台资源遵循 [INSPIRE](../INSPIRE.md)。本次矩阵关闭 runtime hashing 和 W&B。

CPU 检查覆盖36位置、各架构的 cache/速度刷新和 RNG；16-NPU smoke 检查完整256次 reveal、正式 batch 的第二候选块、权重和顺序重放。B 的正式结果与前一轮相同配方的指标及64张图逐项核对。

根目录：`output/evaluation/ablation-matrix/cfg2-heun10-order-20260909-r1/`。保留28组指标、CSV、PNG/PDF、配对图片和检查结果；`scripts/report_sampling_sweep.py` 更新[评测首页](../output/evaluation/index.html)。
