# B：生成顺序比较

固定 [CFG/Heun 扫描](B_SAMPLING_SWEEP.md) 的最低 FID 配方 CFG2 / Heun10，使用 B step95415 final EMA。`cfg-refinement.json` 保存 CFG 1/1.5/2/2.5/3 的邻域复核。

## 固定条件

ImageNet-val 50K、seed42、canonical 图像/位置噪声、BF16 模型、FP32 VAE 与 ODE、温度1、逐位置生成、backbone cache。每组 16 NPUs，全局 batch4096；IS 用十个 synset 分层 split。调参与报告共用该验证集。

## 顺序

基础策略为 `spatial_halton`、`sequential`、`spatial_uniform`（现实现为中心向外/checker 顺序）和 `random`。random 使用 `42 + batch_idx * 1009 + rank * 1000003`；比较保持相同 ranks 和 batch 划分。

四个附加策略以 Halton 的连续 16 个位置为候选块。每块开始时，只根据 caption 和已经生成的内容评分；候选 query 彼此不可见，随后按分数逐个生成。

令 vc / vu 为 t=0、该位置原始噪声 x 的条件/无条件速度，`v = vu + cfg*(vc-vu)`，均值沿 latent 通道计算，评分使用 FP32。

| Policy | Score / ordering |
|---|---|
| `confidence_cfg` | Ascending `mean((vc-vu)^2) / (0.5*mean(vc^2+vu^2)+1e-8)` |
| `confidence_cfg_reverse` | The same score, descending; tests the opposite direction of guidance sensitivity |
| `confidence_stability` | Ascending `mean((v_next-v)^2)/(mean(v^2)+1e-8)`, where `x_next=x+0.1*v`, `t_next=0.1`, and `v_next` uses the same generated context |
| `confidence_halton` | Both velocity probes are computed, but Halton order is retained; numerical/cache and cost control |

平分保持 Halton 原顺序。上述分数分别测量 guidance 敏感度和局部 ODE 稳定性。

## Cache 与复现

探测只提交上一个已完成的 Content 一次，使用其 X0 条件；候选 query 和 Euler proposal 不进入 Content cache。实际生成从原噪声重新求条件，探测不消耗 RNG。保留 64 个相同样本的 PNG、prompt、噪声及 `order_trace`。

CPU 对照将记录顺序交给独立 decode，检查两个以上块边界、cache 提交、RNG 和训练 sigma 独立性；16-NPU smoke 在 `dev-wjx-ascend` 上执行。

```bash
python3 scripts/prepare_unified_order_sweep.py   --previous-sweep <completed> --output-dir <new>
bash <new>/launch/smoke.sh
python3 <new>/launch/submit_unified_t2i_sweep.py   --output-dir <new> --name-prefix <unique>
python3 scripts/report_sampling_sweep.py --output-dir <new> --watch
```

复用 [采样产物结构](B_SAMPLING_SWEEP.md)。相关方法：[MaskGIT](https://arxiv.org/html/2202.04200v1)、[Halton Scheduler](https://arxiv.org/abs/2503.17076)。
