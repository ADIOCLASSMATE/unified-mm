# B + S2-single 调制

检验 S2-single 的条件注入方式能否改善 B。改动前源码为 `a5e0396`；实验标识为 `b-s2-modulation`。

## 改动

设 `P` 为 B 的 latent 投影，`C` 为 backbone 条件投影，`T` 为时间嵌入。query 使用 backbone XT 隐状态，content 使用 backbone X0 隐状态。

| 路径 | B | B + S2-single 调制 |
| --- | --- | --- |
| Query 输入 | `P(xt)` | `P(xt) + C(h_xt)` |
| Query AdaLN | `T(t) + C(h_xt)` | `T(t)` |
| Content 输入 | `P(x0)` | `P(x0) + C(h_x0)` |
| Content AdaLN | `T(1) + C(h_x0)` | `T(1)` |

Backbone 和 flow head 都保留双流。B 的 query 严格可见性 `sigma_j < sigma_i`、content 可见性 `sigma_j <= sigma_i`、随机图像顺序、逐 token 的 t、RF4、head 的 attention/MLP/2D RoPE 和初始化保持原样。新增模型配置为 `image_flow_conditioning_mode: s2_input`；旧 checkpoint 缺失该字段时明确使用 `adaln`。

Head 参数保持 **164,072,976**，与 B 完全相同。数据、任务混合、学习率、优化器、全模型训练、EMA、64 卡、95,415 updates、Heun10、`image_input_noise_strength=0.01` 均取自 B 正式配置。每 token 的 Heun10 执行 20 次速度求值，完整图像仍按 B 的缓存推理顺序生成。

## 入口与验证

- [配置](../configs/selfless/unified_b_s2_modulation_100b_ascend64.yaml)
- [对照配置审计](../utils/b_s2_modulation_protocol.py)
- 正式入口：`bash script/selfless/pretraining_b_s2_modulation_ascend64.sh`
- 16 卡完整验收：在固定 `dev-wjx-ascend` 上加 `--smoke-suite --validation --label <unique-label> --output-dir <report-directory>`。训练到 5 步，在 2/4 步执行生产验证流程，恢复到 6 步，并重载 raw/EMA 权重生成图像、caption 和文本。

训练输出：`output/unified-b-s2-modulation-0p6b-100b-imagenet-split-s42-r1/`。

每 10,000 updates 验证时额外生成 16 张固定 prompt 和初始噪声的 EMA 图像，CFG 3.5、Heun10、`spatial_halton`、256 个 latent。单图、`overview.png`、`index.html` 与 `summary.json` 位于：

```text
output/evaluation/training-validation/
  unified-b-s2-modulation-0p6b-100b-imagenet-split-s42-r1/
  validation_generation/step-<step>/
```

Full HF checkpoint、训练恢复配置与 adapter 都保留调制方式；不同调制方式的 head cache 和 adapter 禁止混用。
