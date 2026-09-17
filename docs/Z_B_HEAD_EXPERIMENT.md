# Z + B flow head（单流）

> 当前入口已切换至 31,800-step（约 33B）配方：warmup 199、decay 7950、每 3180 步验证，随机序语言建模项目，基座 step 0 初始化。新输出目录使用 `33b`；下文 100B 配置及启动记录为原配方历史。详见[当前训练设置](TRAINING.md#当前四项短预算消融)。

在 Z 的双流 backbone、整图共享 sigma 和共享 t 上，使用 B 的完整 flow-head 模块，并让 head 以单流方式双向读取全部 256 个 noisy token。改动前源码为 `ec20ff1`。

## Head

```text
x = B.input_proj(x_t)
c = B.cond_embed(h) + B.time_embed(t * 1000)
for block in B.blocks:
    K, V = block.prepare_cross_cache(x)   # 当前层的同一个 noisy stream
    x = block(x, c, K, V, full_attention)
velocity = B.final_layer(x, c)
```

复用 B 的全部参数与初始化：8 层、宽度 1280、8 个 attention heads、LayerNorm、独立 Q/K/V/out 投影、MLP ratio 1、SiLU、row/column 2D RoPE、6 路 AdaLN 调制和输出层 AdaLN。与 B 相同，attention 的 Q 使用调制后的 Q LayerNorm，K/V 使用 KV LayerNorm，MLP 使用自己的 AdaLN 和残差门。Head 参数量保持 **164,072,976**。

每层只更新一个 hidden state；K/V 从该层更新前的同一 noisy state 构造，并在每次速度求值时重算。Head 不读取目标图像的 clean latent。所有 noisy token 双向互相读取，条件 `h` 保持为 backbone 一次前向的输出。

## Z 的训练与生成设置

- Backbone 保持双流；context 图像双向可见，目标 mask query 不读取自身图像 clean content。
- 同图 sigma 相同，不采样图像生成顺序。
- 每图一个共享 t，RF4；时间只进入 head。
- 每个生成批次 backbone 前向 1 次，Heun10 对应 head 前向 20 次，CFG 两分支合批。
- 输入微噪声 0.01、训练数据、任务混合、学习率、EMA、64 卡与 95,415 updates 沿用 Z。
- 每 10,000 updates 验证时使用当前 raw 权重生成 16 张固定 prompt 和固定噪声的图像，CFG 3.5、Heun10；复用一次 backbone 前向产生的固定条件。下游评分仍使用 EMA。

Checkpoint 通过 `joint_dit_head_type: b_single_stream` 记录 head。旧 Z checkpoint 缺少此字段时，明确恢复原 S2 head。权重来源决定 head 类型。

## 运行

[配置](../configs/selfless/unified_z_b_head_100b_ascend64.yaml) · [实现](../models/modeling_model/image_flow_loss_joint_b.py) · [配置审计](../utils/joint_b_protocol.py)

正式入口：

```bash
bash script/selfless/pretraining_z_b_head_ascend64.sh
```

在固定 16 卡 `dev-wjx-ascend` 上执行完整验收：

```bash
bash script/selfless/pretraining_z_b_head_ascend64.sh \
  --smoke-suite --validation --label <unique-label> --output-dir <report-directory>
```

验收包括第 2/4 步的完整验证和每轮 16 张 raw 图像、从 checkpoint 5 恢复到 6、raw/EMA 重载，以及图像、caption、文本生成。

训练输出为 `output/unified-z-b-head-0p6b-100b-imagenet-split-s42-r1/`。验证画廊位于 `output/evaluation/training-validation/<run>/validation_generation/step-<step>/index.html`，同目录保留单图、`overview.png` 和 `summary.json`。
