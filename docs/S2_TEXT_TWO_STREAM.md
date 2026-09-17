# S2-single 文本双流消融

> 当前入口已切换至 31,800-step（约 33B）配方：warmup 199、decay 7950、每 3180 步验证，随机序语言建模项目，基座 step 0 初始化。新输出目录使用 `33b`；下文 100B 配置及启动记录为原配方历史。详见[当前训练设置](TRAINING.md#当前四项短预算消融)。

只把 S2-single backbone 的文本预测改为 content/query 双流，消除文本的一位 shift。图像前端、noisy-image backbone content 路径和完整 S2 flow head 保持原实现。

| 部分 | S2-single | S2-single + text two-stream |
| --- | --- | --- |
| 文本目标 i | content hidden[i−1] → token[i] | query hidden[i] → token[i] |
| 文本 query 输入 | 无 | 已有的 learned text mask embedding |
| Content 可见性 | 文本因果含自身，图像块双向 | 相同 |
| 文本 query 可见性 | 无 | 仅物理位置 j < i 的 content，同一文档内 |
| 图像速度输入 | noisy-image content hidden | 相同；不构造图像 query |
| Flow head | S2 的 8 层、宽 1280、单流 omni attention、时间 AdaLN | 同一模块，163,295,760 参数 |
| ODE / flow 训练 | 每图共享 t，四份 MC，Heun10，每次刷新 backbone/head | 相同 |

两流共享全部 backbone 层参数，query 不提供 K/V；query 的 residual 路径只含 mask embedding，不含目标 token。前序图像的 content 保留整图可见性。padding、packed segment 和 CFG 限制均保留。图像位置的 query attention 行为空，其结果不参与预测；flow 前向完全不构造 query stream。

文本 CE 在目标位置计算。仍排除每个 packed segment 的首 token，并使用原标签掩码、loss 权重和分母，保持原 S2 的目标曝光量。推理时在待生成位置追加 mask query，从该位置预测，再将生成 token 写入 content；图像生成继续走原 S2 flow 路径。

## 配置与训练

配置为 [unified_s2_single_text_two_stream_100b_ascend64.yaml](../configs/selfless/unified_s2_single_text_two_stream_100b_ascend64.yaml)，实验 ID 为 `s2_single_text_two_stream`。

相对 S2-single，模型配置只改变 `dual_stream_attention_contract: showo2_text_two_stream`。`flow_head_attention_contract: showo2_omni_attention` 和 `flow_condition_contract: backbone_noisy_image_hidden` 保持不变。不增加参数；从 Qwen3-0.6B-Base 和相同的 image/head 初始化开始训练，不从 S2-single 已训练 checkpoint 续训。

训练预算为 64×910B、GA4、95,415 updates、约 100B 名义文本目标；每卡 text/T2I/I2T batch 为 4/16/16。数据、任务顺序、optimizer、scheduler、seed、RF4、EMA 和 checkpoint 节奏与原 S2-single 相同，详见 [S2 协议](SHOWO2_UNIFIED_ABLATION_DESIGN.md)及 [infra](S2_INFRA_20260910.md)。所属项目与路径见 [Inspire](../INSPIRE.md)。

```bash
bash script/selfless/pretraining_s2_text_two_stream_ascend64.sh
```

输出为 `output/unified-s2-single-text-two-stream-0p6b-100b-imagenet-split-s42-r1/`。正式任务使用平台提供的四节点 PET 环境；单节点验收入口：

```bash
bash script/selfless/pretraining_s2_text_two_stream_ascend64.sh \
  --smoke-suite --label 20260916-r1 \
  --output-dir output/experiments/s2-text-two-stream/20260916-r1/smoke
```

验收保持正式每卡 batch，训练 12 步后从完整 checkpoint 恢复到 14 步，检查 optimizer/data/EMA 恢复、raw/EMA 重载、NPU 文本梯度、目标隔离、完整 256-latent Heun10 生成、文本及 I2T 生成。结果与冻结源码保存在 `output/experiments/s2-text-two-stream/20260916-r1/`。

## 评分与验证

评分标识为 `showo2_text_query_same_position_v1`，通过 checkpoint 的 attention contract 自动恢复。理解任务仍确定性地读取完整前序图像，image-order MC=1；文本不再做 hidden shift。图像生成保留无 KV cache 的全模型刷新路径。

[CPU 测试](../tests/test_s2_text_two_stream.py)覆盖三层目标／未来文本隔离、图像条件、packed document、padding、同位置 CE、生成与 teacher-forcing 一致性、评分和 checkpoint 身份。固定全部参数、噪声和时间时，消融模型的 flow loss 和每个参数的 flow 梯度应与 S2-single 逐位一致。

性能及质量结论需待正式训练与相同协议的 final FP32 EMA 评测完成后填写。

## 2026-09-16 验收与启动

- `bash script/check_repo.sh`：1,078 passed、2 skipped；lint 通过。
- 固定 16 卡开发机：12→14 步完整续训通过，恢复后的 8 个 microbatch loss 均有限；末步训练 loss 为 0.652972。
- Raw 与 FP32 EMA 导出重载后，NPU 检查均确认文本没有目标／未来泄漏、生成使用同位置 query、固定权重和输入的 flow 输出与 S2-single 逐位一致。
- 两套权重均完成 256-latent、CFG2、Heun10 图像生成及 VAE 解码；每幅图 40 次 backbone/head 调用，文本与 I2T 的 prediction offset 均为 0。
- 单节点 smoke 第 3–12 步耗时中位数为 3.433 秒/update。此数值只描述开发机验收，不作为配对性能或质量结论。

正式 Job 为 `umm-s2-text2stream-64-0916-r1`，四个 16 卡实例运行，从 step 0 开始执行 95,415-step 配方。开发机已停止。任务详情、源码、CPU 日志及 NPU 报告见 [实验记录](../output/experiments/s2-text-two-stream/20260916-r1/experiment.json)和 [完整验收报告](../output/experiments/s2-text-two-stream/20260916-r1/smoke/report.json)。

启动验收于 2026-09-16 07:19 UTC 观察到 step 20，loss 为 0.625840，最近记录耗时 3.510 秒/update；全部三项任务 loss 有限。该记录仅确认正式训练正常推进，完整训练和效果对比尚未完成。
